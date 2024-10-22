// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/duration.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/Time.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/io.h>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/registrations.hpp>
#include <hdl_graph_slam/ScanMatchingStatus.h>

#include <deque>

namespace hdl_graph_slam {

class ScanMatchingOdometryNodelet : public nodelet::Nodelet {
public:
  typedef pcl::PointXYZ PointT;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanMatchingOdometryNodelet() {}
  virtual ~ScanMatchingOdometryNodelet() {}

  virtual void onInit() {
    NODELET_DEBUG("initializing scan_matching_odometry_nodelet...");
    nh = getNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    if(private_nh.param<bool>("enable_imu_frontend", false)) {
      msf_pose_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/msf_core/pose", 1, boost::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, _1, false));
      msf_pose_after_update_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/msf_core/pose_after_update", 1, boost::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, _1, true));
    }

    points_sub = nh.subscribe("/filtered_points", 256, &ScanMatchingOdometryNodelet::cloud_callback, this);
    read_until_pub = nh.advertise<std_msgs::Header>("/scan_matching_odometry/read_until", 32);
    odom_pub = nh.advertise<nav_msgs::Odometry>(published_odom_topic, 32);
    trans_pub = nh.advertise<geometry_msgs::TransformStamped>("/scan_matching_odometry/transform", 32);
    status_pub = private_nh.advertise<ScanMatchingStatus>("/scan_matching_odometry/status", 8);
    aligned_points_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 32);
    submap_pub = nh.advertise<sensor_msgs::PointCloud2>("/submap", 32);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    auto& pnh = private_nh;
    published_odom_topic = private_nh.param<std::string>("published_odom_topic", "/odom");
    points_topic = pnh.param<std::string>("points_topic", "/velodyne_points");
    odom_frame_id = pnh.param<std::string>("odom_frame_id", "odom");
    robot_odom_frame_id = pnh.param<std::string>("robot_odom_frame_id", "robot_odom");

    // The minimum tranlational distance and rotation angle between keyframes.
    // If this value is zero, frames are always compared with the previous frame
    keyframe_delta_trans = pnh.param<double>("keyframe_delta_trans", 0.25);
    keyframe_delta_angle = pnh.param<double>("keyframe_delta_angle", 0.15);
    keyframe_delta_time = pnh.param<double>("keyframe_delta_time", 1.0);

    // Registration validation by thresholding
    transform_thresholding = pnh.param<bool>("transform_thresholding", false);
    max_acceptable_trans = pnh.param<double>("max_acceptable_trans", 1.0);
    max_acceptable_angle = pnh.param<double>("max_acceptable_angle", 1.0);

    //submap size
    max_submaps_queue_size_ = pnh.param<double>("max_submaps_queue_size", 10.0);
    use_submap_ = pnh.param<bool>("use_submap", false);
    key_frames_path_ = pnh.param<std::string>("key_frames_path", "/home/dreame/catkin_ws/src/hdl_graph_slam/key_frames");


    // select a downsample method (VOXELGRID, APPROX_VOXELGRID, NONE)
    std::string downsample_method = pnh.param<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = pnh.param<double>("downsample_resolution", 0.1);
    if(downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      auto voxelgrid = new pcl::VoxelGrid<PointT>();
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter.reset(voxelgrid);
    } else if(downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
      pcl::PassThrough<PointT>::Ptr passthrough(new pcl::PassThrough<PointT>());
      downsample_filter = passthrough;
    }

    registration = select_registration_method(pnh);
    // registration_test = select_registration_method(pnh);

    local_submaps.reset(new pcl::PointCloud<PointT>());
  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    if(!ros::ok()) {
      return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);


    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    if (use_submap_)
    {
      // auto t1 = std::chrono::high_resolution_clock::now();
      pose = matching_local_map(cloud_msg->header.stamp, cloud);
      // auto t2 = std::chrono::high_resolution_clock::now();
      // ROS_WARN("Matching time: %f ms", std::chrono::duration<double>(t2 - t1).count() * 1000);
    }else{
      pose = matching(cloud_msg->header.stamp, cloud);
    }


    publish_odometry(cloud_msg->header.stamp, cloud_msg->header.frame_id, pose);

    // In offline estimation, point clouds until the published time will be supplied
    std_msgs::HeaderPtr read_until(new std_msgs::Header());
    read_until->frame_id = points_topic;
    read_until->stamp = cloud_msg->header.stamp + ros::Duration(1, 0);
    read_until_pub.publish(read_until);

    read_until->frame_id = "/filtered_points";
    read_until_pub.publish(read_until);
  }

  void msf_pose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg, bool after_update) {
    if(after_update) {
      msf_pose_after_update = pose_msg;
    } else {
      msf_pose = pose_msg;
    }
  }

  /**
   * @brief downsample a point cloud
   * @param cloud  input cloud
   * @return downsampled point cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);

    return filtered;
  }

  /**
   * @brief estimate the relative pose between an input cloud and a keyframe cloud
   * @param stamp  the timestamp of the input cloud
   * @param cloud  the input cloud
   * @return the relative pose between the input cloud and the keyframe cloud
   */
  Eigen::Matrix4f matching(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    if(!keyframe) {
      prev_time = ros::Time();
      prev_trans.setIdentity();
      keyframe_pose.setIdentity();
      keyframe_stamp = stamp;
      keyframe = downsample(cloud);
      registration->setInputTarget(keyframe);
      return Eigen::Matrix4f::Identity();
    }

    auto filtered = downsample(cloud);
    registration->setInputSource(filtered);

    std::string msf_source;
    Eigen::Isometry3f msf_delta = Eigen::Isometry3f::Identity();

    if(private_nh.param<bool>("enable_imu_frontend", false)) {
      if(msf_pose && msf_pose->header.stamp > keyframe_stamp && msf_pose_after_update && msf_pose_after_update->header.stamp > keyframe_stamp) {
        Eigen::Isometry3d pose0 = pose2isometry(msf_pose_after_update->pose.pose);
        Eigen::Isometry3d pose1 = pose2isometry(msf_pose->pose.pose);
        Eigen::Isometry3d delta = pose0.inverse() * pose1;

        msf_source = "imu";
        msf_delta = delta.cast<float>();
      } else {
        std::cerr << "msf data is too old" << std::endl;
      }
    } else if(private_nh.param<bool>("enable_robot_odometry_init_guess", false) && !prev_time.isZero()) {
      tf::StampedTransform transform;
      if(tf_listener.waitForTransform(cloud->header.frame_id, stamp, cloud->header.frame_id, prev_time, robot_odom_frame_id, ros::Duration(0))) {
        tf_listener.lookupTransform(cloud->header.frame_id, stamp, cloud->header.frame_id, prev_time, robot_odom_frame_id, transform);
      } else if(tf_listener.waitForTransform(cloud->header.frame_id, ros::Time(0), cloud->header.frame_id, prev_time, robot_odom_frame_id, ros::Duration(0))) {
        tf_listener.lookupTransform(cloud->header.frame_id, ros::Time(0), cloud->header.frame_id, prev_time, robot_odom_frame_id, transform);
      }

      if(transform.stamp_.isZero()) {
        NODELET_WARN_STREAM("failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        msf_source = "odometry";
        msf_delta = tf2isometry(transform).cast<float>();
        ROS_WARN("msf_delta [%f %f %f %f]", msf_delta(0,3), msf_delta(1,3), msf_delta(2,3), msf_delta(3,3));
      }
    }

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    registration->align(*aligned, prev_trans * msf_delta.matrix());

    publish_scan_matching_status(stamp, cloud->header.frame_id, aligned, msf_source, msf_delta);

    if(!registration->hasConverged()) {
      NODELET_INFO_STREAM("scan matching has not converged!!");
      NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
      return keyframe_pose * prev_trans;
    }
    
    Eigen::Matrix4f trans = registration->getFinalTransformation();
    Eigen::Matrix4f odom = keyframe_pose * trans;

    if(transform_thresholding) {
      Eigen::Matrix4f delta = prev_trans.inverse() * trans;
      double dx = delta.block<3, 1>(0, 3).norm();
      double da = std::acos(Eigen::Quaternionf(delta.block<3, 3>(0, 0)).w());

      if(dx > max_acceptable_trans || da > max_acceptable_angle) {
        NODELET_INFO_STREAM("too large transform!!  " << dx << "[m] " << da << "[rad]");
        NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
        return keyframe_pose * prev_trans;
      }
    }

    prev_time = stamp;
    prev_trans = trans;

    Eigen::Matrix4f T = prev_trans * msf_delta.matrix();
    // ROS_WARN("T [%f %f %f %f]", T(0,3), T(1,3), T(2,3), T(3,3));
    pcl::transformPointCloud (*filtered, *aligned, keyframe_pose * prev_trans);
    aligned->header.frame_id = odom_frame_id;
    aligned_points_pub.publish(*aligned);

    auto keyframe_trans = matrix2transform(stamp, keyframe_pose, odom_frame_id, "keyframe");
    keyframe_broadcaster.sendTransform(keyframe_trans);

    double delta_trans = trans.block<3, 1>(0, 3).norm();
    double delta_angle = std::acos(Eigen::Quaternionf(trans.block<3, 3>(0, 0)).w());
    double delta_time = (stamp - keyframe_stamp).toSec();
    if(delta_trans > keyframe_delta_trans || delta_angle > keyframe_delta_angle || delta_time > keyframe_delta_time) {
      keyframe = filtered;
      registration->setInputTarget(keyframe);

      keyframe_pose = odom;
      keyframe_stamp = stamp;
      prev_time = stamp;
      prev_trans.setIdentity();
    }

    return odom;
  }

Eigen::Matrix4f matching_local_map(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    std::vector<int> indices;
    pcl::PointCloud<PointT>::Ptr valid_cloud(new pcl::PointCloud<PointT>());
    pcl::removeNaNFromPointCloud(*cloud, *valid_cloud, indices);
    keyframe_stamp = stamp;
    auto source_cloud = downsample(valid_cloud);

    //debug
    // pcl::PointCloud<PointT>::Ptr transform_filtered(new pcl::PointCloud<PointT>());
    // pcl::transformPointCloud(*filtered, *transform_filtered, keyframe_pose);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGB>());
    // source->points.resize(transform_filtered->points.size());
    // for (int i; i < transform_filtered->points.size(); i++) 
    // {
    //   source->points[i].r = 255;
    //   source->points[i].g = 0;
    //   source->points[i].b = 0;
    //   source->points[i].x = transform_filtered->points[i].x;
    //   source->points[i].y = transform_filtered->points[i].y;
    //   source->points[i].z = transform_filtered->points[i].z;
    // }
    // viewer.showCloud(source, "source");

    static Eigen::Matrix4f step_pose = Eigen::Matrix4f::Identity();
    static Eigen::Matrix4f last_pose = init_pose_;
    static Eigen::Matrix4f predict_pose = init_pose_;
    static Eigen::Matrix4f last_key_frame_pose = init_pose_;

    //first frame
    if (submaps.size() == 0) {
        keyframe_pose = init_pose_;
        update_local_map(keyframe_pose, source_cloud);
        return Eigen::Matrix4f::Identity();
    }

    //not first frame, just matching
    registration->setInputSource(source_cloud);
    pcl::PointCloud<PointT>::Ptr result_cloud(new pcl::PointCloud<PointT>());
    registration->align(*result_cloud, predict_pose);

    keyframe_pose = registration->getFinalTransformation();

    if(!registration->hasConverged()) {
      NODELET_ERROR_STREAM("scan matching has not converged!!");
      NODELET_ERROR_STREAM("ignore this frame(" << stamp << ")");
      return predict_pose;
    }

    pcl::PointCloud<PointT>::Ptr align_source(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*source_cloud, *align_source, keyframe_pose);
    // ROS_WARN("keyframe_pose [%f %f %f]", keyframe_pose(0,3), keyframe_pose(1,3), keyframe_pose(2,3));

    auto keyframe_trans = matrix2transform(stamp, keyframe_pose, odom_frame_id, "keyframe");
    keyframe_broadcaster.sendTransform(keyframe_trans);

    // auto odom = last_pose * keyframe_pose;

    //update predict pose
    step_pose = last_pose.inverse() * keyframe_pose;
    if (transform_thresholding)
    {
      double dx = step_pose.block<3, 1>(0, 3).norm();
      double da = std::acos(Eigen::Quaternionf(step_pose.block<3, 3>(0, 0)).w());

      if(dx > max_acceptable_trans || da > max_acceptable_angle) {
        NODELET_ERROR_STREAM("too large transform!!  " << dx << "[m] " << da << "[rad]");
        NODELET_ERROR_STREAM("ignore this frame(" << stamp << ")");
        //before predict_pose update wrong
        return predict_pose;
      }
    }    
    predict_pose = keyframe_pose * step_pose;
    last_pose = keyframe_pose;

    double traver_dist = fabs(last_key_frame_pose(0,3) - keyframe_pose(0,3)) + fabs(last_key_frame_pose(1,3) - keyframe_pose(1,3)) + fabs(last_key_frame_pose(2,3) - keyframe_pose(2,3));
    double delta_angle = fabs(std::acos(Eigen::Quaternionf(last_key_frame_pose.block<3, 3>(0, 0)).w()) - std::acos(Eigen::Quaternionf(keyframe_pose.block<3, 3>(0, 0)).w()));
    // ROS_WARN("traver_dist [%f]", traver_dist);
    // ROS_WARN("delta_angle [%f]", delta_angle);

    if (traver_dist > keyframe_delta_trans || delta_angle > keyframe_delta_angle) {
        update_local_map(keyframe_pose, align_source);
        last_key_frame_pose = keyframe_pose;
        // ROS_WARN("------------------");
        keyframe_nums++;
        std::string file_path = key_frames_path_ + "/key_frame_" + std::to_string(keyframe_nums) + ".pcd";
        pcl::io::savePCDFileBinary(file_path, *valid_cloud);
    }

    return keyframe_pose;
}

void update_local_map(const Eigen::Matrix4f& keyframe_pose, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    //add submap for scan matching 2023-11-27
    std::pair<Eigen::Matrix4f, pcl::PointCloud<PointT>::Ptr> keyframe_data;
    keyframe_data.first = keyframe_pose;
    keyframe_data.second.reset(new pcl::PointCloud<PointT>(*cloud));
    submaps.push_back(keyframe_data);

    if (submaps.size() > max_submaps_queue_size_)
    {
      submaps.pop_front();
    }

    local_submaps.reset(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr transform_key_frame(new pcl::PointCloud<PointT>());
    for (size_t i = 0; i < submaps.size(); i++)
    {

      // pcl::transformPointCloud (*submaps[i].second, *transform_key_frame, submaps[i].first);
      // *local_submaps += *transform_key_frame;
      *local_submaps += *submaps[i].second;
    }
    
    local_submaps->header.frame_id = odom_frame_id;
    submap_pub.publish(local_submaps);

    ROS_INFO("local_submap size %d", local_submaps->size());
    registration->setInputTarget(local_submaps);
    //debug
    // pcl::PointCloud<PointT>::ConstPtr filter_local_submaps = downsample(local_submaps);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr target(new pcl::PointCloud<pcl::PointXYZRGB>());
    // target->points.resize(local_submaps->points.size());
    // for (int i; i < local_submaps->points.size(); i++) 
    // {
    //   target->points[i].r = 0;
    //   target->points[i].g = 255;
    //   target->points[i].b = 0;
    //   target->points[i].x = local_submaps->points[i].x;
    //   target->points[i].y = local_submaps->points[i].y;
    //   target->points[i].z = local_submaps->points[i].z;
    // }
    // viewer.showCloud(target, "target");
    // ROS_INFO("local_submap size %d, filter_local_submaps size %d", filter_local_submaps->size(), filter_local_submaps->size());
    return;
}


  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const std::string& base_frame_id, const Eigen::Matrix4f& pose) {
    // publish transform stamped for IMU integration
    geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose, odom_frame_id, base_frame_id);
    trans_pub.publish(odom_trans);

    // broadcast the transform over tf
    odom_broadcaster.sendTransform(odom_trans);

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_id;

    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;

    odom.child_frame_id = base_frame_id;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    odom_pub.publish(odom);
  }

  /**
   * @brief publish scan matching status
   */
  void publish_scan_matching_status(const ros::Time& stamp, const std::string& frame_id, pcl::PointCloud<pcl::PointXYZ>::ConstPtr aligned, const std::string& msf_source, const Eigen::Isometry3f& msf_delta) {
    if(!status_pub.getNumSubscribers()) {
      return;
    }

    ScanMatchingStatus status;
    status.header.frame_id = frame_id;
    status.header.stamp = stamp;
    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();

    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i=0; i<aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();

    status.relative_pose = isometry2pose(Eigen::Isometry3f(registration->getFinalTransformation()).cast<double>());

    if(!msf_source.empty()) {
      status.prediction_labels.resize(1);
      status.prediction_labels[0].data = msf_source;

      status.prediction_errors.resize(1);
      Eigen::Isometry3f error = Eigen::Isometry3f(registration->getFinalTransformation()).inverse() * msf_delta;
      status.prediction_errors[0] = isometry2pose(error.cast<double>());
    }

    status_pub.publish(status);
  }

private:
  // ROS topics
  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  ros::Subscriber points_sub;
  ros::Subscriber msf_pose_sub;
  ros::Subscriber msf_pose_after_update_sub;

  ros::Publisher odom_pub;
  ros::Publisher trans_pub;
  ros::Publisher aligned_points_pub;
  ros::Publisher status_pub;
  ros::Publisher submap_pub;
  tf::TransformListener tf_listener;
  tf::TransformBroadcaster odom_broadcaster;
  tf::TransformBroadcaster keyframe_broadcaster;

  std::string published_odom_topic;
  std::string points_topic;
  std::string odom_frame_id;
  std::string robot_odom_frame_id;
  ros::Publisher read_until_pub;

  // keyframe parameters
  double keyframe_delta_trans;  // minimum distance between keyframes
  double keyframe_delta_angle;  //
  double keyframe_delta_time;   //

  // registration validation by thresholding
  bool transform_thresholding;  //
  double max_acceptable_trans;  //
  double max_acceptable_angle;

  int max_submaps_queue_size_;
  int keyframe_nums = 0;
  bool use_submap_;
  std::string key_frames_path_;
  Eigen::Matrix4f init_pose_ = Eigen::Matrix4f::Identity();

  // odometry calculation
  geometry_msgs::PoseWithCovarianceStampedConstPtr msf_pose;
  geometry_msgs::PoseWithCovarianceStampedConstPtr msf_pose_after_update;

  ros::Time prev_time;
  Eigen::Matrix4f prev_trans;                  // previous estimated transform from keyframe
  Eigen::Matrix4f keyframe_pose;               // keyframe pose
  ros::Time keyframe_stamp;                    // keyframe time
  pcl::PointCloud<PointT>::ConstPtr keyframe;  // keyframe point cloud
  pcl::PointCloud<PointT>::Ptr local_submaps;
  std::deque<std::pair<Eigen::Matrix4f, pcl::PointCloud<PointT>::Ptr>> submaps;
  //
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;
};

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::ScanMatchingOdometryNodelet, nodelet::Nodelet)
