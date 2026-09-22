#include <ament_index_cpp/get_package_share_directory.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <msgs/msg/april_tag_detection2_d_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr double kEpsilon = 1e-9;

struct Transform
{
  cv::Matx33d r{cv::Matx33d::eye()};
  cv::Vec3d t{0.0, 0.0, 0.0};
};

struct CameraCalibration
{
  bool valid{false};
  cv::Matx33d k{cv::Matx33d::eye()};
  cv::Mat distortion;
  int image_width{0};
  int image_height{0};
};

struct Observation
{
  int tag_id{-1};
  std::string family;
  int hamming{0};
  double decision_margin{0.0};
  double mean_edge_px{0.0};
  std::array<cv::Point2d, 4> corners{};
};

struct Sample
{
  int tag_id{-1};
  cv::Vec3d p_body;
  cv::Point2d pixel;
};

struct PoseEstimate
{
  bool valid{false};
  cv::Vec3d position{0.0, 0.0, 0.0};
  cv::Vec4d quat_xyzw{0.0, 0.0, 0.0, 1.0};
  double residual{0.0};
  int iterations{0};
  std::string reject_reason;
};

struct TagHeightStats
{
  bool valid{false};
  double min_z{0.0};
  double max_z{0.0};
  double mean_z{0.0};
};

double norm3(const cv::Vec3d & v)
{
  return std::sqrt(v.dot(v));
}

cv::Vec3d normalize3(const cv::Vec3d & v)
{
  const double n = norm3(v);
  if (n <= kEpsilon) {
    return {0.0, 0.0, 0.0};
  }
  return v * (1.0 / n);
}

double wrap_angle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

double rad_to_deg(double radians)
{
  return radians * 180.0 / M_PI;
}

cv::Matx33d rpy_to_rotation(double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);
  const cv::Matx33d rx(1, 0, 0, 0, cr, -sr, 0, sr, cr);
  const cv::Matx33d ry(cp, 0, sp, 0, 1, 0, -sp, 0, cp);
  const cv::Matx33d rz(cy, -sy, 0, sy, cy, 0, 0, 0, 1);
  return rz * ry * rx;
}

cv::Vec4d rotation_to_quat_xyzw(const cv::Matx33d & r)
{
  const double trace = r(0, 0) + r(1, 1) + r(2, 2);
  double x = 0.0, y = 0.0, z = 0.0, w = 1.0;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (r(2, 1) - r(1, 2)) / s;
    y = (r(0, 2) - r(2, 0)) / s;
    z = (r(1, 0) - r(0, 1)) / s;
  } else if (r(0, 0) > r(1, 1) && r(0, 0) > r(2, 2)) {
    const double s = std::sqrt(1.0 + r(0, 0) - r(1, 1) - r(2, 2)) * 2.0;
    w = (r(2, 1) - r(1, 2)) / s;
    x = 0.25 * s;
    y = (r(0, 1) + r(1, 0)) / s;
    z = (r(0, 2) + r(2, 0)) / s;
  } else if (r(1, 1) > r(2, 2)) {
    const double s = std::sqrt(1.0 + r(1, 1) - r(0, 0) - r(2, 2)) * 2.0;
    w = (r(0, 2) - r(2, 0)) / s;
    x = (r(0, 1) + r(1, 0)) / s;
    y = 0.25 * s;
    z = (r(1, 2) + r(2, 1)) / s;
  } else {
    const double s = std::sqrt(1.0 + r(2, 2) - r(0, 0) - r(1, 1)) * 2.0;
    w = (r(1, 0) - r(0, 1)) / s;
    x = (r(0, 2) + r(2, 0)) / s;
    y = (r(1, 2) + r(2, 1)) / s;
    z = 0.25 * s;
  }
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  if (n <= kEpsilon) {
    return {0.0, 0.0, 0.0, 1.0};
  }
  return {x / n, y / n, z / n, w / n};
}

std::array<double, 3> quat_to_rpy_xyzw(double x, double y, double z, double w)
{
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  if (n > kEpsilon) {
    x /= n; y /= n; z /= n; w /= n;
  } else {
    x = y = z = 0.0; w = 1.0;
  }
  const double sinr = 2.0 * (w * x + y * z);
  const double cosr = 1.0 - 2.0 * (x * x + y * y);
  const double roll = std::atan2(sinr, cosr);
  const double sinp = 2.0 * (w * y - z * x);
  const double pitch = std::abs(sinp) >= 1.0 ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);
  const double siny = 2.0 * (w * z + x * y);
  const double cosy = 1.0 - 2.0 * (y * y + z * z);
  const double yaw = std::atan2(siny, cosy);
  return {roll, pitch, yaw};
}

cv::Matx33d quat_xyzw_to_rotation(const std::array<double, 4> & q)
{
  double x = q[0], y = q[1], z = q[2], w = q[3];
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  if (n > kEpsilon) {
    x /= n; y /= n; z /= n; w /= n;
  } else {
    x = y = z = 0.0; w = 1.0;
  }
  return cv::Matx33d(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
    2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
    2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y));
}

Transform inverse_transform(const Transform & tf)
{
  Transform out;
  out.r = tf.r.t();
  out.t = -(out.r * tf.t);
  return out;
}

Transform compose(const Transform & a, const Transform & b)
{
  Transform out;
  out.r = a.r * b.r;
  out.t = a.r * b.t + a.t;
  return out;
}

cv::Vec3d transform_point(const Transform & tf, const cv::Vec3d & p)
{
  return tf.r * p + tf.t;
}

std::string json_escape(const std::string & input)
{
  std::ostringstream out;
  for (const char c : input) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c; break;
    }
  }
  return out.str();
}

std::string json_string(const std::string & input)
{
  return "\"" + json_escape(input) + "\"";
}

std::string json_int_array(const std::vector<int> & values)
{
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

std::string json_string_array(const std::vector<std::string> & values)
{
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) {
      out << ",";
    }
    out << json_string(values[i]);
  }
  out << "]";
  return out.str();
}

std::string json_rpy_deg(double roll, double pitch, double yaw)
{
  std::ostringstream out;
  out << std::fixed;
  out << "{";
  out << "\"roll\":" << rad_to_deg(roll) << ",";
  out << "\"pitch\":" << rad_to_deg(pitch) << ",";
  out << "\"yaw\":" << rad_to_deg(yaw);
  out << "}";
  return out.str();
}

std::string json_pose_rpy_deg(const PoseEstimate & estimate)
{
  if (!estimate.valid) {
    return "null";
  }
  const auto rpy = quat_to_rpy_xyzw(
    estimate.quat_xyzw[0], estimate.quat_xyzw[1], estimate.quat_xyzw[2], estimate.quat_xyzw[3]);
  return json_rpy_deg(rpy[0], rpy[1], rpy[2]);
}

std::optional<std::array<double, 4>> yaml_quat_xyzw(const YAML::Node & node)
{
  if (!node || !node["rotation_xyzw"]) {
    return std::nullopt;
  }
  const auto values = node["rotation_xyzw"];
  return std::array<double, 4>{
    values[0].as<double>(), values[1].as<double>(), values[2].as<double>(), values[3].as<double>()};
}

Transform transform_from_yaml(const YAML::Node & node)
{
  Transform tf;
  if (!node) {
    return tf;
  }
  if (node["matrix"]) {
    const auto values = node["matrix"];
    cv::Matx44d m;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        m(r, c) = values[r * 4 + c].as<double>();
      }
    }
    tf.r = cv::Matx33d(m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2), m(2, 0), m(2, 1), m(2, 2));
    tf.t = cv::Vec3d(m(0, 3), m(1, 3), m(2, 3));
    return tf;
  }
  if (node["translation_xyz"]) {
    const auto t = node["translation_xyz"];
    tf.t = cv::Vec3d(t[0].as<double>(), t[1].as<double>(), t[2].as<double>());
  }
  if (auto q = yaml_quat_xyzw(node)) {
    tf.r = quat_xyzw_to_rotation(*q);
  } else if (node["rotation_rpy_deg"]) {
    const auto rpy = node["rotation_rpy_deg"];
    tf.r = rpy_to_rotation(
      rpy[0].as<double>() * M_PI / 180.0,
      rpy[1].as<double>() * M_PI / 180.0,
      rpy[2].as<double>() * M_PI / 180.0);
  }
  return tf;
}

CameraCalibration load_camera_calibration(const fs::path & path)
{
  CameraCalibration calibration;
  const YAML::Node root = YAML::LoadFile(path.string());
  calibration.image_width = root["image_width"] ? root["image_width"].as<int>() : 0;
  calibration.image_height = root["image_height"] ? root["image_height"].as<int>() : 0;
  auto matrix_node = root["camera_matrix"];
  if (!matrix_node) {
    return calibration;
  }
  std::vector<double> data;
  if (matrix_node["data"]) {
    data = matrix_node["data"].as<std::vector<double>>();
  } else {
    data = matrix_node.as<std::vector<double>>();
  }
  if (data.size() != 9) {
    return calibration;
  }
  calibration.k = cv::Matx33d(
    data[0], data[1], data[2],
    data[3], data[4], data[5],
    data[6], data[7], data[8]);
  if (auto dist = root["distortion_coefficients"]) {
    std::vector<double> d = dist["data"] ? dist["data"].as<std::vector<double>>() : dist.as<std::vector<double>>();
    calibration.distortion = cv::Mat(d, true).reshape(1, 1);
  } else {
    calibration.distortion = cv::Mat::zeros(1, 5, CV_64F);
  }
  calibration.valid = true;
  return calibration;
}

std::optional<cv::Vec3d> pixel_to_world_at_z(
  const cv::Point2d & pixel,
  double z_target,
  const CameraCalibration & camera,
  const Transform & t_world_camera,
  const cv::Vec3d & water_normal_world,
  double water_plane_d,
  double n_air,
  double n_water)
{
  std::vector<cv::Point2d> src{pixel};
  std::vector<cv::Point2d> undistorted;
  cv::Mat k_mat(3, 3, CV_64F, const_cast<double *>(camera.k.val));
  cv::undistortPoints(src, undistorted, k_mat, camera.distortion);
  if (undistorted.empty()) {
    return std::nullopt;
  }
  cv::Vec3d ray_cam = normalize3({undistorted[0].x, undistorted[0].y, 1.0});
  cv::Vec3d ray_air = normalize3(t_world_camera.r * ray_cam);
  cv::Vec3d water_n = normalize3(water_normal_world);
  const double denom = water_n.dot(ray_air);
  if (std::abs(denom) <= kEpsilon) {
    return std::nullopt;
  }
  const cv::Vec3d c = t_world_camera.t;
  const double s = -(water_n.dot(c) + water_plane_d) / denom;
  if (s <= 0.0) {
    return std::nullopt;
  }
  const cv::Vec3d surface = c + s * ray_air;

  cv::Vec3d normal_to_incident = water_n;
  double cos_i = -normal_to_incident.dot(ray_air);
  if (cos_i < 0.0) {
    normal_to_incident = -normal_to_incident;
    cos_i = -normal_to_incident.dot(ray_air);
  }
  const double eta = n_air / n_water;
  const double k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
  if (k < 0.0) {
    return std::nullopt;
  }
  const cv::Vec3d ray_water = normalize3(eta * ray_air + (eta * cos_i - std::sqrt(k)) * normal_to_incident);
  if (std::abs(ray_water[2]) <= kEpsilon) {
    return std::nullopt;
  }
  const double lambda = (z_target - surface[2]) / ray_water[2];
  if (lambda <= 0.0) {
    return std::nullopt;
  }
  return surface + lambda * ray_water;
}

}  // namespace

class RefractiveAprilTagPoseNode : public rclcpp::Node
{
public:
  RefractiveAprilTagPoseNode()
  : Node("refractive_apriltag_pose")
  {
    detection_topic_ = declare_parameter<std::string>("detection_topic", "/finsrov/vision/tag_detections_2d");
    depth_topic_ = declare_parameter<std::string>("depth_topic", "/finsrov/hardware/depth_raw");
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/finsrov/hardware/imu_raw");
    constrained_pose_topic_ =
      declare_parameter<std::string>("constrained_pose_topic", "/finsrov/vision/refracted_pose_6d");
    pure_pose_topic_ =
      declare_parameter<std::string>("pure_pose_topic", "/finsrov/vision/refracted_pose_6d_pure");
    status_topic_ = declare_parameter<std::string>("status_topic", "/finsrov/vision/refracted/status");
    world_frame_id_ = declare_parameter<std::string>("world_frame_id", "pool_world");
    body_frame_id_ = declare_parameter<std::string>("body_frame_id", "finsrov_base_link");
    camera_calibration_file_ =
      declare_parameter<std::string>("camera_calibration_file", "perception/calibration/rgb_camera.yaml");
    extrinsics_file_ = declare_parameter<std::string>(
      "extrinsics_file", "state_estimation/config/state_fusion_extrinsics.yaml");
    marker_length_m_ = declare_parameter<double>("marker_length_m", 0.098);
    water_surface_z_m_ = declare_parameter<double>("water_surface_z_m", 1.0);
    depth_sign_ = declare_parameter<double>("depth_sign", -1.0);
    pressure_sensor_offset_z_body_ = declare_parameter<double>("pressure_sensor_offset_z_body", -0.0678);
    air_depth_threshold_m_ = declare_parameter<double>("air_depth_threshold_m", 0.02);
    underwater_depth_threshold_m_ = declare_parameter<double>("underwater_depth_threshold_m", 0.08);
    n_air_ = declare_parameter<double>("n_air", 1.0003);
    n_water_ = declare_parameter<double>("n_water", 1.333);
    water_plane_normal_ = yaml_vector3_parameter("water_plane_normal", {0.0, 0.0, 1.0});
    water_plane_d_ = declare_parameter<double>("water_plane_d", -water_surface_z_m_);
    max_geometry_error_ratio_ = declare_parameter<double>("max_geometry_error_ratio", 0.20);
    low_confidence_geometry_error_ratio_ = declare_parameter<double>("low_confidence_geometry_error_ratio", 0.08);
    max_iterations_ = declare_parameter<int>("max_iterations", 8);
    min_visible_tags_ = declare_parameter<int>("min_visible_tags", 1);
    publish_air_pose_on_constrained_topic_ =
      declare_parameter<bool>("publish_air_pose_on_constrained_topic", true);
    publish_surface_transition_pose_on_constrained_topic_ =
      declare_parameter<bool>("publish_surface_transition_pose_on_constrained_topic", true);
    mode_selection_source_ = declare_parameter<std::string>("mode_selection_source", "tag_height");
    air_tag_margin_m_ = declare_parameter<double>("air_tag_margin_m", 0.01);
    surface_tag_transition_margin_m_ = declare_parameter<double>("surface_tag_transition_margin_m", 0.04);
    underwater_tag_margin_m_ = declare_parameter<double>("underwater_tag_margin_m", 0.02);
    publish_pinhole_fallback_on_constrained_topic_ =
      declare_parameter<bool>("publish_pinhole_fallback_on_constrained_topic", true);
    pinhole_fallback_use_pressure_depth_z_ = declare_parameter<bool>("pinhole_fallback_use_pressure_depth_z", true);
    pinhole_fallback_max_reprojection_error_px_ =
      declare_parameter<double>("pinhole_fallback_max_reprojection_error_px", 2.0);
    pinhole_fallback_reject_reasons_ = declare_parameter<std::vector<std::string>>(
      "pinhole_fallback_reject_reasons", {"no_valid_refracted_rays", "geometry_error"});
    constrained_covariance_xy_ = declare_parameter<double>("constrained_covariance_xy", 0.0025);
    constrained_covariance_z_ = declare_parameter<double>("constrained_covariance_z", 0.0025);
    constrained_covariance_rp_ = declare_parameter<double>("constrained_covariance_roll_pitch", 0.02);
    constrained_covariance_yaw_ = declare_parameter<double>("constrained_covariance_yaw", 0.03);
    pure_covariance_xyz_ = declare_parameter<double>("pure_covariance_xyz", 0.05);
    pure_covariance_rpy_ = declare_parameter<double>("pure_covariance_rpy", 0.20);

    camera_ = load_camera_calibration(resolve_path(camera_calibration_file_));
    if (!camera_.valid) {
      throw std::runtime_error("failed to load camera calibration: " + camera_calibration_file_);
    }
    load_extrinsics(resolve_path(extrinsics_file_));
    if (body_tag_transforms_.empty()) {
      RCLCPP_WARN(get_logger(), "no T_body_tag entries loaded; refractive pose will reject detections");
    }
    RCLCPP_INFO(get_logger(), "loaded T_body_tag ids for refractive pose: %s", json_int_array(known_tag_ids()).c_str());
    RCLCPP_INFO(
      get_logger(),
      "refractive depth convention: depth_raw=pressure_sensor_depth, output_z=body, "
      "water_surface_z_m=%.4f, depth_sign=%.2f, pressure_sensor_offset_z_body=%.4f, water_plane_d=%.4f",
      water_surface_z_m_, depth_sign_, pressure_sensor_offset_z_body_, water_plane_d_);
    RCLCPP_INFO(
      get_logger(),
      "refractive mode selection: source=%s, air_tag_margin_m=%.3f, "
      "surface_tag_transition_margin_m=%.3f, underwater_tag_margin_m=%.3f, pinhole_fallback=%s",
      mode_selection_source_.c_str(), air_tag_margin_m_, surface_tag_transition_margin_m_,
      underwater_tag_margin_m_, publish_pinhole_fallback_on_constrained_topic_ ? "true" : "false");

    constrained_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(constrained_pose_topic_, 10);
    pure_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(pure_pose_topic_, 10);
    status_pub_ = create_publisher<std_msgs::msg::String>(status_topic_, 10);
    detection_sub_ = create_subscription<msgs::msg::AprilTagDetection2DArray>(
      detection_topic_, 10, [this](const msgs::msg::AprilTagDetection2DArray::SharedPtr msg) {
        handle_detections(*msg);
      });
    depth_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      depth_topic_, 10, [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        latest_pressure_sensor_z_ = water_surface_z_m_ + depth_sign_ * msg->pose.pose.position.z;
        latest_depth_stamp_ = stamp_to_sec(msg->header.stamp);
      });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, 10, [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
        const auto rpy = quat_to_rpy_xyzw(
          msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
        latest_roll_ = rpy[0];
        latest_pitch_ = rpy[1];
        latest_yaw_ = rpy[2];
        latest_imu_stamp_ = stamp_to_sec(msg->header.stamp);
      });

    RCLCPP_INFO(
      get_logger(),
      "refractive_apriltag_pose started: detections=%s, constrained=%s, pure=%s, camera=%s, extrinsics=%s",
      detection_topic_.c_str(), constrained_pose_topic_.c_str(), pure_pose_topic_.c_str(),
      resolve_path(camera_calibration_file_).c_str(), resolve_path(extrinsics_file_).c_str());
  }

private:
  std::vector<double> yaml_vector3_parameter(const std::string & name, const std::vector<double> & fallback)
  {
    auto values = declare_parameter<std::vector<double>>(name, fallback);
    if (values.size() != 3) {
      throw std::runtime_error(name + " must contain 3 values");
    }
    return values;
  }

  double stamp_to_sec(const builtin_interfaces::msg::Time & stamp) const
  {
    const double value = static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
    return value > 0.0 ? value : now().seconds();
  }

  fs::path resolve_path(const std::string & value) const
  {
    fs::path path(value);
    if (path.is_absolute() && fs::exists(path)) {
      return path;
    }
    if (const char * root = std::getenv("FINSSIM_REPO_ROOT")) {
      fs::path candidate = fs::path(root) / "ros2_ws" / "src" / "perception" / path;
      if (fs::exists(candidate)) {
        return candidate;
      }
      candidate = fs::path(root) / "ros2_ws" / "src" / path;
      if (fs::exists(candidate)) {
        return candidate;
      }
    }
    fs::path source = fs::current_path() / "src" / "perception" / path;
    if (fs::exists(source)) {
      return source;
    }
    source = fs::current_path() / "src" / path;
    if (fs::exists(source)) {
      return source;
    }
    try {
      fs::path share = ament_index_cpp::get_package_share_directory("perception");
      fs::path candidate = share / path;
      if (fs::exists(candidate)) {
        return candidate;
      }
      candidate = share / path.filename();
      if (fs::exists(candidate)) {
        return candidate;
      }
    } catch (const std::exception &) {
    }
    try {
      fs::path share = ament_index_cpp::get_package_share_directory("state_estimation");
      fs::path candidate = share / path;
      if (fs::exists(candidate)) {
        return candidate;
      }
      candidate = share / path.filename();
      if (fs::exists(candidate)) {
        return candidate;
      }
    } catch (const std::exception &) {
    }
    return path;
  }

  void load_extrinsics(const fs::path & path)
  {
    const YAML::Node root = YAML::LoadFile(path.string());
    t_world_camera_ = transform_from_yaml(root["T_world_camera"]);
    if (root["T_tag_body"]) {
      throw std::runtime_error(
        "state_fusion_extrinsics.yaml uses deprecated T_tag_body. "
        "Use T_body_tag only: tag pose expressed in finsrov_base_link/body frame.");
    }
    const auto body_tag_map = root["T_body_tag"];
    if (!body_tag_map) {
      return;
    }
    for (const auto & item : body_tag_map) {
      const int tag_id = item.first.as<int>();
      body_tag_transforms_[tag_id] = transform_from_yaml(item.second);
    }
  }

  std::vector<Observation> parse_observations(const msgs::msg::AprilTagDetection2DArray & msg) const
  {
    std::vector<Observation> observations;
    observations.reserve(msg.detections.size());
    for (const auto & detection : msg.detections) {
      if (!body_tag_transforms_.count(detection.tag_id)) {
        continue;
      }
      Observation obs;
      obs.tag_id = detection.tag_id;
      obs.family = detection.family;
      obs.hamming = detection.hamming;
      obs.decision_margin = detection.decision_margin;
      obs.mean_edge_px = detection.mean_edge_px;
      for (size_t i = 0; i < 4; ++i) {
        obs.corners[i] = {detection.corner_pixels_xy[2 * i], detection.corner_pixels_xy[2 * i + 1]};
      }
      observations.push_back(obs);
    }
    return observations;
  }

  std::vector<int> known_tag_ids() const
  {
    std::vector<int> ids;
    ids.reserve(body_tag_transforms_.size());
    for (const auto & item : body_tag_transforms_) {
      ids.push_back(item.first);
    }
    return ids;
  }

  double body_z_from_pressure_sensor_z(double sensor_z) const
  {
    const double roll = latest_roll_.value_or(0.0);
    const double pitch = latest_pitch_.value_or(0.0);
    const double offset_world_z = pressure_sensor_offset_z_body_ * std::cos(roll) * std::cos(pitch);
    return sensor_z - offset_world_z;
  }

  std::optional<double> body_z_from_pressure_depth() const
  {
    if (!latest_pressure_sensor_z_) {
      return std::nullopt;
    }
    return body_z_from_pressure_sensor_z(*latest_pressure_sensor_z_);
  }

  std::vector<Sample> build_samples(const std::vector<Observation> & observations) const
  {
    const double h = marker_length_m_ * 0.5;
    const std::array<cv::Vec3d, 4> tag_corners = {
      cv::Vec3d(-h, h, 0.0),
      cv::Vec3d(h, h, 0.0),
      cv::Vec3d(h, -h, 0.0),
      cv::Vec3d(-h, -h, 0.0),
    };
    std::vector<Sample> samples;
    for (const auto & obs : observations) {
      const auto found = body_tag_transforms_.find(obs.tag_id);
      if (found == body_tag_transforms_.end()) {
        continue;
      }
      for (size_t i = 0; i < 4; ++i) {
        samples.push_back({obs.tag_id, transform_point(found->second, tag_corners[i]), obs.corners[i]});
      }
    }
    return samples;
  }

  PoseEstimate estimate_constrained(
    const std::vector<Sample> & samples,
    double stamp_sec)
  {
    PoseEstimate estimate;
    if (!latest_pressure_sensor_z_ || !latest_roll_ || !latest_pitch_) {
      estimate.reject_reason = "missing_depth_or_imu";
      return estimate;
    }
    const double depth_age = std::abs(stamp_sec - latest_depth_stamp_);
    const double imu_age = std::abs(stamp_sec - latest_imu_stamp_);
    if (depth_age > 1.0 || imu_age > 1.0) {
      estimate.reject_reason = "stale_depth_or_imu";
      return estimate;
    }

    cv::Vec3d x{0.0, 0.0, latest_yaw_.value_or(0.0)};
    int initialized = 0;
    const auto body_z_value = body_z_from_pressure_depth();
    if (!body_z_value) {
      estimate.reject_reason = "missing_pressure_depth";
      return estimate;
    }
    const double body_z = *body_z_value;
    const cv::Matx33d r0 = rpy_to_rotation(*latest_roll_, *latest_pitch_, x[2]);
    for (const auto & sample : samples) {
      const cv::Vec3d predicted_offset = r0 * sample.p_body;
      const double z_corner = body_z + predicted_offset[2];
      auto q = pixel_to_world_at_z(
        sample.pixel, z_corner, camera_, t_world_camera_,
        {water_plane_normal_[0], water_plane_normal_[1], water_plane_normal_[2]},
        water_plane_d_, n_air_, n_water_);
      if (!q) {
        continue;
      }
      x[0] += (*q)[0] - predicted_offset[0];
      x[1] += (*q)[1] - predicted_offset[1];
      initialized++;
    }
    if (initialized == 0) {
      estimate.reject_reason = "no_valid_refracted_rays";
      return estimate;
    }
    x[0] /= initialized;
    x[1] /= initialized;

    auto residual = [&](const cv::Vec3d & state) {
      std::vector<double> r;
      const cv::Matx33d rot = rpy_to_rotation(*latest_roll_, *latest_pitch_, state[2]);
      const cv::Vec3d origin(state[0], state[1], body_z);
      r.reserve(samples.size() * 3);
      for (const auto & sample : samples) {
        const cv::Vec3d p = origin + rot * sample.p_body;
        auto q = pixel_to_world_at_z(
          sample.pixel, p[2], camera_, t_world_camera_,
          {water_plane_normal_[0], water_plane_normal_[1], water_plane_normal_[2]},
          water_plane_d_, n_air_, n_water_);
        if (!q) {
          r.push_back(10.0);
          r.push_back(10.0);
          r.push_back(10.0);
          continue;
        }
        const cv::Vec3d diff = *q - p;
        r.push_back(diff[0]);
        r.push_back(diff[1]);
        r.push_back(diff[2]);
      }
      return r;
    };

    double lambda = 1e-3;
    for (int iter = 0; iter < max_iterations_; ++iter) {
      const auto r = residual(x);
      const int rows = static_cast<int>(r.size());
      cv::Mat j(rows, 3, CV_64F);
      const std::array<double, 3> eps = {1e-4, 1e-4, 1e-5};
      for (int c = 0; c < 3; ++c) {
        cv::Vec3d xp = x;
        xp[c] += eps[c];
        const auto rp = residual(xp);
        for (int rr = 0; rr < rows; ++rr) {
          j.at<double>(rr, c) = (rp[rr] - r[rr]) / eps[c];
        }
      }
      cv::Mat rv(rows, 1, CV_64F);
      for (int rr = 0; rr < rows; ++rr) {
        rv.at<double>(rr, 0) = -r[rr];
      }
      cv::Mat h = j.t() * j + lambda * cv::Mat::eye(3, 3, CV_64F);
      cv::Mat g = j.t() * rv;
      cv::Mat delta;
      if (!cv::solve(h, g, delta, cv::DECOMP_CHOLESKY)) {
        break;
      }
      cv::Vec3d next = x + cv::Vec3d(delta.at<double>(0), delta.at<double>(1), delta.at<double>(2));
      next[2] = wrap_angle(next[2]);
      const double old_cost = cost(r);
      const double new_cost = cost(residual(next));
      if (new_cost < old_cost) {
        x = next;
        lambda = std::max(lambda * 0.5, 1e-8);
      } else {
        lambda = std::min(lambda * 5.0, 1e3);
      }
      estimate.iterations = iter + 1;
      if (std::abs(delta.at<double>(0)) < 1e-5 && std::abs(delta.at<double>(1)) < 1e-5 && std::abs(delta.at<double>(2)) < 1e-6) {
        break;
      }
    }

    const auto final_r = residual(x);
    estimate.residual = rms(final_r);
    if (estimate.residual / std::max(marker_length_m_, 1e-6) > max_geometry_error_ratio_) {
      estimate.reject_reason = "geometry_error";
      return estimate;
    }
    const cv::Matx33d rot = rpy_to_rotation(*latest_roll_, *latest_pitch_, x[2]);
    estimate.valid = true;
    estimate.position = {x[0], x[1], body_z};
    estimate.quat_xyzw = rotation_to_quat_xyzw(rot);
    estimate.reject_reason = "none";
    return estimate;
  }

  PoseEstimate estimate_pure_pinhole_baseline(const std::vector<Sample> & samples)
  {
    PoseEstimate estimate;
    if (samples.size() < 4) {
      estimate.reject_reason = "not_enough_points";
      return estimate;
    }
    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    object_points.reserve(samples.size());
    image_points.reserve(samples.size());
    for (const auto & sample : samples) {
      object_points.emplace_back(sample.p_body[0], sample.p_body[1], sample.p_body[2]);
      image_points.push_back(sample.pixel);
    }
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat k_mat(3, 3, CV_64F, const_cast<double *>(camera_.k.val));
    bool ok = cv::solvePnP(object_points, image_points, k_mat, camera_.distortion, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    if (!ok) {
      estimate.reject_reason = "solvepnp_failed";
      return estimate;
    }
    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);
    cv::Matx33d r_camera_body;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        r_camera_body(r, c) = rot_mat.at<double>(r, c);
      }
    }
    Transform t_camera_body{r_camera_body, {tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)}};
    Transform t_world_body = compose(t_world_camera_, t_camera_body);
    std::vector<cv::Point2d> projected;
    cv::projectPoints(object_points, rvec, tvec, k_mat, camera_.distortion, projected);
    double error = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
      const auto d = projected[i] - image_points[i];
      error += std::sqrt(d.x * d.x + d.y * d.y);
    }
    estimate.valid = true;
    estimate.position = t_world_body.t;
    estimate.quat_xyzw = rotation_to_quat_xyzw(t_world_body.r);
    estimate.residual = projected.empty() ? 0.0 : error / static_cast<double>(projected.size());
    estimate.reject_reason = "pinhole_multi_tag_baseline";
    return estimate;
  }

  static double cost(const std::vector<double> & r)
  {
    double value = 0.0;
    for (double x : r) {
      value += x * x;
    }
    return value;
  }

  static double rms(const std::vector<double> & r)
  {
    if (r.empty()) {
      return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(cost(r) / static_cast<double>(r.size()));
  }

  std::string mode_for_depth() const
  {
    if (!latest_pressure_sensor_z_) {
      return "missing_depth";
    }
    const double depth_m = water_surface_z_m_ - *latest_pressure_sensor_z_;
    if (depth_m < air_depth_threshold_m_) {
      return "air";
    }
    if (depth_m < underwater_depth_threshold_m_) {
      return "surface_transition";
    }
    return "underwater_refraction";
  }

  TagHeightStats tag_height_from_pressure(const std::vector<Sample> & samples, double stamp_sec) const
  {
    TagHeightStats stats;
    const auto body_z_value = body_z_from_pressure_depth();
    if (!body_z_value || !latest_roll_ || !latest_pitch_ || samples.empty()) {
      return stats;
    }
    const double depth_age = std::abs(stamp_sec - latest_depth_stamp_);
    const double imu_age = std::abs(stamp_sec - latest_imu_stamp_);
    if (depth_age > 1.0 || imu_age > 1.0) {
      return stats;
    }

    const cv::Matx33d rot = rpy_to_rotation(*latest_roll_, *latest_pitch_, latest_yaw_.value_or(0.0));
    double min_z = std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    double sum_z = 0.0;
    size_t count = 0;
    for (const auto & sample : samples) {
      const cv::Vec3d p_world = cv::Vec3d(0.0, 0.0, *body_z_value) + rot * sample.p_body;
      min_z = std::min(min_z, p_world[2]);
      max_z = std::max(max_z, p_world[2]);
      sum_z += p_world[2];
      count++;
    }
    if (count == 0) {
      return stats;
    }
    stats.valid = true;
    stats.min_z = min_z;
    stats.max_z = max_z;
    stats.mean_z = sum_z / static_cast<double>(count);
    return stats;
  }

  std::string select_visual_mode(const std::string & depth_mode, const TagHeightStats & tag_height) const
  {
    if (mode_selection_source_ == "depth" || !tag_height.valid) {
      return depth_mode;
    }
    if (tag_height.min_z >= water_surface_z_m_ - air_tag_margin_m_) {
      return "air";
    }
    if (
      tag_height.max_z >= water_surface_z_m_ - underwater_tag_margin_m_ ||
      std::abs(tag_height.mean_z - water_surface_z_m_) <= surface_tag_transition_margin_m_) {
      return "surface_transition";
    }
    return "underwater_refraction";
  }

  bool pinhole_reprojection_acceptable(const PoseEstimate & pure) const
  {
    return pure.valid && pure.residual <= pinhole_fallback_max_reprojection_error_px_;
  }

  bool snell_reject_reason_allows_fallback(const std::string & reason) const
  {
    return std::find(
      pinhole_fallback_reject_reasons_.begin(),
      pinhole_fallback_reject_reasons_.end(),
      reason) != pinhole_fallback_reject_reasons_.end();
  }

  void handle_detections(const msgs::msg::AprilTagDetection2DArray & msg)
  {
    const auto start = std::chrono::steady_clock::now();
    const double stamp_sec = stamp_to_sec(msg.header.stamp);
    const auto observations = parse_observations(msg);
    const auto samples = build_samples(observations);
    std::vector<int> tag_ids;
    for (const auto & obs : observations) {
      tag_ids.push_back(obs.tag_id);
    }
    std::sort(tag_ids.begin(), tag_ids.end());
    tag_ids.erase(std::unique(tag_ids.begin(), tag_ids.end()), tag_ids.end());
    const std::string depth_mode = mode_for_depth();
    const TagHeightStats tag_height = tag_height_from_pressure(samples, stamp_sec);
    const std::string mode = select_visual_mode(depth_mode, tag_height);

    const PoseEstimate pure = estimate_pure_pinhole_baseline(samples);
    PoseEstimate snell;
    PoseEstimate constrained;
    std::string output_model = "none";
    bool publish_constrained_covariance = false;
    if (mode == "underwater_refraction" && static_cast<int>(tag_ids.size()) >= min_visible_tags_) {
      snell = estimate_constrained(samples, stamp_sec);
      if (snell.valid) {
        constrained = snell;
        publish_constrained_covariance = true;
        output_model = "snell_depth_imu";
      }
    } else {
      constrained.reject_reason = mode == "underwater_refraction" ? "not_enough_visible_tags" : mode;
    }

    if (!constrained.valid && pure.valid) {
      const bool air_fallback = mode == "air" && publish_air_pose_on_constrained_topic_;
      const bool surface_fallback =
        mode == "surface_transition" && publish_surface_transition_pose_on_constrained_topic_;
      const bool underwater_fallback =
        mode == "underwater_refraction" && publish_pinhole_fallback_on_constrained_topic_ &&
        snell_reject_reason_allows_fallback(snell.reject_reason);
      if ((air_fallback || surface_fallback || underwater_fallback) && pinhole_reprojection_acceptable(pure)) {
        constrained = pure;
        constrained.reject_reason = mode + "_pinhole_pose";
        if (pinhole_fallback_use_pressure_depth_z_) {
          if (const auto body_z_value = body_z_from_pressure_depth()) {
            constrained.position[2] = *body_z_value;
            constrained.reject_reason = mode + "_pinhole_pose_pressure_depth_z";
          }
        }
        if (mode == "air") {
          output_model = pinhole_fallback_use_pressure_depth_z_ ? "pinhole_air_pressure_depth_z" : "pinhole_air";
        } else if (mode == "surface_transition") {
          output_model = pinhole_fallback_use_pressure_depth_z_ ?
            "pinhole_surface_transition_pressure_depth_z" : "pinhole_surface_transition";
        } else {
          output_model = pinhole_fallback_use_pressure_depth_z_ ?
            "pinhole_underwater_fallback_pressure_depth_z" : "pinhole_underwater_fallback";
        }
      } else if (!pinhole_reprojection_acceptable(pure)) {
        constrained.reject_reason = mode + "_pinhole_reprojection_reject";
      } else if (mode == "underwater_refraction") {
        constrained.reject_reason = snell.reject_reason.empty() ? "underwater_pinhole_fallback_disabled" : snell.reject_reason;
      }
    }

    if (!constrained.valid && mode == "underwater_refraction" && !snell.reject_reason.empty()) {
      constrained.reject_reason = snell.reject_reason;
    }

    if (constrained.valid) {
      publish_pose(constrained, msg.header.stamp, constrained_pub_, publish_constrained_covariance);
    }

    if (pure.valid) {
      publish_pose(pure, msg.header.stamp, pure_pub_, false);
    }

    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    publish_status(
      msg.header.stamp, mode, depth_mode, tag_height, tag_ids, observations.size(), samples.size(),
      constrained, snell, pure, output_model, elapsed);
  }

  void publish_pose(
    const PoseEstimate & estimate,
    const builtin_interfaces::msg::Time & stamp,
    const rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr & pub,
    bool constrained)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = world_frame_id_;
    msg.pose.pose.position.x = estimate.position[0];
    msg.pose.pose.position.y = estimate.position[1];
    msg.pose.pose.position.z = estimate.position[2];
    msg.pose.pose.orientation.x = estimate.quat_xyzw[0];
    msg.pose.pose.orientation.y = estimate.quat_xyzw[1];
    msg.pose.pose.orientation.z = estimate.quat_xyzw[2];
    msg.pose.pose.orientation.w = estimate.quat_xyzw[3];
    if (constrained) {
      const double multiplier = estimate.residual / std::max(marker_length_m_, 1e-6) > low_confidence_geometry_error_ratio_ ? 10.0 : 1.0;
      msg.pose.covariance[0] = constrained_covariance_xy_ * multiplier;
      msg.pose.covariance[7] = constrained_covariance_xy_ * multiplier;
      msg.pose.covariance[14] = constrained_covariance_z_ * multiplier;
      msg.pose.covariance[21] = constrained_covariance_rp_;
      msg.pose.covariance[28] = constrained_covariance_rp_;
      msg.pose.covariance[35] = constrained_covariance_yaw_ * multiplier;
    } else {
      msg.pose.covariance[0] = pure_covariance_xyz_;
      msg.pose.covariance[7] = pure_covariance_xyz_;
      msg.pose.covariance[14] = pure_covariance_xyz_;
      msg.pose.covariance[21] = pure_covariance_rpy_;
      msg.pose.covariance[28] = pure_covariance_rpy_;
      msg.pose.covariance[35] = pure_covariance_rpy_;
    }
    pub->publish(msg);
  }

  void publish_status(
    const builtin_interfaces::msg::Time & input_stamp,
    const std::string & mode,
    const std::string & depth_mode,
    const TagHeightStats & tag_height,
    const std::vector<int> & tag_ids,
    size_t observation_count,
    size_t sample_count,
    const PoseEstimate & constrained,
    const PoseEstimate & snell,
    const PoseEstimate & pure,
    const std::string & output_model,
    double processing_ms)
  {
    std::ostringstream out;
    out << std::fixed;
    out << "{";
    const int64_t input_stamp_ns =
      static_cast<int64_t>(input_stamp.sec) * 1000000000LL + static_cast<int64_t>(input_stamp.nanosec);
    out << "\"input_stamp_ns\":" << input_stamp_ns << ",";
    out << "\"mode\":" << json_string(mode) << ",";
    out << "\"depth_mode\":" << json_string(depth_mode) << ",";
    out << "\"visual_mode\":" << json_string(mode) << ",";
    out << "\"mode_selection_source\":" << json_string(mode_selection_source_) << ",";
    out << "\"visible_tag_ids\":" << json_int_array(tag_ids) << ",";
    out << "\"known_tag_ids\":" << json_int_array(known_tag_ids()) << ",";
    out << "\"observations\":" << observation_count << ",";
    out << "\"corners\":" << sample_count << ",";
    out << "\"tag_height_valid\":" << (tag_height.valid ? "true" : "false") << ",";
    out << "\"tag_z_min\":" << (tag_height.valid ? std::to_string(tag_height.min_z) : "null") << ",";
    out << "\"tag_z_max\":" << (tag_height.valid ? std::to_string(tag_height.max_z) : "null") << ",";
    out << "\"tag_z_mean\":" << (tag_height.valid ? std::to_string(tag_height.mean_z) : "null") << ",";
    out << "\"constrained_valid\":" << (constrained.valid ? "true" : "false") << ",";
    out << "\"constrained_residual_m\":" << constrained.residual << ",";
    out << "\"constrained_iterations\":" << constrained.iterations << ",";
    out << "\"constrained_rpy_deg\":" << json_pose_rpy_deg(constrained) << ",";
    out << "\"constrained_reject_reason\":" << json_string(constrained.reject_reason) << ",";
    out << "\"constrained_output_model\":" << json_string(output_model) << ",";
    out << "\"fallback_used\":" << ((constrained.valid && output_model.rfind("pinhole_", 0) == 0) ? "true" : "false") << ",";
    out << "\"snell_valid\":" << (snell.valid ? "true" : "false") << ",";
    out << "\"snell_residual_m\":" << snell.residual << ",";
    out << "\"snell_iterations\":" << snell.iterations << ",";
    out << "\"snell_reject_reason\":" << json_string(snell.reject_reason) << ",";
    out << "\"pure_visual_valid\":" << (pure.valid ? "true" : "false") << ",";
    out << "\"pure_visual_model\":\"pinhole_multi_tag_baseline\",";
    out << "\"pure_visual_reprojection_error_px\":" << pure.residual << ",";
    out << "\"pure_visual_rpy_deg\":" << json_pose_rpy_deg(pure) << ",";
    out << "\"pure_visual_reason\":" << json_string(pure.reject_reason) << ",";
    out << "\"water_surface_z_m\":" << water_surface_z_m_ << ",";
    out << "\"water_plane_d\":" << water_plane_d_ << ",";
    out << "\"pressure_sensor_z\":"
        << (latest_pressure_sensor_z_ ? std::to_string(*latest_pressure_sensor_z_) : "null") << ",";
    out << "\"pressure_sensor_depth_m\":"
        << (latest_pressure_sensor_z_ ? std::to_string(water_surface_z_m_ - *latest_pressure_sensor_z_) : "null") << ",";
    const auto body_z_value = body_z_from_pressure_depth();
    out << "\"body_depth_z\":" << (body_z_value ? std::to_string(*body_z_value) : "null") << ",";
    out << "\"pressure_sensor_offset_z_body\":" << pressure_sensor_offset_z_body_ << ",";
    out << "\"imu_used\":" << (latest_roll_ && latest_pitch_ ? "true" : "false") << ",";
    out << "\"imu_rpy_deg\":"
        << (latest_roll_ && latest_pitch_ && latest_yaw_ ? json_rpy_deg(*latest_roll_, *latest_pitch_, *latest_yaw_) : "null")
        << ",";
    out << "\"processing_time_ms\":" << processing_ms;
    out << "}";
    std_msgs::msg::String msg;
    msg.data = out.str();
    status_pub_->publish(msg);
  }

  std::string detection_topic_;
  std::string depth_topic_;
  std::string imu_topic_;
  std::string constrained_pose_topic_;
  std::string pure_pose_topic_;
  std::string status_topic_;
  std::string world_frame_id_;
  std::string body_frame_id_;
  std::string camera_calibration_file_;
  std::string extrinsics_file_;
  double marker_length_m_{0.098};
  double water_surface_z_m_{1.0};
  double depth_sign_{-1.0};
  double pressure_sensor_offset_z_body_{-0.0678};
  double air_depth_threshold_m_{0.02};
  double underwater_depth_threshold_m_{0.08};
  double n_air_{1.0003};
  double n_water_{1.333};
  std::vector<double> water_plane_normal_{0.0, 0.0, 1.0};
  double water_plane_d_{0.0};
  double max_geometry_error_ratio_{0.20};
  double low_confidence_geometry_error_ratio_{0.08};
  int max_iterations_{8};
  int min_visible_tags_{1};
  bool publish_air_pose_on_constrained_topic_{true};
  bool publish_surface_transition_pose_on_constrained_topic_{true};
  std::string mode_selection_source_{"tag_height"};
  double air_tag_margin_m_{0.01};
  double surface_tag_transition_margin_m_{0.04};
  double underwater_tag_margin_m_{0.02};
  bool publish_pinhole_fallback_on_constrained_topic_{true};
  bool pinhole_fallback_use_pressure_depth_z_{true};
  double pinhole_fallback_max_reprojection_error_px_{2.0};
  std::vector<std::string> pinhole_fallback_reject_reasons_{"no_valid_refracted_rays", "geometry_error"};
  double constrained_covariance_xy_{0.0025};
  double constrained_covariance_z_{0.0025};
  double constrained_covariance_rp_{0.02};
  double constrained_covariance_yaw_{0.03};
  double pure_covariance_xyz_{0.05};
  double pure_covariance_rpy_{0.20};

  CameraCalibration camera_;
  Transform t_world_camera_;
  std::map<int, Transform> body_tag_transforms_;
  std::optional<double> latest_pressure_sensor_z_;
  std::optional<double> latest_roll_;
  std::optional<double> latest_pitch_;
  std::optional<double> latest_yaw_;
  double latest_depth_stamp_{0.0};
  double latest_imu_stamp_{0.0};

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr constrained_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pure_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<msgs::msg::AprilTagDetection2DArray>::SharedPtr detection_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int exit_code = 0;
  try {
    rclcpp::spin(std::make_shared<RefractiveAprilTagPoseNode>());
  } catch (const std::exception & exc) {
    std::cerr << "refractive_apriltag_pose fatal: " << exc.what() << std::endl;
    exit_code = 1;
  }
  rclcpp::shutdown();
  return exit_code;
}
