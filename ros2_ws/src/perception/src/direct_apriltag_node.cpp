#include <apriltag/apriltag.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag25h9.h>
#include <apriltag/tag36h10.h>
#include <apriltag/tag36h11.h>
#include <apriltag/common/image_u8.h>
#include <apriltag/common/zarray.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <msgs/msg/april_tag_detection2_d_array.hpp>
#include <msgs/msg/april_tag_detection3_d_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/string.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace
{
struct Settings
{
  std::string device{"/dev/video0"};
  int width{1280};
  int height{720};
  double fps{30.0};
  std::string fourcc{"MJPG"};
  bool enabled{true};
};

struct DetectionResult
{
  bool detected{false};
  bool target_match{false};
  int tag_id{-1};
  int hamming{0};
  double decision_margin{0.0};
  std::string family{"none"};
  std::vector<int> detected_ids;
  cv::Point2d center{0.0, 0.0};
  std::array<cv::Point2d, 4> corners{};
  cv::Point2d world{0.0, 0.0};
  double yaw_rad{0.0};
  bool pnp_valid{false};
  cv::Point3d camera_xyz{0.0, 0.0, 0.0};
  std::array<double, 4> camera_quat_xyzw{0.0, 0.0, 0.0, 1.0};
  std::array<double, 3> camera_rpy_rad{0.0, 0.0, 0.0};
  std::array<double, 3> camera_rpy_deg{0.0, 0.0, 0.0};
  double pnp_reprojection_error_px{0.0};
  double pnp_mean_edge_px{0.0};
  double pnp_edge_z_estimate_m{0.0};
  double pnp_fx_used{0.0};
  double pnp_fy_used{0.0};
};

struct TagFamilyHandle
{
  std::string name;
  apriltag_family_t * family{nullptr};
};

struct CameraCalibration
{
  bool valid{false};
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  double marker_length_m{0.0};
  int image_width{0};
  int image_height{0};
  std::string camera_name;
};

struct TruthWorld
{
  std::optional<double> x_m;
  std::optional<double> y_m;
  std::optional<double> z_m;
  std::optional<double> roll_deg;
  std::optional<double> pitch_deg;
  std::optional<double> yaw_deg;
};

struct CaptureRequest
{
  std::string request_id;
  std::string session_id;
  std::optional<int> tag_id;
  std::string operator_name;
  TruthWorld truth_world;
  std::string note;
  uint64_t after_frame_seq{0};
};

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

std::string json_optional_double(const std::optional<double> & value)
{
  if (!value) {
    return "null";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(9) << *value;
  return out.str();
}

std::string csv_escape(const std::string & input)
{
  if (input.find_first_of(",\"\n\r") == std::string::npos) {
    return input;
  }
  std::ostringstream out;
  out << "\"";
  for (const char character : input) {
    out << (character == '"' ? "\"\"" : std::string(1, character));
  }
  out << "\"";
  return out.str();
}

std::string csv_optional_double(const std::optional<double> & value)
{
  return value ? json_optional_double(value) : "";
}

std::string json_int_array(const std::vector<int> & values)
{
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
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
    if (i > 0) {
      out << ",";
    }
    out << json_string(values[i]);
  }
  out << "]";
  return out.str();
}

std::string json_point_or_null(bool valid, const cv::Point2d & point)
{
  if (!valid) {
    return "null";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << "[" << point.x << "," << point.y << "]";
  return out.str();
}

std::string json_world_yaw_or_null(bool valid, const cv::Point2d & point, double yaw)
{
  if (!valid) {
    return "null";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << "[" << point.x << "," << point.y << "," << yaw << "]";
  return out.str();
}

std::string json_point3_or_null(bool valid, const cv::Point3d & point)
{
  if (!valid) {
    return "null";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << "[" << point.x << "," << point.y << "," << point.z << "]";
  return out.str();
}

std::string json_quat_or_null(bool valid, const std::array<double, 4> & quat_xyzw)
{
  if (!valid) {
    return "null";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << "[" << quat_xyzw[0] << "," << quat_xyzw[1] << ","
      << quat_xyzw[2] << "," << quat_xyzw[3] << "]";
  return out.str();
}

std::string json_double3_or_null(bool valid, const std::array<double, 3> & values)
{
  if (!valid) {
    return "null";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << "[" << values[0] << "," << values[1] << "," << values[2] << "]";
  return out.str();
}

std::string json_detection_details(const std::vector<DetectionResult> & results)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "[";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto & result = results[i];
    if (i > 0) {
      out << ",";
    }
    out << "{";
    out << "\"tag_id\":" << result.tag_id << ",";
    out << "\"target_match\":" << (result.target_match ? "true" : "false") << ",";
    out << "\"family\":" << json_string(result.family) << ",";
    out << "\"hamming\":" << result.hamming << ",";
    out << "\"decision_margin\":" << result.decision_margin << ",";
    out << "\"pixel_xy\":" << json_point_or_null(result.detected, result.center) << ",";
    out << "\"world_xy_yaw\":" << json_world_yaw_or_null(result.detected, result.world, result.yaw_rad) << ",";
    out << "\"pnp_valid\":" << (result.pnp_valid ? "true" : "false") << ",";
    out << "\"pnp_camera_xyz\":" << json_point3_or_null(result.pnp_valid, result.camera_xyz) << ",";
    out << "\"pnp_camera_quat_xyzw\":" << json_quat_or_null(result.pnp_valid, result.camera_quat_xyzw) << ",";
    out << "\"pnp_camera_rpy_deg\":" << json_double3_or_null(result.pnp_valid, result.camera_rpy_deg) << ",";
    out << "\"pnp_reprojection_error_px\":"
        << (result.pnp_valid ? std::to_string(result.pnp_reprojection_error_px) : "null");
    out << "}";
  }
  out << "]";
  return out.str();
}

std::string fourcc_to_string(int value)
{
  std::string out(4, ' ');
  out[0] = static_cast<char>(value & 0xFF);
  out[1] = static_cast<char>((value >> 8) & 0xFF);
  out[2] = static_cast<char>((value >> 16) & 0xFF);
  out[3] = static_cast<char>((value >> 24) & 0xFF);
  while (!out.empty() && out.back() == ' ') {
    out.pop_back();
  }
  return out;
}

std::string canonical_device_path(const std::string & device)
{
  try {
    const fs::path path(device);
    if (fs::exists(path)) {
      return fs::canonical(path).string();
    }
  } catch (const std::exception &) {
  }
  return device;
}

std::string trim_copy(const std::string & value)
{
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

int video_device_index(const std::string & device)
{
  const std::string prefix = "/dev/video";
  if (device.rfind(prefix, 0) != 0 || device.size() <= prefix.size()) {
    return -1;
  }
  const std::string suffix = device.substr(prefix.size());
  if (!std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
    return -1;
  }
  try {
    return std::stoi(suffix);
  } catch (const std::exception &) {
    return -1;
  }
}

bool parse_bool(const YAML::Node & node)
{
  if (node.IsScalar()) {
    const auto text = node.as<std::string>();
    return text == "1" || text == "true" || text == "True" || text == "yes" || text == "on" ||
           text == "enabled";
  }
  return node.as<bool>();
}

fs::path resolve_path(const std::string & value)
{
  fs::path path(value);
  if (path.is_absolute()) {
    return path;
  }

  if (const char * repo_root = std::getenv("FINSSIM_REPO_ROOT")) {
    const fs::path root(repo_root);
    const std::vector<fs::path> candidates = {
      root / "ros2_ws" / "src" / "perception" / path,
      root / "ros2_ws" / "src" / path,
    };
    for (const auto & candidate : candidates) {
      if (fs::exists(candidate)) {
        return candidate;
      }
    }
  }

  try {
    const auto share = fs::path(ament_index_cpp::get_package_share_directory("perception"));
    const auto candidate = share / path;
    if (fs::exists(candidate)) {
      return candidate;
    }
  } catch (const std::exception &) {
  }

  return path;
}

fs::path default_dataset_root()
{
  if (const char * repo_root = std::getenv("FINSSIM_REPO_ROOT")) {
    return fs::path(repo_root) / "artifacts" / "datasets" / "apriltag_pnp_truth";
  }
  return fs::current_path() / "artifacts" / "datasets" / "apriltag_pnp_truth";
}

fs::path resolve_dataset_root(const std::string & value)
{
  if (value.empty()) {
    throw std::runtime_error("dataset_root must not be empty");
  }
  fs::path path(value);
  if (value == "~" || value.rfind("~/", 0) == 0) {
    if (const char * home = std::getenv("HOME")) {
      path = fs::path(home) / value.substr(value == "~" ? 1 : 2);
    }
  }
  if (path.is_relative() && std::getenv("FINSSIM_REPO_ROOT")) {
    path = fs::path(std::getenv("FINSSIM_REPO_ROOT")) / path;
  }
  return fs::absolute(path).lexically_normal();
}

std::string sanitize_path_component(const std::string & value, const std::string & fallback)
{
  std::string out;
  for (const unsigned char character : value) {
    if (std::isalnum(character) || character == '-' || character == '_') {
      out.push_back(static_cast<char>(character));
    } else if (character == '.' || character == ' ') {
      out.push_back('_');
    }
  }
  return out.empty() ? fallback : out;
}

std::string make_time_session_id()
{
  const std::time_t raw = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm time_info{};
  localtime_r(&raw, &time_info);
  std::ostringstream out;
  out << std::put_time(&time_info, "%Y%m%d_%H%M%S");
  return out.str();
}

std::optional<double> yaml_optional_double(const YAML::Node & node)
{
  if (!node || node.IsNull()) {
    return std::nullopt;
  }
  if (node.IsScalar()) {
    const std::string value = trim_copy(node.as<std::string>());
    if (value.empty() || value == "null" || value == "None" || value == "none") {
      return std::nullopt;
    }
  }
  return node.as<double>();
}

std::string yaml_optional_string(const YAML::Node & node)
{
  return (!node || node.IsNull()) ? "" : node.as<std::string>();
}

std::string truth_world_json(const TruthWorld & truth)
{
  std::ostringstream out;
  out << "{\"x_m\":" << json_optional_double(truth.x_m)
      << ",\"y_m\":" << json_optional_double(truth.y_m)
      << ",\"z_m\":" << json_optional_double(truth.z_m)
      << ",\"roll_deg\":" << json_optional_double(truth.roll_deg)
      << ",\"pitch_deg\":" << json_optional_double(truth.pitch_deg)
      << ",\"yaw_deg\":" << json_optional_double(truth.yaw_deg) << "}";
  return out.str();
}

std::vector<std::string> normalize_family_names(std::vector<std::string> names, const std::string & fallback)
{
  if (names.empty()) {
    names.push_back(fallback);
  }
  std::vector<std::string> out;
  for (const auto & name : names) {
    if (name.empty()) {
      continue;
    }
    if (std::find(out.begin(), out.end(), name) == out.end()) {
      out.push_back(name);
    }
  }
  if (out.empty()) {
    out.push_back(fallback);
  }
  return out;
}

std::vector<int> normalize_target_ids(const std::vector<int64_t> & ids, int fallback)
{
  std::vector<int> out;
  if (ids.empty()) {
    out.push_back(fallback);
  }
  for (const auto id : ids) {
    const int normalized = static_cast<int>(id);
    if (std::find(out.begin(), out.end(), normalized) == out.end()) {
      out.push_back(normalized);
    }
  }
  if (out.empty()) {
    out.push_back(-1);
  }
  return out;
}

bool target_matches(const std::vector<int> & target_ids, int id)
{
  if (target_ids.empty()) {
    return true;
  }
  if (std::find(target_ids.begin(), target_ids.end(), -1) != target_ids.end()) {
    return true;
  }
  return std::find(target_ids.begin(), target_ids.end(), id) != target_ids.end();
}

cv::Mat load_homography(const fs::path & path)
{
  const YAML::Node data = YAML::LoadFile(path.string());
  if (data["pixel_to_world_homography"]) {
    const auto values = data["pixel_to_world_homography"].as<std::vector<double>>();
    if (values.size() != 9) {
      throw std::runtime_error("pixel_to_world_homography must contain 9 values");
    }
    return cv::Mat(3, 3, CV_64F, const_cast<double *>(values.data())).clone();
  }

  if (!data["image_points"] || !data["world_points"]) {
    throw std::runtime_error("homography yaml must contain pixel_to_world_homography or image_points/world_points");
  }
  std::vector<cv::Point2d> image_points;
  std::vector<cv::Point2d> world_points;
  for (const auto & node : data["image_points"]) {
    image_points.emplace_back(node[0].as<double>(), node[1].as<double>());
  }
  for (const auto & node : data["world_points"]) {
    world_points.emplace_back(node[0].as<double>(), node[1].as<double>());
  }
  if (image_points.size() < 4 || image_points.size() != world_points.size()) {
    throw std::runtime_error("invalid homography point pairs");
  }
  const cv::Mat h = cv::findHomography(image_points, world_points, 0);
  if (h.empty()) {
    throw std::runtime_error("failed to compute homography");
  }
  return h;
}

cv::Point2d pixel_to_world(const cv::Mat & h, const cv::Point2d & pixel)
{
  const cv::Mat p = (cv::Mat_<double>(3, 1) << pixel.x, pixel.y, 1.0);
  cv::Mat w = h * p;
  const double scale = w.at<double>(2, 0);
  if (std::abs(scale) < 1e-12) {
    throw std::runtime_error("homogeneous world scale is zero");
  }
  return {w.at<double>(0, 0) / scale, w.at<double>(1, 0) / scale};
}

cv::Mat load_matrix_node(const YAML::Node & node, int expected_rows, int expected_cols, const std::string & name)
{
  if (!node || !node["data"]) {
    throw std::runtime_error(name + " missing data");
  }
  const int rows = node["rows"] ? node["rows"].as<int>() : expected_rows;
  const int cols = node["cols"] ? node["cols"].as<int>() : expected_cols;
  const auto values = node["data"].as<std::vector<double>>();
  if (rows <= 0 || cols <= 0 || values.size() != static_cast<size_t>(rows * cols)) {
    throw std::runtime_error(name + " has invalid shape/data length");
  }
  return cv::Mat(rows, cols, CV_64F, const_cast<double *>(values.data())).clone();
}

CameraCalibration load_camera_calibration(const fs::path & path)
{
  const YAML::Node data = YAML::LoadFile(path.string());
  CameraCalibration calibration;
  calibration.camera_matrix = load_matrix_node(data["camera_matrix"], 3, 3, "camera_matrix");
  calibration.distortion_coefficients =
    load_matrix_node(data["distortion_coefficients"], 1, 5, "distortion_coefficients").reshape(1, 1);
  calibration.marker_length_m = data["marker_length_m"] ? data["marker_length_m"].as<double>() : 0.0;
  calibration.image_width = data["image_width"] ? data["image_width"].as<int>() : 0;
  calibration.image_height = data["image_height"] ? data["image_height"].as<int>() : 0;
  calibration.camera_name = data["camera_name"] ? data["camera_name"].as<std::string>() : path.stem().string();
  calibration.valid = true;
  return calibration;
}

std::array<double, 4> rotation_matrix_to_quaternion_xyzw(const cv::Mat & rotation)
{
  const double m00 = rotation.at<double>(0, 0);
  const double m01 = rotation.at<double>(0, 1);
  const double m02 = rotation.at<double>(0, 2);
  const double m10 = rotation.at<double>(1, 0);
  const double m11 = rotation.at<double>(1, 1);
  const double m12 = rotation.at<double>(1, 2);
  const double m20 = rotation.at<double>(2, 0);
  const double m21 = rotation.at<double>(2, 1);
  const double m22 = rotation.at<double>(2, 2);

  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 1.0;
  const double trace = m00 + m11 + m22;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }

  const double norm = std::sqrt(x * x + y * y + z * z + w * w);
  if (norm > 1e-12) {
    x /= norm;
    y /= norm;
    z /= norm;
    w /= norm;
  }
  return {x, y, z, w};
}

std::array<double, 3> rotation_matrix_to_rpy_rad(const cv::Mat & rotation)
{
  const double r00 = rotation.at<double>(0, 0);
  const double r10 = rotation.at<double>(1, 0);
  const double r20 = rotation.at<double>(2, 0);
  const double r21 = rotation.at<double>(2, 1);
  const double r22 = rotation.at<double>(2, 2);
  const double r01 = rotation.at<double>(0, 1);
  const double r11 = rotation.at<double>(1, 1);

  const double sy = std::sqrt(r00 * r00 + r10 * r10);
  const bool singular = sy < 1e-9;

  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  if (!singular) {
    roll = std::atan2(r21, r22);
    pitch = std::atan2(-r20, sy);
    yaw = std::atan2(r10, r00);
  } else {
    roll = std::atan2(-r01, r11);
    pitch = std::atan2(-r20, sy);
    yaw = 0.0;
  }
  return {roll, pitch, yaw};
}

std::array<double, 3> radians_to_degrees(const std::array<double, 3> & values)
{
  constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
  return {values[0] * kRadToDeg, values[1] * kRadToDeg, values[2] * kRadToDeg};
}

double point_distance(const cv::Point2d & a, const cv::Point2d & b)
{
  const auto delta = a - b;
  return std::sqrt(delta.x * delta.x + delta.y * delta.y);
}

double mean_square_edge_px(const std::array<cv::Point2d, 4> & corners)
{
  return 0.25 * (
    point_distance(corners[0], corners[1]) +
    point_distance(corners[1], corners[2]) +
    point_distance(corners[2], corners[3]) +
    point_distance(corners[3], corners[0]));
}

apriltag_family_t * create_family(const std::string & family)
{
  if (family == "tag16h5" || family == "DICT_APRILTAG_16H5") {
    return tag16h5_create();
  }
  if (family == "tag25h9" || family == "DICT_APRILTAG_25H9") {
    return tag25h9_create();
  }
  if (family == "tag36h10" || family == "DICT_APRILTAG_36H10") {
    return tag36h10_create();
  }
  if (family == "tag36h11" || family == "DICT_APRILTAG_36H11") {
    return tag36h11_create();
  }
  throw std::runtime_error("unsupported AprilTag family: " + family);
}

void destroy_family(const std::string & family, apriltag_family_t * tf)
{
  if (!tf) {
    return;
  }
  if (family == "tag16h5" || family == "DICT_APRILTAG_16H5") {
    tag16h5_destroy(tf);
  } else if (family == "tag25h9" || family == "DICT_APRILTAG_25H9") {
    tag25h9_destroy(tf);
  } else if (family == "tag36h10" || family == "DICT_APRILTAG_36H10") {
    tag36h10_destroy(tf);
  } else {
    tag36h11_destroy(tf);
  }
}
}  // namespace

class DirectAprilTagNode : public rclcpp::Node
{
public:
  DirectAprilTagNode()
  : Node("direct_apriltag_node")
  {
    settings_.device = declare_parameter<std::string>("device", settings_.device);
    settings_.width = declare_parameter<int>("width", settings_.width);
    settings_.height = declare_parameter<int>("height", settings_.height);
    settings_.fps = declare_parameter<double>("fps", settings_.fps);
    settings_.fourcc = declare_parameter<std::string>("fourcc", settings_.fourcc);
    settings_.enabled = declare_parameter<bool>("enabled", true);

    // Offline evaluation uses this exact native detector on an immutable list
    // of images.  It deliberately lives in this node (rather than a Python
    // reimplementation) so corner ordering and detector parameters remain
    // identical to the deployed processing path.
    replay_image_list_file_ = declare_parameter<std::string>("replay_image_list_file", "");
    replay_rate_hz_ = declare_parameter<double>("replay_rate_hz", 2.0);
    replay_event_topic_ = declare_parameter<std::string>("replay_event_topic", "");

    family_name_ = declare_parameter<std::string>("family", "tag36h11");
    const auto tag_family_name = declare_parameter<std::string>("tag_family", family_name_);
    const auto configured_families =
      declare_parameter<std::vector<std::string>>("tag_families", std::vector<std::string>{});
    family_names_ = normalize_family_names(configured_families, tag_family_name.empty() ? family_name_ : tag_family_name);

    target_tag_id_ = declare_parameter<int>("target_tag_id", -1);
    target_tag_ids_ = normalize_target_ids(
      declare_parameter<std::vector<int64_t>>("target_tag_ids", std::vector<int64_t>{}), target_tag_id_);
    max_hamming_ = declare_parameter<int>("max_hamming", 0);
    nthreads_ = declare_parameter<int>("nthreads", 4);
    quad_decimate_ = declare_parameter<double>("quad_decimate", 1.5);
    quad_sigma_ = declare_parameter<double>("quad_sigma", 0.0);
    refine_edges_ = declare_parameter<bool>("refine_edges", true);
    decode_sharpening_ = declare_parameter<double>("decode_sharpening", 0.25);

    homography_file_ = declare_parameter<std::string>("homography_file", "calibration/pool_homography.yaml");
    camera_calibration_file_ = declare_parameter<std::string>("camera_calibration_file", "");
    marker_length_m_ = declare_parameter<double>("marker_length_m", 0.0);
    publish_pnp_pose_ = declare_parameter<bool>("publish_pnp_pose", true);
    pnp_pose_topic_ = declare_parameter<std::string>("pnp_pose_topic", "/finsrov/vision/pose_3d_camera");
    pnp_array_topic_ =
      declare_parameter<std::string>("pnp_array_topic", "/finsrov/vision/tag_poses_3d_camera");
    detection2d_topic_ =
      declare_parameter<std::string>("detection2d_topic", "/finsrov/vision/tag_detections_2d");
    pnp_frame_id_ = declare_parameter<std::string>("pnp_frame_id", "finsrov_camera");
    pnp_covariance_xyz_ = declare_parameter<double>("pnp_covariance_xyz", 0.0025);

    status_topic_ = declare_parameter<std::string>("status_topic", "/finsrov/vision/status");
    camera_status_topic_ = declare_parameter<std::string>("camera_status_topic", "/finsrov/camera/status");
    debug_image_topic_ = declare_parameter<std::string>("debug_image_topic", "/finsrov/camera/debug/compressed");
    dataset_root_ = resolve_dataset_root(
      declare_parameter<std::string>("dataset_root", default_dataset_root().string()));
    dataset_capture_request_topic_ = declare_parameter<std::string>(
      "dataset_capture_request_topic", "/finsrov/dataset/capture_request");
    dataset_capture_result_topic_ = declare_parameter<std::string>(
      "dataset_capture_result_topic", "/finsrov/dataset/capture_result");
    dataset_truth_frame_id_ = declare_parameter<std::string>("dataset_truth_frame_id", "pool_world");

    debug_image_enabled_ = declare_parameter<bool>("debug_image_enabled", true);
    debug_image_rate_hz_ = declare_parameter<double>("debug_image_rate_hz", 5.0);
    debug_image_width_ = declare_parameter<int>("debug_image_width", 640);
    debug_jpeg_quality_ = declare_parameter<int>("debug_jpeg_quality", 80);
    status_rate_hz_ = declare_parameter<double>("status_rate_hz", 5.0);

    homography_path_ = resolve_path(homography_file_);
    homography_ = load_homography(homography_path_);
    if (!camera_calibration_file_.empty()) {
      camera_calibration_path_ = resolve_path(camera_calibration_file_);
      camera_calibration_ = load_camera_calibration(camera_calibration_path_);
      if (marker_length_m_ <= 0.0) {
        marker_length_m_ = camera_calibration_.marker_length_m;
      }
    }

    if (!replay_image_list_file_.empty()) {
      replay_image_list_path_ = resolve_path(replay_image_list_file_);
      std::ifstream stream(replay_image_list_path_);
      if (!stream.is_open()) {
        throw std::runtime_error("failed to open replay image list: " + replay_image_list_path_.string());
      }
      std::string line;
      while (std::getline(stream, line)) {
        const std::string path = trim_copy(line);
        if (!path.empty() && path.front() != '#') {
          fs::path image_path(path);
          if (image_path.is_relative()) {
            image_path = replay_image_list_path_.parent_path() / image_path;
          }
          replay_image_paths_.push_back(image_path);
        }
      }
      if (replay_image_paths_.empty()) {
        throw std::runtime_error("replay image list is empty: " + replay_image_list_path_.string());
      }
      replay_mode_ = true;
      if (replay_event_topic_.empty()) {
        throw std::runtime_error("replay_event_topic must be non-empty when replay_image_list_file is set");
      }
    }
    pnp_enabled_ = camera_calibration_.valid && marker_length_m_ > 0.0;
    if (!pnp_enabled_) {
      RCLCPP_WARN(
        get_logger(),
        "AprilTag PnP 3D pose disabled: camera_calibration_file=%s, marker_length_m=%.6f",
        camera_calibration_file_.empty() ? "<empty>" : camera_calibration_file_.c_str(), marker_length_m_);
    }

    detector_ = apriltag_detector_create();
    for (const auto & name : family_names_) {
      auto * family = create_family(name);
      tag_families_.push_back({name, family});
      apriltag_detector_add_family_bits(detector_, family, max_hamming_);
    }
    detector_->nthreads = std::max(1, nthreads_);
    detector_->quad_decimate = static_cast<float>(quad_decimate_);
    detector_->quad_sigma = static_cast<float>(quad_sigma_);
    detector_->refine_edges = refine_edges_;
    detector_->decode_sharpening = decode_sharpening_;

    pnp_pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(pnp_pose_topic_, 10);
    pnp_array_pub_ = create_publisher<msgs::msg::AprilTagDetection3DArray>(pnp_array_topic_, 10);
    detection2d_pub_ = create_publisher<msgs::msg::AprilTagDetection2DArray>(detection2d_topic_, 10);
    status_pub_ = create_publisher<std_msgs::msg::String>(status_topic_, 10);
    camera_status_pub_ = create_publisher<std_msgs::msg::String>(camera_status_topic_, 10);
    debug_image_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(debug_image_topic_, 2);
    dataset_result_pub_ = create_publisher<std_msgs::msg::String>(dataset_capture_result_topic_, 10);
    if (replay_mode_) {
      replay_event_pub_ = create_publisher<std_msgs::msg::String>(replay_event_topic_, 10);
    }
    control_sub_ = create_subscription<std_msgs::msg::String>(
      "/finsrov/camera/control", 10,
      [this](const std_msgs::msg::String::SharedPtr msg) { handle_control(msg->data); });
    dataset_request_sub_ = create_subscription<std_msgs::msg::String>(
      dataset_capture_request_topic_, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) { handle_capture_request(msg->data); });
    dataset_root_parameter_callback_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto & parameter : parameters) {
          if (parameter.get_name() != "dataset_root") {
            continue;
          }
          if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
            result.successful = false;
            result.reason = "dataset_root must be a string";
            return result;
          }
          try {
            const fs::path root = resolve_dataset_root(parameter.as_string());
            std::lock_guard<std::mutex> lock(dataset_root_mutex_);
            dataset_root_ = root;
          } catch (const std::exception & exc) {
            result.successful = false;
            result.reason = exc.what();
            return result;
          }
        }
        return result;
      });

    last_status_time_ = now();
    last_frame_time_ = now();
    last_debug_frame_time_ = now();

    running_.store(true);
    capture_thread_ = std::thread([this]() { capture_loop(); });
    detection_thread_ = std::thread([this]() { detection_loop(); });

    status_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(status_rate_hz_, 0.1)),
      [this]() { publish_camera_status(); });
    debug_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(debug_image_rate_hz_, 0.1)),
      [this]() { publish_debug_image(); });

    RCLCPP_INFO(
      get_logger(),
      "direct_apriltag_node started: source=%s, device=%s, size=%dx%d, fps=%.2f, family=%s, "
      "target_tag_ids=%s, nthreads=%d, quad_decimate=%.2f, pnp=%s, image_raw_pub=<disabled>, debug=%s",
      replay_mode_ ? ("replay(" + std::to_string(replay_image_paths_.size()) + ")").c_str() : "camera",
      settings_.device.c_str(), settings_.width, settings_.height, settings_.fps,
      json_string_array(family_names_).c_str(), json_int_array(target_tag_ids_).c_str(), nthreads_,
      quad_decimate_, pnp_enabled_ ? pnp_pose_topic_.c_str() : "disabled",
      debug_image_enabled_ ? "compressed" : "disabled");
  }

  ~DirectAprilTagNode() override
  {
    running_.store(false);
    frame_cv_.notify_all();
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    if (detection_thread_.joinable()) {
      detection_thread_.join();
    }
    {
      std::lock_guard<std::mutex> lock(capture_mutex_);
      if (capture_.isOpened()) {
        capture_.release();
      }
    }
    if (detector_) {
      apriltag_detector_destroy(detector_);
      detector_ = nullptr;
    }
    for (auto & item : tag_families_) {
      destroy_family(item.name, item.family);
      item.family = nullptr;
    }
    tag_families_.clear();
  }

private:
  void capture_loop()
  {
    if (replay_mode_) {
      capture_replay_loop();
      return;
    }
    while (running_.load()) {
      Settings local_settings;
      bool reopen = false;
      {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        local_settings = settings_;
        reopen = reopen_requested_;
        reopen_requested_ = false;
      }

      if (!local_settings.enabled) {
        close_capture();
        std::this_thread::sleep_for(100ms);
        continue;
      }

      if (!capture_.isOpened() || reopen) {
        open_capture(local_settings);
      }

      cv::Mat frame;
      {
        std::lock_guard<std::mutex> lock(capture_mutex_);
        if (!capture_.isOpened()) {
          std::this_thread::sleep_for(100ms);
          continue;
        }
        if (!capture_.read(frame) || frame.empty()) {
          last_error_ = "camera read failed; will reopen";
          capture_.release();
          std::this_thread::sleep_for(50ms);
          continue;
        }
      }

      const auto frame_ptr = std::make_shared<cv::Mat>(std::move(frame));
      {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = frame_ptr;
        latest_frame_seq_++;
        frame_count_++;
        last_frame_time_ = now();
        const cv::Scalar mean = cv::mean(*frame_ptr);
        last_frame_mean_ = (mean[0] + mean[1] + mean[2]) / 3.0;
        cv::Mat stats_gray;
        if (frame_ptr->channels() == 1) {
          stats_gray = *frame_ptr;
        } else {
          cv::cvtColor(*frame_ptr, stats_gray, cv::COLOR_BGR2GRAY);
        }
        double min_value = 0.0;
        double max_value = 0.0;
        cv::minMaxLoc(stats_gray, &min_value, &max_value);
        last_frame_min_ = min_value;
        last_frame_max_ = max_value;
        last_error_.clear();
      }
      frame_cv_.notify_one();
    }
  }

  void capture_replay_loop()
  {
    const double interval_sec = 1.0 / std::max(replay_rate_hz_, 0.1);
    auto next_frame_time = std::chrono::steady_clock::now();
    for (size_t replay_index = 0; running_.load() && replay_index < replay_image_paths_.size(); ++replay_index) {
      const fs::path & image_path = replay_image_paths_[replay_index];
      cv::Mat frame = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (frame.empty()) {
        last_error_ = "failed to read replay image " + image_path.string();
        publish_replay_event(replay_index, image_path, "image_read_failed", true);
        next_frame_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(interval_sec));
        std::this_thread::sleep_until(next_frame_time);
        continue;
      }

      const auto frame_ptr = std::make_shared<cv::Mat>(std::move(frame));
      uint64_t seq = 0;
      {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = frame_ptr;
        latest_frame_replay_index_ = replay_index;
        latest_frame_seq_++;
        seq = latest_frame_seq_;
        frame_count_++;
        last_frame_time_ = now();
        actual_width_ = frame_ptr->cols;
        actual_height_ = frame_ptr->rows;
        actual_fps_ = replay_rate_hz_;
        actual_fourcc_ = "REPLAY";
        const cv::Scalar mean = cv::mean(*frame_ptr);
        last_frame_mean_ = (mean[0] + mean[1] + mean[2]) / 3.0;
        cv::Mat stats_gray;
        cv::cvtColor(*frame_ptr, stats_gray, cv::COLOR_BGR2GRAY);
        double min_value = 0.0;
        double max_value = 0.0;
        cv::minMaxLoc(stats_gray, &min_value, &max_value);
        last_frame_min_ = min_value;
        last_frame_max_ = max_value;
        last_error_.clear();
      }
      // Publish before exposing the image.  The event is an audit/progress
      // signal; association is finalized with replay_index and frame_stamp in
      // the native detection status after the downstream pose node runs.
      publish_replay_event(replay_index, image_path, "frame_queued", false);
      frame_cv_.notify_one();

      {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        const bool processed = frame_cv_.wait_for(
          lock, 15s,
          [&]() { return !running_.load() || latest_processed_seq_ >= seq; });
        if (!processed && running_.load()) {
          last_error_ = "timeout waiting for replay frame " + std::to_string(replay_index) + " to be processed";
          RCLCPP_WARN(get_logger(), "%s", last_error_.c_str());
        }
      }
      next_frame_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(interval_sec));
      std::this_thread::sleep_until(next_frame_time);
    }
    replay_finished_.store(true);
    publish_replay_event(replay_image_paths_.size(), fs::path(), "complete", false);
    RCLCPP_INFO(get_logger(), "offline replay complete: %zu images", replay_image_paths_.size());
    while (running_.load()) {
      std::this_thread::sleep_for(100ms);
    }
  }

  void publish_replay_event(
    size_t replay_index,
    const fs::path & image_path,
    const std::string & phase,
    bool error)
  {
    if (!replay_event_pub_) {
      return;
    }
    std_msgs::msg::String msg;
    std::ostringstream out;
    out << "{\"replay_index\":" << replay_index << ",\"phase\":" << json_string(phase)
        << ",\"image_path\":" << json_string(image_path.string())
        << ",\"error\":" << (error ? "true" : "false") << "}";
    msg.data = out.str();
    replay_event_pub_->publish(msg);
  }

  void detection_loop()
  {
    uint64_t processed_seq = 0;
    while (running_.load()) {
      std::shared_ptr<cv::Mat> frame;
      uint64_t seq = 0;
      std::optional<size_t> replay_index;
      {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        frame_cv_.wait(lock, [&]() { return !running_.load() || latest_frame_seq_ != processed_seq; });
        if (!running_.load()) {
          return;
        }
        frame = latest_frame_;
        seq = latest_frame_seq_;
        replay_index = latest_frame_replay_index_;
      }
      if (!frame || frame->empty()) {
        continue;
      }
      processed_seq = seq;
      process_frame(frame, seq, replay_index);
      {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_processed_seq_ = seq;
      }
      frame_cv_.notify_all();
    }
  }

  void process_frame(
    const std::shared_ptr<cv::Mat> & frame, uint64_t frame_seq,
    const std::optional<size_t> & replay_index)
  {
    const auto start = now();
    cv::Mat gray;
    if (frame->channels() == 1) {
      gray = *frame;
    } else {
      cv::cvtColor(*frame, gray, cv::COLOR_BGR2GRAY);
    }
    if (!gray.isContinuous()) {
      gray = gray.clone();
    }

    image_u8_t * image = image_u8_create(static_cast<unsigned int>(gray.cols), static_cast<unsigned int>(gray.rows));
    if (!image) {
      last_error_ = "failed to allocate AprilTag grayscale image";
      return;
    }
    for (int y = 0; y < gray.rows; ++y) {
      std::memcpy(&image->buf[y * image->stride], gray.ptr<uint8_t>(y), static_cast<size_t>(gray.cols));
    }

    zarray_t * detections = apriltag_detector_detect(detector_, image);
    DetectionResult result;
    result.family = family_names_.empty() ? "none" : family_names_.front();
    std::vector<int> detected_ids;
    std::vector<DetectionResult> all_results;
    std::vector<DetectionResult> pnp_results;
    double best_margin = -1.0;
    const int count = zarray_size(detections);
    detected_ids.reserve(static_cast<size_t>(count));
    all_results.reserve(static_cast<size_t>(count));
    pnp_results.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      apriltag_detection_t * det = nullptr;
      zarray_get(detections, i, &det);
      if (!det) {
        continue;
      }
      detected_ids.push_back(det->id);
      if (det->hamming > max_hamming_) {
        continue;
      }
      DetectionResult candidate;
      candidate.family = result.family;
      fill_detection_result(det, candidate);
      candidate.detected = true;
      candidate.target_match = target_matches(target_tag_ids_, det->id);
      all_results.push_back(candidate);
      if (candidate.target_match && candidate.pnp_valid) {
        pnp_results.push_back(candidate);
      }
      if (candidate.target_match && det->decision_margin > best_margin) {
        result = candidate;
        best_margin = det->decision_margin;
      }
    }

    result.detected_ids = detected_ids;
    if (result.detected) {
      result.detected = true;
      detection_count_++;
      publish_pnp_pose(result);
    } else {
      miss_count_++;
    }
    const auto frame_stamp = now();
    publish_detection2d_array(all_results, frame_stamp);
    publish_pnp_array(pnp_results);
    publish_detection_status(result, all_results, replay_index, frame_stamp);
    process_pending_capture(*frame, frame_seq, all_results);
    update_debug_frame(*frame, all_results);
    apriltag_detections_destroy(detections);
    image_u8_destroy(image);

    const double dt = std::max(1e-9, (now() - start).seconds());
    detection_fps_ = 0.9 * detection_fps_ + 0.1 * (1.0 / dt);
  }

  void fill_detection_result(const apriltag_detection_t * det, DetectionResult & result)
  {
    result.tag_id = det->id;
    result.hamming = det->hamming;
    result.decision_margin = det->decision_margin;
    if (det->family && det->family->name) {
      result.family = det->family->name;
    }
    result.center = {det->c[0], det->c[1]};
    for (int i = 0; i < 4; ++i) {
      result.corners[static_cast<size_t>(i)] = {det->p[i][0], det->p[i][1]};
    }
    result.world = pixel_to_world(homography_, result.center);
    const auto edge = result.corners[1] - result.corners[0];
    result.yaw_rad = std::atan2(edge.y, edge.x);
    fill_pnp_result(result);
  }

  void fill_pnp_result(DetectionResult & result)
  {
    if (!pnp_enabled_) {
      return;
    }

    cv::Mat camera_matrix = camera_calibration_.camera_matrix.clone();
    if (
      camera_calibration_.image_width > 0 && camera_calibration_.image_height > 0 &&
      actual_width_ > 0 && actual_height_ > 0 &&
      (actual_width_ != camera_calibration_.image_width || actual_height_ != camera_calibration_.image_height))
    {
      const double sx = static_cast<double>(actual_width_) / static_cast<double>(camera_calibration_.image_width);
      const double sy = static_cast<double>(actual_height_) / static_cast<double>(camera_calibration_.image_height);
      camera_matrix.at<double>(0, 0) *= sx;
      camera_matrix.at<double>(0, 1) *= sx;
      camera_matrix.at<double>(0, 2) *= sx;
      camera_matrix.at<double>(1, 1) *= sy;
      camera_matrix.at<double>(1, 2) *= sy;
    }
    result.pnp_fx_used = camera_matrix.at<double>(0, 0);
    result.pnp_fy_used = camera_matrix.at<double>(1, 1);

    const double half = marker_length_m_ * 0.5;
    // Keep the AprilTag detector's canonical corner order. apriltag_detection_t::H
    // maps ideal tag corners (-1,+1), (+1,+1), (+1,-1), (-1,-1) to det->p[0..3].
    // Reordering by image top-left destroys the physical tag frame yaw.
    const std::vector<cv::Point3d> object_points = {
      {-half, half, 0.0},
      {half, half, 0.0},
      {half, -half, 0.0},
      {-half, -half, 0.0},
    };
    std::vector<cv::Point2d> image_points;
    image_points.reserve(4);
    for (const auto & corner : result.corners) {
      image_points.push_back(corner);
    }
    result.pnp_mean_edge_px = mean_square_edge_px(result.corners);
    const double focal_avg = 0.5 * (result.pnp_fx_used + result.pnp_fy_used);
    if (result.pnp_mean_edge_px > 1e-9) {
      result.pnp_edge_z_estimate_m = focal_avg * marker_length_m_ / result.pnp_mean_edge_px;
    }

    cv::Mat rvec;
    cv::Mat tvec;
    bool ok = cv::solvePnP(
      object_points, image_points, camera_matrix,
      camera_calibration_.distortion_coefficients, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
    if (!ok) {
      ok = cv::solvePnP(
        object_points, image_points, camera_matrix,
        camera_calibration_.distortion_coefficients, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    }
    if (!ok) {
      return;
    }

    result.camera_xyz = {
      tvec.at<double>(0, 0),
      tvec.at<double>(1, 0),
      tvec.at<double>(2, 0),
    };
    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    result.camera_quat_xyzw = rotation_matrix_to_quaternion_xyzw(rotation);
    result.camera_rpy_rad = rotation_matrix_to_rpy_rad(rotation);
    result.camera_rpy_deg = radians_to_degrees(result.camera_rpy_rad);

    std::vector<cv::Point2d> projected;
    cv::projectPoints(
      object_points, rvec, tvec, camera_matrix,
      camera_calibration_.distortion_coefficients, projected);
    double error_sum = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
      const auto delta = projected[i] - image_points[i];
      error_sum += std::sqrt(delta.x * delta.x + delta.y * delta.y);
    }
    result.pnp_reprojection_error_px = projected.empty() ? 0.0 : error_sum / static_cast<double>(projected.size());
    result.pnp_valid = true;
  }

  void publish_pnp_pose(const DetectionResult & result)
  {
    if (!publish_pnp_pose_ || !result.pnp_valid) {
      return;
    }
    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = pnp_frame_id_;
    msg.pose.pose.position.x = result.camera_xyz.x;
    msg.pose.pose.position.y = result.camera_xyz.y;
    msg.pose.pose.position.z = result.camera_xyz.z;
    msg.pose.pose.orientation.x = result.camera_quat_xyzw[0];
    msg.pose.pose.orientation.y = result.camera_quat_xyzw[1];
    msg.pose.pose.orientation.z = result.camera_quat_xyzw[2];
    msg.pose.pose.orientation.w = result.camera_quat_xyzw[3];
    msg.pose.covariance[0] = pnp_covariance_xyz_;
    msg.pose.covariance[7] = pnp_covariance_xyz_;
    msg.pose.covariance[14] = pnp_covariance_xyz_;
    msg.pose.covariance[21] = 1e6;
    msg.pose.covariance[28] = 1e6;
    msg.pose.covariance[35] = 1e6;
    pnp_pose_pub_->publish(msg);
  }

  void publish_pnp_array(const std::vector<DetectionResult> & results)
  {
    if (!publish_pnp_pose_ || !pnp_enabled_) {
      return;
    }
    msgs::msg::AprilTagDetection3DArray msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = pnp_frame_id_;
    msg.detections.reserve(results.size());
    for (const auto & result : results) {
      if (!result.pnp_valid) {
        continue;
      }
      msgs::msg::AprilTagDetection3D detection;
      detection.tag_id = result.tag_id;
      detection.pose.pose.position.x = result.camera_xyz.x;
      detection.pose.pose.position.y = result.camera_xyz.y;
      detection.pose.pose.position.z = result.camera_xyz.z;
      detection.pose.pose.orientation.x = result.camera_quat_xyzw[0];
      detection.pose.pose.orientation.y = result.camera_quat_xyzw[1];
      detection.pose.pose.orientation.z = result.camera_quat_xyzw[2];
      detection.pose.pose.orientation.w = result.camera_quat_xyzw[3];
      detection.pose.covariance[0] = pnp_covariance_xyz_;
      detection.pose.covariance[7] = pnp_covariance_xyz_;
      detection.pose.covariance[14] = pnp_covariance_xyz_;
      detection.pose.covariance[21] = 1e6;
      detection.pose.covariance[28] = 1e6;
      detection.pose.covariance[35] = 1e6;
      detection.decision_margin = static_cast<float>(result.decision_margin);
      detection.reprojection_error_px = static_cast<float>(result.pnp_reprojection_error_px);
      msg.detections.push_back(detection);
    }
    pnp_array_pub_->publish(msg);
  }

  CaptureRequest parse_capture_request(const std::string & data)
  {
    const YAML::Node command = YAML::Load(data);
    if (!command.IsMap()) {
      throw std::runtime_error("capture request must be a JSON/YAML object");
    }
    CaptureRequest request;
    request.request_id = sanitize_path_component(
      yaml_optional_string(command["request_id"]), "request_" + std::to_string(now().nanoseconds()));
    request.session_id = sanitize_path_component(
      yaml_optional_string(command["session_id"]), make_time_session_id());
    if (command["tag_id"] && !command["tag_id"].IsNull()) {
      request.tag_id = command["tag_id"].as<int>();
    }
    request.operator_name = yaml_optional_string(command["operator"]);
    request.note = yaml_optional_string(command["note"]);
    const YAML::Node truth = command["truth_world"];
    if (truth && truth.IsMap()) {
      request.truth_world.x_m = yaml_optional_double(truth["x_m"]);
      request.truth_world.y_m = yaml_optional_double(truth["y_m"]);
      request.truth_world.z_m = yaml_optional_double(truth["z_m"]);
      request.truth_world.roll_deg = yaml_optional_double(truth["roll_deg"]);
      request.truth_world.pitch_deg = yaml_optional_double(truth["pitch_deg"]);
      request.truth_world.yaw_deg = yaml_optional_double(truth["yaw_deg"]);
    }
    return request;
  }

  void handle_capture_request(const std::string & data)
  {
    CaptureRequest request;
    try {
      request = parse_capture_request(data);
    } catch (const std::exception & exc) {
      CaptureRequest failed;
      failed.request_id = "invalid_request";
      failed.session_id = make_time_session_id();
      publish_capture_result(failed, false, "invalid_request", exc.what(), "", "", "");
      return;
    }
    {
      std::lock_guard<std::mutex> frame_lock(frame_mutex_);
      request.after_frame_seq = latest_frame_seq_;
    }
    {
      std::lock_guard<std::mutex> request_lock(capture_request_mutex_);
      if (pending_capture_request_) {
        publish_capture_result(
          request, false, "capture_pending",
          "another capture request is already waiting for the next processed frame", "", "", "");
        return;
      }
      pending_capture_request_ = request;
    }
    RCLCPP_INFO(
      get_logger(), "AprilTag dataset capture queued: request_id=%s session=%s after_frame_seq=%lu",
      request.request_id.c_str(), request.session_id.c_str(), static_cast<unsigned long>(request.after_frame_seq));
  }

  const DetectionResult * select_capture_detection(
    const CaptureRequest & request, const std::vector<DetectionResult> & detections) const
  {
    if (request.tag_id) {
      for (const auto & detection : detections) {
        if (detection.tag_id == *request.tag_id && detection.pnp_valid) {
          return &detection;
        }
      }
      return nullptr;
    }
    const DetectionResult * selected = nullptr;
    for (const auto & detection : detections) {
      if (detection.target_match && detection.pnp_valid &&
        (!selected || detection.decision_margin > selected->decision_margin))
      {
        selected = &detection;
      }
    }
    return selected;
  }

  void process_pending_capture(
    const cv::Mat & frame, uint64_t frame_seq, const std::vector<DetectionResult> & detections)
  {
    std::optional<CaptureRequest> request;
    {
      std::lock_guard<std::mutex> lock(capture_request_mutex_);
      if (!pending_capture_request_ || frame_seq <= pending_capture_request_->after_frame_seq) {
        return;
      }
      request = pending_capture_request_;
      pending_capture_request_.reset();
    }
    const DetectionResult * selected = select_capture_detection(*request, detections);
    if (!selected) {
      publish_capture_result(
        *request, false, request->tag_id ? "tag_not_found" : "no_pnp_detection",
        request->tag_id ? "requested tag was not detected with a valid PnP pose" :
        "no configured target tag had a valid PnP pose", "", "", "");
      return;
    }
    try {
      write_dataset_sample(*request, *selected, frame, frame_seq);
    } catch (const std::exception & exc) {
      publish_capture_result(*request, false, "write_failed", exc.what(), "", "", "");
    }
  }

  fs::path active_dataset_root() const
  {
    std::lock_guard<std::mutex> lock(dataset_root_mutex_);
    return dataset_root_;
  }

  std::string make_sample_id(const CaptureRequest & request, uint64_t frame_seq, const rclcpp::Time & stamp) const
  {
    return sanitize_path_component(
      "sample_" + std::to_string(stamp.nanoseconds()) + "_" + std::to_string(frame_seq) + "_" + request.request_id,
      "sample_" + std::to_string(frame_seq));
  }

  void ensure_session_manifest(const fs::path & session_dir, const CaptureRequest & request, const fs::path & root)
  {
    const fs::path manifest_path = session_dir / "manifest.json";
    if (fs::exists(manifest_path)) {
      return;
    }
    std::ofstream out(manifest_path);
    if (!out) {
      throw std::runtime_error("failed to open dataset manifest " + manifest_path.string());
    }
    out << "{\n"
        << "  \"dataset_format\":\"finsrov_apriltag_pnp_truth_v1\",\n"
        << "  \"session_id\":" << json_string(request.session_id) << ",\n"
        << "  \"dataset_root\":" << json_string(root.string()) << ",\n"
        << "  \"camera_frame_id\":" << json_string(pnp_frame_id_) << ",\n"
        << "  \"truth_frame_id\":" << json_string(dataset_truth_frame_id_) << ",\n"
        << "  \"image_format\":\"png\"\n}\n";
  }

  std::string dataset_sample_json(
    const CaptureRequest & request, const DetectionResult & result, uint64_t frame_seq,
    const rclcpp::Time & stamp, const fs::path & image_path, const fs::path & metadata_path) const
  {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9);
    out << "{\"sample_id\":" << json_string(metadata_path.stem().string())
        << ",\"request_id\":" << json_string(request.request_id)
        << ",\"session_id\":" << json_string(request.session_id)
        << ",\"ros_time_sec\":" << stamp.seconds()
        << ",\"frame_seq\":" << frame_seq
        << ",\"image_raw_path\":" << json_string(image_path.string())
        << ",\"metadata_path\":" << json_string(metadata_path.string())
        << ",\"tag_id\":" << result.tag_id
        << ",\"camera_frame_id\":" << json_string(pnp_frame_id_)
        << ",\"truth_frame_id\":" << json_string(dataset_truth_frame_id_)
        << ",\"camera\":{\"actual_width\":" << actual_width_
        << ",\"actual_height\":" << actual_height_
        << ",\"actual_fps\":" << actual_fps_
        << ",\"actual_fourcc\":" << json_string(actual_fourcc_) << "}"
        << ",\"pnp_4d\":{\"camera_x_m\":" << result.camera_xyz.x
        << ",\"camera_y_m\":" << result.camera_xyz.y
        << ",\"camera_z_m\":" << result.camera_xyz.z
        << ",\"camera_quat_w\":" << result.camera_quat_xyzw[3] << "}"
        << ",\"pnp_camera\":{\"translation_xyz\":[" << result.camera_xyz.x << ","
        << result.camera_xyz.y << "," << result.camera_xyz.z << "],\"rotation_xyzw\":["
        << result.camera_quat_xyzw[0] << "," << result.camera_quat_xyzw[1] << ","
        << result.camera_quat_xyzw[2] << "," << result.camera_quat_xyzw[3]
        << "],\"rpy_deg\":[" << result.camera_rpy_deg[0] << "," << result.camera_rpy_deg[1]
        << "," << result.camera_rpy_deg[2] << "],\"reprojection_error_px\":"
        << result.pnp_reprojection_error_px << ",\"decision_margin\":" << result.decision_margin << "}"
        << ",\"truth_world\":" << truth_world_json(request.truth_world)
        << ",\"operator\":" << json_string(request.operator_name)
        << ",\"note\":" << json_string(request.note) << "}";
    return out.str();
  }

  void append_dataset_csv(
    const fs::path & csv_path, const CaptureRequest & request, const DetectionResult & result,
    uint64_t frame_seq, const rclcpp::Time & stamp, const fs::path & image_path,
    const fs::path & metadata_path, const std::string & sample_id)
  {
    const bool write_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
    std::ofstream out(csv_path, std::ios::app);
    if (!out) {
      throw std::runtime_error("failed to open dataset CSV " + csv_path.string());
    }
    if (write_header) {
      out << "sample_id,request_id,session_id,ros_time_sec,frame_seq,image_raw_path,metadata_path,tag_id,"
          << "pnp_camera_x_m,pnp_camera_y_m,pnp_camera_z_m,pnp_camera_quat_w,"
          << "pnp_quat_x,pnp_quat_y,pnp_quat_z,pnp_quat_w,"
          << "pnp_roll_deg,pnp_pitch_deg,pnp_yaw_deg,pnp_reprojection_error_px,decision_margin,"
          << "truth_x_m,truth_y_m,truth_z_m,truth_roll_deg,truth_pitch_deg,truth_yaw_deg,operator,note\n";
    }
    out << csv_escape(sample_id) << "," << csv_escape(request.request_id) << ","
        << csv_escape(request.session_id) << "," << std::fixed << std::setprecision(9) << stamp.seconds() << ","
        << frame_seq << "," << csv_escape(image_path.string()) << "," << csv_escape(metadata_path.string()) << ","
        << result.tag_id << "," << result.camera_xyz.x << "," << result.camera_xyz.y << ","
        << result.camera_xyz.z << "," << result.camera_quat_xyzw[3] << ","
        << result.camera_quat_xyzw[0] << "," << result.camera_quat_xyzw[1]
        << "," << result.camera_quat_xyzw[2] << "," << result.camera_quat_xyzw[3] << ","
        << result.camera_rpy_deg[0] << "," << result.camera_rpy_deg[1] << "," << result.camera_rpy_deg[2]
        << "," << result.pnp_reprojection_error_px << "," << result.decision_margin << ","
        << csv_optional_double(request.truth_world.x_m) << "," << csv_optional_double(request.truth_world.y_m)
        << "," << csv_optional_double(request.truth_world.z_m) << ","
        << csv_optional_double(request.truth_world.roll_deg) << ","
        << csv_optional_double(request.truth_world.pitch_deg) << ","
        << csv_optional_double(request.truth_world.yaw_deg) << "," << csv_escape(request.operator_name)
        << "," << csv_escape(request.note) << "\n";
  }

  void write_dataset_sample(
    const CaptureRequest & request, const DetectionResult & result, const cv::Mat & frame, uint64_t frame_seq)
  {
    const rclcpp::Time stamp = now();
    const fs::path root = active_dataset_root();
    const fs::path session_dir = root / "sessions" / request.session_id;
    const fs::path image_dir = session_dir / "images" / "raw";
    const fs::path metadata_dir = session_dir / "metadata";
    fs::create_directories(image_dir);
    fs::create_directories(metadata_dir);
    ensure_session_manifest(session_dir, request, root);
    const std::string sample_id = make_sample_id(request, frame_seq, stamp);
    const fs::path image_path = image_dir / (sample_id + ".png");
    const fs::path metadata_path = metadata_dir / (sample_id + ".json");
    const fs::path jsonl_path = session_dir / "samples.jsonl";
    const fs::path csv_path = session_dir / "samples.csv";
    const bool jsonl_existed = fs::exists(jsonl_path);
    const bool csv_existed = fs::exists(csv_path);
    const uintmax_t jsonl_size_before = jsonl_existed ? fs::file_size(jsonl_path) : 0;
    const uintmax_t csv_size_before = csv_existed ? fs::file_size(csv_path) : 0;
    try {
      if (!cv::imwrite(image_path.string(), frame)) {
        throw std::runtime_error("failed to write image " + image_path.string());
      }
      const std::string metadata = dataset_sample_json(request, result, frame_seq, stamp, image_path, metadata_path);
      std::ofstream metadata_stream(metadata_path);
      if (!metadata_stream) {
        throw std::runtime_error("failed to open metadata " + metadata_path.string());
      }
      metadata_stream << metadata << "\n";
      {
        std::ofstream jsonl(jsonl_path, std::ios::app);
        if (!jsonl) {
          throw std::runtime_error("failed to open samples.jsonl");
        }
        jsonl << metadata << "\n";
      }
      append_dataset_csv(csv_path, request, result, frame_seq, stamp, image_path, metadata_path, sample_id);
    } catch (...) {
      std::error_code ignored;
      fs::remove(image_path, ignored);
      fs::remove(metadata_path, ignored);
      if (jsonl_existed) {
        fs::resize_file(jsonl_path, jsonl_size_before, ignored);
      } else {
        fs::remove(jsonl_path, ignored);
      }
      if (csv_existed) {
        fs::resize_file(csv_path, csv_size_before, ignored);
      } else {
        fs::remove(csv_path, ignored);
      }
      throw;
    }
    publish_capture_result(request, true, "ok", "sample saved", sample_id, image_path.string(), metadata_path.string());
  }

  void publish_capture_result(
    const CaptureRequest & request, bool success, const std::string & reason, const std::string & message,
    const std::string & sample_id, const std::string & image_path, const std::string & metadata_path)
  {
    std_msgs::msg::String output;
    std::ostringstream json;
    json << "{\"success\":" << (success ? "true" : "false")
         << ",\"reason\":" << json_string(reason)
         << ",\"message\":" << json_string(message)
         << ",\"request_id\":" << json_string(request.request_id)
         << ",\"session_id\":" << json_string(request.session_id)
         << ",\"sample_id\":" << (sample_id.empty() ? "null" : json_string(sample_id))
         << ",\"image_raw_path\":" << (image_path.empty() ? "null" : json_string(image_path))
         << ",\"metadata_path\":" << (metadata_path.empty() ? "null" : json_string(metadata_path))
         << ",\"dataset_root\":" << json_string(active_dataset_root().string()) << "}";
    output.data = json.str();
    dataset_result_pub_->publish(output);
  }

  void publish_detection2d_array(
    const std::vector<DetectionResult> & results,
    const rclcpp::Time & frame_stamp)
  {
    msgs::msg::AprilTagDetection2DArray msg;
    msg.header.stamp = frame_stamp;
    msg.header.frame_id = pnp_frame_id_;
    msg.detections.reserve(results.size());
    for (const auto & result : results) {
      if (!result.target_match) {
        continue;
      }
      msgs::msg::AprilTagDetection2D detection;
      detection.tag_id = result.tag_id;
      detection.family = result.family;
      detection.hamming = result.hamming;
      detection.decision_margin = static_cast<float>(result.decision_margin);
      detection.center_px[0] = result.center.x;
      detection.center_px[1] = result.center.y;
      for (size_t i = 0; i < result.corners.size(); ++i) {
        detection.corner_pixels_xy[2 * i] = result.corners[i].x;
        detection.corner_pixels_xy[2 * i + 1] = result.corners[i].y;
      }
      detection.mean_edge_px = static_cast<float>(mean_square_edge_px(result.corners));
      msg.detections.push_back(detection);
    }
    detection2d_pub_->publish(msg);
  }

  void publish_detection_status(
    const DetectionResult & result,
    const std::vector<DetectionResult> & all_results,
    const std::optional<size_t> & replay_index,
    const rclcpp::Time & frame_stamp)
  {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{";
    out << "\"detected\":" << (result.detected ? "true" : "false") << ",";
    out << "\"replay_index\":"
        << (replay_index ? std::to_string(*replay_index) : "null") << ",";
    out << "\"frame_stamp_ns\":" << frame_stamp.nanoseconds() << ",";
    out << "\"tag_id\":" << (result.detected ? std::to_string(result.tag_id) : "null") << ",";
    out << "\"detected_family\":" << json_string(result.family) << ",";
    out << "\"detected_ids\":" << json_int_array(result.detected_ids) << ",";
    out << "\"detections_detail\":" << json_detection_details(all_results) << ",";
    out << "\"target_tag_ids\":" << json_int_array(target_tag_ids_) << ",";
    out << "\"pixel_xy\":" << json_point_or_null(result.detected, result.center) << ",";
    out << "\"world_xy_yaw\":" << json_world_yaw_or_null(result.detected, result.world, result.yaw_rad) << ",";
    out << "\"pnp_valid\":" << (result.pnp_valid ? "true" : "false") << ",";
    out << "\"pnp_camera_xyz\":" << json_point3_or_null(result.pnp_valid, result.camera_xyz) << ",";
    out << "\"pnp_camera_quat_xyzw\":" << json_quat_or_null(result.pnp_valid, result.camera_quat_xyzw) << ",";
    out << "\"pnp_camera_rpy_rad\":" << json_double3_or_null(result.pnp_valid, result.camera_rpy_rad) << ",";
    out << "\"pnp_camera_rpy_deg\":" << json_double3_or_null(result.pnp_valid, result.camera_rpy_deg) << ",";
    out << "\"pnp_reprojection_error_px\":"
        << (result.pnp_valid ? std::to_string(result.pnp_reprojection_error_px) : "null") << ",";
    out << "\"pnp_mean_edge_px\":"
        << (result.pnp_mean_edge_px > 0.0 ? std::to_string(result.pnp_mean_edge_px) : "null") << ",";
    out << "\"pnp_edge_z_estimate_m\":"
        << (result.pnp_edge_z_estimate_m > 0.0 ? std::to_string(result.pnp_edge_z_estimate_m) : "null") << ",";
    out << "\"pnp_marker_length_m\":" << marker_length_m_ << ",";
    out << "\"pnp_fx\":" << (result.pnp_fx_used > 0.0 ? result.pnp_fx_used : 0.0) << ",";
    out << "\"pnp_fy\":" << (result.pnp_fy_used > 0.0 ? result.pnp_fy_used : 0.0) << ",";
    out << "\"pnp_calibration_size\":[" << camera_calibration_.image_width << "," << camera_calibration_.image_height
        << "],";
    out << "\"pnp_actual_size\":[" << actual_width_ << "," << actual_height_ << "],";
    out << "\"detections\":" << detection_count_.load() << ",";
    out << "\"misses\":" << miss_count_.load() << ",";
    out << "\"hamming\":" << (result.detected ? std::to_string(result.hamming) : "null") << ",";
    out << "\"decision_margin\":" << (result.detected ? std::to_string(result.decision_margin) : "null") << ",";
    out << "\"detection_fps\":" << detection_fps_ << ",";
    out << "\"image_raw_pub\":false";
    out << "}";
    std_msgs::msg::String msg;
    msg.data = out.str();
    status_pub_->publish(msg);
  }

  void update_debug_frame(const cv::Mat & frame, const std::vector<DetectionResult> & results)
  {
    if (!debug_image_enabled_ || debug_image_pub_->get_subscription_count() == 0) {
      return;
    }
    const auto current = now();
    if ((current - last_debug_frame_time_).seconds() < 1.0 / std::max(debug_image_rate_hz_, 0.1)) {
      return;
    }
    last_debug_frame_time_ = current;

    cv::Mat debug = frame.clone();
    for (const auto & result : results) {
      const cv::Scalar color = result.target_match ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 220, 255);
      for (int i = 0; i < 4; ++i) {
        const auto & p0 = result.corners[static_cast<size_t>(i)];
        const auto & p1 = result.corners[static_cast<size_t>((i + 1) % 4)];
        cv::line(debug, p0, p1, color, result.target_match ? 3 : 2);
      }
      cv::drawMarker(debug, result.center, color, cv::MARKER_CROSS, 32, result.target_match ? 3 : 2);
      std::ostringstream label;
      label << (result.target_match ? "target " : "id=") << result.tag_id << " err=" << std::fixed << std::setprecision(1)
            << result.pnp_reprojection_error_px;
      if (result.pnp_valid) {
        label << " z=" << std::setprecision(2) << result.camera_xyz.z;
      }
      cv::putText(debug, label.str(), result.center + cv::Point2d(12, -12), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        color, 2);
    }
    {
      std::lock_guard<std::mutex> lock(debug_mutex_);
      latest_debug_frame_ = std::make_shared<cv::Mat>(std::move(debug));
    }
  }

  void publish_debug_image()
  {
    if (!debug_image_enabled_ || debug_image_pub_->get_subscription_count() == 0) {
      return;
    }
    std::shared_ptr<cv::Mat> debug;
    {
      std::lock_guard<std::mutex> lock(debug_mutex_);
      debug = latest_debug_frame_;
    }
    if (!debug || debug->empty()) {
      return;
    }

    cv::Mat output = *debug;
    if (debug_image_width_ > 0 && output.cols > debug_image_width_) {
      const double scale = static_cast<double>(debug_image_width_) / static_cast<double>(output.cols);
      cv::resize(output, output, cv::Size(debug_image_width_, static_cast<int>(output.rows * scale)));
    }
    std::vector<uchar> encoded;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, std::clamp(debug_jpeg_quality_, 1, 100)};
    if (!cv::imencode(".jpg", output, encoded, params)) {
      return;
    }
    sensor_msgs::msg::CompressedImage msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = "finsrov_camera_debug";
    msg.format = "jpeg";
    msg.data.assign(encoded.begin(), encoded.end());
    debug_image_pub_->publish(msg);
  }

  void publish_camera_status()
  {
    const auto current = now();
    const auto frames = frame_count_.load();
    const double elapsed = std::max(1e-9, (current - last_status_time_).seconds());
    const double capture_fps = static_cast<double>(frames - last_status_frame_count_) / elapsed;
    last_status_time_ = current;
    last_status_frame_count_ = frames;

    Settings local_settings;
    {
      std::lock_guard<std::mutex> lock(settings_mutex_);
      local_settings = settings_;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{";
    out << "\"enabled\":" << (local_settings.enabled ? "true" : "false") << ",";
    bool opened = false;
    {
      std::lock_guard<std::mutex> lock(capture_mutex_);
      opened = capture_.isOpened();
    }
    out << "\"opened\":" << (opened ? "true" : "false") << ",";
    out << "\"device\":" << json_string(local_settings.device) << ",";
    out << "\"width\":" << local_settings.width << ",";
    out << "\"height\":" << local_settings.height << ",";
    out << "\"fps\":" << local_settings.fps << ",";
    out << "\"fourcc\":" << json_string(local_settings.fourcc) << ",";
    out << "\"actual_width\":" << actual_width_ << ",";
    out << "\"actual_height\":" << actual_height_ << ",";
    out << "\"actual_fps\":" << actual_fps_ << ",";
    out << "\"actual_fourcc\":" << json_string(actual_fourcc_) << ",";
    out << "\"frame_count\":" << frames << ",";
    out << "\"publish_fps\":" << capture_fps << ",";
    out << "\"detection_fps\":" << detection_fps_ << ",";
    out << "\"last_frame_mean\":" << last_frame_mean_ << ",";
    out << "\"last_frame_min\":" << last_frame_min_ << ",";
    out << "\"last_frame_max\":" << last_frame_max_ << ",";
    out << "\"last_error\":" << json_string(last_error_) << ",";
    out << "\"image_raw_pub\":false,";
    out << "\"debug_image_topic\":" << json_string(debug_image_topic_);
    out << "}";
    std_msgs::msg::String msg;
    msg.data = out.str();
    camera_status_pub_->publish(msg);
  }

  void open_capture(const Settings & settings)
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (capture_.isOpened()) {
      capture_.release();
    }
    const auto resolved_device = canonical_device_path(settings.device);
    const int device_index = video_device_index(resolved_device);
    if (device_index >= 0) {
      capture_.open(device_index, cv::CAP_V4L2);
    } else {
      capture_.open(resolved_device, cv::CAP_ANY);
    }
    if (!capture_.isOpened()) {
      last_error_ = "failed to open camera " + settings.device + " (resolved " + resolved_device + ")";
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", last_error_.c_str());
      return;
    }
    if (settings.fourcc.size() >= 4) {
      capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(
        settings.fourcc[0], settings.fourcc[1], settings.fourcc[2], settings.fourcc[3]));
    }
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, settings.width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, settings.height);
    capture_.set(cv::CAP_PROP_FPS, settings.fps);
    actual_width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    actual_height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    actual_fps_ = capture_.get(cv::CAP_PROP_FPS);
    actual_fourcc_ = fourcc_to_string(static_cast<int>(capture_.get(cv::CAP_PROP_FOURCC)));
    last_error_.clear();
    RCLCPP_INFO(
      get_logger(), "opened camera %s: actual=%dx%d fps=%.2f fourcc=%s",
      (settings.device + " -> " + resolved_device).c_str(), actual_width_, actual_height_, actual_fps_,
      actual_fourcc_.c_str());
  }

  void close_capture()
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (capture_.isOpened()) {
      capture_.release();
    }
  }

  void handle_control(const std::string & data)
  {
    try {
      const YAML::Node command = YAML::Load(data);
      if (!command.IsMap()) {
        return;
      }
      {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (command["device"]) {
          settings_.device = command["device"].as<std::string>();
        }
        if (command["width"]) {
          settings_.width = command["width"].as<int>();
        }
        if (command["height"]) {
          settings_.height = command["height"].as<int>();
        }
        if (command["fps"]) {
          settings_.fps = command["fps"].as<double>();
        }
        if (command["fourcc"]) {
          settings_.fourcc = command["fourcc"].as<std::string>();
        }
        if (command["enabled"]) {
          settings_.enabled = parse_bool(command["enabled"]);
        }
        reopen_requested_ = true;
      }
      RCLCPP_INFO(get_logger(), "camera control applied");
    } catch (const std::exception & exc) {
      last_error_ = std::string("invalid camera control JSON/YAML: ") + exc.what();
      RCLCPP_WARN(get_logger(), "%s", last_error_.c_str());
    }
  }

  rclcpp::Time now()
  {
    return get_clock()->now();
  }

  Settings settings_;
  std::mutex settings_mutex_;
  bool reopen_requested_{true};

  bool replay_mode_{false};
  std::string replay_image_list_file_;
  fs::path replay_image_list_path_;
  std::vector<fs::path> replay_image_paths_;
  double replay_rate_hz_{2.0};
  std::string replay_event_topic_;
  std::atomic<bool> replay_finished_{false};

  std::string family_name_;
  std::vector<std::string> family_names_;
  int target_tag_id_{-1};
  std::vector<int> target_tag_ids_{-1};
  int max_hamming_{0};
  int nthreads_{4};
  double quad_decimate_{1.5};
  double quad_sigma_{0.0};
  bool refine_edges_{true};
  double decode_sharpening_{0.25};

  std::string homography_file_;
  fs::path homography_path_;
  cv::Mat homography_;
  std::string camera_calibration_file_;
  fs::path camera_calibration_path_;
  CameraCalibration camera_calibration_;
  double marker_length_m_{0.0};
  bool pnp_enabled_{false};
  bool publish_pnp_pose_{true};
  std::string pnp_pose_topic_;
  std::string pnp_array_topic_;
  std::string detection2d_topic_;
  std::string pnp_frame_id_;
  double pnp_covariance_xyz_{0.0025};

  std::string status_topic_;
  std::string camera_status_topic_;
  std::string debug_image_topic_;
  fs::path dataset_root_;
  std::string dataset_capture_request_topic_;
  std::string dataset_capture_result_topic_;
  std::string dataset_truth_frame_id_;
  bool debug_image_enabled_{true};
  double debug_image_rate_hz_{5.0};
  int debug_image_width_{640};
  int debug_jpeg_quality_{80};
  double status_rate_hz_{5.0};

  std::vector<TagFamilyHandle> tag_families_;
  apriltag_detector_t * detector_{nullptr};

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pnp_pose_pub_;
  rclcpp::Publisher<msgs::msg::AprilTagDetection3DArray>::SharedPtr pnp_array_pub_;
  rclcpp::Publisher<msgs::msg::AprilTagDetection2DArray>::SharedPtr detection2d_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr camera_status_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_image_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr replay_event_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr dataset_result_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr control_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr dataset_request_sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dataset_root_parameter_callback_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  rclcpp::TimerBase::SharedPtr debug_timer_;

  std::atomic<bool> running_{false};
  std::thread capture_thread_;
  std::thread detection_thread_;
  cv::VideoCapture capture_;
  std::mutex capture_mutex_;

  std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  std::shared_ptr<cv::Mat> latest_frame_;
  std::optional<size_t> latest_frame_replay_index_;
  uint64_t latest_frame_seq_{0};
  uint64_t latest_processed_seq_{0};
  std::atomic<uint64_t> frame_count_{0};
  rclcpp::Time last_frame_time_;

  mutable std::mutex dataset_root_mutex_;
  std::mutex capture_request_mutex_;
  std::optional<CaptureRequest> pending_capture_request_;

  std::mutex debug_mutex_;
  std::shared_ptr<cv::Mat> latest_debug_frame_;
  rclcpp::Time last_debug_frame_time_;

  int actual_width_{0};
  int actual_height_{0};
  double actual_fps_{0.0};
  std::string actual_fourcc_;
  double last_frame_mean_{0.0};
  double last_frame_min_{0.0};
  double last_frame_max_{0.0};
  std::string last_error_;
  rclcpp::Time last_status_time_;
  uint64_t last_status_frame_count_{0};
  std::atomic<uint64_t> detection_count_{0};
  std::atomic<uint64_t> miss_count_{0};
  double detection_fps_{0.0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<DirectAprilTagNode>());
  } catch (const std::exception & exc) {
    std::cerr << "direct_apriltag_node failed: " << exc.what() << std::endl;
  }
  rclcpp::shutdown();
  return 0;
}
