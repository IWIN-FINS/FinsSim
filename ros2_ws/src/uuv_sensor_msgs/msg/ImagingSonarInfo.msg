## Based on: 
## 1. Marine ROS Conventions Discussion (https://github.com/udgcirs/marine_ros_conventions_discussion)

## This is a message to hold information about a 2D imaging sonar ping.
## Note:
##   1. This information message should be accompanied by a sensor_msgs/Image message. For polar images bearing should change with columns and range with rows.
##   2. Bearing table is optional and should contain the exact bearing for each supplied beam.
##   3. range_resolution is valid in both image dimensions when a cartesian XY image is used.
##   4. For polar images range_resolution is valid only for image rows and bearing_resolution should be used for image columns.
##   5. Sonar frames should be specified in frame_id in a xyz-> forward, left, up. Angles are measured counter-clockwise with x-axis being zero.

std_msgs/Header header

# Basic sonar information
float32 frequency # [kHz]
float32 elevation_angle # [rad]
float32 beam_width # [rad]
float32 gain # [0-100%]
float32 sound_speed # [m/s]

# Data information
uint8 IMAGE_TYPE_AUTO = 0
uint8 IMAGE_TYPE_POLAR = 1
uint8 IMAGE_TYPE_CARTESIAN = 2
uint8 image_type

# Swath information
float32 bearing_start # [rad]
float32 bearing_stop # [rad]
float32 bearing_resolution # [rad]
float64[] bearing_table

# Ranging information
float32 range_start # [m]
float32 range_stop # [m]
float32 range_resolution # [m/px]
