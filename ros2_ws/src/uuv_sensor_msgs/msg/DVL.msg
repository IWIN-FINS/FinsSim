## Based on: 
## - UUV Simulator (https://github.com/uuvsimulator/uuv_simulator)
## - Marine ROS Conventions Discussion (https://github.com/udgcirs/marine_ros_conventions_discussion)
## This is a message to hold data from a Doppler Velocity Logger (DVL).
## 
## Notes:
##   - unknown covariance should be filled with zeros.
##   - select the velocity reference as WATER or BOTTOM.
##   - for valid measurements but unknown reference use the UNKNOWN identifier.
##   - for invalid measurement acquisition don't publish the message. If you need to publish, fill with NaNs.
##   - we assume that DVLs usually provide also an altitude value. In case yours don't use NaN for this value.
##   - if individual beam ranges are available fill the beam_ranges vector. For invalid beam ranges use NaN.
std_msgs/Header header

# Measured velocity [m/s] in header.frame_id.
geometry_msgs/Vector3 velocity
# Row major, xyz axes
float64[9] velocity_covariance

# Altitude from reference layer in header.frame_id, specified in [m].
float64 altitude
float64 altitude_covariance

# Individual DVL beam ranges in [m]
float64[] beam_ranges

# Instrument reference specifier
uint8 VELOCITY_REFERENCE_UNKOWN = 1
uint8 VELOCITY_REFERENCE_WATER = 2
uint8 VELOCITY_REFERENCE_BOTTOM = 3
uint8 velocity_reference
