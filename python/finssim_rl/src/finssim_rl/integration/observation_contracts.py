"""Version-neutral declarations for the current pose-control observation ABI."""

from __future__ import annotations

OBSERVATION16_DIM = 16
OBSERVATION16_FIELDS = (
    "target_offset_body_x_normalized",
    "target_offset_body_y_normalized",
    "target_offset_body_z_normalized",
    "relative_rotation6d_col0_x",
    "relative_rotation6d_col0_y",
    "relative_rotation6d_col0_z",
    "relative_rotation6d_col1_x",
    "relative_rotation6d_col1_y",
    "relative_rotation6d_col1_z",
    "linear_velocity_body_x_normalized",
    "linear_velocity_body_y_normalized",
    "linear_velocity_body_z_normalized",
    "angular_velocity_body_x_normalized",
    "angular_velocity_body_y_normalized",
    "angular_velocity_body_z_normalized",
    "target_distance_normalized",
)
