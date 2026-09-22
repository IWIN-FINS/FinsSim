import numpy as np
import pytest

from hardware_bridge.imu_mounting import ImuMountingTransform


@pytest.fixture
def flipped_mounting() -> ImuMountingTransform:
    return ImuMountingTransform([0.0, 0.0, 1.0, 0.0])


def test_sensor_axes_rotate_to_ros_flu(flipped_mounting: ImuMountingTransform):
    assert flipped_mounting.transform_vector([-1.0, 0.0, 0.0]) == pytest.approx([1.0, 0.0, 0.0])
    assert flipped_mounting.transform_vector([0.0, -1.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
    assert flipped_mounting.transform_vector([0.0, 0.0, 9.80665]) == pytest.approx([0.0, 0.0, 9.80665])


def test_gyro_and_acceleration_share_one_mounting_transform(flipped_mounting: ImuMountingTransform):
    corrected = flipped_mounting.correct_sample(
        orientation_world_sensor_xyzw=[0.0, 0.0, 1.0, 0.0],
        angular_velocity_sensor_xyz=[-0.1, -0.2, 0.3],
        linear_acceleration_sensor_xyz=[-1.0, -2.0, 9.8],
    )

    assert corrected.angular_velocity_xyz == pytest.approx([0.1, 0.2, 0.3])
    assert corrected.linear_acceleration_xyz == pytest.approx([1.0, 2.0, 9.8])
    assert corrected.orientation_xyzw[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-7)
    assert abs(corrected.orientation_xyzw[3]) == pytest.approx(1.0, abs=1e-7)


def test_mounting_rotation_transforms_covariance(flipped_mounting: ImuMountingTransform):
    covariance = [1.0, 0.2, 0.3, 0.2, 2.0, 0.4, 0.3, 0.4, 3.0]

    transformed = np.asarray(flipped_mounting.transform_covariance(covariance)).reshape(3, 3)

    np.testing.assert_allclose(
        transformed,
        [[1.0, 0.2, -0.3], [0.2, 2.0, -0.4], [-0.3, -0.4, 3.0]],
        atol=1e-12,
    )


@pytest.mark.parametrize("quaternion", [[], [0.0, 0.0, 0.0, 0.0], [0.0, float("nan"), 0.0, 1.0]])
def test_invalid_mounting_quaternion_is_rejected(quaternion):
    with pytest.raises(ValueError):
        ImuMountingTransform(quaternion)
