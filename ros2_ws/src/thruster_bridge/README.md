# FinsROV Thruster Bridge

This package subscribes to `std_msgs/Float32MultiArray` thruster commands, by default
`/finsrov/thrusters_out`, and forwards them to the FineSUB V5Streamer motor frame.

The implemented frame matches `reference_code/FineSUB/Services/V5Streamer/Streamer.hpp`:

```text
0xAA + uint8 isEnabled + float32[8] throttles + crc16_modbus(body) + 0xBB
```

The body is packed little-endian as `<B8f`; the CRC is little-endian.

## Build

```bash
cd ./ros2_ws
colcon build --packages-select thruster_bridge
source install/setup.bash
```

## Run

Serial to the lower controller:

```bash
ros2 run thruster_bridge thruster_streamer_bridge --ros-args \
  --params-file ./ros2_ws/src/thruster_bridge/config/thruster_streamer_bridge.yaml
```

Dry-run for checking ROS wiring without touching hardware:

```bash
ros2 run thruster_bridge thruster_streamer_bridge --ros-args \
  -p transport:=dry_run
```

The `motion_controller` output in `[-1, 1]` is suitable if the lower controller
uses `ESC_Motor::SetThrottle` style normalized throttle. If the real motor order
or direction differs, change `motor_order` and `motor_signs` in the YAML rather
than changing the controller.
