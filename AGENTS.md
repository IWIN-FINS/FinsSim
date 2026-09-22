# FinsSim Agent 开发手册

本文是给后续 Codex/Agent 在本仓库继续开发、排障、集成时使用的工作说明。优先遵守这里的项目约定；如果和用户明确的新指令冲突，以用户当前指令为准。

## 1. 项目概述

FinsSim 是面向 FinsROV 水下机器人仿真、强化学习训练、ROS2 控制、实船硬件桥接和感知融合的一体化工程。当前主要链路分为：

```text
Unity / MARUS 仿真
  <-> gRPC / ROS2 adapter
  <-> ROS2 motion_controller
  <-> /finsrov/thrusters_out
  <-> hardware_bridge
  <-> 下位机 / MCU / 电机

实船感知:
  相机 AprilTag / PnP
  + MCU IMU / depth / motor telemetry
  -> state_estimation
  -> /finsrov/pose, /finsrov/imu_link, /finsrov/depth_link, /finsrov/dvl_link
  -> controller / RL observation
```

核心目标是尽量保持控制器和强化学习模型的输入/输出接口稳定，让仿真训练、ROS2 评估和实船实验可以复用同一套 observation/action 语义。

## 2. Agent 工作约定

- 默认使用中文和用户沟通。
- 修改前先读代码和配置，优先用 `rg` / `rg --files` 搜索。
- 手工编辑文件使用 `apply_patch`，不要用 `cat > file` 这类重写方式。
- 不要回滚用户已有改动。遇到 dirty worktree 时，只改本任务相关文件。
- Python 依赖一般由 `uv` 和对应 `pyproject.toml` 管理，不要直接 `uv pip install` 或裸 `pip install`。
- ROS2 workspace 不要直接裸跑 `colcon build`，必须走本仓库脚本。
- 不要把 `__pycache__`、`.pytest_cache`、构建产物、日志等无关文件加入版本控制。
- 实船相关改动要保守，特别是推进器下发、自动使能、坐标系转换、EKF 融合和 timeout 逻辑。

## 3. 目录结构

```text
FinsSim/
  README.md
  pyproject.toml
  uv.lock

  python/
    finssim_core/          # Python 公共核心库
    finssim_cli/           # 顶层 finssim CLI
    finssim_rl/            # 单智能体 RL 训练、评估、模型
    finssim_marl/          # 多智能体 MARL 训练、评估、第三方 MARLlib/CleanMARL

  ros2_ws/
    README.md
    pyproject.toml
    scripts/
      bootstrap_uv_env.sh
      colcon_build_uv.sh
      run_ros2_uv.sh
      source_ros2_uv.sh
    src/
      motion_control/
      hardware_bridge/
      perception/     # C++ runtime + Python monitor/calibration tools
      state_estimation/
      msgs/
      grpc_ros_adapter/
      uuv_sensor_msgs/
      thruster_bridge/
      hydrodynamic_identification/
      thruster_curve_measurement/

  unity/					#  有的主机clone的FinsSim项目有，有的没有。但是如果有也只是一个symlink，只需要修改一份
    marus-example/         # Unity / MARUS 工程
    RLControl/             # Unity 控制、推进器布局、文档

  reference_code/
    FineSUB/               # 下位机 / MCU 代码
    joystick/              # 旧手柄控制参考代码

  configs/
    rl/
    marl/

  docs/
  tools/
  artifacts/
  external/
```

## 4. Python 与 uv

顶层 Python 工程由 `uv` 管理。新增 Python 依赖时，优先加入对应包的 `pyproject.toml`：

- 顶层 CLI / 公共依赖：`pyproject.toml` 或 `python/finssim_cli/pyproject.toml`
- 单智能体 RL：`python/finssim_rl/pyproject.toml`
- 多智能体 MARL：`python/finssim_marl/pyproject.toml`
- ROS2 Python 节点依赖：`ros2_ws/pyproject.toml`

常用命令：

```bash
cd .
uv sync
uv run --package finssim-cli finssim --help
```

不要在仓库里提交 `.venv/`、`__pycache__/`、`.pytest_cache/`。

## 5. ROS2 workspace 约定

ROS2 workspace 在 `ros2_ws/`。必须使用仓库脚本，因为它们会处理 `uv` 环境、ROS 环境和 Python entrypoint。

初始化：

```bash
cd ./ros2_ws
./scripts/bootstrap_uv_env.sh
```

构建全部包：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh
```

构建指定包：

```bash
./scripts/colcon_build_uv.sh --packages-select hardware_bridge state_estimation
```

清 CMake cache 后构建：

```bash
./scripts/colcon_build_uv.sh --cmake-clean-cache
```

运行 ROS2 命令：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic list
./scripts/run_ros2_uv.sh ros2 node list
```

交互式 source：

```bash
cd ./ros2_ws
source ./scripts/source_ros2_uv.sh
```

注意：

- 不要直接 `colcon build`。
- 当前 workspace 不推荐 `--symlink-install`，历史上会因为 `.venv`、Python shebang、entrypoint 和 colcon 安装布局导致环境错乱。
- YAML 参数文件修改后，Python/launch 运行时读取的通常不需要重新 build；新增 package、entrypoint、msg/srv、C++ 编译代码才需要重新 build。

## 6. ROS2 各层概述

### 6.1 `motion_control`

运动控制层，负责订阅状态话题并发布推进器命令。

常见输入：

```text
/finsrov/pose
/finsrov/imu_link
/finsrov/depth_link
/finsrov/dvl_link
/motion_controller/control_mode
```

常见输出：

```text
/finsrov/thrusters_out
```

常用启动：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run motion_control motion_controller --ros-args \
  --params-file ./ros2_ws/src/motion_control/config/FinsROV/ppo_control_for_pose.yaml
```

手动位置目标：

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_position_goal \
  --frame body --x 0.2 --y -0.2 --z 0.2 --yaw 90
```

切控制模式：

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control set_control_mode ros_manual
```

### 6.2 `hardware_bridge`

实船硬件桥接层，负责：

- 订阅 `/finsrov/thrusters_out`
- 做电机顺序、符号、限幅、timeout、command mode 转换
- 通过 UDP/串口链路发送给下位机
- 接收 MCU telemetry
- 发布 IMU、深度、电机转速、debug/status 话题

典型输出：

```text
/finsrov/hardware/imu_raw
/finsrov/hardware/depth_raw
/finsrov/hardware/motor_rpm_raw
/finsrov/hardware/status
/finsrov/hardware/thruster_cmd_echo
```

实船常用启动：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch hardware_bridge hardware_bridge.launch.py \
  params_file:=src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml \
  debug_mask:=1 \
  debug_echo_decimation:=1 \
  enabled:=true
```

如果只做安全 dry-run 或排查，不要直接 `enabled:=true`。

检查状态：

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/hardware/imu_raw
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/depth_raw --once --full-length
```

推进器测试：

```bash
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test forward --amplitude 0.2 --print-only
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test index --index 7 --amplitude 0.15
```

测试推进器前先确认没有多个发布者抢 `/finsrov/thrusters_out`：

```bash
./scripts/run_ros2_uv.sh ros2 topic info /finsrov/thrusters_out -v
```

### 6.3 `perception`

统一的 C++/Python 感知包。C++ 负责实船高性能 AprilTag 3、PnP 与折射约束位姿；Python 仅保留 GUI monitor、相机标定和离线工具，不再提供在线 Python detector。

常用启动：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch perception direct_apriltag.launch.py
./scripts/run_ros2_uv.sh ros2 launch perception direct_apriltag_ir.launch.py
./scripts/run_ros2_uv.sh ros2 launch perception refractive_apriltag.launch.py
./scripts/run_ros2_uv.sh ros2 run perception perception_monitor
```

常见输出：

```text
/finsrov/vision/tag_poses_3d_camera
/finsrov/vision/refracted_pose_6d
/finsrov/vision/status
```

### 6.4 `state_estimation`

状态估计层，融合 AprilTag PnP、相机外参、tag 到机体外参、IMU、深度，发布控制器/RL 可直接使用的状态。

主输出：

```text
/finsrov/pose
/finsrov/imu_link
/finsrov/depth_link
/finsrov/dvl_link
/finsrov/state/status
```

重要文档：

```text
ros2_ws/src/state_estimation/docs/fusion_pipeline.zh-CN.md
```

常用检查：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/pose
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/pose --once --full-length
```

### 6.6 `grpc_ros_adapter`

Unity / MARUS 与 ROS2 之间的桥接层。涉及仿真状态、传感器、控制命令转发时先读该包 README。

### 6.7 `msgs`

自定义 ROS2 message 包。新增/修改 msg 后必须重新 build 依赖包。

### 6.8 其他包

- `uuv_sensor_msgs`：水下传感器相关消息。
- `thruster_bridge`：旧推进器桥接兼容包。
- `hydrodynamic_identification`：水动力辨识。
- `thruster_curve_measurement`：推进器曲线测量。

## 7. 实船网络与硬件链路

当前常见网络约定：

```text
上位机有线网口: 192.168.0.10/24
下位机 / NX:    192.168.0.2
ROS bridge UDP bind:   0.0.0.0:54321
NX raw UDP listener:   192.168.0.2:58766
```

检查网口：

```bash
ip -br addr
ip route get 192.168.0.2
ip neigh show
ping -c 3 -W 1 192.168.0.2
```

如果换了网口，Linux 可能使用不同 interface 名称，需要确认 `192.168.0.10/24` 配在实际连接下位机的有线网卡上。

SSH 到下位机：

```bash
ssh <device-user>@192.168.0.2
```

设备用户名、密码和 SSH key 均属于私有运维凭据。请从受控的安全渠道获取当前凭据；不要将默认密码、轮换后的密码或私钥写入本仓库。设备对外接入或公开发布前必须完成密码轮换。

下位机上常见服务：

```text
streamer.service
/StreamerStart.sh
/V5StreamerNX.py
/GStreamerTest.py
```

检查服务：

```bash
ssh <device-user>@192.168.0.2 "systemctl status streamer.service --no-pager -l"
ssh <device-user>@192.168.0.2 "ss -lunp | egrep '58766|12345|5600|python3|gst'"
```

检查日志：

```bash
ssh <device-user>@192.168.0.2 "journalctl -u streamer.service --no-pager | tail -n 80"
```

## 8. 推进器输出顺序与安全点

`motion_controller` 发布：

```text
/finsrov/thrusters_out
```

当前常用 canonical order 文档见：

```text
unity/RLControl/Docs/UUV_Thruster_Layout_Canonical.md
docs/thruster_mapping*.md
ros2_ws/src/state_estimation/docs/fusion_pipeline.zh-CN.md
```

常见 canonical order：

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

MCU direct command order 通常是：

```text
[M1 LF lower, M2 LF upper, M3 LB upper, M4 LB lower,
 M5 RB lower, M6 RB upper, M7 RF upper, M8 RF lower]
```

桥接层通过参数修正：

```yaml
motor_order: [...]
motor_signs: [...]
```

`finsrov_hardware_bridge_v4_pro1.yaml` 当前是实船优先参考配置。修改电机方向/顺序时，优先改 ROS2 hardware bridge 层参数，不要轻易改 MCU、Unity 或 controller 内部逻辑。

实船测试前检查：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic info /finsrov/thrusters_out -v
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge enabled
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge command_mode
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge motor_order
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge motor_signs
```

如果一启动 bridge 电机就转，优先查：

- 是否已有 `/finsrov/thrusters_out` 发布者。
- 是否残留 `motion_controller` 在发布非零命令。
- `hardware_bridge enabled` 是否为 true。
- timeout 是否会自动发零推力。

## 9. FineSUB 下位机编译与烧录

下位机代码位于：

```text
reference_code/FineSUB/
```

详细排障参考：

```text
reference_code/FineSUB/Docs/OpenOCD_CMSIS-DAP烧写故障排查.md
```

推荐构建：

```bash
cd ./reference_code/FineSUB
cmake --build cmake-build-debug -- -j4
```

确认产物：

```bash
ls -lh cmake-build-debug/bin/Robomaster_A*
file cmake-build-debug/bin/Robomaster_A cmake-build-debug/bin/Robomaster_A.hex
```

OpenOCD 连接测试：

```bash
/usr/bin/openocd -s /usr/share/openocd/scripts \
  -f ./reference_code/FineSUB/OpenOCD/stm32f4+dap-link.cfg \
  -c "init" \
  -c "targets" \
  -c "shutdown"
```

烧写 HEX：

```bash
/usr/bin/openocd -s /usr/share/openocd/scripts \
  -f ./reference_code/FineSUB/OpenOCD/stm32f4+dap-link.cfg \
  -c "program ./reference_code/FineSUB/cmake-build-debug/bin/Robomaster_A.hex verify reset exit"
```

成功标志：

```text
** Programming Finished **
** Verify Started **
** Verified OK **
```

如果 OpenOCD 报：

```text
Error: CMSIS-DAP command CMD_INFO failed.
```

通常是 CMSIS-DAP USB 通信卡住，不是固件内容问题。先执行：

```bash
lsusb | grep -i faed
ps aux | grep -E 'openocd|pyocd|st-util|JLink|probe-rs'
sudo pkill -f openocd
sudo pkill -f pyocd
```

必要时按文档里的 USB reset 流程重置 `/dev/bus/usb/BBB/DDD`，其中 `BBB/DDD` 必须来自当前 `lsusb` 输出。

下位机通信相关重点文件：

```text
reference_code/FineSUB/Services/V5Streamer/Streamer.hpp
reference_code/FineSUB/Interface/Examples/TaskSUB.cpp
```

## 10. 感知、融合、控制常用启动顺序

一个典型实船调试顺序：

1. 确认下位机网络。
2. 启动 hardware bridge。
3. 启动相机 AprilTag 检测。
4. 启动 state fusion。
5. 确认 `/finsrov/pose`、`/finsrov/imu_link`、`/finsrov/depth_link`、`/finsrov/dvl_link`。
6. 启动 motion_controller。
7. 小幅度、短时间测试推进器。

示例：

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 launch hardware_bridge hardware_bridge.launch.py \
  params_file:=src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml \
  debug_mask:=1 debug_echo_decimation:=1 enabled:=false

./scripts/run_ros2_uv.sh ros2 launch perception direct_apriltag_ir.launch.py

./scripts/run_ros2_uv.sh ros2 launch state_estimation state_fusion.launch.py

./scripts/run_ros2_uv.sh ros2 run motion_control motion_controller --ros-args \
  --params-file ./ros2_ws/src/motion_control/config/FinsROV/ppo_control_for_pose.yaml
```

状态检查：

```bash
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/pose
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/thrusters_out
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status --once --full-length
```

## 11. 坐标系与融合注意事项

当前实船融合主设计：

```text
AprilTag PnP pose in camera frame
  + T_world_camera
  + T_tag_body
  -> vision T_world_body

pressure depth
  -> position.z = -depth_m  # ROS/world z-up 约定

IMU quaternion
  -> 高频 roll/pitch/yaw
  -> yaw 可被 AprilTag 低频校正
```

状态估计层输出的是 ROS/world 右手系、z-up 语义。Unity 是左手系、y-up；Unity 训练模型和 ROS 实船输入之间的转换应集中在 controller observation/controller basis 层处理，不要在 perception、hardware bridge、MCU 中分散修坐标。

重要原则：

- `/finsrov/pose` 表示 pool/world 下 ROV body pose。
- `/finsrov/dvl_link` 当前没有真实 DVL 时是由状态估计速度转换出的伪 DVL。
- `/finsrov/imu_link` 应保持 orientation、angular velocity、linear acceleration 单位正确。
- 视觉丢失时 EKF 不能跑炸，应进入预测/衰减/超时状态，status 中标记 stale。

## 12. 单智能体 RL: `python/finssim_rl`

目录：

```text
python/finssim_rl/
```

独立训练示例：

```bash
cd ./python/finssim_rl
uv sync
uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 5005
```

独立评估示例：

```bash
cd ./python/finssim_rl
uv run finssim-rl eval \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 5005
```

顶层 CLI dry-run：

```bash
cd .
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --dry-run
uv run --package finssim-cli finssim rl eval -c configs/rl/example.yaml --dry-run
```

顶层 CLI 实际运行：

```bash
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml
uv run --package finssim-cli finssim rl eval -c configs/rl/example.yaml
```

历史脚本中可能存在：

```bash
cd python/finssim_rl
export DISPLAY=:0
uv run python -m scripts.train --config=ppo_control_for_velocity --exp-name="ppo_control_for_local_velocity" --port-offset=2000 --overwrite
uv run python -m scripts.eval --config=ppo_control_for_velocity --exp-name="ppo_control_for_local_velocity"
```

如果 CLI 和历史脚本参数不一致，先读 `python/finssim_rl/README.zh-CN.md`、`python/finssim_rl/src/` 和对应 config。

## 13. 多智能体 MARL: `python/finssim_marl`

目录：

```text
python/finssim_marl/
```

独立训练示例：

```bash
cd ./python/finssim_marl
uv sync
uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 6000
```

独立评估示例：

```bash
cd ./python/finssim_marl
uv run finssim-marl eval \
  --config chasing_3_chase_1_position_control \
  --exp-name chasing_position_control \
  --output-dir ./artifacts/runs/marl/chasing_position_control \
  --env-base-port 9000 \
  --num-envs 1 \
  --num-episodes 10 \
  --show-graphics
```

顶层 CLI：

```bash
cd .
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --dry-run
uv run --package finssim-cli finssim marl eval -c configs/marl/example.yaml --dry-run
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml
uv run --package finssim-cli finssim marl eval -c configs/marl/example.yaml
```

MARLlib 示例：

```bash
uv run --package finssim-cli finssim marllib train -c configs/marl/marllib_3chase1_smoke.yaml --dry-run
```

历史脚本示例：

```bash
cd python/finssim_marl
uv run python -m scripts.train \
  --config chasing_3_chase_1_position_control \
  --exp-name posctrl_sb3_new_net \
  --env_base_port 1500 \
  --overwrite
```

## 14. 常见排障入口

### 14.1 没有 `/finsrov/hardware/depth_raw`

检查：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 node list
./scripts/run_ros2_uv.sh ros2 topic info /finsrov/hardware/depth_raw -v
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status --once --full-length
ping -c 3 -W 1 192.168.0.2
```

再查下位机服务：

```bash
ssh <device-user>@192.168.0.2 "systemctl status streamer.service --no-pager -l"
ssh <device-user>@192.168.0.2 "journalctl -u streamer.service --no-pager | tail -n 80"
```

### 14.2 `/finsrov/thrusters_out` 频率很低

检查 controller 是否 ready、fusion 是否 fresh、是否只有一个 publisher：

```bash
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/thrusters_out
./scripts/run_ros2_uv.sh ros2 topic info /finsrov/thrusters_out -v
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status --once --full-length
```

## 17. 发布分支与远端约定

- `origin-private` 是私有开发远端：`main`、单一移动候选分支 `release-candidate` 与内部审计用的 `release/<version>` 均推送到这里。
- `origin` 是公开发布远端：其 `main` 永远代表当前已发布版本的无历史快照；绝不向它推送私有 `main` 或候选分支。
- 私有 `main` 保留完整开发历史；从它创建一次 `release-candidate`，并在该分支上完成未发布模块、历史资料和本地实验材料的排除。
- `release-candidate` 是持续迭代、无 tag 的单一候选分支；候选版本仅在实际公开发布时以 `v<version>` tag 标识，不再创建带版本号的 candidate 分支或 RC tag。
- 日常使用 `scripts/release/release_public.sh`；完整流程、分支角色与固定排除规则见 `scripts/release/README.md`。它从 `origin-private/release-candidate` 拉取已提交候选、默认载入版本控制中的 `scripts/release/public-excludes.txt`，再调用底层发布器创建 `release/<version>` orphan 根提交（无父提交），并用 `--force-with-lease` 将该提交更新到 `origin/main`，同时推送不可变的 `v<version>` annotated tag。用 `plan` 先预览；用 `--exclude` / `--exclude-file` 临时追加排除规则；只有明确批准的纠错才能使用 `amend --reason`。
- `AGENTS-PRIVATE.md` 是仅私有 `main` 管理的运维补充文件，可含站点或凭据类内容；不得带入 `release-candidate`。固定公开排除清单也会移除它，作为额外防线。
- 不可将 release 合并回私有 main；正常情况下已公开的 tag 不可改写。只有已获批准的低风险发布修正，才可显式使用 `--amend-release` 重建同版本快照、更新 `origin/main` 和同版本 tag，并必须在提交/发布说明中写明修正原因；任何新功能或行为变化必须使用新版本号。候选分支中未提交的修改不会进入发布快照。

```bash
# 私有开发与候选分支
git push origin-private main
git push origin-private release-candidate

# 检查候选与私有 main 的差异，再预览/发布公开 main。
scripts/release/release_public.sh status
scripts/release/release_public.sh plan --version 0.1.1
scripts/release/release_public.sh publish --version 0.1.1

# 同版本的小型、已批准修复：必须记录原因。
scripts/release/release_public.sh amend --version 0.1.0 \
  --reason "Restore the missing public RL example configuration"
```

如果 motion_controller gate 关闭，它可能不会持续发布正常控制输出。

### 14.3 debug echo 没收到

检查 bridge 参数：

```bash
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge enabled
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge debug_mask
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge debug_echo_decimation
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status --once --full-length
```

如果普通 telemetry 有、IMU/depth 有，但 echo 没有，优先怀疑 MCU 是否发出 echo telemetry type，而不是 ROS parser。

### 14.4 相机没有图像或很卡

检查设备：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

使用 native AprilTag 节点时，优先避免把高分辨率 `image_raw` 通过多个 ROS2 Python 节点序列化传输。GUI 显示应降频或使用独立 debug preview，避免拖慢检测主循环。

### 14.5 EKF / fusion 不正常

先看 status，不要只看 pose：

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/refracted_pose_6d --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/imu_raw --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/depth_raw --once --full-length
```

重点关注：

- `ready`
- `initialized`
- `vision_fresh`
- `imu_fresh`
- `depth_fresh`
- `reject_reason`
- covariance 是否发散
- 是否视觉丢失后仍错误使用 stale detection

## 15. 文档入口

常用文档：

```text
README.md
ros2_ws/README.md
python/finssim_rl/README.zh-CN.md
python/finssim_marl/README.zh-CN.md
ros2_ws/src/perception/README.zh-CN.md
ros2_ws/src/perception/README.zh-CN.md
ros2_ws/src/state_estimation/docs/fusion_pipeline.zh-CN.md
reference_code/FineSUB/Docs/OpenOCD_CMSIS-DAP烧写故障排查.md
```

## 16. 提交前检查建议

文档改动：

```bash
git diff -- AGENTS.md
```

Python 包：

```bash
uv run pytest
```

ROS2 包：

```bash
cd ros2_ws
./scripts/colcon_build_uv.sh --packages-select <changed_pkg>
./scripts/run_ros2_uv.sh ros2 pkg executables <changed_pkg>
```

实船相关改动提交前至少确认：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 topic info /finsrov/thrusters_out -v
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status --once --full-length
```
