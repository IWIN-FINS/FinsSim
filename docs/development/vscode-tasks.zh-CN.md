# VS Code Tasks 使用说明

仓库根目录下的 [`.vscode/tasks.json`](../../.vscode/tasks.json) 为 FinsSim
提供常用 ROS 2 构建、调试和数据工具入口。它们统一从 `${workspaceFolder}/ros2_ws`
执行，并始终通过仓库的 `scripts/colcon_build_uv.sh` 或
`scripts/run_ros2_uv.sh` 初始化 ROS 2 与内部 `uv` Python 环境；不要在 VS Code
终端中改为裸跑 `colcon build` 或系统 Python。

本文假定 VS Code 打开的是 **FinsSim 仓库根目录**，而不是 `ros2_ws/` 子目录。
使用命令面板的 **Tasks: Run Task**（默认快捷键 `Ctrl+Shift+P`）选择任务。

## 构建任务

| Task | 用途 |
| --- | --- |
| `build/ros2-all` | 构建完整 ROS 2 workspace。 |
| `build/ros2-package` | 交互选择一个常用 package 构建；列表包含 `trajectory_data` 与 `experiment_recorder`。 |
| `build/ros2-data-tools` | 用 `--packages-up-to` 构建两个数据工具及其 ROS 依赖；重命名或修改它们后优先使用。 |
| `build/ros2-clean-cache` | 清 CMake cache 后构建；只在 CMake/接口变更或缓存异常时使用。 |

YAML 参数、Python 业务逻辑和 Markdown 文档变更通常不需要重新构建；新增 package、
console entry point、ROS message 或 C++ 改动则需要构建。

## 数据与实验任务

`trajectory_data` 和 `experiment_recorder` 已取代旧的
`finsrov_*` 包名。任务中的命令对应关系如下：

| Task | 实际入口 | 安全边界 |
| --- | --- | --- |
| `run/trajectory-data/record-teleop` | `trajectory_data record_teleop_trajectory` | 仅启动 rosbag 和事件记录；不发布推进器命令，也不使能硬件 bridge。应先手动确认手柄、状态估计和硬件链路已健康。 |
| `run/experiment-recorder/dry-run` | `experiment_recorder record_experiment --dry-run` | 仅显示将要录制的配置；不启动 rosbag、不控制推进器、不使能硬件。 |

这两个任务都会弹出 session ID、label 或 experiment/profile 输入。正式 session ID
应使用可追溯的名称，例如 `20260922_free_drive_001`，不要复用已完成 session 的目录。

实际 T1/T2 实机实验不提供一键 VS Code task。请按照
[`experiment_recorder` 操作手册](../../ros2_ws/src/experiment_recorder/docs/operation.zh-CN.md)
执行，并只在已完成现场检查后显式传入 `--arm`。随仓库提供的 T1/T2、topic 与安全
配置仍是 FinsROV profile；数据工具本身可由新 vehicle profile 复用。

## 调试与硬件任务

`run/hardware-bridge`、perception、fusion 与 controller 任务用于分层启动链路。
`Start All (with debug)` 和 `run/hardware-bridge/debug-enabled` 会传入
`enabled:=true`，可能实际下发推进器指令；仅在实艇、接线、急停和单发布者检查均完成后
使用。日常排障优先使用单独的 perception/fusion 任务，或让 bridge 保持 disabled。

`test/hardware-thruster*` 同样会产生实际执行器输出，不能作为软件 smoke test 使用。
运行前请先用 `monitor/hardware-debug-monitor`、`monitor/fusion-status` 和
`ros2 topic info /finsrov/thrusters_out -v` 确认状态及发布者。

## 输入项与停止任务

`inputs` 区域提供硬件 YAML、AprilTag YAML、融合 YAML 和 controller profile 等交互
选择。选择 controller profile 只会改变参数文件，不会验证 checkpoint、实艇接线或
坐标系标定是否正确。

background task 会保留在 VS Code 的专用终端中。停止长运行节点时，使用终端的
`Ctrl-C`；停止硬件链路前应先运行 `cancel/all-stop` 或明确将
`/hardware_bridge` 的 `enabled` 参数设为 `false`。VS Code 的终止按钮不能替代
推进器安全停机流程。

## 进一步阅读

- [ROS 2 workspace 构建、环境与运行](../../ros2_ws/README.md)
- [轨迹数据工具](../../ros2_ws/src/trajectory_data/README.md)
- [实验记录器](../../ros2_ws/src/experiment_recorder/README.md)
- [实船硬件与安全约定](../operations/hardware/thruster-bridge-mapping.zh-CN.md)
