# experiment_recorder

FinsSim 的通用 ROS2 实验记录 package：负责 rosbag2 录制、事件标记、manifest 与离线分析。基础 recorder 不控制推进器；实机 runner 仅在显式 `--arm` 时请求使能硬件。

当前随仓库提供的 T1/T2 runner、话题集合、硬件安全检查和控制器 YAML 是 FinsROV experiment profile。通用 recorder、事件、manifest 和分析产物可由其他 vehicle profile 复用。

## 文档

- [运行、T1/T2 协议与产物](docs/operation.zh-CN.md)
- [AprilTag 感知稳定性旁路记录器](docs/apriltag-stability-recorder.zh-CN.md)
- [E2 原生链路回放评估](../../../docs/experiments/perception/e2-apriltag-native-replay.zh-CN.md)
