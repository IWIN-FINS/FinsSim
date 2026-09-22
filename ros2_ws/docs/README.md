# ROS2 工作空间文档

本目录只收录跨 ROS2 package 的工作空间约定；与某个节点、launch 文件或参数配置直接绑定的说明，保留在 `src/<package>/README.md` 与该 package 的 `docs/` 下。

## 入口

- [工作空间构建、环境与运行](../README.md)
- [VS Code Tasks 使用说明](../../docs/development/vscode-tasks.zh-CN.md)
- [实船感知与状态融合](../src/state_estimation/docs/fusion_pipeline.zh-CN.md)
- [控制器坐标契约](../src/motion_control/docs/controller_coordinate_contract.zh-CN.md)
- [项目级标定流程](../../docs/operations/calibration/index.zh-CN.md)
- [项目级坐标系与推进器契约](../../docs/reference/)

## 文档归属

- **package 内**：可执行入口、参数、话题、故障排查、实现细节、专属标定。
- **项目级 `docs/`**：跨 package 的系统契约、硬件操作流程、论文实验协议及结果。
