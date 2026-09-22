# state_estimation

FinsROV 实船状态融合 package。它把视觉、IMU 与深度融合为控制器和 RL 统一消费的 pose、IMU、depth、DVL 与状态健康话题。

## 文档

- [状态融合主链路与接口](docs/fusion_pipeline.zh-CN.md)
- [折射 AprilTag 融合算法](docs/refractive_apriltag_fusion_algorithm.zh-CN.md)
- [折射 6D measurement 接入](docs/state_fusion_refractive_6d.zh-CN.md)
- [`T_body_tag` 外参](docs/t_body_tag_extrinsic.zh-CN.md)
- [world-camera 外参](docs/world_camera_extrinsic_calibration.zh-CN.md)
- [项目级标定入口](../../../docs/operations/calibration/index.zh-CN.md)
