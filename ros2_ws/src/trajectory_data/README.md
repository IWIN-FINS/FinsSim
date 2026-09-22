# trajectory_data

FinsSim 的被动轨迹数据工具：记录手柄遥操作轨迹，导出对齐后的训练数据与数据质量报告。它不发布推进器命令，也不会启用 hardware bridge。

当前随仓库提供的 topic 清单和质量检查阈值是 FinsROV profile；记录、事件标记、导出、时间对齐和质量报告本身不绑定某一型号 ROV。接入新 ROV 时应提供其 topic/profile 配置，而不是复制本包。

## 文档

- [手柄轨迹采集、导出与质量检查](docs/teleop_trajectory_pipeline.zh-CN.md)
- [手柄遥操作](../teleop/README.md)
