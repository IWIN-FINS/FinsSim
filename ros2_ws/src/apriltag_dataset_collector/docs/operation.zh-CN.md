# FinsROV AprilTag 数据集采集器

`apriltag_dataset_collector` 是独立的 AprilTag 数据集采集 GUI/CLI。
它不会替代或修改 `perception_monitor`，也不会启动第二个相机节点；原始图像始终由
`perception` 的 C++ `direct_apriltag_node` 在完成 PnP 的采集帧上保存。

## 启动

先启动当前 native AprilTag 链路：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch perception direct_apriltag.launch.py
```

再启动独立采集界面：

```bash
./scripts/run_ros2_uv.sh ros2 run apriltag_dataset_collector apriltag_dataset_collector_gui
```

默认根目录是：

```text
./artifacts/datasets/apriltag_pnp_truth/
```

可在 `config/apriltag_dataset_collector.yaml` 持久修改；GUI 的 `Apply Root` 只改变本次
`direct_apriltag_node` 运行期的 `dataset_root` 参数，不改写 YAML。

## GUI 采集流程

1. 在 `Camera Control` 选择设备并点击 `Open / Apply`；确认预览、PnP 和检测状态有效。
2. 在 `AprilTag Dataset Capture` 填写 session、操作者、可选 tag ID、人工真值和备注。
3. 点击 `Capture Next Valid PnP Frame`。
4. native detector 在请求之后的下一张有效 PnP 原始帧写入样本；GUI 显示成功路径或失败原因。

Tag ID 留空时，使用当前 detector 配置目标 tag 中 decision margin 最大且 PnP 有效的 tag。
人工真值可留空；它们在 JSON 中写为 `null`，在 CSV 中写为空字段。

## CLI

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run apriltag_dataset_collector apriltag_dataset_capture \
  --config src/apriltag_dataset_collector/config/apriltag_dataset_collector.yaml \
  --session 20260828_pool_grid --tag-id 15 --operator fins \
  --x-m 0.25 --y-m -0.40 --z-m -0.50 --yaw-deg 0.0 --note P01
```

可增加 `--dataset-root /data/UnderwaterSim/datasets/apriltag_pnp_truth`，作为当前 native
detector 运行期的根目录覆盖。

## 数据格式和失败状态

```text
<dataset_root>/sessions/<session_id>/
  manifest.json
  samples.csv
  samples.jsonl
  images/raw/<sample_id>.png
  metadata/<sample_id>.json
```

每条样本含原始帧、ROS 时间、帧号、相机实际参数、tag/PnP、重投影误差、人工真值和备注。

失败状态：

- `capture_pending`：上一请求尚在等待下一帧；
- `no_pnp_detection`：没有配置目标 tag 的有效 PnP；
- `tag_not_found`：指定 tag 未在采集帧中得到有效 PnP；
- `write_failed`：图片或元数据写入失败；
- `invalid_request`：请求 JSON/YAML 不合法。
