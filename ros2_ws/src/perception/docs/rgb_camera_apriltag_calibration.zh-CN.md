# FinsROV 相机与 AprilTag 标定工具

本目录的 Python 工具用于生成和审计 `perception/calibration/` 下的标定资产。它们不承担实机在线检测；运行时检测和折射位姿均由 C++ 节点完成。

## 标定产物

| 产物 | 用途 |
| --- | --- |
| `rgb_camera.yaml` / `ir_apriltag_camera.yaml` | C++ AprilTag detector 的相机内参和畸变；PnP 与折射模型共同使用。 |
| `pool_homography.yaml` | RGB 相机像素到固定池面平面坐标的 debug/现场审计映射，不替代折射 6D 主输出。 |
| `pool_homography_ir_unverified.yaml` | 与 RGB 标定隔离的 IR legacy 映射；未完成 IR 专属标定，不能用于定量结论。 |
| `state_estimation/config/state_fusion_extrinsics.yaml` | `T_world_camera`、`T_body_tag` 等刚体外参；由折射位姿节点和状态融合读取。 |

运行相机内参采集和棋盘格标定：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run perception capture_chessboard_images --help
./scripts/run_ros2_uv.sh ros2 run perception calibrate_camera_chessboard --help
```

立体相机、视频采集、homography 采点和 Tag 扫描也以同样方式调用。所有工具均先使用 `--help` 核对输入/输出路径；新的标定结果在替换 `calibration/` 权威文件前应先保存到独立试验目录，并经重投影、已知点或外参审计验证。

### 从 E2 归档数据拟合固定深度 homography

`fit_pool_homography_from_e2` 使用 E2 原生回放的 `native_detector_status.pixel_xy`，而不是
重新用 Python 检测图片。其真值转换显式区分人工记录的 AprilTag 外框左下角网格点和
AprilTag 中心：在当前归档坐标中，中心真值为
`[truth_x - 2.0 + 0.06, truth_y - 1.0 + 0.06]`。工具先对同一物理点的重复帧取中心像素
中位数，再使用空间留出点报告验证误差；最终输出是一个只适用于该相机、水位和底部
tag 平面的 2D debug 映射，不替代 Snell 6D 或状态融合。

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run perception fit_pool_homography_from_e2 \
  --e2-session data/experiments/E2/20260828_002300_E2_AprilTag_NativeReplay \
  --output-dir data/calibration/outputs/e2_pool_homography_20260828 \
  --output-yaml data/calibration/outputs/e2_pool_homography_20260828/pool_homography_candidate.yaml \
  --current-homography src/perception/calibration/pool_homography.yaml
```

先审阅候选目录中的 `fit_report.json`、`point_pairs.csv` 和
`heldout_predictions.csv`。只有在独立留出误差合理、配置的相机/水位没有变化时，才将候选矩阵
写入 `calibration/pool_homography.yaml`；替换后仍只影响 detector 的 2D debug 映射。

当前 RGB `pool_homography.yaml` 已由上述 E2 数据集重新标定。该拟合使用了 170 个物理网格点，
并在 34 个空间留出点上得到 `1.59 cm` RMSE、`2.44 cm` P95；完整审计结果位于
`ros2_ws/data/calibration/outputs/e2_pool_homography_20260828/`。数据集中的人工网格点是
AprilTag 外框左下角，因此拟合前固定转换到 tag 中心，偏移为 `[+0.06,+0.06] m`。该结果只适用于
1280x720 RGB 顶视相机、1 m 水深的底部平面；红外相机必须独立标定。

## 用标定结果启动实机链路

RGB：

```bash
./scripts/run_ros2_uv.sh ros2 launch perception refractive_apriltag.launch.py \
  detector_params:=src/perception/config/direct_apriltag_rgb.yaml \
  refractive_params:=src/perception/config/refractive_apriltag_rgb.yaml
```

红外配置替换为对应的 `_ir.yaml`。启动后检查 `/finsrov/vision/status`、`/finsrov/vision/refracted/status` 和 `/finsrov/state/status`；重投影误差或几何残差只用于诊断，不能直接当作全局定位精度。
