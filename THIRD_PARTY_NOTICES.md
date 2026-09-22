# Third-party notices

The repository is not uniformly Apache-2.0. The following components retain
their upstream licenses and copyright notices:

| Component | Location | License / notice |
| --- | --- | --- |
| MARLlib | `python/finssim_marl/third_party/MARLlib` | MIT; copyright Replicable-MARL. See its `LICENSE` (and nested notices). |
| CleanMarl | `python/finssim_marl/third_party/cleanmarl` | MIT; copyright Cleanmarl team. See its `LICENSE`. |
| finssim_marl package provenance | `python/finssim_marl/LICENSE`, `python/finssim_marl/pyproject.toml`, `src/finssim_marl/algorithms/mappo_multihead.py` | The package remains MIT. `mappo_multihead.py` is adapted from CleanMARL and carries an in-file MIT attribution; the remaining in-tree implementation was reviewed and confirmed by the FinsSim rights holder as FinsSim-owned. |
| `grpc_ros_adapter` | `ros2_ws/src/grpc_ros_adapter` | Apache-2.0; copyright LABUST. See its `LICENSE` and source headers. |
| MARUS protobuf generated modules | `ros2_ws/src/grpc_ros_adapter/grpc_ros_adapter/protobuf` | Apache-2.0; generated from the MARUSimulator `marus-proto` project. See `protobuf/NOTICE.md` and `external/ROS_THIRD_PARTY.md`; retain the upstream attribution when redistributing. |
| `uuv_sensor_msgs` | `ros2_ws/src/uuv_sensor_msgs` | BSD 3-Clause; copyright LABUST. See its `LICENSE`. |
| FinsSimIsaacLab / Isaac AUV environment | Independent FinsSimIsaacLab repository | Not part of this FinsSim-public source distribution. Its own repository must carry the WarpAUV/Isaac AUV provenance and license notices before an independent release. |

Other files can contain their own copyright or license headers. Those notices
control the relevant file or component and must not be replaced by the
repository-level Apache-2.0 or commercial notice. In particular, generated
files and code copied from upstream projects require a separate provenance
and license review before redistribution.
