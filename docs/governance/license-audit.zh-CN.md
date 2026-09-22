# FinsSim-public 许可证与授权链审计

审计日期：2026-09-22（技术性复核更新）
审计范围：当前工作树中的 FinsSim-public 主仓、已检出的子模块，以及根目录许可证/NOTICE 文件。
状态：仓库技术性许可证审计已闭环；最终权利链仍以权利人保存的合同、作者和贡献记录为准。不是法律意见。

## 结论摘要

FinsSim 自有代码可按 Apache License 2.0 发布；另行签署的商业协议只应覆盖权属明确、由 FinsSim 或其合法权利人拥有的部分。仓库不是单一许可证：CleanMarl/MARLlib、LABUST/MARUS ROS 包、BSD 传感器消息均须按各自条款处理。已识别 MARUS generated protobuf 的 Apache-2.0 上游项目并补充本地 NOTICE；但吸收副本的精确生成 commit 未保留，未来修改前应以固定 `.proto` commit 重新生成。FinsSimIsaacLab 已是独立仓库，不属于本仓发行物。

根目录 LICENSE、NOTICE、COMMERCIAL-LICENSE.md、THIRD_PARTY_NOTICES.md 和 ROS 来源记录分别见 [LICENSE](../../LICENSE)、[NOTICE](../../NOTICE)、[COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)、[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) 与 [ROS_THIRD_PARTY.md](../../external/ROS_THIRD_PARTY.md)。

## 建议的授权链条

```text
实际作者 / 员工 / 外包方 / 外部贡献者
  │ 书面确认著作权归属、职务作品/委托开发约定、贡献者许可或转让
  ▼
经核实的 FinsSim 权利主体（必须写完整法定名称）
  ├─ 对权属明确的自有代码、文档、配置：授予 Apache-2.0
  ├─ 对同一自有部分：可基于另行签署的商业协议提供额外商业权利
  └─ 对第三方代码、子模块、模型、数据、二进制：不得宣称自有或代为再许可
       ├─ 开源组件：按原许可证保留 LICENSE、版权和 NOTICE
       ├─ Asset Store / 私有组件：取得明确再分发授权，或从公开发布物中排除
       └─ 来源/权利不清：暂停分发，补齐证据后再纳入
  ▼
发布包：许可证清单 + 固定版本/提交 + 可复核的第三方通知 + 排除项说明
```

授权链执行要求：

1. “IWIN-FINS”目前只是仓库声明中的名称；在签发许可证或商业合同前，核实其完整法定主体、注册地、签约代表和对相关作品的权利来源。Git 作者名或提交记录本身不证明著作权转让。
2. 对员工、学生、承包商、合作机构和外部贡献者分别核对劳动/委托合同、IP 转让或许可、机构/资助项目约束。需要商业双授权的贡献，必须确认权利人确有再次商业许可的权力。
3. 根目录 Apache-2.0 仅覆盖权属明确的 FinsSim 自有部分；第三方组件继续受自己的许可证约束。Apache-2.0 已授予的开源权利不能由后续商业协议撤回。Apache-2.0 允许商业使用，因此商业协议是补充专有分发、支持、服务等额外安排，不是把整个仓库变成“商业专属”。
4. 每次发行固定第三方版本/提交；保存上游仓库、commit、来源日期、原始许可证、修改情况、版权人、分发形式以及签署/采购/员工授权凭证。源代码、二进制、模型、数据集分别建账。

## 逐项资源盘点

| 资源 | 仓库内证据与当前许可证 | 结论 / 发布动作 |
| --- | --- | --- |
| FinsSim 自有 Python、ROS2、配置、工具、文档 | 根目录 Apache-2.0；多个 ROS 包的 package.xml/setup.py 声明 Apache-2.0；finssim_core、finssim_cli、finssim_rl、finssim_irl 的包元数据现声明 Apache-2.0 | 权属核实后可纳入 Apache-2.0。元数据声明不替代对外部来源代码的逐文件 provenance 审查，也不构成权利转让证据。 |
| python/finssim_marl | 包目录的 LICENSE 和 pyproject.toml 声明 MIT，版权文本为 Cleanmarl team；`algorithms/mappo_multihead.py` 有 CleanMARL（MIT）来源注释 | 保留 MIT 及原始版权。权利人已确认其余 in-tree 实现为 FinsSim 自有；不能把来自 CleanMARL 的部分说成公司独有或重标为 Apache。 |
| CleanMarl 子模块 | third_party/cleanmarl 的 MIT LICENSE | 可继续作为 MIT 子模块发布；保留 LICENSE 和子模块提交号。 |
| MARLlib 子模块 | third_party/MARLlib 的 MIT LICENSE；另有 marllib/patch/hns/mujoco-worldgen/LICENSE（MIT，OpenAI） | 可按 MIT 条款分发，保留嵌套许可证和上游归属。不要因它位于 FinsSim 仓库而改标 Apache 或商业专有。 |
| research/MA-AIRL、multiagent-gail | 目录仅含 UPSTREAM.md 来源说明，未 vendoring 源代码 | 当前没有第三方源文件可再分发；继续保持只引用/自行实现的边界，不复制原项目代码时仍应记录实现来源。 |
| ros2_ws/src/grpc_ros_adapter | 包含 LABUST/MARUS 来源 Apache-2.0 LICENSE 与源代码版权头 | 按 Apache-2.0 保留原许可证、版权头和上游来源；FinsSim 修改不抹除 LABUST 原始权利。 |
| ros2_ws/src/uuv_sensor_msgs | BSD-3-Clause LICENSE，package.xml 声明 BSD-3-Clause | 保留 BSD-3-Clause 全文和版权声明；已消除包元数据与 LICENSE 文本的名称歧义。 |
| grpc_ros_adapter 内 generated protobuf | 上游为 `MARUSimulator/marus-proto`（旧 labust URL 已重定向）；默认源分支 `69a3b9188c3ff4def9c590061d30eefc57ab1f38` 声明 Apache-2.0，生成 Python 分支为 `ba41c776f845365f9f09edbb291eb658206bfba3`；本地 `protobuf/NOTICE.md` 已记录归属 | **许可证识别已完成。** 吸收副本早于当前生成分支且未记录精确 commit；可以连同 Apache-2.0 和 NOTICE 分发，但未来修改前须确定 `.proto` source commit 并重新生成，避免手改 generated 文件。 |
| FinsSimIsaacLab 独立仓库 | 已不作为 FinsSim-public 的 git submodule 或目录分发；第三方通知改为独立仓库边界 | 不是本仓公开发行阻断项。FinsSimIsaacLab 的自有代码、WarpAUV/Isaac AUV 参考代码、资产与 checkpoint 仍须在其独立仓库单独审计后发布。 |
| reference_code/FineSUB、joystick | 主仓中是指向仓库外代码的链接，不含可复现源码 | 已加入固定公开排除清单，避免公开快照包含断链。若未来纳入，FineSUB 及其生成代码、音频等需单独审计；不能由本仓 Apache 自动覆盖。 |
| IRL 与数据/模型 | `finsrov_fossen_physical_allocator_cache.pt` 可由仓内脚本生成 | cache 作为可再生的 FinsSim 参数缓存，而非第三方模型权重或训练数据。 |
| uv/ROS 外部依赖 | pyproject.toml、package.xml、uv.lock 等引用外部依赖，源代码并未全部 vendored | 锁文件不是许可证清单。对源码发行维护依赖名称、版本、许可证；对 wheel、容器、二进制发行另生成 SBOM/第三方通知。 |

## 发布前清单

- [ ] 确认最终权利人完整法定名称；核对职务作品、合作机构、学生/外包与外部贡献者授权。
- [x] 识别 generated protobuf 的上游 Apache-2.0 项目并补齐本地归属 NOTICE；后续仍须固定精确 `.proto` commit 后重新生成。
- [x] 将 IsaacLab 从 FinsSim-public 发行物边界中移除；其独立仓库的许可证审计与公开决定另行处理。
- [x] 为 finssim_core、finssim_cli、finssim_rl、finssim_irl 补项目许可证元数据；完成 finssim_marl in-tree 代码审计，并为唯一 CleanMARL 派生文件保留 MIT 来源注释。
- [x] 更新 THIRD_PARTY_NOTICES.md、ROS 来源记录和发行快照排除规则；保留 BSD/MIT/Apache 各自条款，不以商业协议重标。
- [x] 枚举主仓跟踪的二进制/媒体资源与 Git LFS：未发现 Git LFS；子模块外仅有标题图和可再生 allocator cache。标题图、实验数据、曲线和 checkpoint 的权属及公开授权已由权利人确认，不纳入本技术性第三方许可证审计。
- [x] 将 `uuv_sensor_msgs` 的 package.xml 许可证标识从笼统 `BSD` 校正为与 LICENSE 一致的 `BSD-3-Clause`。
- [x] 将公开版中的 FineSUB、joystick 外部链接加入固定发布排除规则；私有 main 保留其开发用途。

## 参考来源

- [Apache License 2.0 正文](https://www.apache.org/licenses/LICENSE-2.0.txt)
- [MARLlib 上游仓库](https://github.com/Replicable-MARL/MARLlib)
- [UUV Simulator LICENSE、NOTICE 与第三方清单](https://github.com/uuvsimulator/uuv_simulator)
- [ECA A9 上游仓库及 Apache-2.0 声明](https://github.com/uuvsimulator/eca_a9)
- [MARUS grpc_ros_adapter（Apache-2.0）](https://github.com/MARUSimulator/grpc_ros_adapter)
- [MARUS marus-proto（Apache-2.0）](https://github.com/MARUSimulator/marus-proto)

本文件记录仓库审计时可观察到的文件与来源信息；它不取代权利人授权文件、供应商合同或法律意见。发现某项许可证与上游当前页面不一致时，应以实际取得该副本时的许可及权利人书面记录进一步核实。
