# 水流与物理波浪域随机化

`unity/marus-example` 当前具备物理波浪训练能力，但默认 flat-water build 会在打包前把 `PhysicalWaveWaterDataProvider` 切到 `FlatFallback` 并清空波浪。需要做水况域随机化时，使用专用 DR water build 入口。

## 结论

- `PhysicalWaveWaterDataProvider` 支持解析物理波浪：水面高度、法向、波浪诱导流速和稳态水流。
- `DomainRandomizationProfile` 支持随机化水流速度/方向，以及波浪数量、振幅、波长、周期、方向、相位、flow scale 和 max flow speed。
- `DomainRandomizationCoordinator` 会在 episode reset 时触发随机化，并支持 Unity 启动参数 `-fins-dr-seed` / `-fins-dr-mode`。
- 域随机化不是所有场景都必须开启；当前推荐用“普通 flat-water build”和“DR-water build”两套环境路径来选择训练环境。

## 相关组件

```text
Assets/Scripts/RL/PhysicalWaveWaterDataProvider.cs
```

水体物理查询入口。它会给 DWP2 / hydrodynamics 提供水面高度、法向、水流速度。启用域随机化后，它作为 `IEpisodeRandomizable` target，在每个 episode 开始时更新 steady current 和 waves。

```text
Assets/marus-core/Scripts/Hydrodynamics/DomainRandomization/DomainRandomizationProfile.cs
```

随机化参数范围。水况相关字段包括：

- `randomizeWater`
- `currentSpeed`
- `allowVerticalCurrent`
- `meanCurrent`
- `maxAngleFromMeanDegrees`
- `randomizeWaves`
- `waveComponentCount`
- `waveAmplitude`
- `waveWavelength`
- `wavePeriod`
- `waveDirectionDeg`
- `randomizeWavePhase`
- `waveFlowScale`
- `waveMaxFlowSpeed`

```text
Assets/marus-core/Scripts/Hydrodynamics/DomainRandomization/DomainRandomizationCoordinator.cs
```

episode 随机化调度器。RL Agent 的 `OnEpisodeBegin()` 会通过 `FinsROVAgentRuntime.RandomizeEpisodeIfPresent(...)` 找到该组件并触发 `RandomizeForEpisode()`。

## 开关语义

### 推荐方式

不需要域随机化时，使用普通 flat-water build：

```text
artifacts/unity_builds/rl/linux/ControlForPosition_SmoothNearTarget_Manual_10Hz_Server/ControlForPosition.x86_64
```

需要域随机化时，使用 DR-water build：

```text
artifacts/unity_builds/rl/linux/ControlForPosition_SmoothNearTarget_Manual_DRWater_10Hz_Server/ControlForPosition.x86_64
```

这样做最清晰：训练配置通过 `unity.env_path` 选择环境，不会影响其他场景。

### Unity 运行时开关

如果同一个 Unity executable 内已经挂了 `DomainRandomizationCoordinator`，可以通过 Unity 启动参数控制：

```text
-fins-dr-mode Disabled
-fins-dr-mode Train
-fins-dr-mode Evaluate
-fins-dr-seed 42
```

含义：

- `Disabled`：完全跳过 `RandomizeForEpisode()`。
- `Train`：训练时按 profile 范围随机化。
- `Evaluate`：保留评估模式标记；当前水况随机化 target 仍按 profile 采样，后续可以在 target 内按 mode 做更保守的评估逻辑。
- `-fins-dr-seed`：覆盖 coordinator 的 `baseSeed`。

注意：当前顶层 `finssim rl train -c ...` 已支持把 `unity.seed` 传到 Unity 的 `-fins-dr-seed`，但还没有把 `-fins-dr-mode` 暴露成 YAML 布尔开关。因此在 Python CLI 训练中，是否使用水况域随机化主要由 `unity.env_path` 是否指向 DR-water build 决定。

## 手动挂载流程

适合你在 Unity Editor 中给某个新场景手动接入水况域随机化。

1. 打开目标训练场景。
2. 找到或创建 `Environment/FlatWaterProvider`。
3. 在 `FlatWaterProvider` 上挂 `PhysicalWaveWaterDataProvider`。
4. 设置 `PhysicalWaveWaterDataProvider`：

```text
mode = AnalyticPhysicalWave
fallbackWaterHeight = 0
stillWaterHeight = 0
targetSurfaceOverride = null
includeWaveNormals = true
includeWaveFlow = true
includeVerticalOrbitalFlow = true
randomizeWaterCurrentFromProfile = true
randomizeWavesFromProfile = true
forceAnalyticModeWhenRandomizingWaves = true
```

5. 创建或复用一个 `DomainRandomizationProfile`。当前自动流程使用路径：

```text
Assets/Generated/DomainRandomization/FinsROVWaterDomainRandomization.asset
```

6. 在 `Environment` 上挂 `DomainRandomizationCoordinator`。
7. 设置 `DomainRandomizationCoordinator`：

```text
profile = Assets/Generated/DomainRandomization/FinsROVWaterDomainRandomization.asset
mode = Train
baseSeed = 12345
incrementSeedPerEpisode = true
randomizeOnStart = false
autoFindTargets = false
targetBehaviours = [FlatWaterProvider.PhysicalWaveWaterDataProvider]
applyCommandLineOverrides = true
seedCommandLineArg = -fins-dr-seed
modeCommandLineArg = -fins-dr-mode
```

8. 如果该 build 用于 headless RL 训练，可以关闭 `Ocean` 渲染对象，保留轻量的 `FlatWaterProvider` 作为物理水体查询入口。

## 自动准备流程

当前仓库已经提供了 Editor 准备方法，会自动完成上面的挂载和默认 profile 配置。

在 Unity Editor 中调用：

```text
FinsSimScenePreparation.PrepareControlForPositionSmoothNearTargetManualDomainRandomizedWater
```

该方法会：

- 打开 `Assets/Scenes/ControlForPosition_smooth_near_target_manual.unity`。
- 关闭 `Ocean`。
- 创建或复用 `FlatWaterProvider`。
- 挂载并配置 `PhysicalWaveWaterDataProvider`。
- 创建或更新 `Assets/Generated/DomainRandomization/FinsROVWaterDomainRandomization.asset`。
- 在 `Environment` 上挂 `DomainRandomizationCoordinator`。
- 将 coordinator 的 target 限定为水体 provider，避免这个 DR-water build 顺手随机化刚体、推进器或水动力参数。

默认水况范围：

```text
currentSpeed = 0.0 ~ 0.35 m/s
allowVerticalCurrent = false
maxAngleFromMeanDegrees = 180 deg
waveComponentCount = 1 ~ 3
waveAmplitude = 0.0 ~ 0.05 m
waveWavelength = 2.5 ~ 12 m
wavePeriod = 2 ~ 7 s
waveDirectionDeg = 0 ~ 360 deg
randomizeWavePhase = true
waveFlowScale = 0.5 ~ 1.2
waveMaxFlowSpeed = 0.1 ~ 0.35 m/s
```

## 打包 DR 训练环境

在 Unity batchmode 或 Editor 中调用：

```text
FinsSimLinuxBuild.BuildControlForPositionSmoothNearTargetManualDomainRandomizedWaterServer
```

输出路径：

```text
artifacts/unity_builds/rl/linux/ControlForPosition_SmoothNearTarget_Manual_DRWater_10Hz_Server/ControlForPosition.x86_64
```

该准备流程会：

- 关闭 Ocean 渲染对象，避免 headless 训练负担。
- 使用 `FlatWaterProvider` 上的 `PhysicalWaveWaterDataProvider` 作为轻量解析物理水面。
- 创建或更新 `Assets/Generated/DomainRandomization/FinsROVWaterDomainRandomization.asset`。
- 在 `Environment` 上挂 `DomainRandomizationCoordinator`，只把水体 provider 作为随机化目标。

batchmode 示例：

```bash
cd .
 -batchmode -quit -nographics \
  -projectPath /marus-example \
  -executeMethod FinsSimLinuxBuild.BuildControlForPositionSmoothNearTargetManualDomainRandomizedWaterServer \
  -logFile /tmp/finssim_drwater_build.log
```

## 开启域随机化训练

训练配置中使用 DR-water build，并设置可复现实验 seed：

```yaml
unity:
  env_path: ./artifacts/unity_builds/rl/linux/ControlForPosition_SmoothNearTarget_Manual_DRWater_10Hz_Server/ControlForPosition.x86_64
  seed: 42
```

当前示例配置：

```text
configs/rl/legacy_pid/hybrid_pid_position_dr.yaml
```

启动：

```bash
cd .
uv run --package finssim-cli finssim rl train -c configs/rl/legacy_pid/hybrid_pid_position_dr.yaml
```

`unity.seed` 会被传给 `finssim-rl --seed`，Python 侧会为每个 Unity worker 派生不同 `-fins-dr-seed`，避免并行 worker 使用完全相同的水况序列。

## 关闭域随机化训练

最稳妥的方式是改回普通 flat-water build：

```yaml
unity:
  env_path: ./artifacts/unity_builds/rl/linux/ControlForPosition_SmoothNearTarget_Manual_10Hz_Server/ControlForPosition.x86_64
```

如果是手动启动 Unity executable，也可以显式传：

```bash
./ControlForPosition.x86_64 -batchmode -nographics -fins-dr-mode Disabled
```

当前 Python CLI 训练入口还没有 YAML 字段直接转发 `-fins-dr-mode Disabled`。在接入该字段之前，配置级开关建议通过 `unity.env_path` 选择普通 build 或 DR-water build。

## 推荐配置命名

建议保留两类配置，避免误开随机化：

```text
configs/rl/pose_control/ppo_control_for_pose_smooth_near_target.yaml       # 普通/平水训练
configs/rl/legacy_pid/hybrid_pid_position_dr.yaml                        # 水况域随机化训练
```

后续如果要做更细粒度的开关，可以在 YAML 中加类似字段：

```yaml
unity:
  domain_randomization: true
  domain_randomization_mode: Train
  seed: 42
```

然后在 Python 启动 Unity 时转成：

```text
-fins-dr-mode Train
-fins-dr-seed 42
```

关闭时转成：

```text
-fins-dr-mode Disabled
```

## 检查清单

- 场景里只有需要随机化的 target 在 `DomainRandomizationCoordinator.targetBehaviours` 中。
- `autoFindTargets = false` 时，不会自动随机化其他 `IEpisodeRandomizable` 组件。
- DR-water build 的 provider 是 `AnalyticPhysicalWave`，普通 flat-water build 的 provider 是 `FlatFallback`。
- `unity.seed` 只是随机种子，不是开关。
- 多环境训练时，worker seed 会按 `seed + rank * 1000003` 派生。
