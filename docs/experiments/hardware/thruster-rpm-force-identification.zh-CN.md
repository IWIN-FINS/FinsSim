# 推进器 RPM-Force 辨识结果汇总

本文记录当前 rpm-sweep 测量后用于 Unity 仿真的推进器转速-推力参数。配套机器可读参数文件为：

```text
ros2_ws/src/thruster_curve_measurement/config/thruster_rpm_force_unity.json
```

## 1. 模型

采用静水 bollard pull 条件下的平方模型：

```text
T = c1 * omega * abs(omega)
```

其中：

```text
T      推进器轴向推力，单位 N
omega  推进器角速度，单位 rad/s
c1     RPM-Force 辨识系数，单位 N/(rad/s)^2
```

为了便于 Unity 接入，最终参数按转速符号分成两个正值系数：

```text
omega >= 0:  thrust_axis_n = +c1_positive_omega * omega^2
omega <  0:  thrust_axis_n = -c1_negative_omega * omega^2
```

也就是说，JSON 文件里的 `c1_positive_omega` 和 `c1_negative_omega` 都是正值幅度；推力方向由 `omega` 的符号以及 Unity 中推进器自身的轴向定义决定。

RPM 与角速度换算：

```text
omega = rpm * 2*pi/60
```

## 2. 最终采用参数

当前测量结论：

```text
M006、M008 为反桨：
  omega < 0 分支 c1 约 9.2e-05
  omega > 0 分支 c1 约 1.20e-04

M005、M007 为正桨：
  omega < 0 分支 c1 约 7.3e-05
  omega > 0 分支 c1 约 8.8e-05
```

汇总表：

| 电机 | canonical index | canonical name | 桨型 | c1_negative_omega | c1_positive_omega |
| ---- | --------------: | -------------- | ---- | ----------------: | ----------------: |
| M005 |               4 | H_LF           | 正桨 |           7.3e-05 |           8.8e-05 |
| M006 |               5 | H_RF           | 反桨 |           9.2e-05 |          1.20e-04 |
| M007 |               6 | H_RB           | 正桨 |           7.3e-05 |           8.8e-05 |
| M008 |               7 | H_LB           | 反桨 |           9.2e-05 |          1.20e-04 |

按桨型分组后，Unity 可以只维护两组参数：

| 桨型组                   | 电机       | c1_negative_omega | c1_positive_omega |
| ------------------------ | ---------- | ----------------: | ----------------: |
| normal_propeller / 正桨  | M005, M007 |           7.3e-05 |           8.8e-05 |
| reverse_propeller / 反桨 | M006, M008 |           9.2e-05 |          1.20e-04 |

## 3. Unity 接入建议

建议 Unity 端使用如下计算逻辑：

```csharp
float OmegaFromRpm(float rpm)
{
    return rpm * 2.0f * Mathf.PI / 60.0f;
}

float ComputeThrusterAxisForce(float rpm, float c1PositiveOmega, float c1NegativeOmega)
{
    float omega = OmegaFromRpm(rpm);
    if (Mathf.Abs(omega) < 1e-6f)
    {
        return 0.0f;
    }

    float c1 = omega >= 0.0f ? c1PositiveOmega : c1NegativeOmega;
    return Mathf.Sign(omega) * c1 * omega * omega;
}
```

如果 Unity 中某个推进器 Transform 的本地正方向与实际正推力方向相反，不要改 `c1` 的正负；应在推进器配置中额外设置 `axis_sign = -1`：

```csharp
float forceForUnity = axisSign * ComputeThrusterAxisForce(rpm, c1Positive, c1Negative);
```

这样可以把“推力大小模型”和“安装方向/坐标方向”分开，避免 M006、M008 这类符号差异污染物理参数。

## 4. 使用限制

这些参数适合作为 Unity 仿真的静态初值，但不包含以下效应：

```text
1. 来流速度和进速比影响。
2. 推进器之间的互扰。
3. 贴近艇体或框架后的流场遮挡。
4. 电池电压变化、ESC 限流、电流饱和。
5. 快速升降速时的动态迟滞。
```

如果后续 Unity 仿真中发现高速段推力过大或过小，应优先检查 RPM 回传到 Unity 的单位、推进器局部轴向、`axis_sign`、以及正反桨分组是否正确。
