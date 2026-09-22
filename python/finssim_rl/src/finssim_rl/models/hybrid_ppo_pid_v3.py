"""
混合 PPO+PID 控制器 v3

核心思想：
- PPO 策略输出 9 维归一化极点参数：3 组 PID，每组 [τ1, τ2, τ3]
- τ 被映射到正值范围 [TAU_MIN, TAU_MAX]，再通过极点配置公式生成 Kp/Ki/Kd
- 控制误差只使用 x/y/z 位置误差，roll/pitch/yaw 姿态误差始终置零
- 单环 PID 根据误差输出 6 维控制分量，再由推力分配器转换为 8 维推进器推力

架构：
   观测 → PPO(三组3维极点 [τ1, τ2, τ3])
        ↓
   [正值映射] → τ∈(τ_min, τ_max)
        ↓
   [极点配置] → Kp, Ki, Kd
        ↓
   [从观测提取 x/y/z 位置误差]
        ↓
   [单环PID执行] → 6维控制分量
        ↓
   [推力分配器] → 8维推进器推力
        ↓
   环境接收8维动作

## 坐标系定义

### Unity 模型局部坐标系
- **X 轴**：潜器前方（Forward）
- **Y 轴**：向上（Up）
- **Z 轴**：潜器左侧（Left）

### 潜器机体坐标系
- **X 轴**：正前方（Forward）
- **Y 轴**：向上（Up）
- **Z 轴**：向左（Left）

### 坐标系映射关系
```
Unity模型局部坐标系 → 潜器控制坐标系
  X(前)   →   X(前)
  Y(上)   →   Y(上)
  Z(左)   →   Z(左)
```

### 旋转对应关系
- **Roll（绕X轴）**：在潜器坐标系中，绕前进方向旋转，左右滚动
- **Pitch（绕Z轴）**：在潜器坐标系中，绕左右轴旋转，上下俯仰
- **Yaw（绕Y轴）**：在潜器坐标系中，绕竖直轴旋转，左右转向

观测中的四元数 [x, y, z, w] 表示潜器本体 local -> world 的旋转。
"""
from typing import Optional, Union, Tuple, Dict, Any, Callable, Iterable
from pathlib import Path
from io import BufferedIOBase
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from finssim_rl.models.ppo_rewrite import PPORewrite
from finssim_rl.models.pid_controller import PIDController


PID_PARAM_DIM = 9
POLE_PARAM_DIM = PID_PARAM_DIM

# PID 参数顺序：depth, surge, sway。当前点位悬停任务不训练 yaw 参数。
DEFAULT_PID_PARAMS = np.array([
    3.0, 0.1, 0.05,  # depth
    3.0, 0.3, 0.05,     # surge
    #500.0, 0.01, 0.5,     # sway
    3.0, 0.3, 0.05,     # sway
], dtype=np.float32)

POSE_ERROR_MASK_6D = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
RATE_ERROR_MASK_3D = np.array([0.0, 0.0, 0.0], dtype=np.float32)
TAU_MIN = 0.1
TAU_MAX = 2.7
TAU_EPS = 1e-6
ACTION_MEAN_CLAMP = 5.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
TRAIN_TENSOR_CLAMP = 1e6


def _sanitize_np_array(array: np.ndarray, clamp_abs: Optional[float] = None) -> np.ndarray:
    """把 NumPy 数组中的 NaN/Inf 转成有限值，避免污染 rollout buffer。"""
    result = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_abs is not None and clamp_abs > 0.0:
        result = np.clip(result, -float(clamp_abs), float(clamp_abs))
    return result.astype(np.float32, copy=False)


def _sanitize_torch_tensor(tensor: torch.Tensor, clamp_abs: Optional[float] = None) -> torch.Tensor:
    """把 torch Tensor 清理为有限值；用于策略前向和训练 minibatch。"""
    result = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_abs is not None and clamp_abs > 0.0:
        result = torch.clamp(result, -float(clamp_abs), float(clamp_abs))
    return result


def _sanitize_torch_obs(obs: Any) -> Any:
    """递归清理 SB3 传入策略的 observation。"""
    if isinstance(obs, torch.Tensor):
        return _sanitize_torch_tensor(obs, clamp_abs=TRAIN_TENSOR_CLAMP)
    if isinstance(obs, dict):
        return {key: _sanitize_torch_obs(value) for key, value in obs.items()}
    return obs


def get_stage_mask(stage: str) -> np.ndarray:
    """返回当前训练阶段允许更新的 PID 参数掩码。"""
    mask = np.zeros(PID_PARAM_DIM, dtype=np.float32)
    if stage == "depth_only":
        mask[0:3] = 1.0
    elif stage == "outer_only":
        # 保留阶段名以兼容训练配置；当前无 yaw 控制，该阶段训练水平位置环。
        mask[3:9] = 1.0
    elif stage == "frozen":
        # 进入 frozen 时，所有 PID 参数保持默认，不接收梯度。
        pass
    else:  # all
        mask[:] = 1.0
    return mask


def map_policy_output_to_pid_params(
    policy_actions: np.ndarray,
    stage: str,
    base_pid_params: np.ndarray,
) -> np.ndarray:
    """
    将 PPO 输出的极点参数映射为 PID 参数（自适应极点布置法）。

    PPO 每3维输出一组归一化极点 [τ1, τ2, τ3]，共 depth/surge/sway 三组。
    先映射到正值范围
    [TAU_MIN, TAU_MAX]，再按极点配置公式生成 PID：
      Kp = (τ1 + τ2 + τ3) / (τ1 τ2 τ3)
      Ki = 1 / (τ1 τ2 τ3)
      Kd = (τ1τ2 + τ2τ3 + τ3τ1) / (τ1 τ2 τ3)

    极点公式生成的 Kp/Ki/Kd 直接作为最终 PID 参数。
    训练阶段 mask 只决定对应通道是否使用生成值；未训练通道保留默认PID参数。
    """
    params_final, _ = map_policy_output_to_pid_params_and_tau(policy_actions, stage, base_pid_params)
    return params_final


def map_policy_output_to_pid_params_and_tau(
    policy_actions: np.ndarray,
    stage: str,
    base_pid_params: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回极点配置后的 PID 参数以及对应的 τ 参数，供控制和调试打印共用。"""
    actions_np = np.asarray(policy_actions, dtype=np.float32)
    if actions_np.ndim == 1:
        actions_np = actions_np.reshape(1, -1)

    # 对齐维度
    if actions_np.shape[1] < PID_PARAM_DIM:
        padded = np.zeros((actions_np.shape[0], PID_PARAM_DIM), dtype=np.float32)
        padded[:, :actions_np.shape[1]] = actions_np
        actions_np = padded
    elif actions_np.shape[1] > PID_PARAM_DIM:
        actions_np = actions_np[:, :PID_PARAM_DIM]

    # 1. 策略动作限制到归一化极点范围 [-1, 1]。
    actions_squashed = np.clip(actions_np, -1.0, 1.0)

    # 2. 对数尺度映射到正极点范围，保证 τ > 0。
    tau_params = map_normalized_actions_to_poles(actions_squashed)

    # 3. 极点配置得到最终 PID 参数。
    generated_params = pole_params_to_pid_params(tau_params)
    base = np.asarray(base_pid_params, dtype=np.float32).reshape(1, -1)

    # 4. 应用训练阶段 Mask：训练通道使用极点公式生成值，未训练通道保持默认PID。
    stage_mask = get_stage_mask(stage).reshape(1, -1)
    params_final = generated_params * stage_mask + base * (1.0 - stage_mask)

    return params_final.astype(np.float32), tau_params.astype(np.float32)


def map_normalized_actions_to_poles(
    normalized_actions: np.ndarray,
    tau_min: float = TAU_MIN,
    tau_max: float = TAU_MAX,
) -> np.ndarray:
    """将 [-1, 1] 的归一化动作映射为正极点 τ。"""
    actions_np = np.asarray(normalized_actions, dtype=np.float32)
    tau_min = max(float(tau_min), TAU_EPS)
    tau_max = max(float(tau_max), tau_min + TAU_EPS)
    ratio = (actions_np + 1.0) * 0.5
    log_tau = np.log(tau_min) + ratio * (np.log(tau_max) - np.log(tau_min))
    return np.exp(log_tau).astype(np.float32)


def pole_params_to_pid_params(tau_params: np.ndarray) -> np.ndarray:
    """按 [τ1, τ2, τ3] 三极点配置公式生成 [Kp, Ki, Kd]。"""
    tau_np = np.asarray(tau_params, dtype=np.float32)
    if tau_np.ndim == 1:
        tau_np = tau_np.reshape(1, -1)
    if tau_np.shape[1] < POLE_PARAM_DIM:
        padded = np.full((tau_np.shape[0], POLE_PARAM_DIM), TAU_MAX, dtype=np.float32)
        padded[:, :tau_np.shape[1]] = tau_np
        tau_np = padded
    elif tau_np.shape[1] > POLE_PARAM_DIM:
        tau_np = tau_np[:, :POLE_PARAM_DIM]

    tau_np = np.maximum(tau_np, TAU_EPS)
    tau_groups = tau_np.reshape(tau_np.shape[0], -1, 3)
    tau1 = tau_groups[:, :, 0]
    tau2 = tau_groups[:, :, 1]
    tau3 = tau_groups[:, :, 2]
    denom = np.maximum(tau1 * tau2 * tau3, TAU_EPS)

    kp = (tau1 + tau2 + tau3) / denom
    ki = 1.0 / denom
    kd = (tau1 * tau2 + tau2 * tau3 + tau3 * tau1) / denom

    pid_groups = np.stack([kp, ki, kd], axis=2)
    return pid_groups.reshape(tau_np.shape[0], POLE_PARAM_DIM).astype(np.float32)


def _normalize_obs_array(obs: Union[np.ndarray, Dict[str, np.ndarray]], default_num_envs: int = 1) -> np.ndarray:
    """把 Unity/Gym 可能返回的 dict/list/tuple/多余单例维度统一成 (batch, obs_dim)。"""
    if isinstance(obs, dict):
        if "state" in obs:
            obs = obs["state"]
        elif "observation" in obs:
            obs = obs["observation"]
        elif "obs" in obs:
            obs = obs["obs"]
        else:
            obs = next(iter(obs.values()))

    if isinstance(obs, (list, tuple)):
        vector_candidates = []
        for item in obs:
            item_array = np.asarray(item, dtype=np.float32)
            if item_array.ndim >= 1 and item_array.shape[-1] >= 14:
                vector_candidates.append(item_array)
        obs = vector_candidates[0] if vector_candidates else obs[0]

    obs_array = np.asarray(obs, dtype=np.float32)
    if obs_array.ndim == 1:
        obs_array = obs_array.reshape(1, -1)
    elif obs_array.ndim > 2:
        obs_array = np.squeeze(obs_array)
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(1, -1)
        elif obs_array.ndim > 2 and obs_array.shape[-1] >= 14:
            obs_array = obs_array.reshape(-1, obs_array.shape[-1])

    if obs_array.ndim != 2:
        if not getattr(_normalize_obs_array, "_warned_bad_shape", False):
            print(f"[ObsParse WARNING] unsupported obs shape={obs_array.shape}, fallback to zeros")
            setattr(_normalize_obs_array, "_warned_bad_shape", True)
        obs_array = np.zeros((default_num_envs, 20), dtype=np.float32)
    return obs_array


def _debug_print_obs_mapping(obs: Union[np.ndarray, Dict[str, np.ndarray]], step_count: int, interval: int, default_num_envs: int = 1) -> None:
    """周期性打印 Unity 20维观测字段，核对 Python 索引是否和 CollectObservations 对齐。"""
    if interval <= 0 or step_count % interval != 0:
        return

    obs_array = _normalize_obs_array(obs, default_num_envs=default_num_envs)
    if obs_array.shape[1] < 14:
        print(f"[ObsParse Step {step_count}] obs_shape={obs_array.shape}, obs_dim<14")
        return

    batch_idx = 0
    target_pos = obs_array[batch_idx, 0:3]
    target_quat = obs_array[batch_idx, 3:7]
    self_pos = obs_array[batch_idx, 7:10]
    self_quat = obs_array[batch_idx, 10:14]
    pos_diff = target_pos - self_pos
    dist = float(np.linalg.norm(pos_diff))
    linear_vel = obs_array[batch_idx, 14:17] if obs_array.shape[1] >= 17 else np.zeros(3, dtype=np.float32)
    angular_vel = obs_array[batch_idx, 17:20] if obs_array.shape[1] >= 20 else np.zeros(3, dtype=np.float32)
    current_yaw_unity = _quat_to_euler_deg_unity(self_quat.reshape(1, 4))[0, 2]
    current_yaw_sub = _quat_to_euler_deg_xyzw(_quat_unity_to_submarine(self_quat.reshape(1, 4)))[0, 2]
    linear_vel_local = _quat_rotate_inverse(self_quat.reshape(1, 4), linear_vel.reshape(1, 3))[0]
    linear_vel_body = _pos_unity_to_submarine(linear_vel_local.reshape(1, 3))[0]
    target_unity_local = _quat_rotate_inverse(self_quat.reshape(1, 4), pos_diff.reshape(1, 3))[0]
    target_body = _pos_unity_to_submarine(target_unity_local.reshape(1, 3))[0]
    target_body_yaw = _calc_yaw_error_from_vec_body(target_body.reshape(1, 3))[0]

    prev_self_pos = getattr(_debug_print_obs_mapping, "_prev_self_pos", None)
    delta_pos = np.zeros(3, dtype=np.float32)
    if prev_self_pos is not None:
        delta_pos = self_pos - prev_self_pos
    setattr(_debug_print_obs_mapping, "_prev_self_pos", self_pos.copy())

    print(
        f"[SubmarineState Step {step_count}] "
        f"self_pos_unity[x,y,z]=[{self_pos[0]:+.4f}, {self_pos[1]:+.4f}, {self_pos[2]:+.4f}] "
        f"delta_since_last_debug=[{delta_pos[0]:+.4f}, {delta_pos[1]:+.4f}, {delta_pos[2]:+.4f}] "
        f"linear_vel_unity=[{linear_vel[0]:+.4f}, {linear_vel[1]:+.4f}, {linear_vel[2]:+.4f}] "
        f"linear_vel_body[x,y,z]=[{linear_vel_body[0]:+.4f}, {linear_vel_body[1]:+.4f}, {linear_vel_body[2]:+.4f}] "
        f"angular_vel_unity=[{angular_vel[0]:+.4f}, {angular_vel[1]:+.4f}, {angular_vel[2]:+.4f}] "
        f"yaw_unity={current_yaw_unity:+.2f}, yaw_sub={current_yaw_sub:+.2f}",
        flush=True,
    )
    print(
        f"[TargetBearing Step {step_count}] "
        f"target_unity_local[x,y,z]=[{target_unity_local[0]:+.4f}, {target_unity_local[1]:+.4f}, {target_unity_local[2]:+.4f}] "
        f"target_body[x,y,z]=[{target_body[0]:+.4f}, {target_body[1]:+.4f}, {target_body[2]:+.4f}] "
        f"yaw_from_body={target_body_yaw:+.2f}",
        flush=True,
    )
    print(
        f"[ObsParse Step {step_count}] obs_shape={obs_array.shape} "
        f"target_pos={target_pos}, target_quat={target_quat}, "
        f"self_pos={self_pos}, self_quat={self_quat}, "
        f"target-self={pos_diff}, dist={dist:.4f}",
        flush=True,
    )


def _get_yaw_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """从四元数提取yaw角（全局坐标系）。
    
    参数：quat_xyzw shape (batch, 4) 或 (4,)，Unity标准坐标系
    返回：yaw angle in degrees, shape (batch,) 或 scalar，航行器坐标系
    """
    # 先从Unity标准系转换到航行器系
    quat_sub = _quat_unity_to_submarine(quat_xyzw)
    euler_deg = _quat_to_euler_deg_xyzw(quat_sub)
    return euler_deg[:, 2] if euler_deg.ndim > 1 else euler_deg[2]


def _wrap_to_180(angle_deg: np.ndarray) -> np.ndarray:
    """将角度包装到 [-180, 180] 范围。"""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _vec_to_yaw_deg(vec_xyz: np.ndarray) -> np.ndarray:
    """计算向量的yaw角（在水平面的投影角）。
    
    坐标系：X前，Z左，Y下 → 在水平面(x-z)计算角度
    参数：vec_xyz shape (batch, 3) 或 (3,)
    返回：yaw angle in degrees
    """
    vec = np.asarray(vec_xyz, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, 3)
    
    # 在x-z平面计算角度
    yaw_rad = np.arctan2(vec[:, 2], vec[:, 0])
    yaw_deg = np.degrees(yaw_rad)
    return yaw_deg.flatten() if yaw_deg.ndim > 1 else yaw_deg


def _quat_unity_to_submarine(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    将四元数从Unity标准坐标系转换到航行器坐标系。
    
    **关键概念**：
    - Unity中的四元数 q_unity 表示从 Unity 全局系 (X右, Y上, Z前) 到潜器本体的旋转
    - 潜器参考系 (X前, Y上, Z左) 是 Unity 全局系绕Y轴旋转-90°得到
    - 转换到潜器坐标系的四元数需要调整，使其表示从潜器参考系到潜器本体的旋转
    
    **转换公式**：q_sub = q_rot * q_unity * q_rot^(-1)
    其中 q_rot 是从 Unity 系到潜器参考系的旋转（绕Y轴-90°）
    
    参数：quat_xyzw shape (batch, 4) 或 (4,)，四元数 [x, y, z, w]
    返回：转换后的四元数，shape 同输入
    """
    q = np.asarray(quat_xyzw, dtype=np.float32)
    is_1d = q.ndim == 1
    if is_1d:
        q = q.reshape(1, 4)
    
    # 绕Y轴旋转-90度对应的四元数：q_rot = [0, sin(-45°), 0, cos(-45°)]
    sqrt2_2 = np.sqrt(2.0) / 2.0
    q_rot = np.array([0.0, -sqrt2_2, 0.0, sqrt2_2], dtype=np.float32)
    
    # 计算 q_rot * q_unity
    x_rot, y_rot, z_rot, w_rot = q_rot[0], q_rot[1], q_rot[2], q_rot[3]
    x_unity = q[:, 0]
    y_unity = q[:, 1]
    z_unity = q[:, 2]
    w_unity = q[:, 3]
    
    # 四元数乘法：q_rot * q_unity
    x1 = w_rot * x_unity + x_rot * w_unity + y_rot * z_unity - z_rot * y_unity
    y1 = w_rot * y_unity - x_rot * z_unity + y_rot * w_unity + z_rot * x_unity
    z1 = w_rot * z_unity + x_rot * y_unity - y_rot * x_unity + z_rot * w_unity
    w1 = w_rot * w_unity - x_rot * x_unity - y_rot * y_unity - z_rot * z_unity
    
    # q_rot 的逆（共轭）：q_rot^(-1) = [0, √2/2, 0, √2/2]
    x_rot_inv = -x_rot
    y_rot_inv = -y_rot
    z_rot_inv = -z_rot
    w_rot_inv = w_rot
    
    # 四元数乘法：(q_rot * q_unity) * q_rot^(-1)
    x_result = w1 * x_rot_inv + x1 * w_rot_inv + y1 * z_rot_inv - z1 * y_rot_inv
    y_result = w1 * y_rot_inv - x1 * z_rot_inv + y1 * w_rot_inv + z1 * x_rot_inv
    z_result = w1 * z_rot_inv + x1 * y_rot_inv - y1 * x_rot_inv + z1 * w_rot_inv
    w_result = w1 * w_rot_inv - x1 * x_rot_inv - y1 * y_rot_inv - z1 * z_rot_inv
    
    result = np.stack([x_result, y_result, z_result, w_result], axis=1).astype(np.float32)
    
    return result[0] if is_1d else result


def _pos_unity_to_submarine(pos_xyz: np.ndarray) -> np.ndarray:
    """
    将位置向量从 Unity 模型局部坐标系转换到航行器控制坐标系。
    
    注意：调用方通常已经先用当前四元数把世界向量转回 selfTransform
    的局部坐标。当前 Unity 潜器模型的局部轴就是：
    X前, Y上, Z左。
    
    映射关系：
    - x_sub = x_unity_local (前)
    - y_sub = y_unity (上)
    - z_sub = z_unity_local (左)
    
    参数：pos_xyz shape (batch, 3) 或 (3,)
    返回：转换后的位置向量，shape同输入
    """
    pos = np.asarray(pos_xyz, dtype=np.float32)
    is_1d = pos.ndim == 1
    if is_1d:
        pos = pos.reshape(1, 3)
    
    result = pos.astype(np.float32)
    
    return result[0] if is_1d else result


def _quat_to_euler_deg_unity(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    将四元数 [x, y, z, w] 转为欧拉角 [roll, pitch, yaw]（度）。
    
    **在Unity全局坐标系中** (X right, Y up, Z forward)
    使用Unity官方的ZXY欧拉角顺序。
    
    - yaw：绕Y轴旋转（右偏为正）
    - 返回的yaw是全局yaw
    
    参考：https://docs.unity3d.com/ScriptReference/Quaternion-eulerAngles.html
    """
    q = np.asarray(quat_xyzw, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, 4)

    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    # ZXY 欧拉角顺序
    sinr = 2.0 * (w * x - y * z)
    sinr = np.clip(sinr, -1.0, 1.0)
    roll = np.arcsin(sinr)
    
    pitch = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # 绕Y轴旋转（yaw）—— 在全局坐标系中
    yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y))

    euler_rad = np.stack([roll, pitch, yaw], axis=1)
    return np.degrees(euler_rad).astype(np.float32)


def _quat_to_euler_deg_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    将四元数 [x, y, z, w] 转为欧拉角 [roll, pitch, yaw]（度）。
    
    **在航行器坐标系中** (X前、Y上、Z左)
    用于已经转换到航行器坐标系的四元数。
    
    使用Unity官方的ZXY欧拉角顺序。
    
    - roll（绕X轴，纵轴）
    - pitch（绕Z轴，横轴）
    - yaw（绕Y轴，竖直轴）
    
    参考：https://docs.unity3d.com/ScriptReference/Quaternion-eulerAngles.html
    """
    q = np.asarray(quat_xyzw, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, 4)

    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    # ZXY 欧拉角顺序
    sinr = 2.0 * (w * x - y * z)
    sinr = np.clip(sinr, -1.0, 1.0)
    roll = np.arcsin(sinr)
    
    pitch = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y))

    euler_rad = np.stack([roll, pitch, yaw], axis=1)
    return np.degrees(euler_rad).astype(np.float32)



def _quat_forward_vector_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """将四元数 [x, y, z, w] 转为机体前向向量（按 local +X 轴）。"""
    q = np.asarray(quat_xyzw, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, 4)

    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    forward_x = 1.0 - 2.0 * (y * y + z * z)
    forward_y = 2.0 * (x * y + w * z)
    forward_z = 2.0 * (x * z - w * y)
    return np.stack([forward_x, forward_y, forward_z], axis=1).astype(np.float32)


def _quat_rotate_inverse(quat_xyzw: np.ndarray, vec_world: np.ndarray) -> np.ndarray:
    """用四元数的共轭（逆旋转）将世界坐标向量转到机体坐标系。
    
    参数：
    - quat_xyzw: shape (batch, 4) 或 (4,)，[x, y, z, w]
    - vec_world: shape (batch, 3)，世界坐标系向量
    
    返回：shape (batch, 3)，机体坐标系向量
    """
    q = np.asarray(quat_xyzw, dtype=np.float32)
    v = np.asarray(vec_world, dtype=np.float32)
    
    if q.ndim == 1:
        q = q.reshape(1, 4)
    if v.ndim == 1:
        v = v.reshape(1, 3)
    
    # 四元数共轭（逆旋转）
    q_inv = np.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], axis=1)
    
    # 四元数乘法：q_inv * v * q
    # Unity 的 transform.rotation 表示 local -> world；世界向量转机体系需要逆旋转。
    # 首先 v 作为四元数 [vx, vy, vz, 0]
    qv = np.stack([
        q_inv[:, 3] * v[:, 0] + q_inv[:, 1] * v[:, 2] - q_inv[:, 2] * v[:, 1],
        q_inv[:, 3] * v[:, 1] + q_inv[:, 2] * v[:, 0] - q_inv[:, 0] * v[:, 2],
        q_inv[:, 3] * v[:, 2] + q_inv[:, 0] * v[:, 1] - q_inv[:, 1] * v[:, 0],
        -q_inv[:, 0] * v[:, 0] - q_inv[:, 1] * v[:, 1] - q_inv[:, 2] * v[:, 2],
    ], axis=1)
    
    # 然后 qv * q
    result = np.stack([
        qv[:, 3] * q[:, 0] + qv[:, 0] * q[:, 3] + qv[:, 1] * q[:, 2] - qv[:, 2] * q[:, 1],
        qv[:, 3] * q[:, 1] + qv[:, 1] * q[:, 3] + qv[:, 2] * q[:, 0] - qv[:, 0] * q[:, 2],
        qv[:, 3] * q[:, 2] + qv[:, 2] * q[:, 3] + qv[:, 0] * q[:, 1] - qv[:, 1] * q[:, 0],
    ], axis=1)
    
    return result.astype(np.float32)


def _calc_yaw_error_from_vec_body(vec_body_xyz: np.ndarray) -> np.ndarray:
    """在机体坐标系中计算向量和x轴的夹角（即yaw误差）。
    
    参数：
    - vec_body_xyz: shape (batch, 3)，机体坐标系向量
    
    返回：shape (batch,)，yaw错误（度）
    """
    # 在x-z平面计算夹角
    yaw_error_rad = np.arctan2(vec_body_xyz[:, 2], vec_body_xyz[:, 0])
    return np.degrees(yaw_error_rad).astype(np.float32)


def extract_pose_error(obs: Union[np.ndarray, Dict[str, np.ndarray]], default_num_envs: int = 1) -> np.ndarray:
    """
    提取6维位姿误差（角度单位：度）。

    控制误差仅保留 x/y/z 三个位置自由度：
    [dx, dy, dz, 0, 0, 0]

    输入约定（Unity CollectObservations，20维）：
    [0:3]  target_pos_rel
    [3:7]  target_rot_quat_xyzw
    [7:10] self_pos_rel
    [10:14] self_rot_quat_xyzw
    [14:17] linear_velocity
    [17:20] angular_velocity
    """
    obs_array = _normalize_obs_array(obs, default_num_envs)
    batch_size, obs_dim = obs_array.shape

    # 维度不足时回退到简单 6 维状态格式。
    current_xyz = np.zeros((batch_size, 3), dtype=np.float32)
    current_rpy_deg = np.zeros((batch_size, 3), dtype=np.float32)
    target_xyz = np.tile(np.array([0.0, -3.0, 0.0], dtype=np.float32), (batch_size, 1))

    if obs_dim >= 14:
        target_xyz_unity = obs_array[:, 0:3]
        current_xyz_unity = obs_array[:, 7:10]
        target_xyz = target_xyz_unity
        current_xyz = current_xyz_unity
        
        pos_diff_unity = target_xyz_unity - current_xyz_unity
        current_quat = obs_array[:, 10:14]

        # Unity 的 transform.rotation 表示物体 local -> world。
        # 先把世界坐标误差转回 Unity 本地坐标(local x右, y上, z前)，
        # 再映射到控制器使用的潜器机体系(x前, y上, z左)。
        pos_error_unity_local = _quat_rotate_inverse(current_quat, pos_diff_unity)
        pos_error_local = _pos_unity_to_submarine(pos_error_unity_local)

        zero_attitude_error = np.zeros((batch_size,), dtype=np.float32)
        ang_error_deg = np.stack([zero_attitude_error, zero_attitude_error, zero_attitude_error], axis=1).astype(np.float32)

        error6 = np.concatenate([pos_error_local, ang_error_deg], axis=1).astype(np.float32)
        
        return _mask_controlled_pose_error(error6)
    else:
        # 简单状态格式：[x, y, z, roll, pitch, yaw]
        state6 = np.zeros((batch_size, 6), dtype=np.float32)
        use_dim = min(obs_dim, 6)
        if use_dim > 0:
            state6[:, :use_dim] = obs_array[:, :use_dim]
        current_xyz = state6[:, :3]
        current_rpy_deg = state6[:, 3:6]

    pos_error = target_xyz - current_xyz

    if obs_dim < 14:
        zero_attitude_error = np.zeros((batch_size,), dtype=np.float32)
        ang_error_deg = np.stack([zero_attitude_error, zero_attitude_error, zero_attitude_error], axis=1).astype(np.float32)

    error6 = np.concatenate([pos_error, ang_error_deg], axis=1).astype(np.float32)
    return _mask_controlled_pose_error(error6)


def _mask_controlled_pose_error(error6: np.ndarray) -> np.ndarray:
    """只保留 x/y/z 位置误差，彻底屏蔽姿态误差。"""
    return (error6 * POSE_ERROR_MASK_6D.reshape(1, -1)).astype(np.float32)


def extract_current_angles_deg(obs: Union[np.ndarray, Dict[str, np.ndarray]], default_num_envs: int = 1) -> np.ndarray:
    """提取当前 [roll, pitch, yaw]（度）。"""
    obs_array = _normalize_obs_array(obs, default_num_envs)
    batch_size, obs_dim = obs_array.shape
    if obs_dim >= 14:
        # Unity: self quaternion at [10:14]，需要先转换坐标系
        quat_sub = _quat_unity_to_submarine(obs_array[:, 10:14])
        return _quat_to_euler_deg_xyzw(quat_sub)

    angles = np.zeros((batch_size, 3), dtype=np.float32)
    if obs_dim > 3:
        use_dim = min(3, obs_dim - 3)
        angles[:, :use_dim] = obs_array[:, 3:3 + use_dim]
    return angles


def build_error9_from_obs(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    prev_angles_deg: Optional[np.ndarray],
    dt: float,
    default_num_envs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造9维误差：[dx, dy, dz, 0, 0, 0, 0, 0, 0]

    当前点位悬停任务不使用姿态误差或角速度误差。
    返回：(error9, current_angles_deg)
    """
    error6 = extract_pose_error(obs, default_num_envs=default_num_envs)
    current_angles_deg = extract_current_angles_deg(obs, default_num_envs=default_num_envs)

    obs_array = _normalize_obs_array(obs, default_num_envs)
    # 优先使用 Unity 提供的角速度 [17:20]，否则回退为角度差分。
    if obs_array.shape[1] >= 20:
        global_angular_vel_rad = obs_array[:, 17:20]
        current_quat_xyzw = obs_array[:, 10:14]

        # 将全局角速度转为机体局部角速度。
        local_angular_vel_rad = _quat_rotate_inverse(current_quat_xyzw, global_angular_vel_rad)
        local_angular_vel_deg = np.degrees(local_angular_vel_rad).astype(np.float32)

        # 机体轴映射：[roll_rate, pitch_rate, yaw_rate]，随后只保留 yaw_rate。
        rates_deg_s = np.stack([
            local_angular_vel_deg[:, 0],   # Roll  (绕 X 轴)
            local_angular_vel_deg[:, 2],   # Pitch (绕 Z 轴)
            local_angular_vel_deg[:, 1]    # Yaw   (绕 Y 轴)
        ], axis=1)
    elif prev_angles_deg is not None and prev_angles_deg.shape == current_angles_deg.shape:
        delta_angles = _wrap_to_180(current_angles_deg - prev_angles_deg)
        rates_deg_s = delta_angles / max(float(dt), 1e-6)
    else:
        rates_deg_s = np.zeros((current_angles_deg.shape[0], 3), dtype=np.float32)

    rates_deg_s = (rates_deg_s * RATE_ERROR_MASK_3D.reshape(1, -1)).astype(np.float32)
    error9 = np.concatenate([error6, rates_deg_s], axis=1).astype(np.float32)
    return error9, current_angles_deg.astype(np.float32)


def build_error4_from_obs(obs: Union[np.ndarray, Dict[str, np.ndarray]], default_num_envs: int = 1) -> np.ndarray:
    """
    构造 PIDController 当前使用的4维误差：[x_error, y_error, z_error, yaw_error]。

    当前任务只要求到目标点并悬停，yaw_error 始终为 0。
    """
    error6 = extract_pose_error(obs, default_num_envs=default_num_envs)
    return error6[:, [0, 1, 2, 5]].astype(np.float32)


def extract_raw_position_error(obs: Union[np.ndarray, Dict[str, np.ndarray]], default_num_envs: int = 1) -> np.ndarray:
    """提取坐标系转换前的位置误差，优先使用 Unity 全局坐标系 [x, y, z]。"""
    obs_array = _normalize_obs_array(obs, default_num_envs)
    batch_size, obs_dim = obs_array.shape

    if obs_dim >= 14:
        return (obs_array[:, 0:3] - obs_array[:, 7:10]).astype(np.float32)

    target_xyz = np.tile(np.array([0.0, -3.0, 0.0], dtype=np.float32), (batch_size, 1))
    current_xyz = np.zeros((batch_size, 3), dtype=np.float32)
    use_dim = min(obs_dim, 3)
    if use_dim > 0:
        current_xyz[:, :use_dim] = obs_array[:, :use_dim]
    return (target_xyz - current_xyz).astype(np.float32)


class HybridPPOPolicy(ActorCriticPolicy):
    """
    自定义PPO策略：使用 Tanh 压缩高斯分布输出归一化极点动作。

    动作会被压缩到 [-1, 1]，随后映射到正极点范围 [TAU_MIN, TAU_MAX]。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换为 Tanh 压缩高斯分布（在父类初始化后获取 action_dim）
        self.action_dist = SquashedDiagGaussianDistribution(int(self.action_space.shape[0]))

    def forward(self, obs, deterministic: bool = False):
        """
        前向传播，输出tanh压缩后的动作

        Args:
            obs: 观测
            deterministic: 是否确定性输出（使用mode）

        Returns:
            action: tanh压缩后的动作（-1, 1）
            values: 价值估计
            log_probs: 对数概率（已修正）
        """
        obs = _sanitize_torch_obs(obs)
        features = self.extract_features(obs)
        features = _sanitize_torch_tensor(features, clamp_abs=TRAIN_TENSOR_CLAMP)
        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi = _sanitize_torch_tensor(latent_pi, clamp_abs=TRAIN_TENSOR_CLAMP)
        latent_vf = _sanitize_torch_tensor(latent_vf, clamp_abs=TRAIN_TENSOR_CLAMP)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        actions = _sanitize_torch_tensor(actions, clamp_abs=1.0)
        values = self.value_net(latent_vf)
        values = _sanitize_torch_tensor(values, clamp_abs=TRAIN_TENSOR_CLAMP)
        log_probs = distribution.log_prob(actions)
        log_probs = _sanitize_torch_tensor(log_probs, clamp_abs=TRAIN_TENSOR_CLAMP)

        return actions, values, log_probs

    def _get_action_dist_from_latent(self, latent_pi):
        """
        从潜在向量获取action分布（tanh压缩高斯）
        """
        mean_actions = self.action_net(latent_pi)
        mean_actions = _sanitize_torch_tensor(mean_actions, clamp_abs=ACTION_MEAN_CLAMP)
        log_std = self.log_std.expand_as(mean_actions)
        log_std = _sanitize_torch_tensor(log_std).clamp(LOG_STD_MIN, LOG_STD_MAX)
        self.action_dist.proba_distribution(mean_actions, log_std)
        return self.action_dist

    def evaluate_actions(self, obs, actions, deterministic: bool = False):
        """
        评估动作的log_prob（用于训练）
        """
        obs = _sanitize_torch_obs(obs)
        actions = _sanitize_torch_tensor(actions, clamp_abs=1.0)
        features = self.extract_features(obs)
        features = _sanitize_torch_tensor(features, clamp_abs=TRAIN_TENSOR_CLAMP)
        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi = _sanitize_torch_tensor(latent_pi, clamp_abs=TRAIN_TENSOR_CLAMP)
        latent_vf = _sanitize_torch_tensor(latent_vf, clamp_abs=TRAIN_TENSOR_CLAMP)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_probs = distribution.log_prob(actions)
        log_probs = _sanitize_torch_tensor(log_probs, clamp_abs=TRAIN_TENSOR_CLAMP)
        values = self.value_net(latent_vf)
        values = _sanitize_torch_tensor(values, clamp_abs=TRAIN_TENSOR_CLAMP)
        entropy = distribution.entropy()
        if entropy is not None:
            entropy = _sanitize_torch_tensor(entropy, clamp_abs=TRAIN_TENSOR_CLAMP)
        return values, log_probs, entropy


class HybridPPOWithPIDv3(PPORewrite):
    """
    第三代混合控制器：通过复写核心方法实现 PPO极点 → PID → 8维推进器推力
    
    关键改进：
    1. 复写 _setup_model() 强制 action_net 输出 9 维极点参数
    2. 复写 forward() 方法绕过 reshape 逻辑
    3. 复写 collect_rollouts() 在数据收集时处理维度转换
    4. 复写 learn() 方法确保梯度流向正确
    
    PPO 学习 depth/surge/sway 三组 PID 对应的 [τ1, τ2, τ3]，再由极点配置公式生成PID参数。
    """
    
    def __init__(self,
                 policy,
                 env: VecEnv,
                 learning_rate: Union[float, Callable] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Callable] = 0.2,
                 clip_range_vf: Optional[Union[float, Callable]] = None,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[torch.device, str] = "auto",
                 _init_setup_model: bool = True,
                 action_dim: Optional[int] = None,
                 stage1_ratio: float = 0.3,
                 stage2_ratio: float = 0.3,
                 freeze_pid_updates: bool = False,
                 random_initial_tau: bool = False,
                 random_initial_tau_log_std: float = -0.5,
                 random_initial_tau_seed: Optional[int] = None,
                 random_initial_tau_reset_weights: bool = True):
        """
        初始化混合控制器。

        算法内部完成 PPO极点 → PID参数 → 8维推进器推力 的转换，
        直接对接原始环境的推进器动作空间。

        Args:
            env: 原始环境
            policy: 策略类型（通常为 'MlpPolicy'）
            （其他参数同标准PPO）
        """
        self.action_dim = action_dim
        self.stage1_ratio = stage1_ratio
        self.stage2_ratio = stage2_ratio
        self.total_timesteps_plan: Optional[int] = None
        self.random_initial_tau = bool(random_initial_tau)
        self.random_initial_tau_log_std = float(random_initial_tau_log_std)
        self.random_initial_tau_seed = random_initial_tau_seed
        self.random_initial_tau_reset_weights = bool(random_initial_tau_reset_weights)

        # 训练阶段只控制哪些极点参数接收梯度。
        self.freeze_pid_updates = freeze_pid_updates
        if self.freeze_pid_updates:
            self.current_stage = "frozen"
        elif stage1_ratio == 0.0 and stage2_ratio == 0.0:
            self.current_stage = "all"
        else:
            self.current_stage = "depth_only"
        self._trainable_pid_mask = torch.from_numpy(get_stage_mask(self.current_stage)).float()
        
        # 获取实际环境action维度
        actual_action_dim = self.action_dim or (
            env.action_space.shape[0] 
            if env and hasattr(env, 'action_space') and env.action_space and hasattr(env.action_space, 'shape') and env.action_space.shape
            else 8
        )
        self.actual_action_dim = actual_action_dim
        
        # 初始化PID控制器（训练/推理共用）
        pid_controller = PIDController(
            action_dim=actual_action_dim,
            device=str(device),
            yaw_deadband_deg=0.5,
            yaw_output_limit=40.0,
            surge_output_limit=120.0,
            sway_output_limit=120.0,
        )
        self.pid_controller = pid_controller
        if policy_kwargs is None:
            policy_kwargs = {}
        else:
            policy_kwargs = dict(policy_kwargs)
        
        if 'net_arch' not in policy_kwargs:
            policy_kwargs['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
        
        # 环境保持原生推进器动作空间；policy_action_dim 指定策略输出9维极点参数。
        policy_for_sb3 = HybridPPOPolicy if policy == "MlpPolicy" else policy
        super().__init__(
            policy=policy_for_sb3,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            policy_action_dim=PID_PARAM_DIM,
            action_postprocess_fn=self._policy_actions_to_env_actions,
        )
        
        self.train_step_count = 0
        self._setup_model()

    def _initialize_random_tau_action_head(self, action_net: nn.Linear) -> None:
        """让策略初始均值对应一组随机 τ，而不是固定在 τ 范围中点。"""
        generator = torch.Generator(device=self.device)
        if self.random_initial_tau_seed is not None:
            generator.manual_seed(int(self.random_initial_tau_seed))
        else:
            generator.seed()

        tau_min = max(float(TAU_MIN), TAU_EPS)
        tau_max = max(float(TAU_MAX), tau_min + TAU_EPS)
        log_tau_min = np.log(tau_min)
        log_tau_max = np.log(tau_max)

        random_ratio = torch.rand(PID_PARAM_DIM, generator=generator, device=self.device)
        random_log_tau = log_tau_min + random_ratio * (log_tau_max - log_tau_min)
        random_tau = torch.exp(random_log_tau)
        random_action = random_ratio * 2.0 - 1.0

        with torch.no_grad():
            if self.random_initial_tau_reset_weights:
                action_net.weight.zero_()
            if action_net.bias is not None:
                action_net.bias.copy_(random_action)

            if hasattr(self.policy, "log_std"):
                log_std = torch.full(
                    (PID_PARAM_DIM,),
                    self.random_initial_tau_log_std,
                    device=self.device,
                    dtype=self.policy.log_std.dtype,
                ).clamp(LOG_STD_MIN, LOG_STD_MAX)
                if tuple(self.policy.log_std.shape) == (PID_PARAM_DIM,):
                    self.policy.log_std.data.copy_(log_std)
                else:
                    self.policy.log_std = nn.Parameter(log_std)

        tau_np = random_tau.detach().cpu().numpy()
        print(
            "[v3 random-tau] initialized policy action bias from random τ: "
            f"DEPTH[{tau_np[0]:.4g}, {tau_np[1]:.4g}, {tau_np[2]:.4g}] "
            f"SURGE[{tau_np[3]:.4g}, {tau_np[4]:.4g}, {tau_np[5]:.4g}] "
            f"SWAY[{tau_np[6]:.4g}, {tau_np[7]:.4g}, {tau_np[8]:.4g}], "
            f"log_std={self.random_initial_tau_log_std:.3g}",
            flush=True,
        )

    def _policy_actions_to_env_actions(
        self,
        policy_actions: np.ndarray,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """
        将策略输出的9维极点参数转换为环境需要的8维推进器推力。
        """
        actions_np = _sanitize_np_array(policy_actions, clamp_abs=1.0)
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)

        params_final, tau_params = map_policy_output_to_pid_params_and_tau(
            policy_actions=actions_np,
            stage=self.current_stage,
            base_pid_params=DEFAULT_PID_PARAMS,
        )
        params_final = _sanitize_np_array(params_final, clamp_abs=TRAIN_TENSOR_CLAMP)
        tau_params = _sanitize_np_array(tau_params, clamp_abs=TRAIN_TENSOR_CLAMP)
        params_tensor = torch.from_numpy(params_final).float().to(self.pid_controller.device)
        tau_tensor = torch.from_numpy(tau_params).float().to(self.pid_controller.device)

        # collect_rollouts 传入的是当前步最新观测。
        error4 = _sanitize_np_array(build_error4_from_obs(obs, default_num_envs=actions_np.shape[0]), clamp_abs=TRAIN_TENSOR_CLAMP)
        raw_error = _sanitize_np_array(extract_raw_position_error(obs, default_num_envs=actions_np.shape[0]), clamp_abs=TRAIN_TENSOR_CLAMP)
        error_tensor = torch.from_numpy(error4).float().to(self.pid_controller.device)
        raw_error_tensor = torch.from_numpy(raw_error).float().to(self.pid_controller.device)
        if error_tensor.ndim == 1:
            error_tensor = error_tensor.unsqueeze(0)
        if raw_error_tensor.ndim == 1:
            raw_error_tensor = raw_error_tensor.unsqueeze(0)

        actions_8d_list = []
        with torch.no_grad():
            for i in range(actions_np.shape[0]):
                error_i = error_tensor[i:i+1]
                raw_error_i = raw_error_tensor[i:i+1]
                params_i = params_tensor[i]
                self.pid_controller.latest_tau_params = tau_tensor[i].detach()
                action_8d_i = self.pid_controller(
                    error=error_i,
                    pid_params=params_i,
                    return_thrust=True,
                    raw_error=raw_error_i,
                )
                actions_8d_list.append(action_8d_i)
            actions_8d = torch.cat(actions_8d_list, dim=0)
            actions_8d = _sanitize_torch_tensor(actions_8d, clamp_abs=1.0).cpu().numpy()

        return actions_8d



    def _update_training_stage(self) -> None:
        if self.freeze_pid_updates:
            stage = "frozen"
        elif self.total_timesteps_plan is None or self.total_timesteps_plan <= 0:
            stage = "all"
        else:
            progress = float(self.num_timesteps) / float(self.total_timesteps_plan)
            if progress < self.stage1_ratio:
                stage = "depth_only"
            elif progress < (self.stage1_ratio + self.stage2_ratio):
                stage = "outer_only"
            else:
                stage = "all"

        if stage != self.current_stage:
            self.current_stage = stage
            self._trainable_pid_mask = torch.from_numpy(get_stage_mask(stage)).float().to(self.device)
            print(f"[v3] Training stage switched to: {stage}")

    def _setup_model(self) -> None:
        """
        设置模型，将 action_net 输出维度改为 9（极点参数维度）
        """
        super()._setup_model()

        # 极点参数维度；转换后的PID保持 depth/surge/sway 三组布局。
        pid_action_dim = PID_PARAM_DIM

        # 获取当前的 action_net
        action_net = getattr(self.policy, "action_net", None)
        if not isinstance(action_net, nn.Linear):
            print("[v3] action_net is not a Linear layer, skipping resize")
            return

        old_action_dim = int(action_net.out_features)
        if old_action_dim == pid_action_dim:
            print(f"[v3] action_net already has {pid_action_dim} dims")
            if self.random_initial_tau:
                self._initialize_random_tau_action_head(action_net)
            return

        old_weight = action_net.weight.data.clone()
        old_bias = action_net.bias.data.clone() if action_net.bias is not None else None

        # 替换为新的 Linear 层（9维输出）
        action_net.out_features = pid_action_dim
        action_net.weight = nn.Parameter(torch.zeros(pid_action_dim, old_weight.shape[1]))
        action_net.bias = nn.Parameter(torch.zeros(pid_action_dim)) if old_bias is not None else None

        # 复制前 min(old, new) 个输出维度的权重
        copy_dim = min(old_action_dim, pid_action_dim)
        action_net.weight.data[:copy_dim] = old_weight[:copy_dim]
        if action_net.bias is not None and old_bias is not None:
            action_net.bias.data[:copy_dim] = old_bias[:copy_dim]

        print(f"[v3] action_net resized: {old_action_dim} -> {pid_action_dim}")

        if self.random_initial_tau:
            self._initialize_random_tau_action_head(action_net)
        else:
            # map_policy_output_to_pid_params 中 action=0 对应 τ 对数范围中点。
            init_action_for_default = 0.0
            nn.init.constant_(action_net.bias, init_action_for_default)
            print("[v3] action_net bias initialized to 0.0 (mid-range τ start)")

            # log_std 维度必须与动作维度一致。
            old_log_std = self.policy.log_std.detach().clone()
            if old_action_dim > 0:
                last_log_std_val = old_log_std[-1] if old_action_dim > 0 else torch.tensor(0.0)
            else:
                last_log_std_val = torch.tensor(0.0)
            
            new_log_std = torch.full(
                (pid_action_dim,),
                last_log_std_val.item(),
                device=self.device,
                dtype=old_log_std.dtype
            )
            new_log_std[:copy_dim] = old_log_std[:copy_dim]
            self.policy.log_std = nn.Parameter(new_log_std)
            print(f"[v3] log_std resized to: {tuple(self.policy.log_std.shape)}")

            # 初始化 log_std 为很小的值，确保初始输出接近 bias（action_net 输出）
            nn.init.constant_(self.policy.log_std, -3.0)  # log_std ≈ 0.05，确保初始动作确定性
            print("[v3] log_std initialized to -3.0 (small variance for stable start)")

        if pid_action_dim >= PID_PARAM_DIM:
            self._trainable_pid_mask = self._trainable_pid_mask.to(self.device)
            if isinstance(action_net, nn.Linear):
                def freeze_weight_hook(grad):
                    if grad is not None:
                        mask = self._trainable_pid_mask[:grad.shape[0]].view(-1, 1)
                        grad = grad * mask
                    return grad

                weight_param: torch.Tensor = action_net.weight
                weight_param.register_hook(freeze_weight_hook)

                def freeze_bias_hook(grad):
                    if grad is not None:
                        mask = self._trainable_pid_mask[:grad.shape[0]]
                        grad = grad * mask
                    return grad

                bias_param: Optional[torch.Tensor] = action_net.bias
                if bias_param is not None:
                    bias_param.register_hook(freeze_bias_hook)

            def freeze_log_std_hook(grad):
                if grad is not None:
                    mask = self._trainable_pid_mask[:grad.shape[0]]
                    grad = grad * mask
                return grad

            self.policy.log_std.register_hook(freeze_log_std_hook)
            print("[v3] staged gradient hooks enabled")
        else:
            print("[v3] action dim is small; no staged hook enabled")
        
        # 移动PID控制器到正确的设备
        if hasattr(self, 'pid_controller'):
            self.pid_controller = self.pid_controller.to(self.device)

    def _build_predict_error4(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        """构造 predict 路径使用的4维误差。"""
        return build_error4_from_obs(observation, default_num_envs=1)

    def _reset_pid_controller(self, env_mask: Optional[np.ndarray] = None) -> None:
        """重置 PID 内部状态，避免 episode 之间的积分/微分状态串扰。"""
        if not hasattr(self, "pid_controller"):
            return
        self.pid_controller.reset(env_mask)

    def _on_rollout_env_step(
        self,
        new_obs: Any,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        del new_obs, rewards, infos
        if np.any(dones):
            self._reset_pid_controller(dones)

    def predict(self,
                observation: Union[np.ndarray, Dict[str, np.ndarray]],
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        使用模型进行推理
        
        从PPO极点参数 → 极点配置PID → 8维推进器推力
        
        Args:
            observation: 观测
            state: RNN状态
            episode_start: episode开始标记
            deterministic: 是否确定性
        
        Returns:
            action: 8维推进器推力
            state: RNN状态
        """
        try:
            if episode_start is not None and np.any(episode_start):
                self._reset_pid_controller(np.asarray(episode_start, dtype=bool))
            self.policy.set_training_mode(False)

            def _get_obs_from_dict(obs_dict: Dict[str, Any]) -> Any:
                if "state" in obs_dict:
                    return obs_dict["state"]
                if "observation" in obs_dict:
                    return obs_dict["observation"]
                if "obs" in obs_dict:
                    return obs_dict["obs"]
                return next(iter(obs_dict.values()))
            
            # 处理观测
            if isinstance(observation, dict):
                obs_ref = _get_obs_from_dict(observation)
                obs_is_batched = np.asarray(obs_ref).ndim > 1
            else:
                obs_is_batched = observation.ndim > 1
            
            with torch.no_grad():
                # 转为tensor
                if isinstance(observation, dict):
                    obs_tensor = {k: torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v
                                 for k, v in observation.items()}
                else:
                    obs_array = np.asarray(observation)
                    obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
                    if obs_tensor.ndim == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                obs_tensor = _sanitize_torch_obs(obs_tensor)
                
                # 获取9维PPO极点参数（三组，每组三个τ）
                features = self.policy.features_extractor(obs_tensor)
                features = _sanitize_torch_tensor(features, clamp_abs=TRAIN_TENSOR_CLAMP)
                latent_pi, _ = self.policy.mlp_extractor(features)
                latent_pi = _sanitize_torch_tensor(latent_pi, clamp_abs=TRAIN_TENSOR_CLAMP)
                ppo_output_raw = self.policy.action_net(latent_pi)
                ppo_output_raw = _sanitize_torch_tensor(ppo_output_raw, clamp_abs=ACTION_MEAN_CLAMP)
                
                # PPO 输出解释为归一化极点，经正值映射和极点配置得到 PID 参数。
                pid_params_np = map_policy_output_to_pid_params(
                    policy_actions=ppo_output_raw.detach().cpu().numpy(),
                    stage=self.current_stage,
                    base_pid_params=DEFAULT_PID_PARAMS,
                )
                pid_params_np = _sanitize_np_array(pid_params_np, clamp_abs=TRAIN_TENSOR_CLAMP)
                pid_params_tensor = torch.from_numpy(pid_params_np).to(self.device)

                error4 = _sanitize_np_array(self._build_predict_error4(observation), clamp_abs=TRAIN_TENSOR_CLAMP)
                raw_error = _sanitize_np_array(extract_raw_position_error(observation, default_num_envs=pid_params_tensor.shape[0]), clamp_abs=TRAIN_TENSOR_CLAMP)
                error_tensor = torch.from_numpy(error4).float().to(self.device)
                raw_error_tensor = torch.from_numpy(raw_error).float().to(self.device)
                if error_tensor.ndim == 1:
                    error_tensor = error_tensor.unsqueeze(0)
                if raw_error_tensor.ndim == 1:
                    raw_error_tensor = raw_error_tensor.unsqueeze(0)

                if error_tensor.shape[-1] < 4:
                    pad_size = 4 - error_tensor.shape[-1]
                    error_tensor = torch.nn.functional.pad(error_tensor, (0, pad_size))

                batch_size = pid_params_tensor.shape[0]
                action_8d_list = []
                for i in range(batch_size):
                    error_i = error_tensor[i:i+1]
                    raw_error_i = raw_error_tensor[i:i+1]
                    params_i = pid_params_tensor[i]
                    action_8d_i = self.pid_controller(
                        error=error_i,
                        pid_params=params_i,
                        return_thrust=True,
                        raw_error=raw_error_i,
                    )
                    action_8d_list.append(action_8d_i)

                action_tensor = torch.cat(action_8d_list, dim=0)
                action_tensor = _sanitize_torch_tensor(action_tensor, clamp_abs=1.0)
            
            # 转回numpy
            action_np = _sanitize_np_array(action_tensor.cpu().numpy(), clamp_abs=1.0)
            
            # 处理维度
            if not obs_is_batched and action_np.ndim == 2 and action_np.shape[0] == 1:
                action_np = action_np[0]
            
            expected_dim = self.actual_action_dim
            if action_np.ndim == 1:
                if len(action_np) < expected_dim:
                    action_np = np.pad(action_np, (0, expected_dim - len(action_np)), constant_values=0.0)
                elif len(action_np) > expected_dim:
                    action_np = action_np[:expected_dim]
            else:
                if action_np.shape[1] < expected_dim:
                    action_np = np.pad(action_np, ((0, 0), (0, expected_dim - action_np.shape[1])), constant_values=0.0)
                elif action_np.shape[1] > expected_dim:
                    action_np = action_np[:, :expected_dim]
            
            return action_np, state
        
        except Exception as e:
            print(f"ERROR in predict: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回安全的默认动作
            if isinstance(observation, dict):
                if "state" in observation:
                    obs_ref = observation["state"]
                elif "observation" in observation:
                    obs_ref = observation["observation"]
                elif "obs" in observation:
                    obs_ref = observation["obs"]
                else:
                    obs_ref = next(iter(observation.values()))
                obs_ref_np = np.asarray(obs_ref)
                batch_size = obs_ref_np.shape[0] if obs_ref_np.ndim > 1 else 1
            else:
                batch_size = observation.shape[0] if observation.ndim > 1 else 1
            
            expected_dim = self.actual_action_dim
            return np.zeros((batch_size, expected_dim) if batch_size > 1 else (expected_dim,), dtype=np.float32), state

    def _repair_nonfinite_policy_state(self, context: str) -> None:
        """训练前后修复非有限参数/优化器状态，避免一次坏 minibatch 继续污染后续训练。"""
        repaired_params = 0
        with torch.no_grad():
            for name, param in self.policy.named_parameters():
                if param is None:
                    continue
                finite_mask = torch.isfinite(param.data)
                if not bool(finite_mask.all().item()):
                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
                    repaired_params += int((~finite_mask).sum().item())

            if hasattr(self.policy, "log_std"):
                self.policy.log_std.data = torch.nan_to_num(
                    self.policy.log_std.data,
                    nan=-3.0,
                    posinf=LOG_STD_MAX,
                    neginf=LOG_STD_MIN,
                ).clamp(LOG_STD_MIN, LOG_STD_MAX)

        repaired_state = 0
        optimizer = getattr(self.policy, "optimizer", None)
        if optimizer is not None:
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        finite_mask = torch.isfinite(value)
                        if not bool(finite_mask.all().item()):
                            state[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                            repaired_state += int((~finite_mask).sum().item())

        if repaired_params > 0 or repaired_state > 0:
            print(
                f"[v3 NaNGuard] repaired non-finite optimizer state at {context}: "
                f"params={repaired_params}, optimizer_state={repaired_state}",
                flush=True,
            )
    
    def train(self) -> None:
        """训练模式"""
        self._update_training_stage()
        self._repair_nonfinite_policy_state("before_train")
        super().train()
        self._repair_nonfinite_policy_state("after_train")
        self.train_step_count += 1
    
    def learn(self,
              total_timesteps: int,
              callback=None,
              log_interval: int = 1,
              tb_log_name: str = "HybridPPOWithPIDv3",
              reset_num_timesteps: bool = True,
              progress_bar: bool = False) -> "HybridPPOWithPIDv3":
        """训练"""
        self.total_timesteps_plan = total_timesteps
        self._update_training_stage()
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def save(self, path: Union[str, Path, BufferedIOBase], 
             exclude: Optional[Iterable[str]] = None, 
             include: Optional[Iterable[str]] = None) -> None:
        """保存模型"""
        exclude_set = set(exclude or [])
        # action_postprocess_fn 是绑定方法，序列化时会捕获进程对象导致 AuthenticationString 错误
        exclude_set.add("action_postprocess_fn")
        super().save(path, exclude=list(exclude_set), include=include)
    
    @classmethod
    def load(cls, path: Union[str, Path, BufferedIOBase], env=None, 
             device: Union[torch.device, str] = "auto",
             custom_objects: Optional[Dict[str, Any]] = None,
             print_system_info: bool = False,
             force_reset: bool = True,
             **kwargs):
        """加载模型"""
        model = super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs
        )
        # 反序列化后恢复动作后处理函数，保持 9D 参数 -> 8D 推力 的训练/推理链路
        if hasattr(model, "set_action_postprocess_fn"):
            model.set_action_postprocess_fn(model._policy_actions_to_env_actions)
        return model
    
    def reset_pid(self, env_mask: Optional[np.ndarray] = None) -> None:
        """重置 PID 状态；可选只清空部分并行环境槽位。"""
        self._reset_pid_controller(env_mask)
