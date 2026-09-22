"""
三组单环PID控制器（depth + surge + sway）
PID 参数共 9 维，每组三个参数 [Kp, Ki, Kd]。
控制器输出6维控制分量 [Fx, Fy, Fz, Mx, My, Mz]，再由推力分配器转换为8维推进器推力。

输出：
1. 6维控制分量 [Fx, Fy, Fz, Mx, My, Mz]
2. 8维推进器推力（经推力分配器）
"""
from typing import Optional, cast

import math
import torch
import torch.nn as nn


DERIVATIVE_FILTER_R = math.exp(-4.0)


class PIDController(nn.Module):
    """单环PID控制器，支持3组PID参数；yaw 输出固定为0。"""

    NUM_GROUPS = 3
    PARAMS_PER_GROUP = 3
    PARAM_DIM = NUM_GROUPS * PARAMS_PER_GROUP

    # 参数顺序（基于4维误差 [x, y, z, yaw]；yaw 不参与PID）：
    # 0 depth, 1 surge, 2 sway
    DEPTH = 0    # y 方向深度/高度误差
    SURGE = 1    # x 方向误差
    SWAY = 2     # z 方向误差

    @staticmethod
    def _resolve_device(device: str) -> str:
        """将auto等别名解析为PyTorch可识别的设备字符串。"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def __init__(
        self,
        action_dim: int = 8,
        device: str = "cpu",
        dt: float = 0.02,
        integral_decay: float = 0.95,
        integral_state_limit: float = 100.0,
        derivative_filter_alpha: float = DERIVATIVE_FILTER_R,
        output_slew_rate_limit: float = 0.0,
        output_range: tuple[float, float] = (-250.0, 250.0),
        yaw_deadband_deg: float = 5.0,
        yaw_output_limit: float = 100.0,
        yaw_integral_state_limit: float = 60.0,
        surge_output_limit: float = 30.0,
        sway_output_limit: float = 30.0,
        attitude_coupling: Optional[torch.Tensor] = None,
        use_thrust_allocator: bool = True,
        allocator_control_linear_range: float = 0.1,
        allocator_control_axis_ranges: Optional[tuple[float, float, float, float, float, float]] = None,
        allocator_allocation_mode: str = "empirical_thruster_mixer",
        allocator_physical_wrench_limits: Optional[tuple[float, float, float, float, float, float]] = None,
        allocator_thruster_force_limit_positive: Optional[tuple[float, float, float, float, float, float, float, float]] = None,
        allocator_thruster_force_limit_negative: Optional[tuple[float, float, float, float, float, float, float, float]] = None,
        debug_print_interval: int = 500,
    ) -> None:
        super().__init__()
        resolved_device = self._resolve_device(device)
        self.action_dim = action_dim
        self.device = resolved_device
        self.dt = float(max(dt, 1e-6))
        self.integral_decay = float(min(max(integral_decay, 0.0), 1.0))
        self.integral_state_limit = float(max(integral_state_limit, 0.0))
        self.derivative_filter_alpha = float(min(max(derivative_filter_alpha, 0.0), 1.0))
        self.output_slew_rate_limit = float(max(output_slew_rate_limit, 0.0))
        self.output_range = output_range
        # yaw 相关入参保留为兼容旧配置；当前点位悬停任务不控制 yaw。
        self.yaw_deadband_deg = float(max(yaw_deadband_deg, 0.0))
        self.yaw_output_limit = float(max(yaw_output_limit, 0.0))
        self.yaw_integral_state_limit = float(max(yaw_integral_state_limit, 0.0))
        self.surge_output_limit = float(max(surge_output_limit, 0.0))
        self.sway_output_limit = float(max(sway_output_limit, 0.0))
        self.use_thrust_allocator = use_thrust_allocator
        self.allocator_control_linear_range = float(max(allocator_control_linear_range, 1e-6))
        self.allocator_control_axis_ranges = allocator_control_axis_ranges
        self.allocator_allocation_mode = str(allocator_allocation_mode)
        self.allocator_physical_wrench_limits = allocator_physical_wrench_limits
        self.allocator_thruster_force_limit_positive = allocator_thruster_force_limit_positive
        self.allocator_thruster_force_limit_negative = allocator_thruster_force_limit_negative
        self.debug_print_interval = max(int(debug_print_interval), 0)
        self.latest_tau_params: Optional[torch.Tensor] = None

        # 兼容旧配置的三轴耦合矩阵参数；当前4DoF单环路径不使用显式姿态耦合。
        if attitude_coupling is None:
            coupling = torch.eye(3, dtype=torch.float32, device=resolved_device)
        else:
            coupling = attitude_coupling.to(device=resolved_device, dtype=torch.float32)

        if coupling.shape != (3, 3):
            raise ValueError(f"Expected attitude_coupling shape (3,3), got {tuple(coupling.shape)}")

        self.register_buffer("attitude_coupling", coupling)

        # 每个环路、每个并行环境一份积分、前一时刻误差、导数低通滤波状态
        # shape: (NUM_GROUPS, batch_size)，batch_size 在首次 forward 时确定
        self.register_buffer("integral_state", torch.zeros(self.NUM_GROUPS, 0, device=resolved_device))
        self.register_buffer("prev_error_state", torch.zeros(self.NUM_GROUPS, 0, device=resolved_device))
        self.register_buffer("derivative_filter_state", torch.zeros(self.NUM_GROUPS, 0, device=resolved_device))
        self.register_buffer("prev_control_6d", torch.zeros(0, 6, device=resolved_device))

        if self.use_thrust_allocator:
            from finssim_rl.models.thrust_allocator import ThrustAllocator

            self.thrust_allocator = ThrustAllocator(
                device=resolved_device,
                deadzone_comp=0.0,
                control_linear_range=self.allocator_control_linear_range,
                control_axis_ranges=self.allocator_control_axis_ranges,
                allocation_mode=self.allocator_allocation_mode,
                physical_wrench_limits=self.allocator_physical_wrench_limits,
                thruster_force_limit_positive=self.allocator_thruster_force_limit_positive,
                thruster_force_limit_negative=self.allocator_thruster_force_limit_negative,
                debug_print_interval=0,
            )
        
        self._debug_step_count = 0

    @staticmethod
    def _reshape_pid_params(pid_params: torch.Tensor) -> torch.Tensor:
        """将输入参数整理为 (batch, 3, 3)。"""
        if pid_params.dim() == 1:
            pid_params = pid_params.unsqueeze(0)

        if pid_params.shape[-1] != PIDController.PARAM_DIM:
            raise ValueError(
                f"Expected pid_params last dim = {PIDController.PARAM_DIM}, got {pid_params.shape[-1]}"
            )
        return pid_params.view(pid_params.shape[0], PIDController.NUM_GROUPS, PIDController.PARAMS_PER_GROUP)

    @staticmethod
    def _normalize_env_mask(
        env_mask: object,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """将局部 reset 掩码统一整理为 shape=(batch_size,) 的 bool Tensor。"""
        if batch_size <= 0:
            return None

        mask = torch.as_tensor(env_mask, dtype=torch.bool, device=device).flatten()
        if mask.numel() == 0:
            return None
        if mask.numel() == 1 and batch_size > 1:
            mask = mask.repeat(batch_size)
        elif mask.numel() != batch_size:
            raise ValueError(f"Expected env_mask length {batch_size}, got {mask.numel()}")
        return mask

    def _axis_output_limit(self, group_idx: int) -> float:
        output_abs_limit = max(abs(float(self.output_range[0])), abs(float(self.output_range[1])))
        if group_idx == self.SURGE and self.surge_output_limit > 0.0:
            return min(output_abs_limit, self.surge_output_limit)
        if group_idx == self.SWAY and self.sway_output_limit > 0.0:
            return min(output_abs_limit, self.sway_output_limit)
        return output_abs_limit

    def _pid_step(
        self,
        group_idx: int,
        error: torch.Tensor,
        gains: torch.Tensor,
    ) -> torch.Tensor:
        """单个PID环路计算，error shape=(batch,), gains shape=(batch,3)。"""
        kp = gains[:, 0]
        ki = gains[:, 1]
        kd = gains[:, 2]
        integral_state = cast(torch.Tensor, getattr(self, "integral_state"))
        prev_error_state = cast(torch.Tensor, getattr(self, "prev_error_state"))
        derivative_filter_state = cast(torch.Tensor, getattr(self, "derivative_filter_state"))

        batch_size = error.shape[0]
        if (
            integral_state.shape[1] != batch_size
            or integral_state.device != error.device
            or prev_error_state.shape[1] != batch_size
            or prev_error_state.device != error.device
            or derivative_filter_state.shape[1] != batch_size
            or derivative_filter_state.device != error.device
        ):
            # 当并行环境数变化时重建状态，避免跨环境串扰
            self.integral_state = torch.zeros(self.NUM_GROUPS, batch_size, device=error.device, dtype=error.dtype)
            self.prev_error_state = torch.zeros(self.NUM_GROUPS, batch_size, device=error.device, dtype=error.dtype)
            self.derivative_filter_state = torch.zeros(self.NUM_GROUPS, batch_size, device=error.device, dtype=error.dtype)
            integral_state = cast(torch.Tensor, getattr(self, "integral_state"))
            prev_error_state = cast(torch.Tensor, getattr(self, "prev_error_state"))
            derivative_filter_state = cast(torch.Tensor, getattr(self, "derivative_filter_state"))

        # 每个环境独立更新积分和微分低通滤波。
        decayed_integral = integral_state[group_idx, :] * self.integral_decay
        candidate_integral = decayed_integral + error * self.dt
        if self.integral_state_limit > 0.0:
            decayed_integral = torch.clamp(
                decayed_integral,
                -self.integral_state_limit,
                self.integral_state_limit,
            )
            candidate_integral = torch.clamp(
                candidate_integral,
                -self.integral_state_limit,
                self.integral_state_limit,
            )

        raw_derivative = (error - prev_error_state[group_idx, :]) / self.dt
        alpha = torch.as_tensor(self.derivative_filter_alpha, device=error.device, dtype=error.dtype)
        filtered_derivative = (1.0 - alpha) * derivative_filter_state[group_idx, :] + alpha * raw_derivative
        derivative_filter_state[group_idx, :] = filtered_derivative
        prev_error_state[group_idx, :] = error

        p_term = kp * error
        integral_limit = torch.as_tensor(
            max(abs(float(self.output_range[0])), abs(float(self.output_range[1]))),
            device=error.device,
            dtype=error.dtype,
        )
        d_term = kd * filtered_derivative
        candidate_i_term = torch.clamp(ki * candidate_integral, -integral_limit, integral_limit)
        candidate_output = p_term + candidate_i_term + d_term

        axis_output_limit = self._axis_output_limit(group_idx)
        if axis_output_limit > 0.0:
            axis_limit = torch.as_tensor(axis_output_limit, device=error.device, dtype=error.dtype)
            integral_push = ki * error
            blocked = (
                ((candidate_output > axis_limit) & (integral_push > 0.0))
                | ((candidate_output < -axis_limit) & (integral_push < 0.0))
            )
            integral_state[group_idx, :] = torch.where(blocked, decayed_integral, candidate_integral)
        else:
            integral_state[group_idx, :] = candidate_integral

        i_term = ki * integral_state[group_idx, :]
        i_term = torch.clamp(i_term, -integral_limit, integral_limit)
        return p_term + i_term + d_term

    def _apply_control_slew_rate_limit(self, control_6d: torch.Tensor) -> torch.Tensor:
        if self.output_slew_rate_limit <= 0.0:
            return control_6d

        prev_control_6d = cast(torch.Tensor, getattr(self, "prev_control_6d"))
        if (
            prev_control_6d.shape != control_6d.shape
            or prev_control_6d.device != control_6d.device
            or prev_control_6d.dtype != control_6d.dtype
        ):
            prev_control_6d = torch.zeros_like(control_6d)
            self.prev_control_6d = prev_control_6d

        max_delta = self.output_slew_rate_limit * self.dt
        limited = torch.clamp(control_6d, prev_control_6d - max_delta, prev_control_6d + max_delta)
        self.prev_control_6d = limited.detach().clone()
        return limited

    def forward(
        self,
        error: torch.Tensor,
        pid_params: torch.Tensor,
        return_thrust: bool = True,
        raw_error: Optional[torch.Tensor] = None,
        apply_slew_rate_limit: bool = True,
    ) -> torch.Tensor:
        """
        参数：
        - error: shape (batch, >=4) 或 (>=4,)
          约定前4维：[x_error, y_error, z_error, yaw_error]
          其中 yaw_error 定义为潜器朝向与目标点之间的夹角
        - pid_params: shape (batch, 9) 或 (9,)
        """
        if error.dim() == 1:
            error = error.unsqueeze(0)

        if error.shape[-1] < 4:
            error = torch.nn.functional.pad(error, (0, 4 - error.shape[-1]), value=0.0)

        if raw_error is not None:
            if raw_error.dim() == 1:
                raw_error = raw_error.unsqueeze(0)
            if raw_error.shape[-1] < 3:
                raw_error = torch.nn.functional.pad(raw_error, (0, 3 - raw_error.shape[-1]), value=0.0)

        pid = self._reshape_pid_params(pid_params)
        
        self._debug_step_count += 1

        # 误差输入定义（4维）
        x_error = error[:, 0]
        y_error = error[:, 1]
        z_error = error[:, 2]
        # 3组PID控制：y深度/高度、x前进、z侧向。yaw 不参与控制。
        u_depth = self._pid_step(self.DEPTH, y_error, pid[:, self.DEPTH, :])
        u_surge = self._pid_step(self.SURGE, x_error, pid[:, self.SURGE, :])
        u_sway = self._pid_step(self.SWAY, z_error, pid[:, self.SWAY, :])
        u_yaw = torch.zeros_like(u_depth)

        # 输出限幅
        if self.surge_output_limit > 0.0:
            u_surge = torch.clamp(u_surge, -self.surge_output_limit, self.surge_output_limit)
        if self.sway_output_limit > 0.0:
            u_sway = torch.clamp(u_sway, -self.sway_output_limit, self.sway_output_limit)
        # 潜器坐标系为 x前、y上、z左，因此6维控制分量为 [Fx, Fy, Fz, Mx, My, Mz]。
        # 当前3DoF平移控制输出 [surge, depth, sway] -> [Fx, Fy, Fz]，My(yaw)=0。
        control_6d = torch.stack([u_surge, u_depth, u_sway, torch.zeros_like(u_depth), u_yaw, torch.zeros_like(u_depth)], dim=1)
        # 这里是控制域限幅；最终推进器推力域[-1,1]由ThrustAllocator内部完成。
        control_6d = torch.clamp(control_6d, self.output_range[0], self.output_range[1])
        if apply_slew_rate_limit:
            control_6d = self._apply_control_slew_rate_limit(control_6d)

        # 周期性输出当前实际使用的 9 个 PID 参数。
        if self.debug_print_interval > 0 and self._debug_step_count % self.debug_print_interval == 0:
            batch_idx = 0
            pid_vals = pid[batch_idx].detach().cpu().numpy()
            tau_vals = None
            if self.latest_tau_params is not None:
                tau_tensor = self.latest_tau_params.detach()
                if tau_tensor.dim() == 1:
                    tau_tensor = tau_tensor.view(self.NUM_GROUPS, self.PARAMS_PER_GROUP)
                elif tau_tensor.dim() == 2:
                    tau_tensor = tau_tensor[batch_idx].view(self.NUM_GROUPS, self.PARAMS_PER_GROUP)
                tau_vals = tau_tensor.cpu().numpy()

            tau_text = ""
            if tau_vals is not None:
                tau_text = (
                    f" TAU_DEPTH[τ1={tau_vals[0, 0]:.6g}, τ2={tau_vals[0, 1]:.6g}, τ3={tau_vals[0, 2]:.6g}] "
                    f"TAU_SURGE[τ1={tau_vals[1, 0]:.6g}, τ2={tau_vals[1, 1]:.6g}, τ3={tau_vals[1, 2]:.6g}] "
                    f"TAU_SWAY[τ1={tau_vals[2, 0]:.6g}, τ2={tau_vals[2, 1]:.6g}, τ3={tau_vals[2, 2]:.6g}]"
                )
            print(
                f"[PID Step {self._debug_step_count}] "
                f"DEPTH[Kp={pid_vals[0, 0]:.6g}, Ki={pid_vals[0, 1]:.6g}, Kd={pid_vals[0, 2]:.6g}] "
                f"SURGE[Kp={pid_vals[1, 0]:.6g}, Ki={pid_vals[1, 1]:.6g}, Kd={pid_vals[1, 2]:.6g}] "
                f"SWAY[Kp={pid_vals[2, 0]:.6g}, Ki={pid_vals[2, 1]:.6g}, Kd={pid_vals[2, 2]:.6g}]"
                f"{tau_text}"
            , flush=True)

        if return_thrust and self.use_thrust_allocator:
            thrusts = self.thrust_allocator(control_6d)
            if self._debug_step_count % 1000 == 0:
                batch_idx = 0
                thrust_vals = thrusts[batch_idx] if thrusts.dim() > 1 else thrusts
                thrust_vals_np = thrust_vals.detach().cpu().numpy()
                thrust_text = ", ".join(f"T{i + 1}={float(v):+.4f}" for i, v in enumerate(thrust_vals_np))
                print(f"[Thruster Input Step {self._debug_step_count}] {thrust_text}", flush=True)
            return thrusts
        return control_6d

    def reset(self, env_mask: Optional[object] = None) -> None:
        """重置 PID 状态；可选只重置部分并行环境槽位。"""
        integral_state = cast(torch.Tensor, getattr(self, "integral_state"))
        prev_error_state = cast(torch.Tensor, getattr(self, "prev_error_state"))
        derivative_filter_state = cast(torch.Tensor, getattr(self, "derivative_filter_state"))
        prev_control_6d = cast(torch.Tensor, getattr(self, "prev_control_6d"))

        if env_mask is None:
            integral_state.zero_()
            prev_error_state.zero_()
            derivative_filter_state.zero_()
            prev_control_6d.zero_()
            self.latest_tau_params = None
            return

        raw_mask = torch.as_tensor(env_mask, dtype=torch.bool, device=integral_state.device).flatten()
        if raw_mask.numel() > 1 and raw_mask.numel() != integral_state.shape[1]:
            batch_size = int(raw_mask.numel())
            self.integral_state = torch.zeros(
                self.NUM_GROUPS,
                batch_size,
                device=integral_state.device,
                dtype=integral_state.dtype,
            )
            self.prev_error_state = torch.zeros(
                self.NUM_GROUPS,
                batch_size,
                device=prev_error_state.device,
                dtype=prev_error_state.dtype,
            )
            self.derivative_filter_state = torch.zeros(
                self.NUM_GROUPS,
                batch_size,
                device=derivative_filter_state.device,
                dtype=derivative_filter_state.dtype,
            )
            self.prev_control_6d = torch.zeros(
                batch_size,
                6,
                device=prev_control_6d.device,
                dtype=prev_control_6d.dtype,
            )
            integral_state = cast(torch.Tensor, getattr(self, "integral_state"))
            prev_error_state = cast(torch.Tensor, getattr(self, "prev_error_state"))
            derivative_filter_state = cast(torch.Tensor, getattr(self, "derivative_filter_state"))
            prev_control_6d = cast(torch.Tensor, getattr(self, "prev_control_6d"))

        mask = self._normalize_env_mask(
            raw_mask,
            batch_size=integral_state.shape[1],
            device=integral_state.device,
        )
        if mask is None or not bool(mask.any().item()):
            return

        integral_state[:, mask] = 0.0
        prev_error_state[:, mask] = 0.0
        derivative_filter_state[:, mask] = 0.0
        if prev_control_6d.shape[0] == mask.numel():
            prev_control_6d[mask, :] = 0.0
        self.latest_tau_params = None
