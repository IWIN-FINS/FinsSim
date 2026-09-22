"""
推力分配器

将6维力/力矩向量 τ = [Fx, Fy, Fz, Mx, My, Mz] 映射到八个推进器的
控制输出，输出范围为 [-1, 1]。

``empirical_thruster_mixer`` 是历史兼容模式：经验混控表直接把无量纲控制量
混合为推进器命令。``physical_wrench_allocator`` 是物理模式：使用当前
FinsROV_Fossen 的推进器几何前向模型 ``tau = B @ force_n``，在各路推力边界内
分配以牛顿/牛顿米表示的目标 wrench。

推力器布局：
- 垂直推进器 1-4：主要用于垂向力 Fy 及滚转/俯仰力矩 Mx, Mz
- 水平推进器 5-8：主要用于纵向力 Fx、横向力 Fz 及偏航力矩 My
"""

from importlib import resources
from typing import Optional, Sequence

import torch


class ThrustAllocator(torch.nn.Module):
    """6D body wrench -> canonical 8D normalized thruster commands."""

    EMPIRICAL_THRUSTER_MIXER = "empirical_thruster_mixer"
    PHYSICAL_WRENCH_ALLOCATOR = "physical_wrench_allocator"
    AXIS_MAP = "axis_map"
    PHYSICAL_WRENCH_ORDER = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
    THRUSTER_ORDER = ("V_LF", "V_LB", "V_RB", "V_RF", "H_LF", "H_LB", "H_RB", "H_RF")
    _OFFLINE_CACHE_RESOURCE = "data/finsrov_fossen_physical_allocator_cache.pt"
    _OFFLINE_CACHE_FORMAT_VERSION = 1

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def __init__(
        self,
        device: str = "cpu",
        deadzone_comp: float = 0.0,
        control_linear_range: float = 50.0,
        control_axis_ranges: Optional[tuple[float, float, float, float, float, float]] = None,
        allocation_mode: str = EMPIRICAL_THRUSTER_MIXER,
        physical_wrench_limits: Optional[Sequence[float]] = None,
        thruster_force_limit_positive: Optional[Sequence[float]] = None,
        thruster_force_limit_negative: Optional[Sequence[float]] = None,
        debug_print_interval: int = 100,
    ) -> None:
        super().__init__()
        resolved_device = self._resolve_device(device)
        self.device = resolved_device
        self.deadzone_comp = float(deadzone_comp)
        self.control_linear_range = max(float(control_linear_range), 1e-6)
        self.allocation_mode = self._normalize_allocation_mode(allocation_mode)
        self.debug_print_interval = max(int(debug_print_interval), 0)
        if control_axis_ranges is None:
            axis_ranges = [self.control_linear_range] * 6
        else:
            if len(control_axis_ranges) != 6:
                raise ValueError(f"control_axis_ranges must contain 6 values, got {len(control_axis_ranges)}")
            axis_ranges = [max(float(value), 1e-6) for value in control_axis_ranges]
        self.register_buffer(
            "control_axis_ranges",
            torch.tensor(axis_ranges, dtype=torch.float32, device=resolved_device),
        )

        physical_limit_values = physical_wrench_limits if physical_wrench_limits is not None else axis_ranges
        if self.allocation_mode == self.PHYSICAL_WRENCH_ALLOCATOR:
            physical_limits = self._validate_positive_vector(
                physical_limit_values,
                length=6,
                name="physical_wrench_limits",
                allow_zero=True,
            )
        else:
            # The historical matrix mode permits an intentionally disabled
            # policy axis (gain/range equal to zero). These values are not
            # consulted by matrix allocation, so keep a nonzero buffer only
            # for module-state consistency.
            if len(physical_limit_values) != 6:
                raise ValueError(
                    f"physical_wrench_limits must contain 6 values, got {len(physical_limit_values)}"
                )
            physical_limits = [max(abs(float(value)), 1e-6) for value in physical_limit_values]
        positive_force_limits = self._validate_positive_vector(
            thruster_force_limit_positive if thruster_force_limit_positive is not None else (7.0,) * 8,
            length=8,
            name="thruster_force_limit_positive",
        )
        negative_force_limits = self._validate_positive_vector(
            thruster_force_limit_negative if thruster_force_limit_negative is not None else (7.0,) * 8,
            length=8,
            name="thruster_force_limit_negative",
        )
        self.register_buffer(
            "physical_wrench_limits",
            torch.tensor(physical_limits, dtype=torch.float32, device=resolved_device),
        )
        self.register_buffer(
            "thruster_force_limit_positive",
            torch.tensor(positive_force_limits, dtype=torch.float32, device=resolved_device),
        )
        self.register_buffer(
            "thruster_force_limit_negative",
            torch.tensor(negative_force_limits, dtype=torch.float32, device=resolved_device),
        )

        # 历史经验混控矩阵 A (8x6)，仅供 empirical_thruster_mixer 使用。
        # controller-body 轴为 x=forward、y=up、z=left；其力矩正负号以
        # Unity 运行时 Vector3.Cross() 后转换到 controller-body 的结果为准。
        # 这里不要用“左手/右手定则”重新推导符号。physical_wrench_allocator
        # 的 B 矩阵才是当前 FinsROV_Fossen 几何的物理前向模型。
        # 控制顺序 [Fx, Fy, Fz, Mx, My, Mz]
        # = [纵向力, 垂向力, 横向力, 滚转力矩, 偏航力矩, 俯仰力矩]
        #
        # 对称垂直推进器的 roll 符号已由 B 回代验证：
        #   +Mx -> [V_LF, V_LB, V_RB, V_RF] = [-, -, +, +].
        #
        # Unity action/engine 顺序（从上往下看，x向前，z向左）:
        #   T1/Vertical1 = 左前, T2/Vertical2 = 左后
        #   T3/Vertical3 = 右后, T4/Vertical4 = 右前
        #   T5/Horizontal1 = 左前, T6/Horizontal2 = 左后
        #   T7/Horizontal3 = 右后, T8/Horizontal4 = 右前
        #
        # 垂直推进器位置（x=前后±0.25, z=左右±0.25）:
        #   垂直1(左前): x=+0.25, z=+0.25  -> Mx=-0.25*Fy, Mz=+0.25*Fy
        #   垂直2(左后): x=-0.25, z=+0.25  -> Mx=-0.25*Fy, Mz=-0.25*Fy
        #   垂直3(右后): x=-0.25, z=-0.25  -> Mx=+0.25*Fy, Mz=-0.25*Fy
        #   垂直4(右前): x=+0.25, z=-0.25  -> Mx=+0.25*Fy, Mz=+0.25*Fy
        #
        # 1Chase1 的 controller-body 与 Unity local 轴对齐：x=+X surge,
        # y=+Y heave, z=+Z left/sway。FinsROV_Fossen 的水平推正向为：
        #   T5/Horizontal1: +x -z
        #   T6/Horizontal2: +x +z
        #   T7/Horizontal3: -x +z
        #   T8/Horizontal4: -x -z
        #
        # 因此纯前进(+Fx)应为 T5/T6 正、T7/T8 负；
        # 纯左移(+Fz)应为 T5/T8 负、T6/T7 正。
        #
        # A 是历史归一化混控增益，不是 B 的伪逆，也不能保证消除当前几何
        # 下的各轴耦合。需要严格重建目标 wrench 时应使用
        # physical_wrench_allocator。
        thrust_allocation_matrix = torch.tensor(
            [
                # T1 / Vertical1: 左前；+Mx 时为负，匹配 B[roll, V_LF] < 0。
                [ 0.000000,  0.249400,  0.000000, -1.601400,  0.000000,   1.883600],
                # T2 / Vertical2: 左后；+Mx 时为负，匹配 B[roll, V_LB] < 0。
                [ 0.000000,  0.249400,  0.000000, -1.601400,  0.000000,  -1.883600],
                # T3 / Vertical3: 右后；+Mx 时为正，匹配 B[roll, V_RB] > 0。
                [ 0.000000,  0.249400,  0.000000,  1.601400,  0.000000, -1.883600],
                # T4 / Vertical4: 右前；+Mx 时为正，匹配 B[roll, V_RF] > 0。
                [ 0.000000,  0.249400,  0.000000,  1.601400,  0.000000, 1.883600],
                # T5 / Horizontal1
                [ 0.353600,  0.000000, -0.353600,  0.000000,  0.951000,  0.000000],
                # T6 / Horizontal2
                [ 0.353600,  0.000000,  0.353600,  0.000000,  0.951000,  0.000000],
                # T7 / Horizontal3
                [-0.353600,  0.000000,  0.353600,  0.000000,  0.951000,  0.000000],
                # T8 / Horizontal4
                [-0.353600,  0.000000, -0.353600,  0.000000,  0.951000,  0.000000],
            ],
            dtype=torch.float32,
            device=resolved_device,
        )
        self.register_buffer('A', thrust_allocation_matrix)

        # Physical forward model for the active 1Chase1 FinsROV_Fossen layout,
        # calculated from the Unity Scene instance and
        # Rigidbody.centerOfMass=[0, -0.08, 0].
        #
        # A positive command on every Vertical1..4 produces +Fy (up).  This
        # was verified by FinsROVChaseStaticAudit against the actual scene;
        # do not infer it from the nested mesh transform orientations.
        # Units: B[:3] is dimensionless direction cosine; B[3:] is metres.
        # Therefore B @ force_n has units [N, N, N, N*m, N*m, N*m].
        physical_wrench_matrix = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.70710678, 0.70710678, -0.70710678, -0.70710678],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.70710678, 0.70710678, 0.70710678, -0.70710678],
                [-0.13304673, -0.13304667, 0.14295308, 0.14295302, -0.02628290, 0.02628290, 0.02628290, -0.02628290],
                [0.0, 0.0, 0.0, 0.0, 0.23037854, 0.26838960, 0.27539489, 0.23738383],
                [0.08312216, -0.13687684, -0.13687672, 0.08312222, -0.02628290, -0.02628290, 0.02628290, 0.02628290],
            ],
            dtype=torch.float32,
            device=resolved_device,
        )
        self.register_buffer("B", physical_wrench_matrix)

        if self.allocation_mode == self.PHYSICAL_WRENCH_ALLOCATOR:
            free_thruster_masks, free_mask_pseudoinverses = self._load_or_build_free_mask_cache(
                physical_wrench_matrix,
                device=resolved_device,
            )
            self.register_buffer("free_thruster_masks", free_thruster_masks)
            self.register_buffer("free_mask_pseudoinverses", free_mask_pseudoinverses)

        # Unity HydrodynamicAxisIdentifier.cs 的默认线性轴映射。
        # 这个模式只用于线速度传统控制，避免在不需要力矩控制时由不准确
        # 的 6D 伪逆矩阵引入偏航力矩。
        linear_axis_map = torch.tensor(
            [
                # Fx/surge, Fy/heave, Fz/sway, Mx/roll, My/yaw, Mz/pitch
                [ 0.0, 1.0,  0.0, -1.0, 0.0, -1.0],  # Vertical1
                [ 0.0, 1.0,  0.0, -1.0, 0.0,  1.0],  # Vertical2
                [ 0.0, 1.0,  0.0,  1.0, 0.0, -1.0],  # Vertical3
                [ 0.0, 1.0,  0.0,  1.0, 0.0,  1.0],  # Vertical4
                [ 1.0, 0.0, -1.0,  0.0, 1.0,  0.0],  # Horizontal1
                [ 1.0, 0.0,  1.0,  0.0, 1.0,  0.0],  # Horizontal2
                [-1.0, 0.0,  1.0,  0.0, 1.0,  0.0],  # Horizontal3
                [-1.0, 0.0, -1.0,  0.0, 1.0,  0.0],  # Horizontal4
            ],
            dtype=torch.float32,
            device=resolved_device,
        )
        self.register_buffer("linear_axis_map", linear_axis_map)

        # 调试计数器
        self._debug_step_count = 0

    @staticmethod
    def _validate_positive_vector(
        values: Sequence[float],
        *,
        length: int,
        name: str,
        allow_zero: bool = False,
    ) -> list[float]:
        if len(values) != length:
            raise ValueError(f"{name} must contain {length} values, got {len(values)}")
        result = [abs(float(value)) for value in values]
        if allow_zero:
            # A zero policy-axis limit deliberately disables that wrench
            # degree of freedom. Keep only the allocator normalization buffer
            # nonzero; the incoming target wrench remains exactly zero.
            return [max(value, 1e-6) for value in result]
        if any(value <= 1e-6 for value in result):
            raise ValueError(f"{name} values must be greater than zero")
        return result

    @classmethod
    def _normalize_allocation_mode(cls, value: str) -> str:
        mode = str(value).strip().lower().replace("-", "_")
        aliases = {
            cls.EMPIRICAL_THRUSTER_MIXER: cls.EMPIRICAL_THRUSTER_MIXER,
            cls.PHYSICAL_WRENCH_ALLOCATOR: cls.PHYSICAL_WRENCH_ALLOCATOR,
            cls.AXIS_MAP: cls.AXIS_MAP,
        }
        if mode not in aliases:
            raise ValueError(
                f"unknown allocation_mode {value!r}; expected one of "
                f"{sorted((cls.EMPIRICAL_THRUSTER_MIXER, cls.PHYSICAL_WRENCH_ALLOCATOR, cls.AXIS_MAP))}"
            )
        return aliases[mode]

    def _load_or_build_free_mask_cache(
        self,
        physical_wrench_matrix: torch.Tensor,
        *,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load calibrated mask pseudo-inverses or build a config-specific cache once.

        The shipped cache belongs to the current FinsROV_Fossen physical matrix
        and pure-axis wrench limits.  A caller may still provide a different
        calibration; that uncommon construction path factorizes once and never
        in the allocator hot path.
        """

        try:
            resource = resources.files(__package__).joinpath(self._OFFLINE_CACHE_RESOURCE)
            with resources.as_file(resource) as cache_path:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError):
            payload = None

        if payload is not None and self._offline_cache_matches(payload, physical_wrench_matrix):
            return (
                payload["free_thruster_masks"].to(device=device),
                payload["free_mask_pseudoinverses"].to(device=device),
            )

        mask_ids = torch.arange(1 << len(self.THRUSTER_ORDER), device=device, dtype=torch.long)
        thruster_bits = 1 << torch.arange(len(self.THRUSTER_ORDER), device=device, dtype=torch.long)
        free_thruster_masks = (mask_ids.unsqueeze(1) & thruster_bits.unsqueeze(0)) != 0
        weighted_b = physical_wrench_matrix / self.physical_wrench_limits.view(6, 1)
        masked_weighted_b = weighted_b.unsqueeze(0) * free_thruster_masks.unsqueeze(1)
        return free_thruster_masks, torch.linalg.pinv(masked_weighted_b)

    def _offline_cache_matches(self, payload: object, physical_wrench_matrix: torch.Tensor) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("format_version") != self._OFFLINE_CACHE_FORMAT_VERSION:
            return False
        required = {
            "physical_wrench_matrix",
            "physical_wrench_limits",
            "free_thruster_masks",
            "free_mask_pseudoinverses",
        }
        if not required.issubset(payload):
            return False
        cached_matrix = payload["physical_wrench_matrix"]
        cached_limits = payload["physical_wrench_limits"]
        cached_masks = payload["free_thruster_masks"]
        cached_pseudoinverses = payload["free_mask_pseudoinverses"]
        return (
            isinstance(cached_matrix, torch.Tensor)
            and isinstance(cached_limits, torch.Tensor)
            and isinstance(cached_masks, torch.Tensor)
            and isinstance(cached_pseudoinverses, torch.Tensor)
            and cached_matrix.shape == physical_wrench_matrix.shape
            and cached_limits.shape == self.physical_wrench_limits.shape
            and cached_masks.shape == (
                1 << len(self.THRUSTER_ORDER),
                len(self.THRUSTER_ORDER),
            )
            and cached_pseudoinverses.shape == (
                1 << len(self.THRUSTER_ORDER),
                len(self.THRUSTER_ORDER),
                6,
            )
            and torch.allclose(cached_matrix, physical_wrench_matrix.cpu(), atol=1e-7, rtol=0.0)
            and torch.allclose(cached_limits, self.physical_wrench_limits.cpu(), atol=1e-7, rtol=0.0)
        )

    def _apply_deadzone_comp(self, thrust: torch.Tensor) -> torch.Tensor:
        if self.deadzone_comp <= 0.0:
            return thrust
        pos = thrust > 0
        neg = thrust < 0
        out = thrust.clone()
        out[pos] = out[pos] + self.deadzone_comp
        out[neg] = out[neg] - self.deadzone_comp
        return out

    def _allocate_physical_wrench(self, tau: torch.Tensor) -> torch.Tensor:
        """Allocate physical wrench with per-thruster force bounds.

        The active-set loop fixes an out-of-bound thruster at its physical
        limit, then resolves the remaining weighted least-squares problem.
        All batch members execute the same eight iterations in parallel. The
        pseudo-inverse for each of the 256 free-thruster masks is cached in
        ``__init__``, so this hot path contains no per-sample Python loop,
        host synchronization, or repeated matrix factorization.

        For a feasible target this preserves ``B @ force_n == tau``; for an
        infeasible combined request it returns the bounded closest wrench,
        weighted by each configured physical wrench limit.
        """

        weights = torch.reciprocal(self.physical_wrench_limits)
        lower = -self.thruster_force_limit_negative
        upper = self.thruster_force_limit_positive
        weighted_target = tau * weights.view(1, 6)
        batch_size = tau.shape[0]
        force_n = torch.zeros(batch_size, len(self.THRUSTER_ORDER), dtype=tau.dtype, device=tau.device)
        free_mask_ids = torch.full(
            (batch_size,),
            (1 << len(self.THRUSTER_ORDER)) - 1,
            dtype=torch.long,
            device=tau.device,
        )

        # A sample can fix at most eight channels.  Keep the fixed iteration
        # count on-device; samples already satisfying the bounds become no-op.
        for _ in range(len(self.THRUSTER_ORDER)):
            free = self.free_thruster_masks[free_mask_ids]
            pseudoinverse = self.free_mask_pseudoinverses[free_mask_ids]

            # ``B @ fixed_force`` is removed from the residual.  The cached
            # pseudo-inverse has zero rows for fixed channels, so its batched
            # product only updates currently-free forces.
            fixed_force = torch.where(free, torch.zeros_like(force_n), force_n)
            weighted_fixed_wrench = (fixed_force @ self.B.T) * weights.view(1, 6)
            rhs = weighted_target - weighted_fixed_wrench
            free_solution = torch.bmm(pseudoinverse, rhs.unsqueeze(2)).squeeze(2)
            force_n = torch.where(free, free_solution, force_n)

            violation = torch.maximum(lower.view(1, -1) - force_n, force_n - upper.view(1, -1))
            violation = torch.where(free, violation, torch.full_like(violation, float("-inf")))
            worst_violation, worst_index = torch.max(violation, dim=1)
            needs_clamp = worst_violation > 0.0

            selected_force = force_n.gather(1, worst_index.unsqueeze(1))
            selected_bound = torch.minimum(
                torch.maximum(selected_force, lower[worst_index].unsqueeze(1)),
                upper[worst_index].unsqueeze(1),
            )
            force_n = force_n.scatter(
                1,
                worst_index.unsqueeze(1),
                torch.where(needs_clamp.unsqueeze(1), selected_bound, selected_force),
            )
            free_mask_ids = free_mask_ids & ~(
                needs_clamp.to(torch.long) << worst_index
            )

        force_n = torch.minimum(torch.maximum(force_n, lower.view(1, -1)), upper.view(1, -1))
        return torch.where(
            force_n >= 0.0,
            force_n / self.thruster_force_limit_positive.view(1, 8),
            force_n / self.thruster_force_limit_negative.view(1, 8),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau: (batch, 6)/(6,) 力/力矩向量
                 [Fx, Fy, Fz, Mx, My, Mz]
                 - Fx: 纵向力 (前进方向)
                 - Fy: 垂向力 (上浮方向)
                 - Fz: 横向力 (向左)
                 - Mx: 滚转力矩
                 - My: 偏航力矩
                 - Mz: 俯仰力矩

        Returns:
            thrust_8d: (batch, 8) 或 (8,)
                       8个推进器推力输出，范围 [-1, 1]
        """
        self._debug_step_count += 1

        is_batched = tau.dim() > 1
        if not is_batched:
            tau = tau.unsqueeze(0)

        tau = tau.to(self.device)

        if tau.shape[1] != 6:
            raise ValueError(f"Expected tau with last dim == 6, got {tau.shape[1]}")

        if self.allocation_mode == self.PHYSICAL_WRENCH_ALLOCATOR:
            thrust_8d = self._allocate_physical_wrench(tau)
        else:
            tau_scaled = tau / self.control_axis_ranges.view(1, 6)
            if self.allocation_mode == self.AXIS_MAP:
                thrust_8d = torch.matmul(tau_scaled, self.linear_axis_map.T)
            else:
                # u = A * tau，A 为 (8, 6) 的推力分配矩阵
                # 结果为 (batch, 8)
                thrust_8d = torch.matmul(tau_scaled, self.A.T)

        thrust_8d = self._apply_deadzone_comp(thrust_8d)

        # 映射到 [-1, 1]。tau 已经按每个控制轴的 range 做过缩放。
        thrust_8d = torch.clamp(thrust_8d, -1.0, 1.0)

        if self.debug_print_interval > 0 and self._debug_step_count % self.debug_print_interval == 0:
            sample = thrust_8d[0].detach().cpu().numpy()
            values = ", ".join(f"T{i + 1}={float(v):+.4f}" for i, v in enumerate(sample))
            print(f"[ThrustAllocator Step {self._debug_step_count}] {values}", flush=True)

        if not is_batched:
            thrust_8d = thrust_8d.squeeze(0)

        return thrust_8d
