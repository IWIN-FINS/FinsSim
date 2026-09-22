import math
from typing import Callable

import numpy as np

# 1. 线性衰减
def linear_schedule(initial_value: float, min_value: float = 1e-5) -> Callable[[float], float]:
    """
    线性衰减，但绝不低于 min_value
    """
    def func(progress_remaining: float) -> float:
        """
        progress_remaining 从 1.0 降到 0.0（代表训练进度）。
        """
        # 计算当前的线性值
        current_lr = progress_remaining * initial_value
        # 返回当前值和最小值的较大者
        return max(current_lr, min_value)
    return func

# 2. 余弦退火
def cosine_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        # progress_remaining 从 1.0 降到 0.0
        return 0.5 * initial_value * (1 + np.cos((1 - progress_remaining) * np.pi))
    return func


def cosine_warm_restarts_schedule(
    peak_value: float,
    min_value: float,
    *,
    start_progress_remaining: float,
    cycles: int,
) -> Callable[[float], float]:
    """Create a continuation-local cosine schedule with warm restarts.

    Stable-Baselines3 supplies progress relative to the model's lifetime. A
    resumed run starts below 1.0 because its checkpoint already contains prior
    timesteps. ``start_progress_remaining`` maps that remaining global range
    back to [0, 1] for this new training invocation.
    """
    if peak_value <= 0.0:
        raise ValueError("peak_value must be positive")
    if min_value <= 0.0:
        raise ValueError("min_value must be positive")
    if min_value > peak_value:
        raise ValueError("min_value must not exceed peak_value")
    if not 0.0 < start_progress_remaining <= 1.0:
        raise ValueError("start_progress_remaining must be in (0, 1]")
    if cycles < 1:
        raise ValueError("cycles must be at least one")

    def func(progress_remaining: float) -> float:
        local_complete = 1.0 - min(
            max(float(progress_remaining), 0.0) / start_progress_remaining,
            1.0,
        )
        if local_complete >= 1.0:
            return min_value

        cycle_progress = (local_complete * cycles) % 1.0
        cosine = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        return min_value + (peak_value - min_value) * cosine

    return func


def apply_continuation_learning_rate_schedule(
    model,
    training_config,
    *,
    additional_steps: int | None = None,
) -> None:
    """Replace a loaded PPO model's scheduler when explicitly requested.

    SB3 checkpoints serialize ``lr_schedule``. Without this step, a new YAML
    cannot change the schedule of a resumed model because loading replaces the
    freshly configured model instance.
    """
    schedule_name = getattr(training_config, "learning_rate_schedule", None)
    if schedule_name in (None, "", "checkpoint"):
        return
    if schedule_name != "cosine_warm_restarts":
        raise ValueError(
            "Unsupported learning_rate_schedule "
            f"{schedule_name!r}; expected 'cosine_warm_restarts'."
        )

    if additional_steps is None:
        additional_steps = int(getattr(training_config, "total_timesteps", 0))
    else:
        additional_steps = int(additional_steps)
    if additional_steps <= 0:
        raise ValueError("total_timesteps must be positive for a continuation scheduler")

    previous_steps = max(int(getattr(model, "num_timesteps", 0)), 0)
    total_steps = previous_steps + additional_steps
    start_progress_remaining = additional_steps / total_steps
    peak_value = getattr(training_config, "learning_rate_peak", None)
    min_value = getattr(training_config, "learning_rate_min", None)
    cycles = getattr(training_config, "learning_rate_warm_restart_cycles", None)
    if peak_value is None or min_value is None or cycles is None:
        raise ValueError(
            "cosine_warm_restarts requires learning_rate_peak, "
            "learning_rate_min, and learning_rate_warm_restart_cycles"
        )

    schedule = cosine_warm_restarts_schedule(
        float(peak_value),
        float(min_value),
        start_progress_remaining=start_progress_remaining,
        cycles=int(cycles),
    )
    model.learning_rate = schedule
    model.lr_schedule = schedule

    initial_lr = schedule(start_progress_remaining)
    for parameter_group in model.policy.optimizer.param_groups:
        parameter_group["lr"] = initial_lr

    print(
        "[LearningRateSchedule] applied cosine_warm_restarts "
        f"previous_steps={previous_steps} additional_steps={additional_steps} "
        f"cycles={int(cycles)} peak={float(peak_value):.6g} "
        f"min={float(min_value):.6g} initial_lr={initial_lr:.6g}"
    )
