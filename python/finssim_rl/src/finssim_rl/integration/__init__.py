"""Stable integration surface for sibling FinsSim backends.

The regular RL backend remains the owner of Unity process management, the
canonical FinsROV thrust allocator, and controller baselines.  Packages such
as :mod:`finssim_irl` should import from this module instead of reaching into
private training implementation details.
"""

from finssim_rl.integration.allocation import (
    CANONICAL_THRUSTER_ORDER,
    DEFAULT_PHYSICAL_WRENCH_LIMITS,
    build_physical_thrust_allocator,
)
from finssim_rl.integration.observation_contracts import OBSERVATION16_DIM, OBSERVATION16_FIELDS
from finssim_rl.integration.pid_experts import build_goal_yaw_thruster8_pid
from finssim_rl.integration.unity_env import make_eval_vec_env, make_train_vec_env

__all__ = [
    "CANONICAL_THRUSTER_ORDER",
    "DEFAULT_PHYSICAL_WRENCH_LIMITS",
    "OBSERVATION16_DIM",
    "OBSERVATION16_FIELDS",
    "build_goal_yaw_thruster8_pid",
    "build_physical_thrust_allocator",
    "make_eval_vec_env",
    "make_train_vec_env",
]
