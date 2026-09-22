# 3Chase1 Body-Subgoal MAPPO

`MAPPO_POSITION_CONTROL` is a hierarchical controller for 3Chase1. The learned
policy never sends world-frame poses to Unity. It emits a local 4D subgoal;
a fixed low-level controller converts that into the 8 normalized thruster
commands expected by Unity.

## Contract

```text
30D local actor observation + Python role id
  -> MAPPO actor action a=[a_forward, a_up, a_left, a_yaw] in [-1, 1]^4
  -> e_body=a[:3]*[1.5 m, 0.5 m, 1.5 m]
  -> e_yaw_deg=a[3]*90 deg
  -> TraditionalPositionPIDModel.predict_from_body_position_error
  -> tau_body=[Fx,Fy,Fz,Mx,My,Mz]
  -> physical_wrench_allocator
  -> 8D normalized thruster action in [-1,1]^8
  -> Unity FinsROV_Fossen
```

The controller-body basis is always `x=forward/surge`, `y=up/heave`,
`z=left/sway`. `e_yaw_deg` is a signed yaw error in degrees. Positive yaw is
mapped by the same `TraditionalPositionPIDModel` and calibrated physical
allocator used by the 1Chase1 pose-yaw hierarchy.

The high-level action enters the PPO buffer. The 8D thruster action is only
the environment action; the fixed PID and allocator do not receive PPO
gradients.

## Observation

Every Chaser has exactly 30 Unity vector observations. There is no controller
suffix and no world pose/quaternion in this interface.

```text
0:3    self linear velocity, body frame
3:6    self angular velocity, body frame
6:9    prey relative position, body frame
9:12   prey relative velocity, body frame
12     distance to prey
13     bearing to prey = atan2(rel_z, rel_x)
14:16  teammate A role one-hot [is_herder, is_netter]
16:19  teammate A relative position, body frame
19:22  teammate A relative velocity, body frame
22:24  teammate B role one-hot
24:27  teammate B relative position, body frame
27:30  teammate B relative velocity, body frame
```

The actor also receives a Python-side role one-hot to select the Herder or
Netter head. Prey remains non-trainable and receives an all-zero action from
this algorithm.

## Low-Level Limits

`body_pid_wrench` is the normal backend. It is intentionally the only real
backend for this interface; `zero` exists only for tests. It always uses
`physical_wrench_allocator`, each thruster is bounded to `[-7, 7] N`, and its
body wrench limits are:

```text
[Fx, Fy, Fz, Mx, My, Mz]
= [19.528527, 22.501886, 18.415027, 3.863995, 7.080833, 3.079985]
  [N,       N,         N,         N*m,      N*m,      N*m]
```

The translational PID gains are `[5, 0.1, 0.5]` for depth, `[5, 0.3, 0.5]`
for surge, and `[5, 0.3, 0.5]` for sway. The yaw PID gains are
`[0.12, 0, 0.01]`, with a `1 deg` deadband. These values and physical limits
match the 1Chase1 pose-yaw hierarchy.

## Running

The primary YAML uses the new contract:

```bash
cd .
uv run --package finssim-cli finssim marl train \
  -c configs/marl/3chase1/position_control.yaml
```

The `chasing_3_chase_1_traditional_pose_controller` config id is retained so
older YAML filenames still resolve, but it is now only a compatibility alias
for `body_pid_wrench`; it no longer means world-pose control or empirical
mixing.

Existing checkpoints from the old world-pose controller are rejected. Their
4D tensor shape happens to match, but its yaw and position semantics do not,
so loading them would silently produce invalid control.

## Verification

Run the focused interface and wrench-sign tests:

```bash
cd ./python/finssim_marl
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_position_control_yaw_toggle.py -q
```

The test checks 30D execution, zero error, positive body-forward command
producing positive `Fx`, and positive yaw error producing positive `My` after
the actual physical allocation matrix.
