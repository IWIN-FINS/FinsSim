# 3Chase1 Position-Control Training Flow

There are two independent action interfaces in the MARL package.

## Direct Thruster MAPPO

The existing end-to-end `MAPPO` path is unchanged:

```text
actor 8D normalized thruster action
  -> rollout buffer 8D action
  -> Unity 8D thruster action
```

## Body-Subgoal PID/Wrench MAPPO

`MAPPO_POSITION_CONTROL` retains the scheduler action interface
`position_controller` and records the controller contract as
`body_subgoal_pid_physical_wrench_v1` in checkpoint metadata:

```text
actor 4D normalized body subgoal action
  -> rollout buffer 4D action
  -> fixed body_pid_wrench backend
  -> Unity 8D normalized thruster action
```

The actor action is `a=[forward, up, left, yaw]`, each component in `[-1,1]`.
It is converted exactly once in
`MAPPOMultiHeadPositionControlAlgorithm.select_action()`:

```python
e_body_m = a[..., :3] * [1.5, 0.5, 1.5]
e_yaw_deg = a[..., 3] * 90.0
thrusters = body_pid_wrench.act(e_body_m, e_yaw_deg)
```

`body_pid_wrench` invokes
`TraditionalPositionPIDModel.predict_from_body_position_error()`. The PID
builds the physical controller-body wrench
`[Fx,Fy,Fz,Mx,My,Mz]`; `physical_wrench_allocator` then produces the
canonical Unity order
`[V_LF,V_LB,V_RB,V_RF,H_LF,H_LB,H_RB,H_RF]`.

The controller is fixed: PPO only optimizes the role-aware high-level actor
and centralized critic. The buffer intentionally stores the sampled 4D actor
action rather than its 8D PID output, so PPO log probabilities remain valid.

## Reset

At episode reset, `reset_controller_state()` resets PID integral, derivative,
and slew-rate state. A partial environment reset supplies an environment mask
so parallel areas do not share controller state.

## CLI Parameters

The low-level path is no longer selected between SB3, hybrid, and empirical
mixers. `body_pid_wrench` is the only physical control backend. Optional
configuration parameters are deliberately limited to tuning quantities:

```text
--target-body-delta-limits FWD UP LEFT
--yaw-error-limit-deg DEG
--body-pid-control-rate-hz HZ
--body-pid-params KP KI KD ...
--body-pid-yaw-pid-params KP KI KD
--body-pid-integral-decay VALUE
--body-pid-yaw-deadband-deg DEG
```

Wrench limits, per-thruster `[-7,7] N` bounds, and
`physical_wrench_allocator` are not CLI-overridable. That prevents an old
configuration from silently substituting an empirical mixer or unrealistic
250 N limits.

## Checkpoints

Checkpoints record `control_interface_version`.
`body_subgoal_pid_physical_wrench_v1` rejects old world-pose controller
checkpoints even though both have a 4D tensor action shape. The old action was
a world-target delta plus yaw delta, whereas the current action is a direct
body PID error plus a yaw error in degrees.
