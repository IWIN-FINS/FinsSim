# 3Chase1 Local Observation and Action Interface

## Status

This document is the implemented 3Chase1 interface contract. It supersedes
the retired `30D actor + 13D controller suffix` plan.

The same Chaser Unity agent exports exactly 30 vector observations. No second
observation channel, virtual agent, world pose, or quaternion is needed by
the fixed low-level controller.

## Coordinate Basis

Every vector consumed by the Chaser actor and low-level PID is expressed in
`controller_body`:

```text
x = forward / surge
y = up / heave
z = left / sway
```

This basis is not converted again in Python. Unity creates the local relative
vectors, then Python passes high-level body errors directly to the PID.

## 30D Actor Observation

```text
[0:3]    self_linear_velocity_body
[3:6]    self_angular_velocity_body
[6:9]    prey_relative_position_body
[9:12]   prey_relative_velocity_body
[12]     distance_to_prey
[13]     bearing_to_prey = atan2(rel_z, rel_x)

[14:16]  teammate A role one-hot [is_herder, is_netter]
[16:19]  teammate A relative position body
[19:22]  teammate A relative velocity body

[22:24]  teammate B role one-hot [is_herder, is_netter]
[24:27]  teammate B relative position body
[27:30]  teammate B relative velocity body
```

The agent's own role is supplied separately by Python as an actor head selector.
It does not consume observation dimensions.

## 4D Action

The high-level MAPPO action is normalized:

```text
a = [a_forward, a_up, a_left, a_yaw], a_i in [-1, 1]
```

It maps to a direct low-level controller input:

```text
position_error_body_m = a[:3] * [1.5, 0.5, 1.5]
yaw_error_deg         = a[3] * 90
```

The fourth component is a signed body-yaw error, not a world yaw target and
not a yaw-only quaternion.

## Low-Level Consumption

```text
4D body error
  -> TraditionalPositionPIDModel.predict_from_body_position_error
  -> body wrench [Fx,Fy,Fz,Mx,My,Mz]
  -> physical_wrench_allocator
  -> Unity canonical 8D thruster action
```

The allocator uses the physical FinsROV_Fossen matrix and bounded individual
thruster forces. It is shared with the successful 1Chase1 hierarchy. The
policy must not apply a second body-to-world conversion.

## Non-Compatibility

Older 43D world-pose controller checkpoints are not compatible. Although both
policies may expose four floats, their values mean different things. New
checkpoints carry `body_subgoal_pid_physical_wrench_v1` and reject the old
interface at load time.
