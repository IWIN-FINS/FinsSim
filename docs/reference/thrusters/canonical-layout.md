# UUV Thruster Layout Canonical Reference

## Purpose

This document is the canonical reference for the 8-thruster UUV layout used by the RLChase scene `Assets/Scenes/3Chase1.unity`.
It is intended for AI tools, controllers, debugging utilities, and reward-analysis scripts.

Use this document as the ground truth for:
- thruster naming
- thruster directions
- thruster positions
- control-axis interpretation
- action-to-thruster mapping

## Scope

This applies to the three chase UUVs in `3Chase1.unity`:
- `UUV7` (`netter1`)
- `UUV8` (`netter2`)
- `UUV9` (`herder`)

These three vehicles use the same `AdvancedShipController.engines` layout for the 8 propulsion engines.
The herder does not have a unique asymmetric thruster configuration.

## Coordinate And Control Conventions

### Vehicle-local axes

For planar control, the important local axes are:
- local `+X`: surge / effective forward direction for the diagonal horizontal thruster set
- local `+Y`: upward direction
- local `+Z`: lateral component that contributes to yaw error in the current baseline controller

Important:
- Do not assume that local `+Z` is the effective planar forward axis for these UUVs.
- For the horizontal thruster layout in this scene, net forward planar force is primarily along local `+X`.

### Control reference transform

High-level chase controllers should use:
- `ChaserAgent.selfTransform` if assigned
- otherwise fall back to `agent.transform`

Reason:
- `ChaserAgent` is authored to expose a dedicated control reference via `selfTransform`.
- Current baseline code uses `selfTransform` first for target-offset conversion.

## Thruster Groups

There are 8 engines total:
- `Vertical1` to `Vertical4`
- `Horizontal1` to `Horizontal4`

### Vertical thrusters

All vertical thrusters use:
- `thrustDirection = (0, 1, 0)`

They generate heave only in the ideal symmetric case.

Approximate local thrust positions:
- `Vertical1`: `(0.36, -0.11, 0.03)`
- `Vertical2`: `(0.138, -0.11, 0.03)`
- `Vertical3`: `(0.14, -0.11, -0.24)`
- `Vertical4`: `(0.36, -0.11, -0.24)`

Top-view interpretation used by existing docs:
- `Vertical1`: left-front
- `Vertical2`: left-rear
- `Vertical3`: right-rear
- `Vertical4`: right-front

## Horizontal Thrusters

The horizontal thrusters are diagonal, not axis-aligned.

### Canonical directions

- `Horizontal1`: `( 1, 0, -1 )`
- `Horizontal2`: `( 1, 0,  1 )`
- `Horizontal3`: `(-1, 0,  1 )`
- `Horizontal4`: `(-1, 0, -1 )`

Approximate local thrust positions:
- `Horizontal1`: `(0.46, -0.16,  0.04)`
- `Horizontal2`: `(0.03, -0.16,  0.04)`
- `Horizontal3`: `(0.04, -0.16, -0.24)`
- `Horizontal4`: `(0.47, -0.16, -0.24)`

Top-view interpretation used by existing docs:
- `Horizontal1`: left-front
- `Horizontal2`: left-rear
- `Horizontal3`: right-rear
- `Horizontal4`: right-front

## Symmetry Conclusion

For `UUV7`, `UUV8`, and `UUV9` in `3Chase1.unity`:
- horizontal thruster directions match
- horizontal thruster positions match
- vertical thruster directions match
- vertical thruster positions match
- `maxThrust` values are scene-unified by `CatchAreaManager`

Therefore:
- if one vehicle spins abnormally while others do not, the first suspect should be controller logic or target-axis interpretation
- it should not be assumed that `UUV9` has a unique propulsion geometry issue

## Action Mapping Used By RLChase Agents

In `ChaserAgent.OnActionReceived()`, the 8 continuous actions are applied as:
- action 0 -> `Vertical1`
- action 1 -> `Vertical2`
- action 2 -> `Vertical3`
- action 3 -> `Vertical4`
- action 4 -> `Horizontal1`
- action 5 -> `Horizontal2`
- action 6 -> `Horizontal3`
- action 7 -> `Horizontal4`

This is direct per-engine throttle assignment.

## Canonical Composite Control Patterns

### Pure heave up/down

Apply the same sign to all vertical thrusters:
- `[v, v, v, v, 0, 0, 0, 0]`

This approximates pure vertical motion.

### Pure planar surge along local `+X`

Apply the horizontal pattern:
- `H1 = +f`
- `H2 = +f`
- `H3 = -f`
- `H4 = -f`

Equivalent action vector tail:
- `[+f, +f, -f, -f]`

This is the canonical "forward" pattern for these diagonal horizontal thrusters.

### Pure planar surge along local `-X`

Apply the horizontal pattern:
- `H1 = -f`
- `H2 = -f`
- `H3 = +f`
- `H4 = +f`

Equivalent action vector tail:
- `[-f, -f, +f, +f]`

### Pure yaw-style pattern

Apply the same sign to all four horizontal thrusters:
- `H1 = s`
- `H2 = s`
- `H3 = s`
- `H4 = s`

Equivalent action vector tail:
- `[s, s, s, s]`

This is the pattern currently used by the RLChase baseline controller as the steering component.

## Baseline Controller Interpretation

The current RLChase baseline controller uses the following planar interpretation:
- planar forward command is derived from `desiredOffsetLocal.x`
- planar yaw error is derived from `atan2(-localZ, localX)`

This is the key correction that fixed the earlier herder instability.

Previous incorrect assumption:
- treating local `+Z` as the planar forward axis

Observed failure mode from the incorrect assumption:
- very large steering demand even when the target is "in front" in the real propulsion frame
- rapid in-place spinning
- lateral drift / runaway motion caused by diagonal-thruster coupling

## Current RLChase Scene Defaults

In the current RLChase setup, `CatchAreaManager` unifies chase-UUV thrust settings to:
- vertical `maxThrust = 150`
- horizontal `maxThrust = 400`

Chaser runtime `maxSpeed` is set to:
- `6`

These are scene/runtime configuration values and are not part of the underlying geometry definition.

## What AI Tools Should Assume

AI tools operating on this UUV should assume:
- there are 8 directly commanded engines
- the 4 horizontal engines are diagonal
- the effective planar forward axis is local `+X`
- direct per-thruster control is valid
- controller bugs are more likely to come from wrong axis assumptions than from UUV9-only asymmetry

## What AI Tools Should Not Assume

AI tools should not assume:
- local `+Z` is the correct planar forward axis
- horizontal thrusters are simple front/back propellers
- `UUV9` uses a different propulsion layout from `UUV7` and `UUV8`
- direct-chase instability automatically implies wrong engine placement

## Source Of Truth

This document was reconciled against:
- `Assets/Scenes/3Chase1.unity`
- `Assets/Scripts/ThreeChaseOne/ChaserAgent.cs`
- `Assets/Scripts/ThreeChaseOne/ThreeChaseOneBaselineController.cs`
- `Packages/com.nwh.dynamicwaterphysics/Runtime/ShipController/Engine.cs`

If these sources change, this document should be updated.
