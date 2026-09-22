# Hydrodynamic Identification Workflow

This workflow estimates per-axis hydrodynamic coefficients from the body-axis model:

`tau = J * v_dot + B * v + D * |v| * v`

The Unity side logs `tau`, `v`, and `v_dot` for one axis at a time. The Python side then solves a least-squares regression for `J`, `B`, and `D`.

The current Unity logger uses DWP runtime thrust values (`Engine.Thrust`, `Engine.ThrustDirection`, `Engine.ThrustPosition`) so the logged `tau` follows the actual DWP thrust formula instead of a simplified `maxThrust * throttle` approximation.

## Files

- `examples/system_identification/HydrodynamicAxisIdentifier.cs`
- `scripts/fit_hydrodynamic_jbd.py`

## Unity setup

1. Copy `HydrodynamicAxisIdentifier.cs` into your Unity project under `Assets/Scripts/`.
2. Attach it to the same GameObject that already has:
   - `Rigidbody`
   - `AdvancedShipController`
3. Assign `bodyFrame` to the vehicle center frame used by your other controllers.
4. Keep `RandomizedWaterCurrentProvider` unassigned or disable random current during identification.
5. Verify the default thruster maps:
   - `SurgeX`: `Horizontal1=-1`, `Horizontal2=-1`, `Horizontal3=+1`, `Horizontal4=+1`
   - `HeaveY`: `Vertical1..4=+1`
   - `SwayZ`: `Horizontal1=+1`, `Horizontal2=-1`, `Horizontal3=-1`, `Horizontal4=+1`

The default maps are based on the existing test combinations in the Unity project, but you should still verify the sign convention once in Play mode.

## Data collection

Run the component from the inspector with `Run Identification`.

You can choose the test axis manually in the inspector:

- `axisSelectionMode = SingleAxis`
- `selectedAxis = SurgeX / HeaveY / SwayZ`

If you want the old behavior, switch `axisSelectionMode` back to `AllConfiguredAxes`.

The script will:

1. Reset pose and rigid-body velocity before each trial.
2. Wait for a `settle` segment with zero thrust so the vehicle can stabilize after reset.
3. Log a short `baseline` segment with zero thrust.
4. Apply one axis excitation level.
5. Log a `rest` segment after thrust is removed.
6. Repeat for positive and negative amplitudes on each axis.

The CSV is written to `./artifacts/results/outputFileName` by default.

Important practice:

- Disable RL agents and any PID / ESO loops while collecting data.
- Disable current randomization for the cleanest `J/B/D` fit.
- Keep the vehicle far from the surface and obstacles.
- If coupling torques are large, adjust the axis thruster weights until `torque_body_*` is near zero during excitation.

## Fitting

Default direct least-squares:

```bash
python3 scripts/fit_hydrodynamic_jbd.py /path/to/hydrodynamic_identification.csv --output-json /tmp/jbd.json
```

If the exported acceleration is noisy, recompute it from smoothed velocity:

```bash
python3 scripts/fit_hydrodynamic_jbd.py /path/to/hydrodynamic_identification.csv --recompute-acc --dt 0.02
```

PySINDy STLSQ sparse regression:

```bash
python3 scripts/fit_hydrodynamic_jbd.py /path/to/hydrodynamic_identification.csv --method pysindy --pysindy-threshold 1e-5
```

The script prints one line per axis:

- `J`: effective inertia / added-mass term
- `B`: linear damping
- `D`: quadratic damping
- `R2`: fit quality
- `RMSE_tau`: force-domain residual
- `cond(X)`: conditioning of the regression matrix

## Notes on PySINDy

The `--method pysindy` mode uses the `PySINDy` `STLSQ` optimizer on:

`v_dot = c_tau * tau + c_v * v + c_q * |v| * v`

and then converts back with:

- `J = 1 / c_tau`
- `B = -c_v / c_tau`
- `D = -c_q / c_tau`

This is useful when you want sparse identification and thresholding, but for the current 3-term model the direct least-squares form is still the simplest baseline.
