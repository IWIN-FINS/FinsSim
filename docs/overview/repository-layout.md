# Repository Layout

FinsSim is organized as a platform repository:

- Shared platform utilities live directly in this repository under `python/finssim_core` and `python/finssim_cli`.
- The ROS2 workspace is managed directly by this repository under `ros2_ws`.
- Existing trainer projects are imported as submodules during the first migration stage.
- Unity will be added later under `unity/`.

The main architectural rule is to share experiment/runtime protocols while keeping algorithm runtimes isolated.
