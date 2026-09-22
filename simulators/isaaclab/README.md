# Isaac Lab integration boundary

`FinsSimIsaacLab` is an optional independent repository, not a submodule.
Before building the ROS2 workspace or running Isaac Lab tasks, clone a reviewed
release into the expected directory:

```bash
git clone https://github.com/IWIN-FINS/FinsSimIsaacLab.git FinsSimIsaacLab
```

Set `ISAACLAB_PATH` to the separate NVIDIA Isaac Lab installation directory.

Use `CUDA_VISIBLE_DEVICES=1` for Isaac Lab training on this host so the RTX
3090, rather than the display-oriented T1000, is selected.
