# Simulator integrations

`simulators/unity/` is a local ignored directory containing links to the Unity
projects.  The legacy root `unity` path is a compatibility link to it.

`simulators/isaaclab/FinsSimIsaacLab` is an optional, independently versioned
repository containing FinsSim's Isaac Lab extension and runtime adapter. It is
not a Git submodule. Clone a reviewed release into this fixed local path:

```bash
git clone https://github.com/IWIN-FINS/FinsSimIsaacLab.git \
  simulators/isaaclab/FinsSimIsaacLab
```

The upstream NVIDIA Isaac Lab checkout is separate; configure its path through
`ISAACLAB_PATH`.
