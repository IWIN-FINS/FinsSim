# multiagent-gail Upstream

- Repository: https://github.com/ermongroup/multiagent-gail
- Upstream path: `multiagent-gail/`
- Intended use: reference implementation for MAGAIL discriminator/training logic.
- Integration policy: port algorithm ideas into `finssim_marl.imitation` using
  PyTorch/TorchRL; do not import the original TensorFlow 1.x runner in FinsSim.
- Legacy requirements observed upstream: `tensorflow>=1.2`, old `gym`, `mpi4py`,
  `ray`, `zmq`, MuJoCo/Atari extras.
