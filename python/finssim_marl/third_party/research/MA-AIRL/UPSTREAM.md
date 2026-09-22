# MA-AIRL Upstream

- Repository: https://github.com/ermongroup/MA-AIRL
- Upstream path: `multi-agent-irl/`
- Intended use: reference implementation for MA-AIRL reward/discriminator logic.
- Integration policy: port algorithm ideas into `finssim_marl.imitation` using
  PyTorch/TorchRL; do not import the original TensorFlow 1.x runner in FinsSim.
- Legacy requirements observed upstream: `tensorflow>=1.2`, old `gym`, `mpi4py`,
  `ray`, `zmq`, MuJoCo/Atari extras.
