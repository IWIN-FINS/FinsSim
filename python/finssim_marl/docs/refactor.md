"""
Legacy entry point for MAPPO 3chase1 Unity training.

This file is kept for backward compatibility.
For new training, use the unified script:

    python cleanmarl/scripts/train.py --help

For quick testing with xvfb (headless Unity):
    xvfb-run --auto-servernum --server-args='-screen 0 1280x1024x24' \
        python cleanmarl/scripts/train.py --env-type unity --num-envs 2 --total-timesteps 10000

The new architecture separates concerns into:
- algorithms/: MAPPO algorithm, networks (Actor/Critic)
- envs/: Environment abstractions and Unity wrappers
- scripts/train.py: Unified training runner
- utils/: Logging, checkpointing, configuration

See scripts/train.py for full training capabilities including
--overwrite, --resume, and configurable hyperparameters.
"""

# This file is deprecated. Use scripts/train.py instead.
