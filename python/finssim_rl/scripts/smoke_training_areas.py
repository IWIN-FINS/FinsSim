"""Progressively validate a TrainingAreaReplicator Server binary.

Example:
    uv run python scripts/smoke_training_areas.py --env-path /path/to/HoldForPosition_Parallel.x86_64
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from finssim_rl.models import make_unity_training_areas_env
from finssim_rl.training.config import BaseEnvironmentConfig


def _rss_mb(env) -> float | None:
    unity_env = getattr(env, "unity_env", None)
    process = getattr(unity_env, "process", None) or getattr(unity_env, "_process", None)
    pid = getattr(process, "pid", None)
    if not pid:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-path", required=True)
    parser.add_argument("--areas", nargs="+", type=int, default=[4, 256, 512, 1024, 2048])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--env-base-port", type=int, default=39105)
    parser.add_argument("--timeout-wait", type=int, default=600)
    parser.add_argument("--time-scale", type=float, default=10.0)
    args = parser.parse_args()

    results = []
    for offset, areas in enumerate(args.areas):
        started = time.monotonic()
        env = None
        try:
            config = BaseEnvironmentConfig(
                env_path=args.env_path,
                # One Unity process exposes this many replicated area slots.
                num_envs=int(areas),
                env_base_port=int(args.env_base_port) + offset * 10,
                timeout_wait=int(args.timeout_wait),
                time_scale=float(args.time_scale),
                no_graphics=True,
            )
            env = make_unity_training_areas_env(args.env_path, config)
            obs = env.reset()
            if obs.shape[0] != areas:
                raise RuntimeError(f"expected {areas} vector slots, got {obs.shape[0]}")
            loop_started = time.monotonic()
            for _ in range(args.steps):
                actions = np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32)
                env.step(actions)
            elapsed = time.monotonic() - loop_started
            result = {
                "areas": areas,
                "startup_seconds": round(loop_started - started, 3),
                "steps": args.steps,
                "vector_steps_per_second": round(args.steps / elapsed, 3),
                "agent_steps_per_second": round(args.steps * areas / elapsed, 3),
                "unity_rss_mb": _rss_mb(env),
                "status": "passed",
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
        except Exception as exc:
            result = {"areas": areas, "status": "failed", "error": str(exc)}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
            break
        finally:
            if env is not None:
                env.close()

    print(json.dumps({"ladder": results}, ensure_ascii=False, indent=2))
    return 0 if results and results[-1]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
