from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STEP_CHECKPOINT_RE = re.compile(r"^step_(\d+)\.pt$")


@dataclass(frozen=True)
class StepCheckpoint:
    path: Path
    step: int

    @property
    def training_state_path(self) -> Path:
        return self.path.with_name(f"{self.path.stem}_training_state.pt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune low-step MARL checkpoints under one or more roots. "
            "Only step_*.pt files are considered; named checkpoints such as best.pt/final.pt are preserved."
        )
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=["artifacts/runs/marl"],
        help="Root directories to scan recursively. Defaults to artifacts/runs/marl.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        required=True,
        help="Delete step checkpoints with step < min_step unless they are kept by another rule.",
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=2,
        help="Always keep the latest N step checkpoints in each checkpoint directory. Default: 2.",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=0,
        help=(
            "Among checkpoints below --min-step, keep those whose step is a multiple of this value. "
            "Set 0 to disable. Example: --keep-every 100000."
        ),
    )
    parser.add_argument(
        "--keep-first",
        action="store_true",
        help="Always keep the earliest step checkpoint in each checkpoint directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag the script only prints a dry-run plan.",
    )
    return parser.parse_args()


def discover_step_checkpoints(roots: Iterable[Path]) -> dict[Path, list[StepCheckpoint]]:
    grouped: dict[Path, list[StepCheckpoint]] = {}
    for root in roots:
        if not root.exists():
            print(f"[WARN] Root does not exist, skipped: {root}", file=sys.stderr)
            continue

        for path in root.rglob("*.pt"):
            if path.name.endswith("_training_state.pt"):
                continue
            match = STEP_CHECKPOINT_RE.match(path.name)
            if match is None:
                continue
            checkpoint = StepCheckpoint(path=path, step=int(match.group(1)))
            grouped.setdefault(path.parent, []).append(checkpoint)

    for checkpoints in grouped.values():
        checkpoints.sort(key=lambda ckpt: ckpt.step)
    return grouped


def select_prune_candidates(
    checkpoints: list[StepCheckpoint],
    *,
    min_step: int,
    keep_last_n: int,
    keep_every: int,
    keep_first: bool,
) -> tuple[list[StepCheckpoint], list[StepCheckpoint]]:
    if not checkpoints:
        return [], []

    keep_paths: set[Path] = set()

    if keep_last_n > 0:
        for checkpoint in checkpoints[-keep_last_n:]:
            keep_paths.add(checkpoint.path)

    if keep_first:
        keep_paths.add(checkpoints[0].path)

    if keep_every > 0:
        for checkpoint in checkpoints:
            if checkpoint.step < min_step and checkpoint.step % keep_every == 0:
                keep_paths.add(checkpoint.path)

    to_delete: list[StepCheckpoint] = []
    to_keep: list[StepCheckpoint] = []
    for checkpoint in checkpoints:
        if checkpoint.step >= min_step or checkpoint.path in keep_paths:
            to_keep.append(checkpoint)
        else:
            to_delete.append(checkpoint)
    return to_keep, to_delete


def prune_checkpoints(
    roots: Iterable[Path],
    *,
    min_step: int,
    keep_last_n: int,
    keep_every: int,
    keep_first: bool,
    apply: bool,
) -> int:
    grouped = discover_step_checkpoints(roots)
    if not grouped:
        print("No step checkpoints found.")
        return 0

    total_deleted_files = 0
    total_deleted_checkpoints = 0
    total_kept_checkpoints = 0

    for checkpoint_dir in sorted(grouped):
        checkpoints = grouped[checkpoint_dir]
        kept, deleted = select_prune_candidates(
            checkpoints,
            min_step=min_step,
            keep_last_n=keep_last_n,
            keep_every=keep_every,
            keep_first=keep_first,
        )
        total_kept_checkpoints += len(kept)
        total_deleted_checkpoints += len(deleted)

        print(f"\n[{checkpoint_dir}]")
        print(f"  keep={len(kept)} delete={len(deleted)} total={len(checkpoints)}")

        if kept:
            kept_text = ", ".join(str(checkpoint.step) for checkpoint in kept)
            print(f"  kept steps: {kept_text}")
        if deleted:
            deleted_text = ", ".join(str(checkpoint.step) for checkpoint in deleted)
            print(f"  delete steps: {deleted_text}")

        if not apply:
            continue

        for checkpoint in deleted:
            checkpoint.path.unlink(missing_ok=True)
            total_deleted_files += 1
            if checkpoint.training_state_path.exists():
                checkpoint.training_state_path.unlink()
                total_deleted_files += 1

    mode = "APPLY" if apply else "DRY-RUN"
    print(
        f"\n[{mode}] kept_checkpoints={total_kept_checkpoints} "
        f"deleted_checkpoints={total_deleted_checkpoints} deleted_files={total_deleted_files}"
    )
    if not apply:
        print("Add --apply to actually delete the planned checkpoints.")
    return 0


def main() -> int:
    args = _parse_args()
    if args.min_step < 0:
        print("--min-step must be >= 0", file=sys.stderr)
        return 2
    if args.keep_last_n < 0:
        print("--keep-last-n must be >= 0", file=sys.stderr)
        return 2
    if args.keep_every < 0:
        print("--keep-every must be >= 0", file=sys.stderr)
        return 2

    roots = [Path(root).expanduser().resolve() for root in args.roots]
    return prune_checkpoints(
        roots,
        min_step=args.min_step,
        keep_last_n=args.keep_last_n,
        keep_every=args.keep_every,
        keep_first=args.keep_first,
        apply=args.apply,
    )


if __name__ == "__main__":
    raise SystemExit(main())
