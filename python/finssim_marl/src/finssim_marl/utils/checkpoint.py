"""
Checkpoint management for saving and loading model states.
"""

import os
import shutil
import torch
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 10  # Save every N environment steps


class CheckpointManager:
    """Manages saving and loading of model checkpoints.

    Handles:
    - Creating checkpoint directories
    - Saving model state, optimizer state, and training progress
    - Loading from checkpoint for resuming training
    """

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, path: str, algorithm, training_state: dict):
        """Save checkpoint.

        Args:
            path: Full path to save checkpoint
            algorithm: Algorithm instance with save() method
            training_state: Dict with training progress info (step, etc.)
        """
        algorithm.save(path)
        # Also save training state alongside
        state_path = path.replace(".pt", "_training_state.pt")
        torch.save(training_state, state_path)

    def load(self, path: str, algorithm):
        """Load checkpoint.

        Args:
            path: Full path to checkpoint
            algorithm: Algorithm instance with load() method

        Returns:
            training_state: Dict with training progress info, or empty dict
        """
        algorithm.load(path)
        state_path = path.replace(".pt", "_training_state.pt")
        if os.path.exists(state_path):
            return torch.load(state_path)
        return {}

    def get_latest(self, run_name: str):
        """Get the latest checkpoint path for a run.

        Args:
            run_name: Name of the run

        Returns:
            Path to latest checkpoint or None
        """
        run_dir = self.checkpoint_dir / run_name
        if not run_dir.exists():
            return None
        checkpoints = [
            path
            for path in run_dir.glob("step_*.pt")
            if not path.name.endswith("_training_state.pt")
        ]
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))

    def save_interval(self, run_name: str, iteration: int, algorithm, training_state: dict):
        """Save checkpoint for the current interval crossing.

        Args:
            run_name: Name of the run
            iteration: Current environment step
            algorithm: Algorithm instance
            training_state: Training progress dict
        """
        run_dir = self.checkpoint_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"step_{iteration}.pt"
        self.save(str(path), algorithm, training_state)
        return str(path)

    def save_named(self, run_name: str, checkpoint_name: str, algorithm, training_state: dict):
        """Save a named checkpoint such as ``best.pt`` or ``final.pt``.

        Args:
            run_name: Name of the run
            checkpoint_name: Checkpoint stem or filename
            algorithm: Algorithm instance
            training_state: Training progress dict
        """
        run_dir = self.checkpoint_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        filename = checkpoint_name if checkpoint_name.endswith(".pt") else f"{checkpoint_name}.pt"
        path = run_dir / filename
        self.save(str(path), algorithm, training_state)
        return str(path)

    def promote_snapshot(
        self,
        run_name: str,
        snapshot_path: str | Path,
        training_state: dict,
        *,
        keep_source: bool,
    ) -> str:
        """Promote an already-evaluated immutable snapshot to ``best.pt``.

        The asynchronous evaluator may finish after training has advanced, so
        saving the live training model here would associate the evaluation score
        with the wrong policy. Promote the evaluated snapshot itself instead.
        """
        source = Path(snapshot_path)
        run_dir = self.checkpoint_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        destination = run_dir / "best.pt"
        source_state = self._training_state_path(source)
        destination_state = self._training_state_path(destination)

        if keep_source:
            temporary_destination = destination.with_name(f".{destination.name}.tmp")
            shutil.copy2(source, temporary_destination)
            temporary_destination.replace(destination)
        else:
            source.replace(destination)
            if source_state.exists():
                source_state.replace(destination_state)

        # Snapshot state was created before this evaluation score existed.
        # Store the updated best-score metadata beside the promoted policy.
        torch.save(training_state, destination_state)
        return str(destination)

    @staticmethod
    def discard_snapshot(snapshot_path: str | Path) -> None:
        """Delete an asynchronous evaluation model/state pair."""
        snapshot = Path(snapshot_path)
        snapshot.unlink(missing_ok=True)
        CheckpointManager._training_state_path(snapshot).unlink(missing_ok=True)

    @staticmethod
    def _training_state_path(path: str | Path) -> Path:
        checkpoint_path = Path(path)
        return checkpoint_path.with_name(f"{checkpoint_path.stem}_training_state.pt")

    def exists(self, run_name: str) -> bool:
        """Check if any checkpoint exists for a run."""
        run_dir = self.checkpoint_dir / run_name
        return run_dir.exists() and any(run_dir.glob("step_*.pt"))
