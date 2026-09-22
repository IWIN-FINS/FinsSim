"""Background evaluation of immutable MARL policy snapshots."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class AsyncEvaluationJob:
    """One immutable policy snapshot waiting for evaluation."""

    step: int
    snapshot_path: Path
    training_state: dict[str, Any]
    environment_parameters: dict[str, float]
    submitted_at: float


@dataclass(frozen=True)
class AsyncEvaluationResult:
    """Completed asynchronous evaluation, consumed by the training thread."""

    job: AsyncEvaluationJob
    started_at: float
    completed_at: float
    metrics: dict[str, Any] | None
    error: str | None = None


class AsyncEvaluationWorker:
    """Evaluate queued snapshots on a worker-owned Unity evaluation environment.

    The evaluator callback must own all interaction with its Unity environment.
    The training thread only saves snapshots and consumes completed results.
    """

    def __init__(
        self,
        *,
        snapshot_dir: Path,
        evaluator: Callable[[Path, dict[str, float]], dict[str, Any]],
        keep_snapshots: bool = False,
    ) -> None:
        self.snapshot_dir = snapshot_dir
        self.evaluator = evaluator
        self.keep_snapshots = keep_snapshots
        self._jobs: queue.Queue[AsyncEvaluationJob | None] = queue.Queue()
        self._results: queue.Queue[AsyncEvaluationResult] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if not self.keep_snapshots:
            self.discard_residual_snapshots()
        self._thread = threading.Thread(
            target=self._run,
            name="finssim-marl-async-eval",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        *,
        step: int,
        snapshot_path: Path,
        training_state: dict[str, Any],
        environment_parameters: dict[str, float],
    ) -> None:
        if self._thread is None:
            raise RuntimeError("AsyncEvaluationWorker.start() must be called before submit().")
        self._jobs.put(
            AsyncEvaluationJob(
                step=step,
                snapshot_path=snapshot_path,
                training_state=dict(training_state),
                environment_parameters=dict(environment_parameters),
                submitted_at=time.time(),
            )
        )

    def drain_results(self) -> list[AsyncEvaluationResult]:
        results: list[AsyncEvaluationResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def close(self) -> None:
        if self._thread is None:
            return
        self._jobs.put(None)
        self._thread.join()
        self._thread = None

    def discard_residual_snapshots(self) -> None:
        """Remove callback-owned model/state pairs from aborted default runs."""
        for snapshot_path in self.snapshot_dir.glob("snapshot_*.pt"):
            if snapshot_path.name.endswith("_training_state.pt"):
                continue
            self.discard_snapshot(snapshot_path)
        for state_path in self.snapshot_dir.glob("snapshot_*_training_state.pt"):
            state_path.unlink(missing_ok=True)

    @staticmethod
    def discard_snapshot(snapshot_path: Path) -> None:
        snapshot_path.unlink(missing_ok=True)
        snapshot_path.with_name(f"{snapshot_path.stem}_training_state.pt").unlink(missing_ok=True)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return

            started_at = time.time()
            try:
                metrics = self.evaluator(job.snapshot_path, job.environment_parameters)
                result = AsyncEvaluationResult(
                    job=job,
                    started_at=started_at,
                    completed_at=time.time(),
                    metrics=metrics,
                )
            except Exception as exc:  # surfaced to the training thread as a failed eval result
                result = AsyncEvaluationResult(
                    job=job,
                    started_at=started_at,
                    completed_at=time.time(),
                    metrics=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._results.put(result)
