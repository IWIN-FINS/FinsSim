"""Curriculum scheduling for Unity environment parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping


def normalize_environment_parameters(
    parameters: Mapping[str, Any] | None,
    *,
    context: str = "environment_parameters",
) -> dict[str, float]:
    """Validate and convert ML-Agents environment parameters to float values."""
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError(f"{context} must be a mapping")

    normalized: dict[str, float] = {}
    for key, value in parameters.items():
        if not isinstance(value, Real):
            raise TypeError(f"{context}.{key} must be numeric, got {type(value).__name__}")
        normalized[str(key)] = float(value)
    return normalized


@dataclass(frozen=True)
class CurriculumLesson:
    """A single curriculum lesson selected by global environment step."""

    name: str
    start_step: int
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CurriculumUpdate:
    """Result returned when the scheduler wants parameters to be sent."""

    lesson_index: int
    lesson_name: str
    start_step: int
    parameters: dict[str, float]
    changed_lesson: bool


class CurriculumScheduler:
    """Step-based scheduler for ML-Agents EnvironmentParametersChannel values."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        update_interval_steps: int = 1,
        base_parameters: Mapping[str, Any] | None = None,
        lessons: list[CurriculumLesson] | None = None,
    ):
        self.enabled = bool(enabled)
        self.update_interval_steps = max(1, int(update_interval_steps))
        self.base_parameters = normalize_environment_parameters(
            base_parameters,
            context="curriculum.base_parameters",
        )
        self.lessons = sorted(lessons or [], key=lambda lesson: lesson.start_step)
        self._validate_lessons()
        self._last_update_step: int | None = None
        self._last_lesson_index: int | None = None
        self._last_parameters: dict[str, float] | None = None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        base_parameters: Mapping[str, Any] | None = None,
    ) -> "CurriculumScheduler":
        """Create a scheduler from the `curriculum` section of a FinsSim YAML file."""
        if mapping is None:
            return cls(enabled=False, base_parameters=base_parameters)
        if not isinstance(mapping, Mapping):
            raise TypeError("curriculum must be a mapping")

        merged_base = normalize_environment_parameters(
            base_parameters,
            context="unity.environment_parameters",
        )
        merged_base.update(
            normalize_environment_parameters(
                mapping.get("base_parameters"),
                context="curriculum.base_parameters",
            )
        )

        lessons_raw = mapping.get("lessons") or []
        if not isinstance(lessons_raw, list):
            raise TypeError("curriculum.lessons must be a list")

        lessons: list[CurriculumLesson] = []
        for index, lesson_raw in enumerate(lessons_raw):
            if not isinstance(lesson_raw, Mapping):
                raise TypeError(f"curriculum.lessons[{index}] must be a mapping")
            name = str(lesson_raw.get("name", f"lesson_{index}"))
            start_step = int(lesson_raw.get("start_step", 0))
            if start_step < 0:
                raise ValueError(f"curriculum.lessons[{index}].start_step must be >= 0")
            parameters = normalize_environment_parameters(
                lesson_raw.get("parameters"),
                context=f"curriculum.lessons[{index}].parameters",
            )
            lessons.append(
                CurriculumLesson(
                    name=name,
                    start_step=start_step,
                    parameters=parameters,
                )
            )

        enabled = bool(mapping.get("enabled", bool(lessons)))
        update_interval_steps = int(mapping.get("update_interval_steps", 1))
        return cls(
            enabled=enabled,
            update_interval_steps=update_interval_steps,
            base_parameters=merged_base,
            lessons=lessons,
        )

    def _validate_lessons(self) -> None:
        previous_start = -1
        for index, lesson in enumerate(self.lessons):
            if lesson.start_step < previous_start:
                raise ValueError("curriculum lessons must be sorted by start_step")
            if index > 0 and lesson.start_step == previous_start:
                raise ValueError(
                    f"duplicate curriculum start_step={lesson.start_step}; "
                    "each lesson must start at a distinct step"
                )
            previous_start = lesson.start_step

    @property
    def active(self) -> bool:
        return self.enabled or bool(self.base_parameters)

    def lesson_for_step(self, step: int) -> tuple[int, CurriculumLesson | None]:
        """Return the active lesson index and lesson for a global training step."""
        if not self.lessons:
            return -1, None

        step = max(0, int(step))
        active_index = 0
        for index, lesson in enumerate(self.lessons):
            if step >= lesson.start_step:
                active_index = index
            else:
                break
        return active_index, self.lessons[active_index]

    def parameters_for_step(self, step: int) -> tuple[int, str, int, dict[str, float]]:
        """Return merged base + active lesson parameters for a global step."""
        lesson_index, lesson = self.lesson_for_step(step)
        parameters = dict(self.base_parameters)
        if self.enabled and lesson is not None:
            parameters.update(lesson.parameters)
            return lesson_index, lesson.name, lesson.start_step, parameters
        return -1, "base", 0, parameters

    def maybe_update(self, step: int, *, force: bool = False) -> CurriculumUpdate | None:
        """Return an update when a lesson changes or the update interval elapses."""
        lesson_index, lesson_name, start_step, parameters = self.parameters_for_step(step)
        changed_lesson = lesson_index != self._last_lesson_index
        changed_parameters = parameters != self._last_parameters
        interval_elapsed = (
            self._last_update_step is None
            or int(step) - self._last_update_step >= self.update_interval_steps
        )

        if not force and not changed_lesson and not changed_parameters and not interval_elapsed:
            return None

        self._last_update_step = int(step)
        self._last_lesson_index = lesson_index
        self._last_parameters = dict(parameters)
        return CurriculumUpdate(
            lesson_index=lesson_index,
            lesson_name=lesson_name,
            start_step=start_step,
            parameters=parameters,
            changed_lesson=changed_lesson,
        )
