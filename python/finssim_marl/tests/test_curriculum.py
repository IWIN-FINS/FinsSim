import pytest

from finssim_marl.training.curriculum import CurriculumScheduler, normalize_environment_parameters


def test_curriculum_scheduler_merges_base_and_active_lesson():
    scheduler = CurriculumScheduler.from_mapping(
        {
            "enabled": True,
            "base_parameters": {
                "finsim_3c1.capture.criterion": 2,
                "finsim_3c1.reward.chaser_capture": 20,
            },
            "lessons": [
                {
                    "name": "bootstrap",
                    "start_step": 0,
                    "parameters": {
                        "finsim_3c1.lesson_id": 0,
                    },
                },
                {
                    "name": "strict",
                    "start_step": 100,
                    "parameters": {
                        "finsim_3c1.lesson_id": 1,
                        "finsim_3c1.capture.criterion": 0,
                    },
                },
            ],
        },
        base_parameters={"finsim_3c1.prey.move_speed": 5},
    )

    index, name, start_step, params = scheduler.parameters_for_step(150)

    assert index == 1
    assert name == "strict"
    assert start_step == 100
    assert params == {
        "finsim_3c1.prey.move_speed": 5.0,
        "finsim_3c1.capture.criterion": 0.0,
        "finsim_3c1.reward.chaser_capture": 20.0,
        "finsim_3c1.lesson_id": 1.0,
    }


def test_curriculum_scheduler_updates_on_lesson_change_and_interval():
    scheduler = CurriculumScheduler.from_mapping(
        {
            "enabled": True,
            "update_interval_steps": 50,
            "lessons": [
                {"name": "l0", "start_step": 0, "parameters": {"lesson": 0}},
                {"name": "l1", "start_step": 100, "parameters": {"lesson": 1}},
            ],
        }
    )

    first = scheduler.maybe_update(0, force=True)
    assert first is not None
    assert first.changed_lesson
    assert first.lesson_name == "l0"

    assert scheduler.maybe_update(25) is None

    interval = scheduler.maybe_update(50)
    assert interval is not None
    assert not interval.changed_lesson
    assert interval.lesson_name == "l0"

    changed = scheduler.maybe_update(100)
    assert changed is not None
    assert changed.changed_lesson
    assert changed.lesson_name == "l1"


def test_normalize_environment_parameters_rejects_non_numeric_values():
    with pytest.raises(TypeError, match="must be numeric"):
        normalize_environment_parameters({"valid": 1.0, "bad": "slow"})


def test_curriculum_rejects_duplicate_start_steps():
    with pytest.raises(ValueError, match="duplicate curriculum start_step"):
        CurriculumScheduler.from_mapping(
            {
                "enabled": True,
                "lessons": [
                    {"name": "a", "start_step": 0},
                    {"name": "b", "start_step": 0},
                ],
            }
        )
