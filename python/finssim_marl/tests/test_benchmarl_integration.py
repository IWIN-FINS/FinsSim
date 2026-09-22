from __future__ import annotations

from finssim_marl.integrations.benchmarl.registry import available_tasks
from finssim_marl.integrations.benchmarl.tasks import build_task_from_config


def test_benchmarl_task_registry_exposes_chasing_task():
    assert available_tasks() == ["chasing_3chase1"]


def test_benchmarl_task_reports_continuous_actions():
    task = build_task_from_config({"unity_env_binary_path": "/tmp/fake.x86_64"})

    assert task.supports_continuous_actions()
    assert not task.supports_discrete_actions()
    assert task.env_name() == "finssim"
