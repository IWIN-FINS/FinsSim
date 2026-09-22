from __future__ import annotations

import numpy as np
import pytest

from finssim_marl.imitation.datasets import load_dataset, save_dataset, validate_dataset
from finssim_marl.imitation.role_bc import split_role_transitions
from finssim_marl.imitation.schemas import MultiAgentDataset


def _dataset() -> MultiAgentDataset:
    return MultiAgentDataset(
        obs=np.zeros((2, 4, 3, 5), dtype=np.float32),
        states=np.zeros((2, 4, 15), dtype=np.float32),
        actions=np.ones((2, 3, 3, 2), dtype=np.float32),
        rewards=np.zeros((2, 3, 3), dtype=np.float32),
        dones=np.zeros((2, 3), dtype=np.bool_),
        truncateds=np.zeros((2, 3), dtype=np.bool_),
        role_ids=np.array([0, 1, 1], dtype=np.int64),
        metadata={"scenario": "test"},
    )


def test_dataset_roundtrip(tmp_path):
    dataset = _dataset()
    output = save_dataset(dataset, tmp_path / "demo")
    loaded = load_dataset(output)

    assert loaded.metadata["scenario"] == "test"
    assert loaded.obs.shape == (2, 4, 3, 5)
    assert loaded.actions.shape == (2, 3, 3, 2)
    assert loaded.valid_steps.shape == (2, 3)


def test_dataset_validation_rejects_bad_time_dimension():
    dataset = _dataset()
    bad = MultiAgentDataset(
        obs=dataset.obs[:, :-1],
        states=dataset.states,
        actions=dataset.actions,
        rewards=dataset.rewards,
        dones=dataset.dones,
        truncateds=dataset.truncateds,
        role_ids=dataset.role_ids,
    )
    with pytest.raises(ValueError, match="T\\+1"):
        validate_dataset(bad)


def test_split_role_transitions_shares_netter_samples():
    transitions = split_role_transitions(_dataset())

    assert sorted(transitions) == [0, 1]
    assert transitions[0].obs.shape == (2 * 3 * 1, 5)
    assert transitions[1].obs.shape == (2 * 3 * 2, 5)
    assert transitions[1].actions.shape == (12, 2)


def test_split_role_transitions_ignores_padded_steps():
    dataset = MultiAgentDataset(
        obs=np.zeros((2, 4, 3, 5), dtype=np.float32),
        states=np.zeros((2, 4, 15), dtype=np.float32),
        actions=np.ones((2, 3, 3, 2), dtype=np.float32),
        rewards=np.zeros((2, 3), dtype=np.float32),
        dones=np.zeros((2, 3), dtype=np.bool_),
        truncateds=np.zeros((2, 3), dtype=np.bool_),
        role_ids=np.array([0, 1, 1], dtype=np.int64),
        valid_steps=np.array(
            [
                [True, True, True],
                [True, False, False],
            ],
            dtype=np.bool_,
        ),
    )

    transitions = split_role_transitions(dataset)

    assert transitions[0].obs.shape == (4, 5)
    assert transitions[1].obs.shape == (8, 5)
