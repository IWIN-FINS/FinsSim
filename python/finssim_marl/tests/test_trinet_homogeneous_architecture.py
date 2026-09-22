import numpy as np
import torch

from finssim_marl.algorithms.networks.trinet import (
    TriNetSharedPositionActor,
    build_trinet_critic,
)
from finssim_marl.algorithms.trinet_capture_mappo import (
    TriNetCaptureMAPPOAlgorithm,
    TriNetCaptureMAPPOConfig,
)


def test_actor_is_invariant_to_teammate_slot_swap():
    actor = TriNetSharedPositionActor()
    observation = torch.randn(4, 3, 34)
    swapped = observation.clone()
    swapped[..., 14:20] = observation[..., 20:26]
    swapped[..., 20:26] = observation[..., 14:20]
    action, _ = actor.act(observation, deterministic=True)
    swapped_action, _ = actor.act(swapped, deterministic=True)
    assert torch.allclose(action, swapped_action, atol=1e-6)


def test_permutation_aware_critics_keep_team_value_under_rov_reorder():
    state = torch.randn(4, 57)
    reordered = state.clone()
    reordered[:, :45] = state[:, :45].reshape(4, 3, 15)[:, [2, 0, 1]].reshape(4, 45)
    for variant in ("trinet_deepsets", "trinet_attention"):
        critic = build_trinet_critic(variant)
        value, _ = critic(state)
        reordered_value, _ = critic(reordered)
        assert torch.allclose(value, reordered_value, atol=1e-5), variant


def test_flat_critic_is_a_separate_order_sensitive_ablation():
    critic = build_trinet_critic("trinet_flat_mlp")
    assert critic.global_encoder[0].in_features == 57


def test_dual_value_update_accepts_team_and_individual_returns():
    config = TriNetCaptureMAPPOConfig(device="cpu", controller_backend="zero", epochs=1)
    algorithm = TriNetCaptureMAPPOAlgorithm(
        obs_dim=34,
        action_dim=8,
        state_dim=57,
        n_agents=3,
        role_ids=np.zeros(3, dtype=np.int64),
        config=config,
    )
    episodes, steps = 2, 3
    batch = (
        torch.randn(episodes, steps, 3, 34),
        torch.tanh(torch.randn(episodes, steps, 3, 4)),
        torch.zeros(episodes, steps, 3),
        torch.randn(episodes, steps),
        torch.randn(episodes, steps, 57),
        torch.zeros(episodes, steps),
        torch.ones(episodes, steps, dtype=torch.bool),
        torch.zeros(episodes, 3, dtype=torch.long),
        torch.randn(episodes, steps, 3 * 34),
        torch.randn(episodes, steps, 3),
    )
    metrics = algorithm.update(batch)
    assert np.isfinite(metrics["team_critic_loss"])
    assert np.isfinite(metrics["individual_critic_loss"])
    assert metrics["advantage_std"] > 0.0
