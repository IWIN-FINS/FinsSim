import numpy as np

from finssim_marl.envs.unity.wrapper import Unity3Chase1TrainingAreasEnv


class FakeParallelEnv:
    def __init__(self):
        self.possible_agents = {
            "Herder?team=0?agent_id=0",
            "Netter?team=1?agent_id=1",
            "Prey?team=3?agent_id=2",
            "Netter?team=2?agent_id=3",
            "Prey?team=3?agent_id=4",
            "Netter?team=1?agent_id=5",
            "Herder?team=0?agent_id=6",
            "Netter?team=2?agent_id=7",
        }
        self.agents = sorted(self.possible_agents)
        self.received_actions = None

    @staticmethod
    def _observation(agent):
        size = 15 if "Prey" in agent else 30
        return np.full(size, int(agent.rsplit("=", 1)[1]), dtype=np.float32)

    def reset(self):
        self.agents = sorted(self.possible_agents)
        return {agent: self._observation(agent) for agent in self.agents}

    def step(self, actions):
        self.received_actions = actions
        observations = {agent: self._observation(agent) for agent in self.agents}
        rewards = {agent: float(int(agent.rsplit("=", 1)[1])) for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        return observations, rewards, dones, {}

    def close(self):
        pass


def test_training_areas_groups_roles_and_scatters_actions():
    raw_env = FakeParallelEnv()
    env = Unity3Chase1TrainingAreasEnv(raw_env, num_areas=2)

    observations, states = env.reset()

    assert observations.shape == (2, 4, 30)
    assert states.shape == (2, 120)
    # Per-role sorted ML-Agents ids produce the two independent teams.
    assert observations[:, :, 0].tolist() == [[0.0, 1.0, 3.0, 2.0], [6.0, 5.0, 7.0, 4.0]]

    actions = np.zeros((2, 4, 8), dtype=np.float32)
    actions[0, 0, 0] = 0.1
    actions[0, 1, 0] = 0.2
    actions[0, 2, 0] = 0.3
    actions[1, 0, 0] = 0.4
    actions[1, 1, 0] = 0.5
    actions[1, 2, 0] = 0.6
    response = env.step(actions)

    assert np.isclose(raw_env.received_actions["Herder?team=0?agent_id=0"][0], 0.1)
    assert np.isclose(raw_env.received_actions["Netter?team=1?agent_id=1"][0], 0.2)
    assert np.isclose(raw_env.received_actions["Netter?team=2?agent_id=3"][0], 0.3)
    assert np.isclose(raw_env.received_actions["Herder?team=0?agent_id=6"][0], 0.4)
    assert np.isclose(raw_env.received_actions["Netter?team=1?agent_id=5"][0], 0.5)
    assert np.isclose(raw_env.received_actions["Netter?team=2?agent_id=7"][0], 0.6)
    assert response["next_obs"].shape == (2, 4, 30)
    assert response["reward"].tolist() == [4.0, 18.0]
