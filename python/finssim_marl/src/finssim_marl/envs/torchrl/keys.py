"""Shared TensorDict keys for FinsSim MARL TorchRL integration."""

AGENTS = "agents"
OBSERVATION = "observation"
ACTION = "action"
REWARD = "reward"
ROLE_ID = "role_id"
STATE = "state"
DONE = "done"
TERMINATED = "terminated"
TRUNCATED = "truncated"

AGENT_OBS_KEY = (AGENTS, OBSERVATION)
AGENT_ACTION_KEY = (AGENTS, ACTION)
AGENT_REWARD_KEY = (AGENTS, REWARD)
AGENT_ROLE_ID_KEY = (AGENTS, ROLE_ID)
