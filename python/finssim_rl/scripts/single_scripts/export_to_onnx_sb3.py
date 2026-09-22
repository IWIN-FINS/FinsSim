import torch
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy


class OnnxableSB3Policy(torch.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)


# Example: model = PPO("MlpPolicy", "Pendulum-v1")
PPO("MlpPolicy", "Pendulum-v1").save("PathToTrainedModel")
model = PPO.load("PathToTrainedModel.zip", device="cpu")

onnx_policy = OnnxableSB3Policy(model.policy)

observation_size = model.observation_space.shape
dummy_input = torch.randn(1, *observation_size)
torch.onnx.export(
    onnx_policy,
    dummy_input,
    "my_ppo_model.onnx",
    opset_version=17,
    input_names=["input"],
)

##### Load and test with onnx

import onnx
import onnxruntime as ort
import numpy as np

onnx_path = "my_ppo_model.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

observation = np.zeros((1, *observation_size)).astype(np.float32)
ort_sess = ort.InferenceSession(onnx_path)
actions, values, log_prob = ort_sess.run(None, {"input": observation})

print(actions, values, log_prob)

# Check that the predictions are the same
with torch.no_grad():
    print(model.policy(torch.as_tensor(observation), deterministic=True))