import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from config.config import Config
from .base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, config: Config, env: VecEnv):
        BaseAgent.__init__(self, config, env)

    def act(self, state: torch.tensor):
        return self.env.action_space.sample()

    def test(self):
        self.env.reset()
        while True:
            self.env.step(self.act(None))
