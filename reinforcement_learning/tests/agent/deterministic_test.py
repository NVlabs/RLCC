import pytest
import torch
import numpy as np
from config.config import Config
from agents.deterministic import Deterministic
from tests.dummy_env import make_vec_env


def test_agent():
    config = Config()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.training.max_timesteps = 100
    config.agent.deterministic.rollout_length = 20
    config.logging.wandb = None
    config.env.scenarios = ['a', 'b']
    config.env.envs_per_scenario = 2
    config.env.history_length = 1

    env = make_vec_env(config)
    agent = Deterministic(config, env)
    agent.train()
    env.close()
