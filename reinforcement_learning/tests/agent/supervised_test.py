import pytest
import torch
import numpy as np
from config.config import Config
from agents.supervised import Supervised
from tests.dummy_env import make_vec_env


def test_agent():
    config = Config()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.training.max_timesteps = 100
    config.agent.supervised.batch_size = 32
    config.logging.wandb = None
    config.env.scenarios = ['a', 'b']
    config.env.envs_per_scenario = 4

    env = make_vec_env(config)
    agent = Supervised(config, env)
    agent.train()
    env.close()
