import pytest
import torch
from config.config import Config
from agents.dqn import DQN
from tests.dummy_env import make_vec_env


def test_agent():
    config = Config()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.training.max_timesteps = 1000
    config.logging.wandb = None
    config.env.scenarios = ['a', 'b']
    config.env.envs_per_scenario = 4
    config.agent.dqn.target_update_interval = 16

    env = make_vec_env(config)
    agent = DQN(config, env)
    agent.train()
    env.close()
