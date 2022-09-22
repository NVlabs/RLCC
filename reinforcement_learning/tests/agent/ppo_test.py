import pytest
import torch
import numpy as np
from config.config import Config
from agents.ppo import PPO
from tests.dummy_env import make_vec_env


def test_agent():
    config = Config()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.training.max_timesteps = 100
    config.logging.wandb = None
    config.agent.ppo.rollout_length = 10
    config.env.scenarios = ['a', 'b']
    config.env.envs_per_scenario = 4
    config.agent.discount = 0

    env = make_vec_env(config)
    agent = PPO(config, env)
    policy_loss, v_loss, entropy_loss = agent.train()
    env.close()


def test_discrete_agent():
    config = Config()
    config.agent.ppo.discrete_actions = True
    config.agent.ppo.action_weights = [0, 1, 2]
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.training.max_timesteps = 100000
    config.training.learning_rate = 1e-4
    config.logging.wandb = None
    config.agent.ppo.rollout_length = 5
    config.env.scenarios = ['a', 'b']
    config.env.envs_per_scenario = 4

    env = make_vec_env(config)
    agent = PPO(config, env)
    policy_loss, v_loss, entropy_loss = agent.train()
    env.close()

    # np.testing.assert_almost_equal(policy_loss, 0, decimal=4)
    # np.testing.assert_almost_equal(v_loss, 0, decimal=4)
    # assert entropy_loss == 0
