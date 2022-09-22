import pytest
import torch
from config.config import Config
from tests.dummy_env import make_vec_env
from models.actor_critic import ActorCritic


def test_model():
    config = Config()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = make_vec_env(config)
    model = ActorCritic(env.observation_space, config)

    state, info = env.reset()

    value, action, action_log_probs = model(state)

    total_envs = config.env.envs_per_scenario * len(config.env.scenarios)

    assert value.shape[0] == total_envs and value.shape[1] == 1 and len(value.shape) == 2
    assert action.shape[0] == total_envs and action.shape[1] == 1 and len(action.shape) == 2
    assert action_log_probs.shape[0] == total_envs and action_log_probs.shape[1] == 1 and len(action_log_probs.shape) == 2

    value2, action_log_probs2, dist_entropy = model.evaluate(state, action)

    assert ((value2 - value) ** 2).mean().item() < 1e-10
    assert ((action_log_probs2 - action_log_probs) ** 2).mean().item() < 1e-10
    assert isinstance(dist_entropy.cpu().item(), float)


def test_discrete_model():
    config = Config()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.agent.ppo.discrete_actions = True

    env = make_vec_env(config)
    model = ActorCritic(env.observation_space, config)

    state, info = env.reset()

    value, action, action_log_probs = model(state)

    total_envs = config.env.envs_per_scenario * len(config.env.scenarios)

    assert value.shape[0] == total_envs and value.shape[1] == 1 and len(value.shape) == 2
    assert action.shape[0] == total_envs and action.shape[1] == 1 and len(action.shape) == 2
    assert action_log_probs.shape[0] == total_envs and action_log_probs.shape[1] == 1 and len(action_log_probs.shape) == 2

    value2, action_log_probs2, dist_entropy = model.evaluate(state, action)

    assert ((value2 - value) ** 2).mean().item() < 1e-10
    assert ((action_log_probs2 - action_log_probs) ** 2).mean().item() < 1e-10
    assert isinstance(dist_entropy.cpu().item(), float)
