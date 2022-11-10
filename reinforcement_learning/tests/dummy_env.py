import gym
import numpy as np
from baselines.common.vec_env import VecEnv
from config.config import Config
from env.utils.env_utils import VecPyTorch, DummyVecEnvWithResetInfo, SubprocVecEnvWithResetInfo


def make_vec_env(config: Config) -> VecEnv:
    envs = [make_env(config, scenario, i) for scenario in config.env.scenarios for i in range(config.env.envs_per_scenario)]
    if len(envs) == 1:
        envs = DummyVecEnvWithResetInfo(envs)
    else:
        envs = SubprocVecEnvWithResetInfo(envs)

    envs = VecPyTorch(envs, device=config.device)

    return envs


def make_env(config, scenario, i):
    def _thunk():
        return DummyEnv(config, scenario, i)

    return _thunk


class DummyEnv(gym.Env):
    def __init__(self, config: Config, scenario: str, index: int):
        self.config = config
        self.scenario = scenario
        self.host = str(index)
        self.qp = '0'
        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.tile(-np.inf, len(self.config.agent.agent_features)),
                                                np.tile(np.inf, len(self.config.agent.agent_features)),
                                                dtype=np.float32)

    def seed(self, seed=None) -> None:
        pass

    def sample_random_features(self):
        return self.observation_space.sample()

    def reset(self):
        return np.zeros_like(self.sample_random_features()), dict(agent_key=self.scenario + ' ' + self.host + ' ' + self.qp)

    def step(self, action: float):
        state = np.zeros_like(self.sample_random_features())
        if action > 1:
            reward = .1
        else:
            reward = -.1
        return state, reward, False, dict(agent_key=self.scenario + ' ' + self.host + ' ' + self.qp)

    def render(self, mode: str = 'human') -> None:
        pass

    def close(self) -> None:
        pass
