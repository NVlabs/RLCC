import torch
import numpy as np
from multiprocessing import Process, Pipe
import gym
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from baselines.common.vec_env.subproc_vec_env import worker
from env import OMNeTpp
from config.config import Config


def make_vec_env(config: Config) -> VecEnv:
    """
    Create a vector list of environments based on the configuration provided.

    :param config: A config object containing all the test parameters and requirements.
    :return: A vectorized environment.
    """
    envs = []
    i = 0
    for scenario in config.env.scenarios:
        num_envs = config.env.envs_per_scenario
        for j in range(num_envs):
            envs.append(make_env(scenario, i, j, config))
            i += 1

    if True:  
        envs = DummyVecEnvWithResetInfo(envs)
    else:
        envs = SubprocVecEnvWithResetInfo(envs)

    envs = VecPyTorch(envs, device=config.device)

    return envs


def make_env(scenario: str, index: int, env_index: int, config: Config):
    def _thunk():
        return OMNeTpp.OMNeTpp(scenario, index, env_index, config)

    return _thunk


class VecPyTorch(VecEnvWrapper):
    """
    Convert state, reward and done signals from numpy arrays into torch tensors and place on the relevant device.
    """
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.tensor(np.array(obs)).squeeze(0).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.tensor(np.array(obs)).squeeze(0).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        done = torch.from_numpy(done).unsqueeze(dim=1).float().to(self.device)
        return obs, reward, done, info


class DummyVecEnvWithResetInfo(VecEnv):
    """
    Tweaked the original VecEnv to support async envs that return information on reset.
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns] #output a list of omnetpp obj
        env = self.envs[0]  #start training from env 0 (first env)
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

        obs_spaces = self.observation_space.spaces if isinstance(self.observation_space, gym.spaces.Tuple) else (self.observation_space,)  #TODO what this checks? isinstance(self.observation_space, gym.spaces.Tuple)
        self.buf_obs = [np.zeros((self.num_envs,) + tuple(s.shape), s.dtype) for s in obs_spaces]
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool) #TODO what it would be used for?
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32) #TODO what it would be used for?
        self.buf_infos = [{} for _ in range(self.num_envs)] #TODO what it would be used for?
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for i in range(self.num_envs):
            # ignore = True
            # j = 0 # try counter
            # while ignore and j < 10: #if ignorable flow than just perform another step until we get non-ignorable flow
            data = self.envs[i].step(self.actions[i])
            if len(data) == 4:
                obs_tuple, self.buf_rews[i], self.buf_dones[i], self.buf_infos[i] = data
            else:
                obs_tuple, self.buf_infos[i] = data # reset
                self.buf_rews[i] = 0
            if isinstance(obs_tuple, (tuple, list)):
                for t, x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple
        return self.buf_obs, self.buf_rews, self.buf_dones, self.buf_infos

    def reset(self):
        for i in range(self.num_envs):
            obs_tuple, self.buf_infos[i] = self.envs[i].reset()
            if isinstance(obs_tuple, (tuple, list)):
                for t, x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple
        return self.buf_obs, self.buf_infos

    def close(self):
        for i in range(self.num_envs):
            self.envs[i].close()


class SubprocVecEnvWithResetInfo(VecEnv):
    """
    Tweaked the original VecEnv to support async envs that return information on reset.
    """
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
