import random
import torch
import numpy as np
from collections import deque
from types import SimpleNamespace
from typing import List, Dict, Tuple
from config.config import Config


def random_sample(indices, batch_size):
    """
    Given a set of indices, returns an iterator that at each iteration provides a random permutation of the indices
    without resampling previously used indices.

    :param indices: List of indices.
    :param batch_size: Size of each returned batch.
    :return: returns an iterator over batch of indices.
    """
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def flatten(l):
    """
    Flattens the provided object.

    :param l: list of list of X.
    :return: list of X.
    """
    return [item for sublist in l for item in sublist]


class AsyncronousRollouts:
    """
    While previous works considered asyncronous interaction with multiple environments, the OMNeTpp env is itself
    asyncronous. The AsyncronousRollouts class enables proper accumulation of asyncronous information across multiple
    envs where each env is a multi-agent asyncronous env.
    """
    def __init__(self, config: Config):
        self.keys = ['value', 'action', 'action_log_probs', 'state', 'reward', 'mask']
        self.config = config
        self.rollout = dict()

    def _update_dict(self, env_info: Dict) -> str:
        """
        Make sure the rollouts dictionary has an entry for the current (env, host, qp) key.

        :param env_info: A dictionary containing the info returned from the environment for the current key.
        :return: The key of the (env, host, qp) combination.
        """
        instance_key = env_info['agent_key']
        if instance_key not in self.rollout:
            self.rollout[instance_key] = dict()

        return instance_key

    def add(self, transition_info: Dict, env_info: List[Dict], build_rollout_if_ready: bool) -> List:
        """
        Receives the state information and the environment info for the entire batch. This method maintains the rollouts
        and returns those that are ready.

        :param transition_info: Contains information such as state, reward, mask, value, action and action_log_probs.
            Each element in the dict is of size batch x dim.
        :param env_info: Contains the information provided by the environment. This is a list of size batch, where each
            entry contains information about a (env, host, qp) combination.
        :param build_rollout_if_ready: When set to True, returns a rollout if longer than rollout_length.
        :return: All rollouts that are ready (if requested).
        """
        rollout_data = []
        for i in range(len(env_info)):
            instance_key = self._update_dict(env_info[i])

            for data_key, data in transition_info.items():
                if data_key not in self.rollout[instance_key]:
                    self.rollout[instance_key][data_key] = []
                self.rollout[instance_key][data_key].append(data[i].unsqueeze(0))

            if build_rollout_if_ready and len(self.rollout[instance_key]['state']) > self.config.agent.ppo.rollout_length:
                ret_data = dict()
                ret_data['state'] = self.rollout[instance_key]['state'][:-1]
                ret_data['reward'] = self.rollout[instance_key]['reward'][1:]
                ret_data['mask'] = self.rollout[instance_key]['mask'][1:]
                ret_data['action'] = self.rollout[instance_key]['action'][:]
                ret_data['action_log_probs'] = self.rollout[instance_key]['action_log_probs'][:]
                ret_data['value'] = self.rollout[instance_key]['value'][:]
                ret_data = SimpleNamespace(**ret_data)

                del self.rollout[instance_key]['state'][:-1]
                del self.rollout[instance_key]['reward'][:-1]
                del self.rollout[instance_key]['mask'][:-1]
                del self.rollout[instance_key]['action'][:]
                del self.rollout[instance_key]['value'][:]
                del self.rollout[instance_key]['action_log_probs'][:]

                rollout_data.append((ret_data, self.rollout[instance_key]['state'][-1]))
        return rollout_data


class AsynchronousReplay:
    """
    As the environment is asynchronous, the replay maintains the last (s, a) for each (env, host, qp) combination and
    builds the transition once receiving the next (s', r, done).
    """
    def __init__(self, config: Config):
        self.keys = ['state', 'action', 'reward', 'next_state', 'mask']
        self.config = config
        self.partial_info = dict()
        self.replay = deque(maxlen=self.config.training.replay_size)

    @staticmethod
    def _get_key(env_info: Dict) -> str:
        return env_info['agent_key']

    def add_state_action(self, state_info: Dict, env_info: List[Dict]) -> None:
        """
        Given a state-action combination, adds them to the replay.

        :param state_info: Dictionary where each item is of size batch x dim.
        :param env_info: List of length batch where each element is an env info dictionary.
        """
        for i in range(len(env_info)):
            instance_key = self._get_key(env_info[i])

            if instance_key not in self.partial_info or self.partial_info[instance_key] is None:
                self.partial_info[instance_key] = dict(
                    state=state_info['state'][i].unsqueeze(0),
                    action=state_info['action'][i].unsqueeze(0)
                )
            else:
                self.partial_info[instance_key]['next_state'] = state_info['state'][i].unsqueeze(0)
                self.replay.append(self.partial_info[instance_key])

                self.partial_info[instance_key] = dict(
                    state=state_info['state'][i].unsqueeze(0),
                    action=state_info['action'][i].unsqueeze(0)
                )

    def add_reward_mask(self, state_info: Dict, env_info: List[Dict]) -> None:
        """
        Similar to the add_state_action, only here it adds the reward and mask.

        :param state_info: Dictionary where each item is of size batch x dim.
        :param env_info: List of length batch where each element is an env info dictionary.
        """
        for i in range(len(env_info)):
            instance_key = self._get_key(env_info[i])

            if instance_key not in self.partial_info or self.partial_info[instance_key] is None:
                pass 
            else:
                self.partial_info[instance_key]['reward'] = state_info['reward'][i].unsqueeze(0)
                self.partial_info[instance_key]['mask'] = state_info['mask'][i].unsqueeze(0)

    def sample(self, batch_size: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Samples a batch from the replay memory and returns it in a batched tensor form.

        :param batch_size: The number of transitions to return.
        :return: A batch of 5-tuple (s, a, r, s', done).
        """
        dict_batch = random.sample(self.replay, batch_size)
        state = torch.cat([dict_batch[i]['state'] for i in range(batch_size)])
        action = torch.cat([dict_batch[i]['action'] for i in range(batch_size)])
        reward = torch.cat([dict_batch[i]['reward'] for i in range(batch_size)])
        next_state = torch.cat([dict_batch[i]['next_state'] for i in range(batch_size)])
        mask = torch.cat([dict_batch[i]['mask'] for i in range(batch_size)])

        return state, action, reward, next_state, mask


class Replay:
    """
    A standard replay memory that receives the entire transition data at once.
    """
    def __init__(self, config: Config):
        self.config = config
        self.replay = deque(maxlen=self.config.training.replay_size)

    def add(self, data: Tuple[torch.tensor, ...]) -> None:
        """
        Receives a tuple of N elements where each element is batch x dim. Stores the data as batch tuples where each
        tuple contains N elements and each element is of size dim.
        :param data: A tuple of information to store. Each element is of size batch x dim.
        """
        for j in range(data[0].shape[0]):
            self.replay.append(tuple(data[i][j] for i in range(len(data))))

    def sample(self, batch_size: int) -> Tuple[torch.tensor, ...]:
        """
        Returns a batch of randomly sampled data.

        :param batch_size: The number of samples to randomly sample from memory.
        :return: A tuple of tensors. Each tensor is of size batch x dim.
        """
        tuple_batch = random.sample(self.replay, batch_size)
        if len(tuple_batch[0]) > 1:
            tensor_batch = (torch.stack([tuple_batch[i][j] for i in range(batch_size)]) for j in range(len(tuple_batch[0])))
        else:
            tensor_batch = (torch.stack([tup[0] for tup in tuple_batch]),)

        return tensor_batch


class KeySeparatedTemporalReplay:
    """
    Maintain the data for each key separately. The sample function returns a sequence (rollout).
    """
    def __init__(self, config: Config):
        self.config = config
        self.replay = dict()

    def add(self, data: Tuple[torch.tensor, ...], env_info: List[Dict]) -> None:
        for j in range(data[0].shape[0]):
            instance_key = AsynchronousReplay._get_key(env_info[j])
            if instance_key not in self.replay:
                self.replay[instance_key] = deque(maxlen=self.config.training.replay_size)
            self.replay[instance_key].append(tuple(data[i][j] for i in range(len(data))))

    def sample(self, batch_size: int, rollout_length: int) -> Tuple[torch.tensor, ...]:
        batch_data = None
        replay_keys = list(self.replay.keys())
        for _ in range(batch_size):
            key = random.sample(replay_keys, 1)[0]
            if batch_data is None:
                batch_items = len(self.replay[key][0])
                batch_data = [[] for _ in range(batch_items)]

            for i in range(batch_items):
                batch_data[i].append([])

            index = np.random.randint(low=0, high=len(self.replay[key]) - rollout_length, size=1)[0]
            for i in range(index, index+rollout_length):
                for j in range(batch_items):
                    batch_data[j][-1].append(self.replay[key][i][j])
            for j in range(batch_items):
                batch_data[j][-1] = torch.stack(batch_data[j][-1])

        tensor_batch = tuple(torch.stack(batch_data[i]) for i in range(batch_items))

        return tensor_batch
