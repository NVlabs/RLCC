from collections import defaultdict
import os
import torch
import numpy as np
from baselines.common.vec_env import VecEnv
from config.config import Config
from collections import defaultdict


class BaseAgent:
    """
    Each agent should implement the act and train steps.
    Act is used for inference whereas the train loop is used for training the agent.
    """
    def __init__(self, config: Config, env: VecEnv):
        self.config: Config = config
        self.env: VecEnv = env
        self.logging_data = {}
        self.timesteps = 0
        self.log_key_hist = defaultdict(lambda: 0)
        self.save_path = self.config.env.save_path
        if self.save_path[-1] != '/':
            self.save_path += '/'

    def act(self, state: torch.tensor):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_model(self, checkpoint, loss=None):
        save_path = f'{self.save_path}{self.config.agent.save_name}/'
        name = ''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        checkpoint_name = str(checkpoint)
        if len(self.config.agent.save_name) > 0:
            name = name + '_checkpoint_' + checkpoint_name
            if loss is not None:
                name += f'_{loss:.4f}'
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path + self.config.agent.agent_type + name)

    def load_model(self):
        name = ''
        if len(self.config.agent.save_name) > 0:
            name = self.config.agent.save_name
        checkpoint = ''
        if len(self.config.agent.checkpoint) > 0:
            checkpoint = '_checkpoint_' + self.config.agent.checkpoint
            try:
                self.timesteps = int(self.config.agent.checkpoint)
            except:
                pass
        file_list = os.listdir(f'{self.save_path}' + name)
        filename = [f for f in file_list if self.config.agent.agent_type + checkpoint in f and '.txt' not in f]
        checkpoint_state_dict = torch.load(f'{self.save_path}' + name + '/' + filename[0])
        self.model.load_state_dict(checkpoint_state_dict['model_state_dict'])

    def test(self):
        raise NotImplementedError

    def log_data(self, timesteps, infos):
        if self.config.logging.wandb is not None:
            for env_info in infos:
                test, host, qp = env_info['key'].split('_')[-1].split('/')
                flow_limit_check = True
                if self.config.logging.limit_flows is not None:
                    flow_limit_check = (int(host) < self.config.logging.limit_hosts and int(qp) < self.config.logging.limit_qps)

                for key, value in env_info.items():
                    # env_info items to ignore during logging
                    if key not in ['flow_tag', 'host', 'qp', 'rtt_reward']:
                        if int(test) < self.config.logging.num_tests_to_log and flow_limit_check:
                            if key not in ['key']:
                                data_name = key + '/' + env_info['key']

                                if data_name not in self.logging_data:
                                    self.logging_data[data_name] = []
                                self.logging_data[data_name].append(value)

            if timesteps % self.config.logging.min_log_interval == 0:
                for key, value in self.logging_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value, torch.Tensor):
                            value = value.cpu()
                        self.config.logging.wandb.log({key: np.mean(value)}, step=timesteps)
                        del self.logging_data[key][:]
