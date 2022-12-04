import random
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from baselines.common.vec_env import VecEnv
from typing import List
from config.config import Config
from models.mlp import MLP
from .utils import flatten, Replay
from .base import BaseAgent


class Supervised(BaseAgent):
    """
    An online-learning supervised agent. The supervision signal is an action similar to what a rule-based ECN-based
    algorithm would do.
    """
    def __init__(self, config: Config, env: VecEnv):
        BaseAgent.__init__(self, config, env)
        self.model = MLP(
            input_size=env.observation_space.shape[0] - len(self.config.agent.agent_features),
            output_size=1,
            hidden_sizes=self.config.agent.supervised.architecture
        ).to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate, eps=1e-5)
        self.replay = Replay(config=self.config)

    def train(self) -> None:
        timesteps = 0

        state, info = self.env.reset()

        while timesteps < self.config.training.max_timesteps:
            action = self._policy(state[:, len(self.config.agent.agent_features):])
            state, _, done, info = self.env.step(self._parse_action(action))
            self.replay.add((state,))

            if self.config.logging.wandb is not None:
                for env_info in info:
                    for key, value in env_info.items():
                        if key not in ['agent_key']:
                            self.config.logging.wandb.log({env_info['agent_key'] + '/' + key: value}, step=timesteps)

            timesteps += state.shape[0]

            if len(self.replay.replay) >= self.config.agent.supervised.batch_size:
                loss = self._calculate_loss()

                if self.config.logging.wandb is not None:
                    self.config.logging.wandb.log({"Loss": loss}, step=timesteps)

    def _parse_action(self, actions: torch.tensor) -> List[float]:
        actions = flatten(actions.cpu().numpy())
        for i, action in enumerate(actions):
            if action < 0:
                action = 1. / (1. - action * self.config.agent.supervised.action_multiplier_dec)
            else:
                action = 1. + action * self.config.agent.supervised.action_multiplier_inc
            actions[i] = action
        return actions

    def _calculate_supervised_actions(self, state: torch.tensor) -> torch.tensor:
        actions = []
        for i in range(state.shape[0]):
            if state[i][self.config.agent.agent_features.index('nack_indicator')] > 0:
                actions.append(-1)
            elif state[i][self.config.agent.agent_features.index('cnp_ratio')] > 0:
                actions.append(-1 * min(state[i][self.config.agent.agent_features.index('cnp_ratio')] * 0.01, 1.))
            else:
                actions.append(max(1 - state[i][self.config.agent.agent_features.index('latency_mean_min_ratio')] * 0.01, -1))
        return torch.tensor(actions, dtype=torch.float32, device=self.config.device).unsqueeze(-1)

    def _policy(self, state: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            action, _ = self.model(state)
            action = torch.tanh(action)
        return action

    def _calculate_loss(self) -> float:
        state = self.replay.sample(self.config.agent.supervised.batch_size)[0].detach()

        action, _ = self.model(state[:, :-len(self.config.agent.agent_features)])
        action = torch.tanh(action)
        supervised_action = self._calculate_supervised_actions(state[:, -len(self.config.agent.agent_features):])

        # Compute Huber loss
        loss = F.smooth_l1_loss(action, supervised_action)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
        self.optimizer.step()

        return loss.item()

    def act(self, state: torch.tensor) -> torch.tensor:
        action, _ = self._policy(state[:, len(self.config.agent.agent_features):])
        action = torch.tanh(action)
        return self._parse_action(action)

    def test(self) -> None:
        timesteps = 0

        state, info = self.env.reset()

        with torch.no_grad():
            while True:
                action = self.act(state)
                state, reward, done, infos = self.env.step(self._parse_action(action))

                self.log_data(timesteps, infos)

                timesteps += state.shape[0]
