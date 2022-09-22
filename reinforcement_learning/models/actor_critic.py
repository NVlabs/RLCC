import torch
from torch import nn
from gym import spaces
from typing import Tuple
from .model_utils import DiagGaussian, Categorical
from .mlp import MLP
from config.config import Config


class Actor(nn.Module):
    def __init__(self, observation_space: spaces.Box, config: Config):
        super(Actor, self).__init__()
        self.config = config

        self.net = MLP(
            input_size=observation_space.shape[0],
            output_size=self.config.agent.ppo.actor_architecture[-1],
            activation_function=self.config.agent.activation_function,
            hidden_sizes=self.config.agent.ppo.actor_architecture
        )

        if self.config.agent.ppo.discrete_actions:
            self.output_layer = Categorical(self.config.agent.ppo.actor_architecture[-1], len(self.config.agent.ppo.action_weights))
        else:
            self.output_layer = DiagGaussian(self.config.agent.ppo.actor_architecture[-1], 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x, _ = self.net(x)
        x = self.output_layer(x)
        return x


class Critic(nn.Module):
    def __init__(self, observation_space: spaces.Box, config: Config):
        super(Critic, self).__init__()
        self.config = config

        self.net = MLP(
            input_size=observation_space.shape[0],
            output_size=1,
            activation_function=self.config.agent.activation_function,
            hidden_sizes=self.config.agent.ppo.critic_architecture
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x, _ = self.net(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, observation_space: spaces.Box, config: Config):
        super(ActorCritic, self).__init__()
        self.config = config
        self.actor = Actor(observation_space, self.config)
        self.critic = Critic(observation_space, self.config)

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        value = self.critic(x)
        dist = self.actor(x)

        action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def evaluate(self, x: torch.tensor, action: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        value = self.critic(x)
        dist = self.actor(x)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    def act(self, x: torch.tensor) -> torch.tensor:
        dist = self.actor(x)
        action = dist.sample()
        return action
