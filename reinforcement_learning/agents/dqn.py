import random
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from baselines.common.vec_env import VecEnv
from ..config.config import Config
from ..models.mlp import MLP
from .utils import AsynchronousReplay, flatten
from .base import BaseAgent


class DQN(BaseAgent):
    """
    A standard DQN agent with an asyncronous experience replay (the ordering at which each flow is called, per env, is
    unknown).
    """
    def __init__(self, config: Config, env: VecEnv):
        BaseAgent.__init__(self, config, env)
        self.model = MLP(
            input_size=env.observation_space.shape[0],
            output_size=len(self.config.agent.dqn.action_weights),
            hidden_sizes=self.config.agent.dqn.architecture
        ).to(self.config.device)
        self.target_model = MLP(
            input_size=env.observation_space.shape[0],
            output_size=len(self.config.agent.dqn.action_weights),
            hidden_sizes=self.config.agent.dqn.architecture
        ).to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate, eps=1e-5)
        self.replay = AsynchronousReplay(config=self.config)

    def train(self) -> None:
        timesteps = 0

        assert self.config.agent.dqn.target_update_interval % (len(self.config.env.scenarios) * self.config.env.envs_per_scenario) == 0

        state, info = self.env.reset()

        while timesteps < self.config.training.max_timesteps:
            action = self._epsilon_greedy(state, timesteps)
            self.replay.add_state_action(dict(state=state, action=action), info)

            state, reward, done, infos = self.env.step(self._parse_action(action))
            self.replay.add_reward_mask(dict(reward=reward, mask=1. - done), info)

            self.log_data(timesteps, infos)

            timesteps += state.shape[0]

            if len(self.replay.replay) >= self.config.agent.dqn.batch_size:
                loss = self._calculate_loss()

                if self.config.logging.wandb is not None:
                    self.config.logging.wandb.log(
                        {"Loss": loss},
                        step=timesteps
                    )

            if timesteps % self.config.agent.dqn.target_update_interval == 0:
                self.target_model.load_state_dict(self.model.state_dict())

    def _parse_action(self, actions: torch.tensor) -> float:
        actions = flatten(actions.cpu().numpy())
        for i, action in enumerate(actions):
            actions[i] = self.config.agent.dqn.action_weights[action]
        return actions

    def _epsilon_greedy(self, state: torch.tensor, timesteps: int) -> torch.tensor:
        eps_threshold = self.config.agent.dqn.eps_end + (self.config.agent.dqn.eps_start - self.config.agent.dqn.eps_end) * \
                        math.exp(-1. * timesteps / self.config.agent.dqn.eps_decay)
        with torch.no_grad():
            actions = self.model(state).max(1)[1].view(-1, 1)
        for i in range(state.shape[0]):
            if random.random() > eps_threshold:
                actions[i] = random.randrange(len(self.config.agent.ppo.action_weights))
        return actions

    def _calculate_loss(
            self,
    ) -> float:
        state, action, reward, next_state, mask = self.replay.sample(self.config.agent.dqn.batch_size)

        state_action_values = self.model(state).gather(1, action)
        next_state_values = self.target_model(next_state).max(1)[0].unsqueeze(-1).detach()
        expected_state_action_values = reward + self.config.agent.discount * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
        self.optimizer.step()

        return loss.item()

    def act(self, state: torch.tensor) -> torch.tensor:
        return self.model.act(state)
