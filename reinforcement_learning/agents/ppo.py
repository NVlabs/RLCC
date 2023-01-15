import torch
from torch import optim
from torch import nn
import numpy as np
from typing import List, Tuple
from types import SimpleNamespace
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from config.config import Config
from models.actor_critic import ActorCritic
from .utils import random_sample, AsyncronousRollouts, flatten
from .base import BaseAgent


class PPO(BaseAgent):
    def __init__(self, config: Config, env: VecEnv):
        BaseAgent.__init__(self, config, env)
        self.model = ActorCritic(self.env.observation_space, self.config).to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate, eps=1e-5)
        self.rollout = AsyncronousRollouts(self.config)

        if self.config.agent.evaluate:
            self.load_model()

    def test(self) -> None:
        timesteps = 0

        state, info = self.env.reset()

        with torch.no_grad():
            while True:
                action = self.model.act(state)
                state, reward, done, infos = self.env.step(self._parse_action(action))

                self.log_data(timesteps, infos)

                timesteps += state.shape[0]

    def train(self) -> Tuple[float, float, float]:
        timesteps = 0

        state, info = self.env.reset()
        reward = done = torch.tensor([0. for _ in range(state.shape[0])])
        policy_loss = v_loss = entropy_loss = -1

        while timesteps < self.config.training.max_timesteps:
            rollouts = []
            final_states = []
            while len(rollouts) < self.config.agent.ppo.rollouts_per_batch:
                rollout = self.rollout.add(dict(state=state, reward=reward, mask=1. - done), info, True)
                if len(rollout) > 0:
                    for r in rollout:
                        rollouts.append(r[0])
                        final_states.append(r[1])

                value, action, action_log_probs = self.model(state)
                self.rollout.add(dict(value=value, action=action, action_log_probs=action_log_probs), info, False)

                state, reward, done, infos = self.env.step(self._parse_action(action.cpu().detach()))

                self.log_data(timesteps, infos)

                timesteps += state.shape[0]

            states, actions, log_probs, returns, advantages = self._process_data(rollouts, final_states)

            policy_loss, v_loss, entropy_loss = self._calculate_loss(states, actions, log_probs, returns, advantages)

            if self.config.logging.wandb is not None:
                self.config.logging.wandb.log(
                    {"Policy loss": policy_loss, "Value loss": v_loss, "Entropy loss": entropy_loss},
                    step=timesteps
                )
        return policy_loss, v_loss, entropy_loss

    def _parse_action(self, actions: torch.tensor) -> float:
        actions = actions.view(-1).numpy().tolist()
        for i, action in enumerate(actions):
            if self.config.agent.ppo.discrete_actions:
                action = self.config.agent.ppo.action_weights[action]
            else:
                action = np.tanh(action)
                if action < 0:
                    action = 1. / (1. - action * self.config.agent.ppo.action_multiplier_dec)
                else:
                    action = 1. + action * self.config.agent.ppo.action_multiplier_inc
            actions[i] = action
        return actions

    def _process_data(
            self,
            rollouts: List[SimpleNamespace],
            final_states: List[torch.tensor]
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:

        batch_size = len(final_states)

        final_states = torch.cat(final_states)
        with torch.no_grad():
            final_values = self.model.critic(final_states)

        states = []
        actions = []
        log_probs = []
        returns = []
        advantages = []

        for batch_index in range(batch_size):
            reward_to_go = final_values[batch_index]
            rollouts[batch_index].value.append(reward_to_go)

            states.append(rollouts[batch_index].state)
            actions.append(rollouts[batch_index].action)
            log_probs.append(rollouts[batch_index].action_log_probs)

            advantages.append([])
            returns.append([])

            adv = 0
            for i in reversed(range(len(actions[-1]))):
                reward_to_go = rollouts[batch_index].reward[i] + self.config.agent.discount * rollouts[batch_index].mask[i] * reward_to_go
                if not self.config.agent.ppo.use_gae:
                    adv = reward_to_go - rollouts[batch_index].value[i].detach()
                else:
                    td_error = rollouts[batch_index].reward[i] + self.config.agent.discount * rollouts[batch_index].mask[i] * rollouts[batch_index].value[i + 1] - rollouts[batch_index].value[i]
                    adv = td_error + adv * self.config.agent.ppo.gae_tau * self.config.agent.discount * rollouts[batch_index].mask[i]
                advantages[-1].insert(0, adv)
                returns[-1].insert(0, reward_to_go)

        states = torch.cat(flatten(states)).detach()
        actions = torch.cat(flatten(actions)).detach()
        log_probs = torch.cat(flatten(log_probs)).detach()
        returns = torch.cat(flatten(returns)).detach()
        advantages = torch.cat(flatten(advantages)).detach()

        return states, actions, log_probs, returns, advantages

    def _calculate_loss(
            self,
            states: torch.tensor,
            actions: torch.tensor,
            old_log_probs: torch.tensor,
            returns: torch.tensor,
            advantages: torch.tensor
    ) -> Tuple[float, float, float]:

        policy_loss = 0
        entropy_loss = 0
        value_loss = 0

        for _ in range(self.config.agent.ppo.params.ppo_optimization_epochs):
            sample = random_sample(np.arange(self.config.agent.ppo.rollout_length), self.config.agent.ppo.params.ppo_batch_size)
            for batch_indices in sample:
                batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=self.config.device)
                sampled_states, sampled_actions, sampled_old_log_probs, sampled_returns, sampled_advantages = (arr[batch_indices] for arr in [states, actions, old_log_probs, returns, advantages])

                v, log_probs, entropy = self.model.evaluate(sampled_states, sampled_actions)

                ratio = (log_probs - sampled_old_log_probs).exp()
                pg_obj1 = ratio * sampled_advantages
                pg_obj2 = ratio.clamp(1.0 - self.config.agent.ppo.params.ppo_ratio_clip, 1.0 + self.config.agent.ppo.params.ppo_ratio_clip) * sampled_advantages
                pg_loss = torch.min(pg_obj1, pg_obj2).mean()

                v_loss = 0.5 * torch.square(sampled_returns - v).mean()

                loss = - pg_loss - self.config.agent.ppo.entropy_coeff * entropy + self.config.agent.ppo.baseline_coeff * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.optimizer.step()

                policy_loss += pg_loss.item() / (self.config.agent.ppo.params.ppo_optimization_epochs * self.config.agent.ppo.params.ppo_batch_size)
                entropy_loss += entropy.item() / (self.config.agent.ppo.params.ppo_optimization_epochs * self.config.agent.ppo.params.ppo_batch_size)
                value_loss += v_loss.item() / (self.config.agent.ppo.params.ppo_optimization_epochs * self.config.agent.ppo.params.ppo_batch_size)

        return policy_loss, value_loss, entropy_loss

    def act(self, state: torch.tensor) -> torch.tensor:
        return torch.tanh(self.model.act(state))
