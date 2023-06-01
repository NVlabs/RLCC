"""
Deterministic model that runs a rollout and optimizes the action accordingly using a simple attractor deflector based on
the outcome.

Play K steps in each env and save rollouts.
Action should be an optimization task maximizing the value but minimizing the CNP/NACK packets.

The optimization can look over the entire rollout and optimize for some statistic or on a per-state.
"""

import torch
import os

from pathlib import Path
from torch import nn
from torch import optim
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from typing import List, Tuple
from config.config import Config
from .utils import flatten

from .base import BaseAgent
from collections import defaultdict

from models.mlp import MLP

import random

WARM_UP_PRINT = {True: 'warm-up stage', False: 'second stage'}


class ADPG(BaseAgent):
    def __init__(self, config: Config, env: VecEnv):
        BaseAgent.__init__(self, config, env)

        self.model = MLP(
            input_size=env.observation_space.shape[0],
            output_size=1,
            hidden_sizes=self.config.agent.adpg.architecture,
            use_rnn=self.config.agent.adpg.use_rnn,
            bias=self.config.agent.adpg.bias,
            device=self.config.device,
        ).to(self.config.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate, eps=1e-5)

        if self.config.agent.evaluate:
            self.load_model()
                
        if self.config.logging.wandb is not None:
            self.config.logging.wandb.watch(self.model, log=None, log_freq=1000)

    def _init_hidden(self) -> Tuple[torch.tensor, torch.tensor]:
        """
            :return: A tuple representing a newly initialized hidden LSTM state for a newly encountered agent.
        """
        return (
            torch.zeros(self.config.agent.adpg.architecture[-1], device=self.config.device),
            torch.zeros(self.config.agent.adpg.architecture[-1], device=self.config.device)
        )

    def test(self) -> None:
        """
            Evaluate the agent. Viewing the performance is performed either using the vector files created by the
            simulator, or by utilizing weights and biases logging.
            :return:
        """
        state, infos = self.env.reset()
        hc_dict = {}
        timesteps = 0
        print('Starting tests')
        with torch.no_grad():
            while True:
                hc = []
                for info in infos:
                    if info['agent_key'] in hc_dict:
                        hc.append(hc_dict[info['agent_key']])
                    else:
                        hc.append(self._init_hidden())

                h, c = zip(*hc)
                hc = (torch.stack(h), torch.stack(c))

                action, hc = self._policy(state, hc)

                for i, info in enumerate(infos):
                    hc_dict[info['agent_key']] = (hc[0][i], hc[1][i])

                state, _, done, infos = self.env.step(self._parse_action(action))

                if len(done) == 0:
                    break

                self.log_data(timesteps, infos)

                timesteps += state.shape[0]

        print('Finished tests')
        
    # def distil(self) -> None:

    def train(self) -> None:
        timesteps = self.timesteps + 1 if self.timesteps > 0 else 0
        state, infos = self.env.reset()
        hc_dict = {}

        num_updates = 1
        self.rollout = {}
        
        rollout_counter = {}
        print(20*'-')
        print(f'Initiating training')
        print(20*'-')

        warmup_updates = self.config.agent.adpg.warmup_updates
        warmup_length = self.config.agent.adpg.warmup_length

        print(f"Two stage policy update:\n1) warmup stage {warmup_updates} after {warmup_length} steps per env")
        print(f"2) second stage: policy update after a minimum of {self.config.agent.adpg.rollout_length} steps per agent")

        while num_updates <= self.config.training.max_num_updates:
            # Perform a rollout
            min_counter = 0
            # max_counter = 0
            warmup_step = 0

            is_warmup = num_updates < warmup_updates

            with torch.no_grad():
                while warmup_step < warmup_length if is_warmup else min_counter < self.config.agent.adpg.rollout_length:
                    hc = []
                    for info in infos:
                        if info['agent_key'] in hc_dict:
                            hc.append(hc_dict[info['agent_key']])
                        else:
                            hc.append(self._init_hidden())
                    h, c = zip(*hc)
                    hc = (torch.stack(h), torch.stack(c)) 
                    action, hc = self._policy(state, hc)

                    for i, info in enumerate(infos):
                        hc_dict[info['agent_key']] = (hc[0][i], hc[1][i])
                    for i, info in enumerate(infos):
                        if info['agent_key'] not in self.rollout:
                            self.rollout[info['agent_key']] = dict(state=[], action=[], reward=[])
                            rollout_counter[info['agent_key']] = 0
                        self.rollout[info['agent_key']]['state'].append(state[i])
                    
                    parsed_action = self._parse_action(action.detach())
                    state, reward, _, infos = self.env.step(parsed_action)
                
                    for i, info in enumerate(infos):
                        if info['agent_key'] in self.rollout:
                            self.rollout[info['agent_key']]['reward'].append(reward[i].detach().cpu().item())
                            rollout_counter[info['agent_key']] += 1
                        infos[i]['reward'] = reward[i].detach().cpu().item()

                    warmup_step += 1
                    min_counter = min(rollout_counter.values())
                    timesteps += 1 

                    self.log_data(timesteps, infos)
                    if warmup_step > self.config.agent.adpg.max_step_size:
                        print(f"break steps : {warmup_step}>{self.config.agent.adpg.max_step_size}")
                        break
            
            loss_stats = self._calculate_loss()

            if loss_stats is not None:
                reward_loss, action_loss, scenario_loss = loss_stats
                self.rollout = {}
                if self.config.logging.wandb is not None:
                    self.config.logging.wandb.log({"Loss": reward_loss + action_loss, "reward_loss": reward_loss, "action_loss": action_loss, "num_updates": num_updates}, step=timesteps)
                    for key in scenario_loss.keys():
                        self.config.logging.wandb.log({f"Loss_scenario_{key}": scenario_loss[key]}, step=timesteps)
                print(f"Policy Update {num_updates}/{self.config.training.max_num_updates} {WARM_UP_PRINT[is_warmup]}: after {timesteps} total timesteps. Loss: {(reward_loss + action_loss):.5f}")
                print(20*'-')
                num_updates += 1

                self.save_model(checkpoint=timesteps)
            else:
                print(f'Update failed: finished rollout at {timesteps} timesteps without any update')

        self.save_model(checkpoint=timesteps)

    def _parse_action(self, actions: torch.tensor) -> List[float]:
        """
            Convert the action from a network output value to a rate multiplier value accepted by the environment.

            :param actions: A tensor of size batch x 1.
            :return: List (batch elements) of multipliers. Each element represents how much the current rate should
                change.
        """
        actions = flatten(actions.cpu().numpy())
        for i, action in enumerate(actions):
            if action < 0:
                action = 1. / (1. - action * self.config.agent.adpg.action_multiplier_dec)
            else:
                action = 1. + action * self.config.agent.adpg.action_multiplier_inc
            actions[i] = action
        return actions

    def _policy(
            self,
            state: torch.tensor,
            hc: Tuple[torch.tensor, torch.tensor]
    ) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        """
            :param state: A batch of observations from the environments.
            :param hc: The hidden state of the LSTM (if applicable).
            :return: The action to be performed at this state and the updated LSTM hidden state (if applicable).
        """
        action, hc = self.model(state, hc)
        action = torch.tanh(action)
        return action, hc

    def _calculate_loss(self):
        """
            Calculate the loss for the adpg agent:
            Action smoothing loss is added to push sequential actions to be similar (stabilize the output).
            :return: Loss metrics.
        """
        agents = [key for key in list(self.rollout.keys()) if len(self.rollout[key]['reward']) > 1]
        if self.config.agent.adpg.loss_batch > 0:
            random.shuffle(agents)
            agents = random.sample(agents, min(self.config.agent.adpg.loss_batch, len(agents)))
        
        def get_loss(action, reward, batch_size, scenario, sl, a_sl, scenario_loss, gamma):
            """
                Memory efficiency -> tensors allocated within the scope will be de-allocated before we use backward()
            """
            if 0 < gamma < 1:
                   avg_reward_to_go = torch.stack([torch.stack([(r*(gamma**idx)) for idx, r in enumerate(reward[i:])], dim=0).mean() for i in range(batch_size)], dim=0).squeeze()
            else:
                avg_reward_to_go = torch.stack([(reward[i:]).mean() for i in range(batch_size)], dim=0).squeeze()
        
            flow_reward_loss = sl*(action * avg_reward_to_go).mean()

            action_diff = torch.stack([((a_sl*0.5*(action[i] - action[i+1:])**2).sum()) / len(action[i:]) for i in range(batch_size) if len(action[i+1:]) > 0  ], dim=0).sum()
            flow_action_loss = action_diff / batch_size

            # For visualization, we separate the loss by scenario
            scenario_loss[scenario] += flow_reward_loss.detach().item()
            if batch_size > 1:
                scenario_loss[scenario] += flow_action_loss.detach().item()
 
            return flow_reward_loss, flow_action_loss
        
        scenario_loss = defaultdict(lambda: 0)
        scenario_loss_agents = defaultdict(lambda: 0)

        total_reward_loss = 0
        total_action_loss = 0
        num_agents = len(agents)
        if num_agents == 0:
            return None

        self.optimizer.zero_grad()

        # ADPG considers multiple agents sharing the same control policy.
        # This loop aggregates the losses over agents, performing a joint optimization step.
        for agent_key in agents:
            agent_rollout = self.rollout[agent_key]

            scenario, _, _ = agent_key.split('/')
            rewards = torch.tensor(agent_rollout['reward'], device=self.config.device)
            
            batch_size = min(len(rewards), self.config.agent.adpg.max_batch_size, len(agent_rollout['state']))
            rewards = rewards[:batch_size]

            if self.config.agent.adpg.use_rnn:
                hc = None
                actions = []
                for state in agent_rollout['state'][:batch_size]:
                    action, hc = self.model(state, hc)
                    actions.append(action)
                actions = torch.stack(actions, dim=0)
            else:
                states = torch.stack(agent_rollout['state'][:batch_size], dim=0)
                actions, _ = self.model(states, None)

            actions = torch.tanh(actions).squeeze()

            scenario_loss_agents[scenario] += 1

            action_loss_coeff = self.config.agent.adpg.action_loss_coeff
            reward_loss_coeff = self.config.agent.adpg.reward_loss_coeff
            
            agent_reward_loss, agent_action_loss = get_loss(actions, rewards, batch_size, scenario, reward_loss_coeff, action_loss_coeff, scenario_loss, self.config.agent.discount)
            agent_reward_loss = agent_reward_loss / num_agents
            agent_action_loss = agent_action_loss / num_agents

            (agent_reward_loss + agent_action_loss).backward()

            total_reward_loss += agent_reward_loss.detach().item()
            total_action_loss += agent_action_loss.detach().item()

        if self.config.training.gradient_clip > 0:
            nn.utils.clip_grad_value_(self.model.parameters(), self.config.training.gradient_clip)
        self.optimizer.step()
        
        for key in scenario_loss_agents.keys():
            scenario_loss[key] = scenario_loss[key] / num_agents

        return total_reward_loss, total_action_loss, scenario_loss
