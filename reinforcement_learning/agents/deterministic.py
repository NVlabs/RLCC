"""
Deterministic model that runs a rollout and optimizes the action accordingly using a simple attractor deflector based on
the outcome.

Play K steps in each env and save rollouts.
Action should be an optimization task maximizing the value but minimizing the CNP/NACK packets.

The optimization can look over the entire rollout and optimize for some statistic or on a per-state.
"""

import numpy as np
import torch
import os
from pathlib import Path
from torch import nn
from torch import optim
from baselines.common.vec_env import VecEnv
from typing import List, Tuple
from config.config import Config
from .utils import flatten
# from .base import BaseAgent, ROOT_PATH
from .base import BaseAgent
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from models.mlp import MLP
from models.mlp_quant import MLPQ
import random


class Deterministic(BaseAgent):
    def __init__(self, config: Config, env: VecEnv): #TODO module it,make it simple to follow
        BaseAgent.__init__(self, config, env)
        if self.config.agent.m_quantization or self.config.agent.quantization:
            self.config_quantization()
            from models.mlp_amax import MLPAMAX
        if self.config.agent.m_quantization:
            self.config_m_quant_model(env)
        else:
            mlp_model = MLP if not (self.config.agent.quantization and self.config.quantization.lstm_LUT) else MLPAMAX
            self.model = mlp_model(
                input_size=env.observation_space.shape[0],
                output_size=1,
                # activation_function=self.config.agent.activation_function,
                hidden_sizes=self.config.agent.deterministic.architecture,
                use_rnn=self.config.agent.deterministic.use_rnn,
                bias=self.config.agent.deterministic.bias,
                lrelu_coeff=self.config.agent.deterministic.leaky_relu,
                device=self.config.device,
            ).to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate, eps=1e-5)
            # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.training.learning_rate)
            if self.config.agent.evaluate or self.config.quantization.fine_tune or self.config.agent.quantization:
                self.load_model()
            if self.config.agent.quantization and self.config.quantization.lstm_LUT:
                self.config_quantization_model()
                
        save_path = f'{self.save_path}{self.config.agent.save_name}/info/'
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True)
        self.writer = SummaryWriter(f'{save_path}')

        if self.config.logging.wandb is not None:
            # self.config.logging.wandb.watch(self.model, log='all')
            self.config.logging.wandb.watch(self.model, log=None, log_freq=1000)

    def config_quantization_model(self):
        lstm_model = torch.load('../../reinforcement_learning/saved_models/' + self.config.agent.agent_type + '_' + self.config.agent.save_name)
        self.model.lstm.weight_ih = torch.nn.Parameter(lstm_model['lstm.weight_ih'])
        self.model.lstm.weight_hh = torch.nn.Parameter(lstm_model['lstm.weight_hh'])
        self.model.net[0].weight = torch.nn.Parameter(lstm_model['net.0.weight'])
        self.model.output_layer.weight = torch.nn.Parameter(lstm_model['output_layer.weight'])

    def config_quantization(self):
        self._import_rel_packages()
        #Set default QuantDescriptor to use histogram based calibration for activation
        quant_desc_input = self.QuantDescriptor(calib_method='histogram')
        self.quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        if self.config.agent.deterministic.use_lstm:
            self.quant_nn.QuantLSTMCell.set_default_quant_desc_input(quant_desc_input)
        self.quant_modules.initialize()

    def config_m_quant_model(self, env):
            quant_model = torch.load('../../reinforcement_learning/saved_models/' + self.config.agent.agent_type + '_' + self.config.agent.save_quant_name)
            self.model = MLPQ(
                input_size=env.observation_space.shape[0],
                output_size=1,
                activation_function=self.config.agent.activation_function,
                hidden_sizes=self.config.agent.deterministic.architecture,
                use_lstm=self.config.agent.deterministic.use_lstm
            ).to(self.config.device)
            self.model.param_dict = {key:value.to(self.config.device) for key,value in quant_model.items()}
            self.model.param_dict['net.0.weight'] = self.quant_tensor(self.model.param_dict['net.0.weight'], self.model.param_dict['net.0._weight_quantizer._amax']).to('cpu') #TODO understand why it is on gpu?
            self.model.param_dict['output_layer.weight'] =  self.quant_tensor(self.model.param_dict['output_layer.weight'], self.model.param_dict['output_layer._weight_quantizer._amax']).to('cpu')
            self.model.param_dict['lstm.weight_ih'] =  self.quant_tensor(self.model.param_dict['lstm.weight_ih'],  self.model.param_dict['lstm._weight_quantizer._amax']).to('cpu')
            self.model.param_dict['lstm.weight_hh'] =  self.quant_tensor(self.model.param_dict['lstm.weight_hh'],  self.model.param_dict['lstm._weight_quantizer._amax']).to('cpu')


    def quant_tensor(self, input, amax): #TODO redundant see how to change
        scale = 127/amax
        output = torch.clamp((input * scale).round_(), -127, 127)
        return output.type(torch.int8)

    def _import_rel_packages(self):
        import sys
        sys.path.append(self.config.quantization.quantization_path)
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization import calib
        from pytorch_quantization.tensor_quant import QuantDescriptor
        from pytorch_quantization import quant_modules
        self.quant_modules = quant_modules
        self.quant_nn = quant_nn
        self.calib = calib
        self.QuantDescriptor = QuantDescriptor


    def _init_hidden(self) -> Tuple[torch.tensor, torch.tensor]:
        """
        :return: A tuple representing a newly initialized hidden LSTM state for a newly encountered agent.
        """
        return (
            torch.zeros(self.config.agent.deterministic.architecture[-1], device=self.config.device),
            torch.zeros(self.config.agent.deterministic.architecture[-1], device=self.config.device)
        )

    def calibration(self) -> None: #TODO the pretraining on all 3 env? like we are doing in the traning
        with torch.no_grad():
            #gather stats for quantization
            self.collect_stats()
            if self.config.quantization.quantization_method == 'percentile':
                self.compute_quantization(method="percentile", percentile=99.9)
            else:
                self.compute_quantization(method=self.config.quantization.quantization_method)
            self.save_model(checkpoint=0)


    def compute_quantization(self, **kwargs):
        # Load calib result
        for name, module in self.model.named_modules():
            if isinstance(module, self.quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, self.calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
                print(F"{name:40}: {module}")
        # model.cuda()

    def collect_stats(self):
        #Enable calibration
        state, infos = self.env.reset()
        hc_dict = {}

        timesteps = self.timesteps
        max_timesteps = self.config.quantization.max_timesteps + timesteps

        for name, module in self.model.named_modules():
            if isinstance(module, self.quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                    if isinstance(module._calibrator, self.calib.HistogramCalibrator):
                        module._calibrator._num_bins = self.config.quantization.num_bins
                else:
                    module.disable()
        #gather stats data
        while timesteps < max_timesteps:
            hc = []
            for info in infos:
                if info['key'] in hc_dict:
                    hc.append(hc_dict[info['key']])
                else:
                    hc.append(self._init_hidden())

            h, c = zip(*hc)
            hc = (torch.stack(h), torch.stack(c)) #TODO understand why we do this?

            action, hc = self._policy(state, hc)

            for i, info in enumerate(infos):
                hc_dict[info['key']] = (hc[0][i], hc[1][i])

            state, __, done, infos = self.env.step(self._parse_action(action))
            self.log_data(timesteps, infos)
            # timesteps += state.shape[0]  #TODO understand how it reached 1024 without getting stopped
            timesteps += 1 #TODO understand how it reached 1024 without getting stopped
        #disable calibration (gather of stats)
        for name, module in self.model.named_modules():
            if isinstance(module, self.quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()


    def test(self) -> None:
        state, infos = self.env.reset()
        hc_dict = {}
        timesteps = 0

        with torch.no_grad():
            while True:
                hc = []
                for info in infos:
                    if info['key'] in hc_dict:
                        hc.append(hc_dict[info['key']])
                    else:
                        hc.append(self._init_hidden())

                h, c = zip(*hc)
                hc = (torch.stack(h), torch.stack(c))

                action, hc = self._policy(state, hc)

                for i, info in enumerate(infos):
                    hc_dict[info['key']] = (hc[0][i], hc[1][i])

                state, _, done, infos = self.env.step(self._parse_action(action))

                self.log_data(timesteps, infos)

                timesteps += state.shape[0]
            print(state, _, done, infos)

    def train(self) -> None:
        timesteps = self.timesteps + 1 if self.timesteps > 0 else 0

        state, infos = self.env.reset()
        hc_dict = {}

        num_updates = 1
        self.rollout = {}
        
        flows = [env_i.nb_flows for env_i in self.env.envs]
        max_num_flows = max(flows)
        total_num_flows = sum(flows)
        num_envs = len(self.env.envs)
        
        rollout_counter = {}

        warmup_updates = self.config.agent.deterministic.warmup_updates
        warmup_length = self.config.agent.deterministic.warmup_length

        print(f'Update policy after a minimum of : {self.config.agent.deterministic.rollout_length} steps per flow after a warmup of: {warmup_updates} updates of {warmup_length} total steps')
        while num_updates <= self.config.training.max_num_updates:
            # Perform a rollout
            # steps_per_agent = 0
            min_counter = 0
            # max_counter = 0
            warmup_step = 0

            is_warmup = num_updates < warmup_updates

            with torch.no_grad():
                while warmup_step < warmup_length if is_warmup else min_counter < self.config.agent.deterministic.rollout_length:
                    hc = []
                    for info in infos:
                        if info['key'] in hc_dict:
                            hc.append(hc_dict[info['key']])
                        else:
                            hc.append(self._init_hidden())

                    h, c = zip(*hc)
                    hc = (torch.stack(h), torch.stack(c)) 
                    # print(state)
                    action, hc = self._policy(state, hc)

                    for i, info in enumerate(infos):
                        hc_dict[info['key']] = (hc[0][i], hc[1][i])

                    for i, info in enumerate(infos):
                        if info['key'] not in self.rollout:
                            self.rollout[info['key']] = dict(state=[], action=[], reward=[], num_flows=info['num_flows'])
                            rollout_counter[info['key']] = 0
                        self.rollout[info['key']]['state'].append(state[i])
                    
                    parsed_action = self._parse_action(action.detach())

                    state, reward, done, infos = self.env.step(parsed_action)
                    
                    for i, info in enumerate(infos):
                        if info['key'] in self.rollout:
                            self.rollout[info['key']]['reward'].append(reward[i].detach().cpu().item())
                            # self.rollout[info['key']]['num_flows'] = info['num_flows']
                            rollout_counter[info['key']] += 1
                        infos[i]['unparsed_action'] = action[i].detach().cpu().item()
                        infos[i]['reward'] = reward[i].detach().cpu().item()
                        # print(f"key: {info['key']} reward: {infos[i]['reward']}")

                    timesteps += 1  #TODO understand how it reached 1024 without getting stopped 
                    warmup_step += 1
                    min_counter = min(rollout_counter.values())
                    # max_counter = max(rollout_counter.values())

                    self.log_data(timesteps, infos)
                    if warmup_step > self.config.agent.deterministic.max_step_size:
                        print(f"break steps : {warmup_step}>{self.config.agent.deterministic.max_step_size}")
                        break

            agent_steps = [len(self.rollout[agent_key]['reward']) for agent_key in self.rollout.keys() if len(self.rollout[agent_key]['reward']) > 0]
            print(f"Policy Update: {num_updates}/{self.config.training.max_num_updates} after {timesteps} total timesteps and {warmup_step} simulator steps -- steps per agent min {min(agent_steps)} max {max(agent_steps)} mean: {np.mean(agent_steps)} std: {np.std(agent_steps)}")
            loss_stats = self._calculate_loss(timesteps)


            if loss_stats is not None:
                reward_loss, action_loss, scenario_loss, num_agents = loss_stats
                print(f"Updated based on: {num_agents} / {total_num_flows} agents")

                for name, weight in self.model.named_parameters():
                    self.writer.add_histogram(name,weight, timesteps)
                    self.writer.add_histogram(f'{name}.grad', weight.grad, timesteps)

                self.rollout = {}

                if self.config.logging.wandb is not None:
                    self.config.logging.wandb.log({"Loss": reward_loss + action_loss, "reward_loss": reward_loss, "action_loss": action_loss, "num_updates": num_updates}, step=timesteps)
                    for key in scenario_loss.keys():
                        self.config.logging.wandb.log({f"Loss_scenario_{key}": scenario_loss[key]}, step=timesteps)
                num_updates += 1
                self.save_model(checkpoint=timesteps, loss=reward_loss + action_loss)
            else:
                print(f'Update failed: finished rollout at {timesteps} timesteps without any update')

        self.save_model(checkpoint=timesteps, loss=reward_loss + action_loss)


    def _parse_action(self, actions: torch.tensor) -> List[float]:
        """
        Convert the action from a network output value to a rate multiplier value accepted by the environment.

        :param actions: A tensor of size batch x 1.
        :return: List (batch elements) of multipliers. Each element represents how much the current rate should change.
        """
        actions = flatten(actions.cpu().numpy())
        for i, action in enumerate(actions):
            if action < 0:
                action = 1. / (1. - action * self.config.agent.deterministic.action_multiplier_dec)
            else:
                action = 1. + action * self.config.agent.deterministic.action_multiplier_inc
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
        action, hc = self.model(state, hc) #TODO need to pass float32 or anything else in the end so it would add it to tanh
        # action = torch.tanh(action)  #factor the output to be between -1 to 1
        return action, hc


    def _calculate_loss(self, timesteps, eps=1e-8) -> float:
        """
        Calculate the loss for the deterministic agent:
        Action smoothing loss is added to push sequential actions to be similar (stabilize the output).
        :return: The loss (scalar value).
        """
        agents = [key for key in list(self.rollout.keys()) if len(self.rollout[key]['reward']) > 1]
        if self.config.agent.deterministic.loss_batch > 0:
            random.shuffle(agents)
            agents = random.sample(agents, min(self.config.agent.deterministic.loss_batch, len(agents)))

        
        def get_loss(action, reward, batch_size, scenario, sl, a_sl, scenario_loss, gamma):
            """
            Memory efficiency -> tensors allocated within the scope will be de-allocated before we use backward()
            """
            if gamma != 0 and gamma != 1:
                   avg_reward_to_go = torch.stack([torch.stack([(r*(gamma**idx)) for idx, r in enumerate(reward[i:])], dim=0).mean() for i in range(batch_size)], dim=0).squeeze()
            else:
                avg_reward_to_go = torch.stack([(reward[i:]).mean() for i in range(batch_size)], dim=0).squeeze()
            # avg_reward_to_go = torch.where(avg_reward_to_go > 0, avg_reward_to_go, avg_reward_to_go * 10)

            flow_reward_loss = sl*(action * avg_reward_to_go).mean()

            action_diff = torch.stack([((a_sl*0.5*(action[i] - action[i+1:])**2).sum()) / len(action[i:]) for i in range(batch_size) if len(action[i+1:]) > 0  ], dim=0).sum()
            flow_action_loss = action_diff / batch_size# number of elements

            scenario_loss[scenario] += flow_reward_loss.detach().item() # classifiy reward by scenario for visualization
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

        total_steps = np.sum([len(self.rollout[agent_key]['reward']) for agent_key in agents])
        self.optimizer.zero_grad()

        
        for agent_key in agents:
            # print(f"calculating loss for agent: {agent_key}")
            agent_rollout = self.rollout[agent_key]

            scenario, host, qp = agent_key.split('/')
            rewards = torch.tensor(agent_rollout['reward'], device=self.config.device)
            
            batch_size = min(len(rewards), self.config.agent.deterministic.max_batch_size, len(agent_rollout['state']))
            rewards = rewards[:batch_size]

            if self.config.agent.deterministic.use_rnn:
                hc = None
                actions = []
                for state in agent_rollout['state'][:batch_size]:
                    action, hc = self.model(state, hc)
                    actions.append(action)
                actions = torch.stack(actions, dim=0)
            else:
                states = torch.stack(agent_rollout['state'][:batch_size], dim=0)
                actions, _ = self.model(states, None)

            # actions = actions.squeeze()
            actions = torch.tanh(actions).squeeze()

            # actions = actions.squeeze()
            scenario_loss_agents[scenario] += 1

            # action_loss_coeff = 0.1*self.config.agent.deterministic.action_loss_coeff if 'a2a' in scenario else self.config.agent.deterministic.action_loss_coeff
            action_loss_coeff = self.config.agent.deterministic.action_loss_coeff
            agent_reward_loss, agent_action_loss = get_loss(actions, rewards, batch_size, scenario, self.config.agent.deterministic.loss_scale, action_loss_coeff, scenario_loss, self.config.agent.discount)
            # loss = loss / num_agents
            agent_reward_loss = agent_reward_loss / num_agents
            agent_action_loss = agent_action_loss / num_agents
            if self.config.agent.deterministic.balance_loss:
                ideal_timesteps = total_steps / agent_rollout['num_flows']
                agent_reward_loss = agent_reward_loss * (ideal_timesteps / batch_size)
                agent_action_loss = agent_action_loss * (ideal_timesteps / batch_size)

            # self.logger.info(f"loss: {loss}")
            (agent_reward_loss + agent_action_loss).backward()

            total_reward_loss += agent_reward_loss.detach().item()
            total_action_loss += agent_action_loss.detach().item()

        if self.config.training.gradient_clip > 0:
            # nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
            nn.utils.clip_grad_value_(self.model.parameters(), self.config.training.gradient_clip)
        self.optimizer.step()
        
        for key in scenario_loss_agents.keys():
            scenario_loss[key] = scenario_loss[key] / num_agents# divide scenario loss by number of agents per scenario
        # print('update')
        # return parts of loss for debugging
        return  total_reward_loss, total_action_loss , scenario_loss, num_agents

    def log_data_to_file(self, checkpoint):
        save_path = f'{{self.save_path}}{self.config.agent.save_name}/info/'
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True)

        frequency_file = f'{save_path}frequency.csv'

        #log frequency
        flow_counters = {'checkpoint': checkpoint}
        summary_flow_counters = defaultdict(list)
        for key in self.rollout.keys():
            test, host, qp = key.split('_')[-1].split('/')
            nb_updates = len(self.rollout[key]['reward'])
            avg_reward = [reward.cpu().item() for reward in self.rollout[key]['reward']]
            if avg_reward:
                avg_reward = np.mean(avg_reward)
                flow_counters[key] = [(nb_updates, avg_reward)]
                summary_flow_counters[test].append((nb_updates, avg_reward))
        
        freq_df = pd.DataFrame(flow_counters, index=[0])
        
        try:
            old_df = pd.read_csv(frequency_file)
            old_df = old_df.append(freq_df)
            old_df.to_csv(frequency_file, index=False)
        except:
            freq_df.to_csv(frequency_file, index=False)

    def log_state_to_file(self, data):
        save_path = '/swgwork/bfuhrer/projects/rlcc/distillation/simulator_inference/sim/results/'
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True)

        datafile = f'{save_path}training_log_data_hl2_lstm_remove.csv'

        # str = ''
        # for i in range(len(state)):
        #     str += f'{state[i].item()}'
        with open(datafile, 'a') as f:
            np.savetxt(f, data, delimiter=',')
        #     f.write(f'{data[0].item()}, {data[1].item()}, {data[0]}, {data[0]}, {data[0]},')

        

        



        
