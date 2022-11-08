import os
import subprocess
import time
from typing import Dict, Tuple

import gym
import numpy as np
from config.config import Config
from config.constants import py_to_c_scenarios
from .utils.feature_history import FeatureHistory
from .utils.server import Server
from .utils import DEFAULT_PORT

class OMNeTpp(gym.Env):
    """
    A GYM wrapper for the OMNeTpp simulator.
    This environment is ASYNCHRONOUS. As opposed to the standard synchronous GYM envs, the OMNeTpp simulator can be
    viewed as a multi-agent environment. At each step the simulator will return an observation for ONE of the agents.
    This wrapper will maintain the history and information over all observed agents.
    """
    def __init__(self, scenario: str, simulation_number: int, env_number: int, config: Config):
        self.config = config
        self.simulation_number = simulation_number
        self.scenario = scenario
        if config.agent.evaluate:
            scenario += '_test'

        scenario_name_used = scenario

        self.scenario_name, test_number, config_num_tests = py_to_c_scenarios[scenario]

        algo_name = 'RL_'

        self.scenario_name = algo_name + self.scenario_name
        print(f"Scenario: {scenario_name_used} -> {self.scenario_name} -r {test_number} ")

        self.port = DEFAULT_PORT + simulation_number + self.config.env.port_increment

        self.test_number = test_number + (simulation_number + self.config.env.port_increment) * config_num_tests 
        self.env_running = False
        
        self.feature_history = FeatureHistory(self.config, simulation_number)

        self.server = Server(self.config, self.port)
        self.env_number = env_number
        self._configure_omnet()

        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)  #FIXME float32
        number_of_features = self.feature_history.number_of_features * self.config.env.history_length#TODO why to calculate it like this?
        self.observation_space = gym.spaces.Box(np.tile(-np.inf, number_of_features),
                                                np.tile(np.inf, number_of_features),
                                                dtype=np.float32)
        self.previous_host_flow_tag = None
        self.previous_cur_rate = 0

        self.last_bw_request = {}

    def _configure_omnet(self) -> None:
        """
        In order to run OMNeT we need to ensure that several environment variables are set.
        """
        if 'NVIDIACCSim/sim' not in os.getcwd():
            os.chdir(self.config.env.omnet.simulator_path)
            os.environ['LD_LIBRARY_PATH'] = '../lib:../lib/python_2.7.11/lib/'
            os.environ['NEDPATH'] = '../ned/algo/:../ned/prog_cc:../ned/dctg/'
            os.environ['PATH'] = '../lib/python_2.7.11/bin:' + os.environ['PATH']

    def seed(self, seed: int = None) -> None:
        """
        Environment randomness is set by the simulator and is not controlled from here.
        """
        pass

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset will close the running env and open a new instance..

        :return: The state and info received from the new simulator.
        """
        if self.env_running:
            self.close()
            time.sleep(0.001)

        print(f"restarting env: {self.scenario}")
        subprocess.Popen([
            self.config.env.omnet.exe_path, self.config.env.omnet.config_path,
            '-c', self.scenario_name,
            '-r', str(self.test_number), '-u' ,'Cmdenv',
             ], close_fds=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            #  ])
        #    
        
        self.env_running = True

        self.feature_history.reset()
        self.last_bw_request = {}

        raw_features = self.server.reset()
        self.feature_history.update_history(raw_features)

        key = self.scenario + '_' + str(self.env_number) + '/' + raw_features.host + '/' + raw_features.flow_tag

        self.previous_host_flow_tag = (raw_features.host, raw_features.flow_tag)
        self.previous_cur_rate = raw_features.cur_rate

        return self.feature_history.process_observation(raw_features.host, raw_features.flow_tag)[0],\
               dict(key=key, reward=0, env_num=self.env_number, host=raw_features.host, qp=raw_features.flow_tag)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Send the action to the environment. The action provided to the env is sent to the previous observed (flow_tag)
        combination.

        :param action: A float representing the relative increase/decrease (multiplier) for the agent's requested send
            rate.
        :return: The updated state, reward, done and information for a given (flow_tag) combination.
        """
        if self.previous_host_flow_tag[0] not in self.last_bw_request:
            self.last_bw_request[self.previous_host_flow_tag[0]] = {}
        self.last_bw_request[self.previous_host_flow_tag[0]][self.previous_host_flow_tag[1]] = min(self.previous_cur_rate * action, 1.)
        
        self.feature_history.update_action(self.previous_host_flow_tag[0], self.previous_host_flow_tag[1], action)

        # Perform a step in the simulator and process the received raw features.
        raw_features = self.server.step(action)
        print(f"updating action {action} for: {self.scenario + '_' + str(self.env_number) + '/' + raw_features.host + '/' + raw_features.flow_tag}")
        if raw_features is None:
            return self.reset()
        self.feature_history.update_history(raw_features)
      
        state, info, _ = self.feature_history.process_observation(raw_features.host, raw_features.flow_tag)

        # Calculate the reward.
        reward = self._calculate_reward(action, info)

        # Store the historical info.
        self.previous_flow_tag = raw_features.flow_tag
        self.previous_cur_rate = raw_features.cur_rate

        # Update the information dict.
        key = self.scenario + '_' + str(self.env_number) + '/' + raw_features.host + '/' + raw_features.flow_tag

        info.update(dict(key=key, reward=reward, host=raw_features.host, qp=raw_features.flow_tag, env_num=self.env_number))
        return state, reward, False, info

    def _calculate_reward(self, action: float, info: Dict) -> float:
        """
        Returns a reward signal for the agent.

        :param total_requested_rate: The total combined rate requested by all the flow_tags (flows).
        :param info: Information dictionary with local information on the current flow_tag (flow).
        :return: A scalar reward.
        """

        if self.config.env.reward == 'general':
            reward = (action - 1) - info['rtt_inflation'] * 0.1 - info['cnp_ratio'] - int(info['nack_ratio'] > 0) * 1000
        elif self.config.env.reward == 'distance':
            optimal_rate = 1. / int(self.scenario.split('_')[0])
            reward = - (info['bandwidth'] * 1. / 12.5 - optimal_rate) ** 2
        elif self.config.env.reward == 'constrained':
            if info['cnp_ratio'] > 2:
                reward = -1
            else:
                reward = action - 1. - info['rtt_inflation'] * 0.1
        else:
            reward = info[self.config.env.reward]
        return reward

    def render(self, mode: str = 'human') -> None:
        pass

    def close(self) -> None:
        self.server.end_connection()
