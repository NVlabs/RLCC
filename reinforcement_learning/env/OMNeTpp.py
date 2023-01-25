import os
import subprocess
import time
from typing import Dict, Tuple

import gym

gym.logger.set_level(40)
import numpy as np
from config.config import Config
from config.constants import py_to_c_scenarios

from .utils import DEFAULT_PORT
from .utils.feature_history import FeatureHistory
from .utils.parse_results import parse_results
from .utils.server import Server

scenario_types = {"l": 'LongSimult', 'm': 'MediumSimult', 's': 'ShortSimult'}

def parse_scenario(scenario):
    scenario_name = 'RL_'
    if 'vec' in scenario:
        assert len(scenario) > 4 and scenario[-4:] == '_vec', "scenario name must end with _vec to use vectors" 
        if 'm2o' in scenario or 'a2a' in scenario:
            assert '_m_vec' in scenario in '_s_vec', "must specify scenario type: _s_vec, _m_vec, no long simult with vectors"
        scenario_name += scenario_types[scenario.replace('_vec', '')[-1]] + '_Vectors'
    elif 'm2o' in scenario or 'a2a' in scenario:
        assert len(scenario) > 2 and scenario[-2:] in ['_s', '_m', '_l'], "must specify scenario type: _l, _s, _m" 
        scenario_name += scenario_types[scenario[-1]]
    if 'm2o' in scenario:
        scenario_name += '_ManyToOne'
    elif 'a2a' in scenario:
        scenario_name += '_AllToAll'
    else:
        scenario_name += 'LongShort'
    return scenario_name

class OMNeTpp(gym.Env):
    """
        A GYM wrapper for the OMNeTpp simulator.
        This environment is ASYNCHRONOUS. As opposed to the standard synchronous GYM envs, the OMNeTpp simulator can be
        viewed as a multi-agent environment. At each step the simulator will return an observation for ONE of the agents.
        This wrapper will maintain the history and information over all observed agents.
    """
    def __init__(self, scenario: str, simulation_number: int, env_number: int, config: Config):
        assert 'm2o' in scenario or 'a2a' in scenario or 'ls' in scenario, "Scenario name must contain either: m2o/a2a/ls"
        self.config = config
        self.simulation_number = simulation_number
        self.scenario = scenario

        scenario_name_used = scenario

        host_qps = f"{scenario.split('_')[0]}_{scenario.split('_')[1]}"

        
        if 'm2o' in scenario:
            test_number, config_num_tests = py_to_c_scenarios['m2o'][host_qps]
        elif 'a2a' in scenario:
                    test_number, config_num_tests = py_to_c_scenarios['a2a'][host_qps]
        elif 'ls' in scenario:
                    test_number, config_num_tests = py_to_c_scenarios['ls'][host_qps]
 

        self.scenario_name = parse_scenario(scenario)
       

        self.port = DEFAULT_PORT + simulation_number + self.config.env.port_increment

        self.test_number = test_number + (simulation_number + self.config.env.port_increment) * config_num_tests 
        print(f"Scenario: {scenario_name_used} -> {self.scenario_name} -r {self.test_number} ")

        self.scenario_raw_name = self.scenario_name + '-' + str(self.test_number)

        self.env_running = False
        
        self.feature_history = FeatureHistory(self.config, simulation_number)

        self.server = Server(self.config, self.port)
        self.env_number = env_number
        self._configure_omnet()

        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)
        number_of_features = self.feature_history.number_of_features * self.config.env.history_length
        self.observation_space = gym.spaces.Box(np.tile(-np.inf, number_of_features),
                                                np.tile(np.inf, number_of_features),
                                                dtype=np.float32)

        self.previous_host_flow_tag = None
        self.previous_cur_rate = 0

        self.last_bw_request = {}

    def _configure_omnet(self) -> None:
        """
            To initialize the simulator, we need to ensure that several environment variables are set.
        """
        if 'nv_ccsim/sim' not in os.getcwd():
            os.chdir('../nv_ccsim/sim')
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

            :return: The state and info received from the new simulator instance.
        """
        if self.env_running:
            self.close()
            time.sleep(0.001)

        # The scenario name and test number will define the test config, including which port the simulator will listen
        # on for communication with our gym env.
        print(f"Resetting env: {self.scenario_raw_name}")
        subprocess.Popen([
            '../bin/ccsim_release', 'omnetpp.ini',
            '-c', self.scenario_name,
            '-r', str(self.test_number),
            '-u', 'Cmdenv',
        ], stdout=None if self.config.env.verbose else subprocess.DEVNULL , stderr=None if self.config.env.verbose else subprocess.DEVNULL)
        
        self.env_running = True

        self.feature_history.reset()
        self.last_bw_request = {}

        raw_features = self.server.reset()
        self.feature_history.update_history(raw_features)

        agent_key = self.scenario + '_' + str(self.env_number) + '/' + raw_features.host + '/' + raw_features.flow_tag

        self.previous_host_flow_tag = (raw_features.host, raw_features.flow_tag)
        self.previous_cur_rate = raw_features.cur_rate

        return self.feature_history.process_observation(raw_features.host, raw_features.flow_tag)[0],\
               dict(agent_key=agent_key, reward=0, env_num=self.env_number, host=raw_features.host, qp=raw_features.flow_tag)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
            Send the action to the environment. The action provided to the env is sent to the previous observed (flow_tag)
            combination.

            :param action: A float representing the relative increase/decrease (multiplier) for the agent's requested send
                rate.
            :return: The updated state, reward, done and information for a given (flow_tag) combination.
        """
        # The simulator is asyncronous. A single instance will control multiple hosts (servers) and within them
        # multiple QPs (flows).
        # Each (host, flow) tuple corresponds to a different and unique agent. As such, we perform internal bookkeeping
        # to differentiate between the observations and actions performed by each agent.
        if self.previous_host_flow_tag[0] not in self.last_bw_request:
            self.last_bw_request[self.previous_host_flow_tag[0]] = {}
        self.last_bw_request[self.previous_host_flow_tag[0]][self.previous_host_flow_tag[1]] = min(self.previous_cur_rate * action, 1.)
        
        self.feature_history.update_action(self.previous_host_flow_tag[0], self.previous_host_flow_tag[1], action)

        # Perform a step in the simulator and process the received raw features.
        raw_features = self.server.step(action)
        #print(f"updating action {action} for: {self.scenario + '_' + str(self.env_number) + '/' + self.previous_host_flow_tag[0] + '/' + self.previous_host_flow_tag[1]}")
        if raw_features is None:
            if self.config.env.restart_on_end:
                return self.reset()
            else:
                parse_results(self.scenario_raw_name)
                return None, None, True, None

        self.feature_history.update_history(raw_features)
      
        state, info, _ = self.feature_history.process_observation(raw_features.host, raw_features.flow_tag)

        # Calculate the reward.
        reward = self._calculate_reward(action, info)

        # Store the historical info.
        self.previous_flow_tag = raw_features.flow_tag
        self.previous_cur_rate = raw_features.cur_rate

        # Update the information dict.
        agent_key = self.scenario + '_' + str(self.env_number) + '/' + raw_features.host + '/' + raw_features.flow_tag

        info.update(dict(agent_key=agent_key, reward=reward, host=raw_features.host, qp=raw_features.flow_tag, env_num=self.env_number))
        return state, reward, False, info

    def _calculate_reward(self, action: float, info: Dict) -> float:
        """
            Returns a reward signal for the agent. Either uses one of the pre-defined options in here, or one of the
            values from the feature_history, such as rtt_inflation.

            We provide several options:
                - General - combination of latency and packet loss.
                - Distance - assuming the optimal rate is 1/(total number of flows), the reward is the distance from
                    this value.
                - Constrained - the agent should focus on rtt_inflation but ensure cnp ratio is below threshold.
                - Other - defined in the config and passed through the info dict.

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
