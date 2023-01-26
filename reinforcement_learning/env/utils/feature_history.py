from collections import deque, namedtuple
from typing import Dict, Tuple

import numpy as np

from .server import RawFeatures

ProcessedFeatures = namedtuple(
    'ProcessedFeatures',
    [
        'nack_ratio', 'cnp_ratio', 'bandwidth',
        'rtt_inflation', 'cur_rate', 'action',
        'adpg_reward', 'bytes_sent',
    ]
)


def calc_adpg_reward(config, rtt_inflation, rate):
    """
        Calculates RTT reward function (rtt_inflation * sqrt(rate))
        reward function needs to be equal to target

        RTT and rtt_inflation are in microseconds

        Args:
            rtt ([float]): normalized rtt values
            rate ([float]): current rate
            factor ([float]):
            max_factor ([float]): threshold queue length value (below this threshold set reward to 0)
            power ([float]): take power of function

        Returns:
            [float]: inflation value
    """
    rtt_inflation = max(rtt_inflation - config.agent.adpg.beta, 0)
    reward = rtt_inflation * np.sqrt(rate)
    reward = (reward - config.agent.adpg.target) * config.agent.adpg.scale
    return reward


class FeatureHistory:
    """
        The features history for each host-flow_tag pair should contain the past N monitor intervals (MI),
        where each MI contains:
            1. RTT diff (for K RTT packets in each MI, will contain RTT_last - RTT_first)
            2. Time difference between when first and last RTT were requested.
            3. Width of the entire MI
            4. Proportion of ACKs (ACKs received / packets sent)
    """
    def __init__(self, config, simulation_number):
        self.config = config
        self.number_of_features = len(self.config.agent.agent_features) 
        self.state_history_dict = {}
        self.action_history_dict = {}
        self.simulation_number = simulation_number

    def reset(self) -> None:
        self.state_history_dict = {}

    def update_history(self, raw_features: RawFeatures) -> None:
        agent_key = raw_features.host + ' ' + raw_features.flow_tag
        if agent_key not in self.state_history_dict:
            self.state_history_dict[agent_key] = deque(maxlen=self.config.env.history_length)
            self.update_action(raw_features.host, raw_features.flow_tag, 1.)

        features = self._process_features(raw_features)
        self.state_history_dict[agent_key].append(features)
        while len(self.state_history_dict[agent_key]) < self.config.env.history_length:
            self.state_history_dict[agent_key].append(features)

    def update_action(self, host: str, flow_tag: str, action: float):
        agent_key = host + ' ' + flow_tag
        self.action_history_dict[agent_key] = action

    def _get_action(self, host: str, flow_tag: str):
        """

            :param host: The host ID
            :param flow_tag: flow_tag ID
            :return: The previous action the agent controlling this flow took.
        """
        
        agent_key = host + ' ' + flow_tag
        if agent_key not in self.action_history_dict:
            return 1
        return self.action_history_dict[agent_key]

    def _process_features(self, raw_features: RawFeatures) -> ProcessedFeatures:
        """
            Converts the raw features provided by the simulator to a processed format that can be provided to the
            agents.

            :param raw_features: A namedtuple of features received by the environment simulator.
            :return: The processed features.
        """
        return ProcessedFeatures(
            nack_ratio=raw_features.nacks_received * 1. / max(raw_features.packets_sent, 1.),
            cnp_ratio=raw_features.cnps_received * 1. / max(raw_features.packets_sent, 1.),
            bandwidth=raw_features.bytes_sent * 1. / (raw_features.monitor_interval_width) ,
            bytes_sent=raw_features.bytes_sent,
            rtt_inflation=raw_features.rtt_packet_delay,
            cur_rate=raw_features.cur_rate,
            action=self._get_action(raw_features.host, raw_features.flow_tag),
            adpg_reward=calc_adpg_reward(self.config, raw_features.rtt_packet_delay, raw_features.cur_rate),
        )

    def process_observation(self, host: str, flow_tag: str) -> Tuple[np.ndarray, Dict, ProcessedFeatures]:
        """
            Given a host and flow_tag combination, returns the env information.

            :param host: The host id.
            :param flow_tag: The flow_tag (flow) id. This identifier is unique per each host but multiple hosts can have
                identical flow_tags.
            :return: The features provided to the agent, important information that can be logged and the last processed
                features.
        """
        agent_key = host + ' ' + flow_tag

        features = []
        for idx in range(len(self.state_history_dict[agent_key])):
            for required_feature in self.config.agent.agent_features:
                feature = getattr(self.state_history_dict[agent_key][idx], required_feature)
                features.append(feature)

        logging_information = dict(
            nack_ratio=self.state_history_dict[agent_key][-1].nack_ratio,
            cnp_ratio=self.state_history_dict[agent_key][-1].cnp_ratio,
            rate=self.state_history_dict[agent_key][-1].cur_rate,
            adpg_reward=self.state_history_dict[agent_key][-1].adpg_reward,
            rtt_inflation=self.state_history_dict[agent_key][-1].rtt_inflation,
            bandwidth=self.state_history_dict[agent_key][-1].bandwidth,
            action=self.state_history_dict[agent_key][-1].action,
            bytes_sent=self.state_history_dict[agent_key][-1].bytes_sent,
        )

        return np.array(features).flatten(), logging_information, self.state_history_dict[agent_key][-1]
