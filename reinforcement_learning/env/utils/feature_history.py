from collections import deque, namedtuple
from typing import Dict, Tuple

import numpy as np

from .server import RawFeatures

ProcessedFeatures = namedtuple(
    'ProcessedFeatures',
    [
         'latency_mean_min_ratio', 'nacks_received', 'bandwidth',
        'rtt_packet_delay', 'cur_rate', 'action', 'qlength_reward', 'input_rate', 'qlength',
        'rtt_reward', 'bytes_sent', 'tx_rate',
    ]
)

def normalize_max_min(value, min_value, max_value, a=-1, b=1):
    """ 
    Normalize values using max and min such that they lie within the interval [a, b]
    """
    norm_value = (b-a)*(value - min_value) / (max_value - min_value) + a
    return np.clip(norm_value, a_min=a, a_max=b)

def unnoramlize_min_max(value, min_value, max_value, a=-1, b=1):
    """ 
    Reverse normalization values using max and min
    """
    return (value - a)*(max_value - min_value) / (b - a) + min_value

def calc_rtt_reward(config, rtt, rate):
    """Calculates RTT reward function (rtt_inflation * sqrt(rate))
    reward function needs to be equal to target

    We used to divide rtt (nanoseconds) by 8192 -> this makes issues with fixed-point on HW as we have a high risk for overflow.
    For example: max_factor = 1.5 when we divide by 8192 and factor = 12.5, can result in large overflows when we try to do the multiplication
    in fixed-point, but 1.5*8192 = 12288 which is a number in nanoseconds --> a/x -b = (a - bx) / x. This means that our baseline is around 12 microseconds
    It is much simpler to use rtt in microseconds instead of nanoseconds -> this is done by dividing by 1024, if we replace 8192 with 1024
    we can divide by 1024 and multiply the factor by 8 and use maxfactor = 12288 and we get the same exact result but now it is much more stable for HW
    We could have done the same by dividing by 8192 and using maxfactor = 12288 but now we have a value in ~ microseconds which is interpretable
    --> our target rtt_inflation is in microseconds 

    Args:
        rtt ([float]): normalized rtt values
        rate ([float]): current rate
        factor ([float]): 
        max_factor ([float]): threshold qlength value (below this threshold set reward to 0)
        power ([float]): take power of function

    Returns:
        [float]: inflation value
    """
        # rate = 1 - rate
    # rtt_inflation = (rtt - max_factor) / 1024
    rtt_inflation = max((rtt / config.agent.deterministic.base_rtt) - config.agent.deterministic.max_factor, 0)
    reward = rtt_inflation * np.sqrt(rate)
    reward = (reward - config.agent.deterministic.target) * config.agent.deterministic.factor
    return reward

def calc_tele_reward(config, qlen, rate):
    """Calculates qlength reward value in telemetry mode (qlength_inflation * sqrt(rate))
    reward function needs to be equal to target

    Args:
        qlen ([float]): normalized queue length values
        rate ([float]): current rate
        factor ([float]): 
        max_factor ([float]): threshold qlength value (below this threshold set reward to 0)
        power ([float]): take power of function
        calc_method ([str], optional): different methods to calculate reward Defaults to None.

    Returns:
        [float]: reward value
    """
    # max_rate = 0.0488 # constant calculatd as 100 Gbits in units of 256bytes/nanoseconds
    # rtt_inflation = 1 + qlen / (max_rate * config.agent.deterministic.base_rtt)
    # reward = max(rtt_inflation - config.agent.deterministic.max_factor,0) * np.sqrt(rate)
    # return (reward - config.agent.deterministic.target)*config.agent.deterministic.factor
    # print(config.env)
    qlen_percent = qlen / config.env.buffer_size # 4096 = 10 MiB which is current buffer size in .ini
    reward = max(qlen_percent - config.agent.deterministic.max_factor,0) * np.sqrt(rate)
    return (reward - config.agent.deterministic.target)*config.agent.deterministic.factor

class FeatureHistory:
    """
    The features history for each host-flow_tag pair should contain the past N monitor intervals, where each MI contains:
        1. RTT diff (for K RTT packets in each MI, will contain RTT_last - RTT_first)
        2. Time difference between when first and last RTT were requested.
        3. Width of the entire MI
        4. Proportion of ACKs (ACKs received / packets sent)
    """
    def __init__(self, config, simulation_number):
        self.config = config
        self.number_of_features = len(self.config.agent.agent_features) #TODO understand why do we need to do len for a string???
        self.state_history_dict = {}
        self.action_history_dict = {}
        self.prev_action_history_dict = {}
        self.simulation_number = simulation_number # for debugging

    def reset(self) -> None:
        self.state_history_dict = {}

    def update_history(self, raw_features: RawFeatures) -> None:
        key = raw_features.host + ' ' + raw_features.flow_tag
        if key not in self.state_history_dict:
            self.state_history_dict[key] = deque(maxlen=self.config.env.history_length)
            self.update_action(raw_features.host, raw_features.flow_tag, 1.)

        features = self._process_features(raw_features)
        self.state_history_dict[key].append(features)
        while len(self.state_history_dict[key]) < self.config.env.history_length:
            self.state_history_dict[key].append(features)

    def update_action(self, host: str, flow_tag: str, action: float):
        key = host + ' ' + flow_tag
        if key in self.prev_action_history_dict:
            self.prev_action_history_dict[key] = self.action_history_dict[key]
        else:
            self.prev_action_history_dict[key] = action
        self.action_history_dict[key] = action

    def _get_action(self, host: str, flow_tag: str):
        """

        :param host: The host ID
        :param flow_tag: flow_tag ID
        :return: The previous action the agent controlling this flow took.
        """
        key = host + ' ' + flow_tag
        if key not in self.action_history_dict:
            return 1
        return self.action_history_dict[key]

    def _process_features(self, raw_features: RawFeatures) -> ProcessedFeatures:
        """
        Converts the raw features provided by the simulator to a processed format that can be provided to the agents.

        :param raw_features: A namedtuple of features received by the environment simulator.
        :return: The processed features.
        """
        return ProcessedFeatures(
            latency_mean_min_ratio=raw_features.rtt_packet_delay * 1. / .44,  # 0.44 - empty system RTT
            nacks_received=raw_features.nacks_received,
            bandwidth=raw_features.bytes_sent * 1. / (raw_features.rtt_packet_delay) ,
            bytes_sent=raw_features.bytes_sent,
            rtt_packet_delay=raw_features.rtt_packet_delay,
            cur_rate = raw_features.cur_rate,
            action=self._get_action(raw_features.host, raw_features.flow_tag),
            tx_rate=raw_features.tx_rate,
            rtt_reward=calc_rtt_reward(self.config, raw_features.rtt_packet_delay, raw_features.cur_rate),
            input_rate=raw_features.input_rate, # we only concentrate on cases where the switch rate is under the optimal value (1.0)
            qlength=raw_features.qlength,
            qlength_reward=calc_tele_reward(self.config, raw_features.qlength, raw_features.cur_rate),
        )

    def process_observation(self, host: str, flow_tag: str, ip_mode: float) -> Tuple[np.ndarray, Dict, ProcessedFeatures]:
        """
        Given a host and flow_tag combination, returns the env information.

        :param host: The host id.
        :param flow_tag: The flow_tag (flow) id. This identifier is unique per each host but multiple hosts can have identical flow_tags.
        :return: The features provided to the agent, important information that can be logged and the last processed
            features.
        """
        key = host + ' ' + flow_tag

        features = []
        for idx in range(len(self.state_history_dict[key])):
            for required_feature in self.config.agent.agent_features:
                feature = getattr(self.state_history_dict[key][idx], required_feature)
                features.append(feature)
        # features.append(ip_mode)

        logging_information = dict(
            # nacks_received=self.state_history_dict[key][-1].nacks_received,
            rate=self.state_history_dict[key][-1].cur_rate,
            rtt_reward=self.state_history_dict[key][-1].rtt_reward,
            qlength_reward=self.state_history_dict[key][-1].qlength_reward,
            input_rate=self.state_history_dict[key][-1].input_rate,
            rtt_latency=self.state_history_dict[key][-1].rtt_packet_delay,
            qlength=self.state_history_dict[key][-1].qlength,
            action=self.state_history_dict[key][-1].action,
            tx_rate=self.state_history_dict[key][-1].tx_rate,
        )

        return np.array(features).flatten(), logging_information, self.state_history_dict[key][-1]
