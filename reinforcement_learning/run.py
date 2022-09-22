import argparse
import sys

import torch

from config.config import Config, str2bool
from config.constants import AGENTS
from env.utils.env_utils import make_vec_env

try:
    import wandb
except:
    wandb = None

wandb = None

sys.path.append(r'/swgwork/bfuhrer/projects/rlcc/new_simulator/reinforcement_learning/')

def main(config: Config) -> None:
    """
    After initializing the environments and the agent (based on the provided configuration requriements), runs the
    selected procedure - training or evaluation (test).
    :param config:
    :return:
    """
    env = make_vec_env(config)
    try:
        agent = AGENTS[config.agent.agent_type](config, env)
        if config.agent.evaluate:
            agent.test()
        elif config.agent.quantization and not config.quantization.fine_tune:
            agent.calibration()
        else:
            agent.train()
        env.close()
    except KeyboardInterrupt:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Config
    parser.add_argument('--config', default='rtt_deterministic', help='Use config file') #

    # General Parameters
    parser.add_argument('--agent', dest='agent_type', type=str,
                        choices=['PPO', 'DQN', 'SUPERVISED', 'random', 'CONSTRAINED', 'DETERMINISTIC', 'None'],
                        default='DETERMINISTIC')
    parser.add_argument('--port_increment', type=int, default=10)  # To enable multiple simulations at once, #0 default

    # Env Parameters
    parser.add_argument('--scenarios', type=str, default=['4_1_qp'], nargs='*') #['1024_to_1', '2048_to_1', '4096_to_1', '8192_to_1','2_to_1', '4_to_1', '8_to_1']
    # parser.add_argument('--scenarios', type=str, default=['2_1_qp', '4_1_qp', '8_1_qp', '4_8_a2a_qp', '8_16_a2a_qp', '32_1_qp', '8_16_a2a_qp', '16_32_a2a_qp'], nargs='*') #['1024_to_1', '2048_to_1', '4096_to_1', '8192_to_1','2_to_1', '4_to_1', '8_to_1']
    parser.add_argument('--envs_per_scenario', type=int, default=1)
    parser.add_argument('--max_timesteps', type=int, default=-1)
    parser.add_argument('--history_length', type=int, default=2)
    parser.add_argument('--evaluate', action='store_true') #FIXME
    parser.add_argument('--quantization', action='store_true')
    parser.add_argument('--quantization_method', type=str, default='mse') #'percentile'
    parser.add_argument('--m_quantization', action='store_true')
    parser.add_argument('--lstm_LUT', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--num_bins', type=int, default=-1)
    parser.add_argument('--log_data', action='store_true')
 

    # OMNeT Parameters
    parser.add_argument('--recv_len', type=int, default=-1)  # The number of features received from the OMNeT simulator
    parser.add_argument('--size_of_data', type=int, default=-1)  # The number of bytes for each feature
    parser.add_argument('--run_path', default='None', type=str)  # Path for the OMNeT executable
    parser.add_argument('--config_path', default='None', type=str)  # Path for the OMNeT config file

    # Learning Parameters
    parser.add_argument('--save_name', default='rtt_debug', type=str) 
    parser.add_argument('--save_quant_name', default='quant_rtt_based_model_wo_b', type=str) 
    parser.add_argument('--checkpoint', default='', type=str) #FIXME

    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--discount', default=-1, type=float)
    parser.add_argument('--use_gae', action='store_true')
    parser.add_argument('--use_rnn', type=str)
    parser.add_argument('--baseline_coeff', default=-1, type=float)
    parser.add_argument('--entropy_coeff', default=-1, type=float)
    parser.add_argument('--gae_tau', default=-1, type=float)
    parser.add_argument('--gradient_clip', default=-1, type=float)
    parser.add_argument('--linear_lr_decay', action='store_true')
    parser.add_argument('--use_dynamic_target', action='store_true')
    parser.add_argument('--actor_architecture', type=int, default=-1, nargs='*')
    parser.add_argument('--critic_architecture', type=int, default=-1, nargs='*')
    parser.add_argument('--architecture', type=int, default=-1, nargs='*')
    parser.add_argument('--activation_function', type=str, default='None', choices=['relu', 'tanh'])
    parser.add_argument('--target', type=float)
    parser.add_argument('--base_rtt', type=float)
    parser.add_argument('--factor', type=float)
    parser.add_argument('--power', default=1, type=float)
    parser.add_argument('--max_factor', type=float)
    parser.add_argument('--max_num_updates', default=80, type=int)
    parser.add_argument('--action_loss_coeff', default=1, type=float)
    parser.add_argument('--loss_scale', default=10, type=float)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--input_rate_loss_coeff', default=1, type=float)
    parser.add_argument('--per_scenario_norm', action='store_true', default=None)
    parser.add_argument('--per_scenario_downsampling', action='store_true', default=None)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--reward_calc_method', type=str, default=None)

    # PPO
    parser.add_argument('--agent_features', type=str, default=['action', 'rtt_reward'], nargs='*')
    parser.add_argument('--reward', type=str)
    parser.add_argument('--ppo_ratio_clip', default=-1, type=float)
    parser.add_argument('--ppo_batch_size', default=-1, type=int)
    parser.add_argument('--ppo_num_minibatch', default=-1, help='Will be ignored if ppo_batch_size > 0', type=int)
    parser.add_argument('--ppo_optimization_epochs', default=-1, type=int)
    parser.add_argument('--action_multiplier_dec', default=-1, type=float)
    parser.add_argument('--action_multiplier_inc', default=-1, type=float)
    parser.add_argument('--rollout_length', type=int, default=-1)
    parser.add_argument('--rollouts_per_batch', type=int, default=-1)
    parser.add_argument('--discrete_actions', action='store_true')
    parser.add_argument('--loss_batch', type=int, default=-1) # batch size of agents when we calculate loss (0 means all agenst)
    parser.add_argument('--warmup_updates', type=int, default=-1) # batch size of agents when we calculate loss (0 means all agenst)
    parser.add_argument('--warmup_length', type=int, default=-1) # batch size of agents when we calculate loss (0 means all agenst)
    parser.add_argument('--max_batch_size', type=int, default=-1) # batch size of agents when we calculate loss (0 means all agenst)
    parser.add_argument('--max_step_size', type=int, default=-1) # maximum number of steps to take in simulator for each epoch

    #DQN
    parser.add_argument('--target_update_interval', type=int, default=-1)

    # Logging
    parser.add_argument('--wandb', default='rlcc-distillation', type=str)  # Logging using weights and biases
    parser.add_argument('--run_id', default='', type=str)  # Logging using weights and biases
    parser.add_argument('--wandb_run_name', default='eval_2_to_1_quantization', type=str)  # Logging using weights and biases
    parser.add_argument('--frequency', default=-1, type=int)
    parser.add_argument('--num_tests_to_log', default=1, type=int)
    parser.add_argument('--limit_flows', action='store_true')
    parser.add_argument('--limit_hosts', type=int)
    parser.add_argument('--limit_qps', type=int)
    parser.add_argument('--max_rollout', type=int) # maximum number of steps in rollout per agent - n in TD-learning vs MonteCarlo
    parser.add_argument('--leaky_relu', type=float)
    parser.add_argument('--rtt_inflation_max', type=float)
    parser.add_argument('--rtt_inflation_min', type=float)
    parser.add_argument('--buffer_size', type=float)
    parser.add_argument('--norm_reward', action='store_true')
    parser.add_argument('--balance_loss', action='store_true')



    args = parser.parse_args()
    
    if args.config:
        config = Config(args.config, override=args.__dict__)
        # config = Config(args.config)
    else:
        config = Config(override=args.__dict__)
        config.logging.run_id = False
    config.agent.save_name = config.logging.wandb_run_name
    if config.logging.wandb and wandb:
        wandb.init(project=args.wandb, name=args.wandb_run_name, resume=config.logging.run_id,
                   config=args)
    else:
        wandb = None
    config.logging.wandb = wandb
    if not config.agent.m_quantization:
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        config.device = 'cpu'
    # print(config.agent.save_name)
    print(config)
    main(config)
