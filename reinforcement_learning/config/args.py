import argparse

from config.config import str2bool, str_parser


def parse_args():
    parser = argparse.ArgumentParser()

    # Config
    parser.add_argument('--config', default='rlcc', help='Name of the config file to be loaded')

    # General Parameters
    parser.add_argument('--agent', dest='agent_type', type=str, choices=['PPO', 'DQN', 'SUPERVISED', 'random', 'CONSTRAINED', 'ADPG', 'None'])
    parser.add_argument('--port_increment', type=int, default=-1, help='Increase the default port number. Useful if a prior run has crashed and the ports are currently in use')

    # Env Parameters
    parser.add_argument('--scenarios', type=str, default=-1, nargs='*', help='List of scenarios to parallelize during training')
    parser.add_argument('--envs_per_scenario', type=int, default=1, help='Number of instances of each scenario')
    parser.add_argument('--max_num_updates', default=-1, type=int)
    parser.add_argument('--history_length', type=int, default=-1, help='Agent state contains history of history_length-1 past observations')
    parser.add_argument('--evaluate', action='store_true', default=-1)
    parser.add_argument('--verbose', action='store_true', default=-1)
    parser.add_argument('--restart_on_end', action='store_true', default=-1)
    parser.add_argument('--multiprocess', action='store_true', default=-1)

    parser.add_argument('--action_multiplier_dec', default=-1, type=float) # percent of action multiplyer for decreasing the transmission rate
    parser.add_argument('--action_multiplier_inc', default=-1, type=float) # percent of action multiplyer for increasing the transmission rate

    # NV CCsim Parameters
    parser.add_argument('--run_path', default='None', type=str)  # Path for the NV CCsim  executable
    parser.add_argument('--config_path', default='None', type=str)  # Path for the NV CCsim  config file

    # Learning Parameters
    parser.add_argument('--save_name', default=-1, type=str)
    parser.add_argument('--checkpoint', default=-1, type=str)

    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--discount', default=-1, type=float)
    parser.add_argument('--use_gae', action='store_true', default=-1)
    parser.add_argument('--use_rnn', type=str, choices=['LSTM', 'GRU', 'RNN'], default=-1)
    parser.add_argument('--baseline_coeff', default=-1, type=float)
    parser.add_argument('--entropy_coeff', default=-1, type=float)
    parser.add_argument('--gae_tau', default=-1, type=float)
    parser.add_argument('--gradient_clip', default=-1, type=float)
    parser.add_argument('--linear_lr_decay', action='store_true', default=-1)
    parser.add_argument('--actor_architecture', type=int, default=-1, nargs='*')
    parser.add_argument('--critic_architecture', type=int, default=-1, nargs='*')
    parser.add_argument('--architecture', type=int, default=-1, nargs='*')
    parser.add_argument('--rollout_length', type=int, default=-1)
    parser.add_argument('--reward', type=str, choices=['general', 'distance', 'constrained', 'adpg_reward'])

    # ADPG loss function parameters
    parser.add_argument('--target', type=float)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--beta', type=float, default=-1)
    parser.add_argument('--scale', type=float, default=-1)

    parser.add_argument('--action_loss_coeff', default=1, type=float)
    parser.add_argument('--reward_loss_coeff', default=10, type=float)

    parser.add_argument('--loss_batch', type=int, default=-1)
    parser.add_argument('--max_batch_size', type=int, default=-1)
    parser.add_argument('--max_step_size', type=int, default=-1)

    # PPO
    parser.add_argument('--agent_features', type=str, default=-1, nargs='*', choices=['nack_ratio', 'cnp_ratio', 'bandwidth', 'bytes_sent', 'rtt_inflation', 'cur_rate', 'action', 'adpg_reward'])
    parser.add_argument('--ppo_ratio_clip', default=-1, type=float)
    parser.add_argument('--ppo_batch_size', default=-1, type=int)
    parser.add_argument('--ppo_optimization_epochs', default=-1, type=int)
    parser.add_argument('--rollouts_per_batch', type=int, default=-1)
    parser.add_argument('--discrete_actions', action='store_true', default=-1)

    # DQN
    parser.add_argument('--target_update_interval', type=int, default=-1)

    # Logging
    parser.add_argument('--wandb', default=-1, type=str_parser)  # Logging using weights and biases
    parser.add_argument('--run_id', default='', type=str)  # Logging using weights and biases
    parser.add_argument('--wandb_run_name', default='train_default_config', type=str_parser)  # Logging using weights and biases
    parser.add_argument('--log_interval', default=-1, type=int)  # Logging using weights and biases
    parser.add_argument('--limit_flows', action='store_true')  # used to avoid over logging wandb
    parser.add_argument('--limit_hosts', type=int)  # max host id to log
    parser.add_argument('--limit_qps', type=int)  # max qp id per host to log

    args = parser.parse_args()

    return args
