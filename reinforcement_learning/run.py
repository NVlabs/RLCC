import sys
import os

import torch

from config.args import parse_args
from config.config import Config
from config.constants import AGENTS
from env.utils.env_utils import make_vec_env
import pickle

try:
    import wandb
except:
    wandb = None

sys.path.append(os.getcwd())

def load_config(config: Config) -> Config:
    with open(f'{config.env.save_path}{config.agent.save_name}/config.pkl', 'rb') as f:
        old_config = pickle.load(f)
    old_config.agent.checkpoint = config.agent.checkpoint
    old_config.agent.evaluate = True
    old_config.env.scenarios = config.env.scenarios
    old_config.env.verbose = config.env.verbose
    old_config.env.restart_on_end = config.env.restart_on_end
    old_config.env.port_increment = config.env.port_increment
    return old_config

def save_config(config: Config):
    if not os.path.exists(f'{config.env.save_path}{config.agent.save_name}'):
        os.makedirs(f'{config.env.save_path}{config.agent.save_name}')
    with open(f'{config.env.save_path}{config.agent.save_name}/config.pkl', 'wb') as f:
        pickle.dump(config, f)


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
        else:
            agent.train()
        env.close()
    except KeyboardInterrupt:
        env.close()


if __name__ == '__main__':
    args = parse_args()

    if args.config:
        config = Config(args.config, override=args.__dict__)
    else:
        config = Config(override=args.__dict__)
        config.logging.run_id = False
    
    if config.agent.evaluate:
        config = load_config(config)
    else:
        save_config(config)

    if config.logging.wandb and wandb:
        wandb.init(project=args.wandb, name=args.wandb_run_name, resume=config.logging.run_id, config=args)
        if not config.agent.save_name:
            config.agent.save_name = config.logging.wandb_run_name
    else:
        wandb = None
    config.logging.wandb = wandb
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(config)
    main(config)
