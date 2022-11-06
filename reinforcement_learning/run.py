import sys
import os

import torch

from config.args import parse_args
from config.config import Config, str2bool
from config.constants import AGENTS
from env.utils.env_utils import make_vec_env

try:
    import wandb
except:
    wandb = None

sys.path.append(os.getcwd())

wandb = None 

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
    config.agent.save_name = config.logging.wandb_run_name
    if config.logging.wandb and wandb:
        wandb.init(project=args.wandb, name=args.wandb_run_name, resume=config.logging.run_id, config=args)
    else:
        wandb = None
    config.logging.wandb = wandb
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(config)
    main(config)
