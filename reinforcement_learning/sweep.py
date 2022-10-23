import shutil
import os
import glob
import random
import string
import argparse
from multiprocessing import Process
import torch
from config.config import Config
from config.constants import AGENTS
from env.utils.env_utils import make_vec_env

try:
    import wandb
except:
    wandb = None


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str


def run_evaluation(config, i, num_flows):
    run = None
    if config.logging.wandb:
        run = config.logging.wandb.init(project='congestion control sweep',
                                        name=f'{config.agent.save_name}_test_{num_flows}_to_1', reinit=True)
    config.env.scenarios = [f'{num_flows}_to_1']
    config.env.port_increment = 3 + i
    config.agent.evaluate = True
    env = make_vec_env(config)
    agent = AGENTS[config.agent.agent_type](config, env)
    print('Evaluating {} scenario...'.format(config.env.scenarios))
    agent.test()
    env.close()
    if run is not None:
        run.finish()


def main(config: Config) -> None:
    """
    After initializing the environments and the agent (based on the provided configuration requriements), runs the
    selected procedure - training or evaluation (test).
    :param config:
    :return:
    """

    random_path = get_random_string(8)
    if not os.path.exists(f'../{random_path}'):
        os.makedirs(f'../{random_path}')
    shutil.copytree('../simulator', f'../{random_path}/simulator')

    config.env.omnet.simulator_path = f'../{random_path}/simulator'

    try:
        run = None
        if config.logging.wandb:
            run = config.logging.wandb.init(project='congestion control sweep', name=config.agent.save_name, reinit=True)
        config.env.scenarios = ['2_to_1', '4_to_1', '8_to_1']
        env = make_vec_env(config)
        agent = AGENTS[config.agent.agent_type](config, env)
        print('Training {} agent...'.format(config.agent.agent_type))
        agent.train()
        env.close()
        if run is not None:
            run.finish()
    except Exception as e:
        print(e)

    shutil.rmtree('results')

    processes = []
    for i, num_flows in enumerate([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
        processes.append(Process(target=run_evaluation, args=(config, i, num_flows)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    os.chdir('../../..')

    results_path = f'results/{config.agent.save_name}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for file in glob.glob(rf'{random_path}/simulator/sim/results/*.sca'):
        shutil.copy(file, results_path)

    shutil.rmtree(random_path)


if __name__ == '__main__':
    archtecture = [[32, 16], [16, 16], [8, 8], [4, 4]]

    hyperparameter_defaults = dict(
        default_config='rtt_adpg',
        additional_agent_feature_0=None,
        additional_agent_feature_1=None,
        additional_agent_feature_2=None,
        architecture_index=0,
        use_lstm=True,
    )

    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults, project='congestion control sweep', name='init')
    sweep_config = wandb.config

    config = Config(sweep_config.default_config)
    config.logging.wandb = wandb
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_name = sweep_config.default_config
    config.agent.adpg.architecture = archtecture[sweep_config.architecture_index]
    save_name += '_arch_' + '_'.join(map(str, config.agent.adpg.architecture))

    config.agent.adpg.use_lstm = sweep_config.use_lstm
    if sweep_config.additional_agent_feature_0 is not None:
        config.agent.agent_features.append(sweep_config.additional_agent_feature_0)
        save_name += f'_{sweep_config.additional_agent_feature_0}'
    if sweep_config.additional_agent_feature_1 is not None:
        config.agent.agent_features.append(sweep_config.additional_agent_feature_1)
        save_name += f'_{sweep_config.additional_agent_feature_1}'
    if sweep_config.additional_agent_feature_2 is not None:
        config.agent.agent_features.append(sweep_config.additional_agent_feature_2)
        save_name += f'_{sweep_config.additional_agent_feature_2}'

    config.agent.save_name = save_name

    main(config)
