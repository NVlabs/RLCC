# Reinforcement Learning for Datacenter Congestion Control
RL-CC is an RDMA congestion control algorithm trained with the analytical deterministic policy gradient method (ADPG) (Tessler et al. 2021) [1].  
This repository conatins the source code and simulator used to train RL-CC for [1] and [2]. 


## 1. About the NVIDIA ConnectX-6Dx CC Simulator

The CCsim simulator was used to develop the RL-CC algorithm and is based on the omnest 4.6 simulator.
The simulator is pre-compiled **to run on linux distributions** and located in the dir: `nv_ccsim/`. 
It contains the following ingredients:  
		- `nv_ccsim/sim/omnetpp.ini` - configuration file of the simulation parameters that can be
			          modified by the user. (network parameters, traffic patterns, congestion parameters)  
		- `nv_ccsim/sim/ccsim_release` - the executable of the simulator in release mode.

The configuration file contains the three available scenarios: many-to-one, all-to-all, long-short.

***it is advised to modify the configurations only if you are familiar with the omnest simulator!***

## 2. Installation   
First, clone the project: 
```
git clone https://github.com/NVlabs/RLCC.git
```
### 2a. Docker
The docker setup here assumes access to an nvidia-based docker with pytorch and cuda. However, any other standard pytorch docker should suffice too.

Prerequisites:
- Docker 19 or newer.
- Access to NVIDIA Docker Catalog. Visit the [NGC website](https://ngc.nvidia.com/signup) and follow the instructions. This will grant you access to the base docker image (from the Dockerfile) and ability to run on NVIDIA GPU using the nvidia runtime flag.

1. CD to project dir and build the docker image:
```
docker build -t rlcc .
```

2. Run the image: 
```
docker run --runtime=nvidia -it rlcc /bin/bash
```

### 2b. Local installation
Before installing RL-CC make sure to install Python version >= 3.7.

Install requirements by running
```
pip install -r requirements.txt
```
 
## 3. Running RL-CC 
Running RL-CC is done from  `reinforcement_learning/run.py`.  
When running RL-CC there are two separate phases: training and evaluation.
Both phases require configuring the RL-CC agent and the CCsim environment.

Default configurations per RL-CC agent type are available at `reinforcement_learning/configs` as yaml files. All parameters are modifiable and can be overloaded through the CLI. Below is a detailed list of all available parameters and how to configure them.
### 3.1 Agent Configuration Parameters
<!-- RL-CC may be trained with five agent types/algorithms:
- DQN,  PPO, Random Agent, Supervised Learning, ADPG.   -->

```yaml
## Agent parameters
agent:                  "RL-CC may be trained with five agent types/algorithms:
                         PPO, DQN, SUPERVISED, random, ADPG"
agent_features:         "features used as input to policy, choices: 
                         nack_ratio, cnp_ratio, bandwidth, bytes_sent, 
                         rtt_inflation, cur_rate, action, adpg_reward"
history_length:         "number of past observations to use as input to the policy"
evaluate:               "run agent in evaluation mode" 
action_multiplier_dec:  "percent of action multiplyer for decreasing the transmission rate"
action_multiplier_inc:  "percent of action multiplyer for increasing the transmission rate"
```

### 3.2 CCsim Environment Parameters
```yaml
## CCsim parameters
scenarios:           "list of scenarios to run on (see below for detailed explanation)"
envs_per_scenario:   "number of environments per scenario"
verbose:             "verbosity level of NVIDIA CCsim (recommended verbose=False)"
restart_on_end:      "restart scenario if it finishes before training ends"
multiprocess:        "run RL-CC on multiple processes in parallel"
port_increment:      "RL-CC interacts with the NVIDIA CC simulator via a TCP socket. 
                      port_increment specifies which port to connect to."
```
Specifying CC scenarios is done the following format: `<num_hosts>_<num_qps_per_hosts>_<scenario_type>_<test_type>`.
The `<scenario_type>` choices are: `m2o` for many-to-one, `a2a` for all-to-all, `longshort` for long-short. The `<test_type>` choices are: `train` for training, `eval` for evaluation. When choosing CC scenarios, only a specific set of <num_hosts>_<num_qps_per_hosts> combinations ara possible, see `reinforcement_learning/configs/constants.py` for available combinations.  
Users that are familiar with the OMNeT++ simulator can generate vector files in evaluation mode by specifying `_eval_vec` as the test type. 

Usage Examples: 
- many-to-one with 2 hosts and 1 qp per host for training - `2_1_m2o_train` 
- all-to-all with 4 hosts and 8 qp per host for training - `4_8_a2a_train`
- long-short with  1 long flow, 7 short flows and 8 qps per flow for evaluation - `8_8_longshort_eval`

### 3.3 Training RL-CC
RL-CC is trained for a pre-specified number of policy updates. After each update, the policy is saved as a checkpoint that corresponds to the total number of steps taken in the environment since the beginning.
RL-CC training monitoring is done through weights and biases. The following parameters are logged and are used to determine model convergance: ***nack_ratio, cnp_ratio, rate, adpg_reward, rtt_inflation, bandwidth, bytes_sent, action, and loss***.  
Below is the full list of training parameters. 
```yaml
## Training parameters
save_name:               "model save name"
reward:                  "reward function: general, distance,
                          constrained, adpg_reward"
max_num_updates:         "maximum number of policy updates used for training"
learning_rate:           "optimizer learning rate"
discount:                "discount factor"
linear_lr_decay:         "use linear learning rate decay"
use_rnn:                 "use a recursive layer in the model architecture:
                          RNN, GRU, LSTM"
gradient_clip:           "gradient clipping value"
architecture:            "neural network mlp architecture
                          ex: 12 will generate an mlp with a single
                          12 node hidden layer" 
activation_function:     "activation function between hidden layers: relu or tanh
                          output activation is always tanh."
bias:                    "use bias in mlp layers"
## ADPG  
rollout_length:          "length of rollout buffer"
target:                  "adpg target value"
beta:                    "adpg beta value"
scale:                   "adpg reward scale factor"
action_loss_coeff:       "coefficient of action loss"
reward_loss_coeff:       "coefficient of reward loss"
loss_batch:              "use batch of agents in rollout instead of all agents"
max_batch_size:          "maximum number of agent steps to use from the rollout
                          buffer while calculating the reward rollout"
max_step_size:           "maximum number of steps in the environment when 
                          collecting a rollout"
warmup_length:           "maximum number of steps in the environment when 
                          collecting a rollout in the warmup stage"
warmup_updates:          "number of policy updates for warmup stage"
## PPO
actor_architecture:      "actor mlp architecture (see architecture above)"
critic_architectur:      "critic mlp architecture (see architecture above)"
baseline_coeff:          "value function loss coefficient"
entropy_coeff:           "entropy loss coefficient"
use_gae:                 "use general advantage estimation"
gae_tau:                 "tau parameter used in gae calculation"
ppo_ratio_clip:          "ppo clipping value"
ppo_batch_size:          "ppo minibatch size"
ppo_optimization_epochs: "number of optimization epochs per rollout"
rollouts_per_batch:      "number of rollouts per batch"
discrete_actions:        "use discrete actions instead of continuous"
## DQN 
replay_size:             "size of replay buffer"
target_update_interval:  "interval for target network update"
eps_start:               "epsilon start value"
eps_end:                 "epsilon end value"
eps_decay:               "epsilon decay value"
## Supervised Learning
batch_size:              "batch size"
## Logging parameters for weights and biases
wandb:                   "project name for logging using weights and biases"
run_id:                  "wandb run_id for resuming a run"
wandb_run_name:          "run name"
log_interval:            "The number of environment steps between logs (average values are logged)"
limit_flows:             "whether to limit the number of flows logged"
limit_hosts:             "maximum number of hosts to log"
limit_qps:               "maximum number of qps per host to log"
```   

### 3.4 Evaluating RL-CC
An RL-CC model is evaluated by loading a trained model and running its policy in inference on desired scenarios. Each successfull CCsim run on ends with a .sca file that holds:  
* information regarding the test that was preformed, how many hosts we had, how many flows, run timestamp, etc.
* Run statistics: bandwidth, packet latency, packet drops, and much more.  

At the end of the evaluation phase the .sca files are automatically parsed to output key summary statistics to a .csv file for each scenario. The results can be found at: `nv_ccsim/sim/results`.  
Below are the evaluation parameters
```yaml
## Evaluation parameters
save_name:               "name of model to load"
checkpoint:              "specific checkpoint of model"
evaluate:                "must be set to true for evaluation"
verbose:                 "verbosity level of CCsim output (recommended verbose=True)"
```

### 3.5 CLI Examples
Example of training an ADPG model from the command line on the following scenarios:
- 2 hosts 1 qp per host many-to-one
- 16 hosts 8 qps per host many-to-one
- 4 hosts 4 qps per host all-to-all
```bash
python3 run.py --envs_per_scenario 1 --agent ADPG --scenarios 2_1_m2o_train 16_8_m2o_train 4_4_a2a_train --agent_features action adpg_reward --port_increment 0 --config rlcc
```
Example of evaluating an ADPG model from the command line on the following scenario:
- 64 hosts 128 QPs per host many-to-one short simulation scenario.
```bash
python3 run.py --envs_per_scenario 1  --learning_rate 0.01 --history_length 2 --agent ADPG --scenarios 64_128_m2o_eval --save_name <model_name> --agent_features action adpg_reward --port_increment 0 --config rlcc_evaluate  --evaluate
```



## Citing the repository
```
@article{10.1145/3512798.3512815,
author = {Tessler, Chen and Shpigelman, Yuval and Dalal, Gal  
and Mandelbaum, Amit and Haritan Kazakov, Doron and Fuhrer, Benjamin   
and Chechik, Gal and Mannor, Shie},
title = {Reinforcement Learning for Datacenter Congestion Control},
year = {2022},
issue_date = {September 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {49},
number = {2},
issn = {0163-5999},
url = {https://doi.org/10.1145/3512798.3512815},
doi = {10.1145/3512798.3512815},
journal = {SIGMETRICS Perform. Eval. Rev.},
month = {jan},
pages = {43–46},
numpages = {4}
}
```

## Reference Papers

[1] Tessler, C., Shpigelman, Y., Dalal, G., Mandelbaum, A., Kazakov, D. H., Fuhrer, B., Chechik, G., & Mannor, S. (2021). Reinforcement Learning for Datacenter Congestion Control. http://arxiv.org/abs/2102.09337. arXiv:2102.09337.  

[2] Fuhrer, B., Shpigelman, Y., Tessler, C., Mannor, S., Chechik, G., Zahavi, E., Dalal, G. (2022). Implementing Reinforcement Learning Datacenter Congestion Control in NVIDIA NIC. https://arxiv.org/abs/2207.02295. 	arXiv:2207.02295.
