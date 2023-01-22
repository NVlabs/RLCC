# Reinforcement Learning for Datacenter Congestion Control
RL-CC is an RDMA congestion control algorithm trained with the analytical deterministic policy gradient method (ADPG) (Tessler et al. 2021) [1].  
This repository conatins the source code and simulator used to train RL-CC for [1] and [2]. 

## Installation

### Dependancies
* Python version >= 3.7
* linux OS
* gcc > 4.9 

### Installation steps
1. clone repository: ```https://gitlab-master.nvidia.com/bfuhrer/rl-cc-demo.git```
2. install necessary python packages via pip:  ```pip install -r requirements.txt```



**Note: the congestion control simulator nv_ccsim is pre-complied for linux (only) and does not require installation.**

## NVIDIA ConnectX-6Dx CC Simulator

* CCsim simulator enables Congestion Control algorithm development.
* The simulator is based on the omnest 4.6 simulator.
* The simulator is pre-compiled **for linux** and located in the dir: `nv_ccsim/`. 
* It contains the following ingredients:  
		- `nv_ccsim/sim/omnetpp.ini` - configuration file of the simulation parameters that can be
			          modified by the user. (Network parameters, Traffic patterns, Algorithms parameters)  
		- `nv_ccsim/sim/ccsim_release` - executables of the simulation in release mode.

* The configuration file contains the three available scenarios:
    - many-to-one
    - all-to-all
    - long-short

    **it is advised to modify the configurations only if you are familiar with the omnest simulator!**
 
## RL-CC
RL-CC may be trained with the following agents/algorithms:
- DQN
- PPO
- Random Agent
- Supervised Learning
- ADPG

Running RL-CC done from `reinforcement_learning/run.py`.

### Configurations
Default algorithm configurations are available at `reinforcement_learning/configs` as yaml files.
#### Configuration Parameters
There are many available configuration parameters depending on the agent used. Here is a list of the general configuration parameters. 
See `reinforcement_learning/config/args.py` for all available arguments. 
```
# Config
--config    name of the yaml config file to be loaded from `reinforcement_learning/config`

# General Parameters
--agent     type of RL agent, choices are: 'PPO', 'DQN', 'SUPERVISED', 'random', 'ADPG'
--agent_features  features used as input to policy, default choices for ADPG are: 'action', 'adpg_reward', where 'action' is the previous action taken by the agent and 'adpg_reward' is the resulting reward calculated via the adpg equation 

--port_increment    RL-CC interacts with the NVIDIA CC simulator via a TCP socket. port_increment allows to specify which port to connect to.

# Env Parameters
--scenarios     list of scenarios to run on
--envs_per_scenario     number of environments per scenario
--max_num_updates      maximum number of policy updates used for training
--history_length       number of past observations to use as input to the policy
--evaluate             run rl-cc in evaluation mode (default is False)       
--verbose              verbosity level of NVIDIA CCsim (recommended verbosity=0)
--restart_on_end       restart scenario if it finishes
--multiprocess         run RL-CC on multiple processes

--action_multiplier_dec     percent of action multiplyer for decreasing the transmission rate
--action_multiplier_inc     percent of action multiplyer for increasing the transmission rate

# Learning Parameters
--save_name            model save name
--checkpoint           model checkpoint to load from

# RL-CC (ADPG) loss function parameters
--target        adpg target value
--beta          adpg beta value
--scale         adpg loss scale value

# Logging
--wandb         project name for logging using weights and biases
--run_id        wandb run_id for resuming a run
--wandb_run_name 
```


***Configurations may be overloaded through the CLI.*** 
#### Specifying CC scenarios
Scenarios are specified in the following format: `<num_hosts>_<num_qps_per_hosts>_<scenario_type>_<test_duration>`  
Examples: 
- many-to-one with 2 hosts and 1 qp per host for training - `2_1_m2o_l` 
- all-to-all with 4 hosts and 8 qp per host for training - `4_8_a2a_l`
- long-short with  1 long flow, 7 short flows and 8 qps per flow for test - `8_8_ls`

* available `<scenario_type>` values are: `m2o` for many-to-one, `a2a` for all-to-all, `ls` for long-short.
* available `<test_duration>` values are: `s` for a short duration (200 ms), `m` for a medium duration (1 sec), `l` for a long duration (10000 sec).
* Scenarios may only use specific combinations of hosts and QPs, see `reinforcement_learning/configs/constants.py` for available combinations. 

Notes
- **training is recommended on scenarios with a long duration**  
- **test duration is not specified for the long-short scenario**  
- **long-short should not be used for training** 
#### CLI Examples
##### Example of training an ADPG model from the command line on the follwoing scenarios:
- 2 hosts 1 qp per host many-to-one long simulation
- 16 hosts 8 qps per host many-to-one long simulation
- 4 hosts 4 qps per host all-to-all long simulation
```
python3 run.py --envs_per_scenario 1 --wandb <project_name> --wandb_run_name <wandb_run_name>   --agent ADPG --scenarios 2_1_m2o_l 16_8_m2o_l 4_4_a2a_l --save_name <model_name> --agent_features action adpg_reward --port_increment 0 --config rlcc
```
##### Example of evaluating an ADPG model from the command line on the follwowing scenario:
- 64 hosts 128 QPs per host many-to-one short simulation scenario.
```
python3 run.py --envs_per_scenario 1 --wandb <project_name> --wandb_run_name <wandb_run_name>  --learning_rate 0.01 --history_length 2 --agent ADPG --scenarios 64_128_m2o_s --save_name <model_name> --agent_features action adpg_reward --port_increment 0 --config rlcc_evaluate  --evaluate
```

### Simulation Results

* Simulator run results:
    * Each successfull run on the simulator ends with two basic files:
        * .params
        * .sca file this file holds data regarding:
            * The test that was preformed, how many hosts we had, how many flows and etc.
            * The timestamp of the run.
            * Run statistics. This data includes: bw, packet latency, packet drops, and much more. 
            * .sca file is created automatically only at the end of the run
    * Additional file that can be created during the run is a vector file. a vector files holds the data that was sampled during all of the run and not only run statistics .sca fille does.
* Run results can be found at: `nv_ccsim/sim/results`.
* **Training is typically done on very long simulations and is monitored through wandb/tensorboard and not via the .sca files**
* At test time, sca files are automatically parsed to output key summary statistics to a .csv file.

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


<!-- Vector files takes a lot of memory space (~7GB per file).

To define if the run will have vectors file output or not we use the configuration set in the relevant ccsim.ini file (located in ./simulator/sim/ccsim.ini). For example to run our algo without vectors we will config the run to be Config RL_ShortSimult_ManyToOne and with vectors we will define the run to be Config  RL_ShortSimult_ManyToOne_Vectors. While runing the code from python we will config the run using the relevant configuration file located in ./config -->

<!-- # TO DOs -->
<!-- * Write advanced simulator usage (vector files, explain .ini file (or maybe we shouldn't)) -->
<!-- * test and train other algos than RL-CC -->