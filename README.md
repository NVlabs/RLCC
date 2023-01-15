# Reinforcement Learning for Datacenter Congestion Control
RL-CC is an RDMA congestion control algorithm trained with the analytical deterministic policy gradient method (ADPG).  
This repository conatins the source code and simulator used to train RL-CC.


## NVIDIACCSim Simulator
* NVIDIACCSim simulates the NVIDIA ConnectX-6Dx NIC and is based on the omnest 4.6 simulator.  
* The simulator is pre-compiled and located in the dir: `NVIDIACCsim/`. 
* The simulator configuration file is located at `NVIDIACCSim/sim/omnetpp.ini`.
    The configuration file contains the three available scenarios:
    - many-to-one
    - all-to-all
    - long-short

    **It is strongly recommended to not modify any of the configurations!**
 
## Installation

### Prerequisites
RL-CC requires Python 3.7+.  
### Installing using pip
install RL-CC:  
```pip install -r requirements.txt```

## Configuration
### Running RL-CC
Running RL-CC is done through the command-line
```
cd reinforcement_learning/
python3 run.py
```
RL-CC may be trained with the following algorithms:
- DQN
- PPO
- Random Agent
- Supervised Learning
- ADPG

Algorithm configurations are available at `reinforcement_learning/configs` as yaml files.
Configurations may be overloaded through the CLI. See `reinforcement_learning/config/args.py` for available arguments.


### Specifying scenarios
Scenarios are specified in the following format: `<num_hosts>_<num_qps_per_hosts>_<scenario_type>_<test_duration>`

Scenarios may only use specific combinations of hosts and QPs, see `reinforcement_learning/configs/constants.py` for available combinations. 
* available `<scenario_type>` values are: `m2o` for many-to-one, `a2a` for all-to-all, `ls` for long-short.
* available `<test_duration>` values are: `s` for a short duration (200 ms), `m` for a medium duration (1 sec), `l` for a long duration (10000 sec).

**For training it is recommended to use a long duration**

Note: 
* **test duration is not specified for the long-short scenario**
* **long-short should not be used for training** 

Usage examples: 
- many-to-one with 2 hosts and 1 qp per host for training - `2_1_m2o_l` 
- all-to-all with 4 hosts and 8 qp per host for training - `4_8_a2a_l`
- long-short with  1 long flow, 7 short flows and 8 qps per flow for test - `8_8_ls`

It is possible to train/test on multiple scenarios at the same time.


## Training
Model training is done by running the script: `reinforcement_learning/run.py`.
To train the model run the following line (notice it would train the model on 3 scenarios: 2 hosts 1 qp per host many-to-one, 16 hosts 8 qps per host many-to-one, 4 hosts 4 qps per host all-to-all). all of the parameterss bellow can be changes

```
python3 run.py --envs_per_scenario 1 --wandb <project_name> --wandb_run_name <wandb_run_name>   --agent ADPG --scenarios 2_1_m2o_l 16_8_m2o_l 4_4_a2a_l --save_name <model_name> --port_increment 0 --config rlcc_evaluate
```

## Test and visualizing results
Testing the model is similar to training the model and is done by running the script:  `reinforcement_learning/run.py` and settings the parameter `evaluate: True` in the yaml file or adding `--evaluate` in the command line. 

Example for test scenario: 64 hosts 128 QPs per host many-to-one short simulation.
```
python3 run.py --envs_per_scenario 1 --wandb <project_name> --wandb_run_name <wandb_run_name>  --learning_rate 0.01 --history_length 2 --agent ADPG --scenarios 64_128_m2o_s --save_name <model_name> --port_increment 0 --config rlcc_evaluate --agent_features action rtt_reward --evaluate
```

### Simulator Results

* Simulator run results:
    * Each successfull run on the simulator ends with two basic files:
        * .params
        * .sca file this file holds data regarding:
            * The test that was preformed, how many hosts we had, how many flows and etc.
            * The timestamp of the run.
            * Run statistics. This data includes: bw, packet latency, packet drops, and much more. 
            * .sca file is created automatically only at the end of the run
    * Additional file that can be created during the run is a vector file. a vector files holds the data that was sampled during all of the run and not only run statistics .sca fille does.
* Run results can be found at: `NVIDIACCSim/sim/results`.
* **Training is typically done on very long simulations and is monitored through wandb/tensorboard and not via the .sca files**
* At test time, sca files are automatically parsed to output key summary statistics to a .csv file.

## Citing the repository
To cite this repository:  
```
@misc{https://doi.org/10.48550/arxiv.2102.09337,
  doi = {10.48550/ARXIV.2102.09337},
  url = {https://arxiv.org/abs/2102.09337},
  author = {Tessler, Chen and Shpigelman, Yuval and Dalal, Gal and Mandelbaum, Amit and Kazakov, Doron Haritan and Fuhrer, Benjamin and Chechik, Gal and Mannor, Shie},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Networking and Internet Architecture (cs.NI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Reinforcement Learning for Datacenter Congestion Control},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
## Reference Paper

Tessler, C., Shpigelman, Y., Dalal, G., Mandelbaum, A., Kazakov, D. H., Fuhrer, B., Chechik, G., & Mannor, S. (2021). Reinforcement Learning for Datacenter Congestion Control. http://arxiv.org/abs/2102.09337. arXiv:2102.09337.

Fuhrer, B., Shpigelman, Y., Tessler, C., Mannor, S., Chechik, G., Zahavi, E., Dalal, G. (2022). Implementing Reinforcement Learning Datacenter Congestion Control in NVIDIA NIC. https://arxiv.org/abs/2207.02295. 	arXiv:2207.02295.


<!-- Vector files takes a lot of memory space (~7GB per file).

To define if the run will have vectors file output or not we use the configuration set in the relevant ccsim.ini file (located in ./simulator/sim/ccsim.ini). For example to run our algo without vectors we will config the run to be Config RL_ShortSimult_ManyToOne and with vectors we will define the run to be Config  RL_ShortSimult_ManyToOne_Vectors. While runing the code from python we will config the run using the relevant configuration file located in ./config -->

<!-- # TO DOs -->
<!-- * Write advanced simulator usage (vector files, explain .ini file (or maybe we shouldn't)) -->
<!-- * test and train other algos than RL-CC -->