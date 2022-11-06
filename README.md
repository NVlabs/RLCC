# OLD README from NBU AI repo - TO DO change this
# RLCC / Analytic Deterministic Policy Gradient (ADPG) DestIP Mode  - Telemetry-based version
## Project Commitment
### Project Definition
* Team Name:  Networking Research 
* Manager: Yuval Shpigelman 
* Challenge: Mellanox CC algorithms don’t work
* AI Goal: Develop a good AI CC algorithm.
* Definition of success: Perform better than the exiting Mellanox CC algorithms (emphasis on SWIFT).
* Data Source and Availability:  Using a CX6DX simulator 
* Value: Competitive Mellanox CC ability 
* Due Date: With the release of CX8, a year from now

### Project’s team
* AI: Doron Haritan, Benjamin Fuhrer
* Domain Expert: Yuval Shpigelman 
* Arch: Gal Yefet
* Coding: NA

## Project overview:
Implementation of RL Congestion control in production environemnt by converting RLCC/ADPG to work in DestIP mode with Telemetry.

## Simulator
* The simulator that was used in the Training/Testing of the model is an omnet simulator that mimics CX6DX, and was programmed by Yuval  Shpigelman
* In each new terminal we open we need to define the MknxCCSimsetop (TO enable the simulator). To do so run:
    ```
    cd ./simulator
    source MlnxCCSimSetup
    ```

* The simulator is located in the dir: ./simulator. The relevant C files are located in the ./simulator/FW/Algorithm
* To compile the relevant c files on the simulator run the following commends from the dir ./simulator/FW/Algorithm
    ```
    make MODE=debug -B
    make MODE=release -B
    ```
    After each change in the c files we need to compile them again.
* To config the runs on the simulator (no need in recompiling after each change) use the file ./simulator/sim/ccsim.ini.

    Examples of things we can config in this file are: duration of the test  and per algorithm define its relevant c file (be defining the algorithm name) and the scenarios it can run on.
* How to run the simulator.
    ```
    cd ./simulator/sim/
    ../bin/ccsim_release ccsim.ini -c <Config> -r <Run number>
    ```

    c in the relevant ccsim.ini file and r is the run number.
    The run number will define the test case that would run on the simulator. To define the run number we will use the following calculation ( the test num and the config_num_tests are taken from the file ./config/constants.py):
     test_number + (simulation_number + config.env.port_increment) * config_num_tests

    The python scripts calculate the run number automatically, and opens a socket with the simulator during the run (can be seen in the file env/OMNeTpp.py)

    We can set a seed for thr simulator run, this way we can be sure that all of the host and the start point of the flows were the same during each scenario run, per scenario (done in model evaluation not while we train it)

* Simulator run results:
    * Each good run on the simulator ends with two basic files:
        * .params
        * .sca file this file holds data regarding:
            * The test that was preformed, how many hosts we had, how many flows and etc.
            * The timestamp of the run
            * The average statistic data for the run. Including data regarding the Utalization, Latency and fairness of the tests. This is the File that would be parsed in the plot_many_to_one.py script, and according to it the relevant plots/csv files would be created.
            * .sca file would be created only in the end of the run
    * Additional file that can be created during the run is a vector file. a vector files holds the data that was sampled during all of the run and not only the average data like the .sca fille does.

        To define which data we want to be saved in the vectors file we need to config the relevant .xml file. In our case it would be ./simulator/FW/Algorithm/rl.xml

        Vector files takes a lot of memory space (~7GB per file).

        To define if the run will have vectors file output or not we use the configuration set in the relevant ccsim.ini file (located in ./simulator/sim/ccsim.ini). For example to run our algo without vectors we will config the run to be Config RL_ShortSimult_ManyToOne and with vectors we will define the run to be Config  RL_ShortSimult_ManyToOne_Vectors. While runing the code from python we will config the run using the relevant configuration file located in ./config
## requirements:
* Baselines. To install use the following link: https://github.com/openai/baselines
* wandb
* Torch (to open some of the saved model you'll need torch 1.7)
* For quantization only - Hao Pytorch_quantization package (https://gitlab-master.nvidia.com/TensorRT/Tools/pytorch-quantization)

## How to run the python code
The configuration of the run is set by configuring the args of the ./reinforcement_learning/run.py file
### Train
To train the model run the following line (notice it would train the model on 3 scenarios 2_to_1, 4_to_1, 8_to_1). all of the params bellow can be changes

    ```
    python run.py --envs_per_scenario 1 --wandb <project_name> --wandb_run_name <wandb_run_name>  --learning_rate 0.01 --history_length 1 --agent DETERMINISTIC --scenarios 2_to_1 4_to_1 8_to_1 --save_name <model_name> --port_increment 0 --config rtt_deterministic --agent_features action rtt_rate_signal
    ```

### Test
To test/evaluate  the model run the following line

```
python ./run.py --envs_per_scenario 1 --wandb <project_name>--wandb_run_name <wandb_run_name> --scenario <test_scenario> --config rtt_deterministic --agent_features action rtt_rate_signal --save_name <name_of_model_we_want_to_eval> --evaluate --port_increment <num_of_port>

Example for test scenario: 8192_to_1
```
<num_of_port> each port will run independently to one another, meaning defining the port_num will enable us to run in parallel on a few scenarios
## Quantization (not used)
The quantization is done according to the article of Hao Wu (https://arxiv.org/abs/2004.09602) and the relevant package (https://gitlab-master.nvidia.com/TensorRT/Tools/pytorch-quantization)
The Quantization process is divided into 2 steps:
1. Learn the scale (amax) needed for the quantization.
    * Run the model in eval mode and enable the PyTorch quantization package to gather statistics on the model.
        ```
        python run.py --envs_per_scenario 1 --wandb <project_name> --wandb_run_name <wandb_run_name> --learning_rate 0.01 --history_length 1 --agent DETERMINISTIC --scenarios 2_to_1 4_to_1 8_to_1 --save_name <name_of_model_we_want_to_eval> --port_increment <num_of_port> --config rtt_deterministic --agent_features action rtt_rate_signal --quantization
        ```
2. Quantization of the model based on the learned scales.
    * **Run fake Quantization** : run quantization using Hao PyTorch quantization package, in this case each input/weight will be quantized (Q) and de-quantized (DQ) right after, showing the potential of the model to be quantized (cause we will loss the precision on the Q DQ process) but the calculation itself will be done using float32

        ```
         python /run.py --envs_per_scenario 1 --wandb  <project_name> --wandb_run_name <wandb_run_name>  --scenario <test_scenario> --config rtt_deterministic --agent_features action rtt_rate_signal --save_name <name_of_qunat_model_we_want_to_eval>  --evaluate --port_increment <num_of_port> --quantization
        ```
        Example for test scenario: 8192_to_1
    * **Run manual quantization**: Use the learned scales (amax) and preform the quantization manually based on Hao principles. In this case the multiplication and other linear operations will be done in int8. The only process that will still run in float32 is the non linear activations (sigmoid and tanh)

        ```
         python /run.py --envs_per_scenario 1 --wandb  <project_name> --wandb_run_name <wandb_run_name>  --scenario <test_scenario> --config rtt_deterministic --agent_features action rtt_rate_signal --save_name <name_of_qunat_model_we_want_to_eval>  --evaluate --port_increment <num_of_port> --m_quantization
        ```