#!/auto/sw_tools/OpenSource/python/INSTALLS/python_3.8.1/linux_x86_64/bin/python3.8

## usage : ./run.py --topo FT_S02_L04_H08 --AR --algo DCQCN 
# parse flags/args
# args : topology, cc_algo, simluation pattern, vectors, histograms, GBN/selective_repeat ,features: AR, --help

# for each flag, extend relevant config
## example : if algo == DCQCN 
# config_str += 'DCQCN'
#extends += 'DCQCN_tuned'
# network = 

'''
add_to_config(algo, dict[algo])



'''
# to run_configs.ini, add config:
# check if config already exists
#[Config config_str]
#extends = , , , , 

#../prog_cc/simulatiom_path/ run_configs.ini -C config_str

extension_dict = {
    'DCQCN': 'DCQCN_Tuned',
    'DC2QCN': 'DC2QCN_Tuned',
    'TIMELY': 'TIMELY_Tuned',
    'HPCC': 'HPCC_Tuned',
    'DCTC2': 'DCTC2_Tuned',
    'SWIFT': 'SWIFT_Tuned',
    'SWINT': 'SWINT_Tuned',
    'DCTCP_SW': 'DCTCP_SW_Tuned',
    'WRAPPER': 'WRAPPER_TUNED',
    'RTT_TEMPLATE': 'RTT_TEMPLATE_TUNED',
    'LongShort_ManyToOne': 'LongShort',
    'PerDest_LongShort_ManyToOne': 'PerDest_LongShort',
}
'''a dictionary holding extension syntax for configuration strings'''

config_dict = {
    'vectors': 'Vectors',
    'ar': 'AR',
} 
'''a dictionary holding configuration syntax for flag args'''

config_filter = ['algo', 'time_pat', 'traffic_pat', 'topo', 'other']
'''a list that determines which strings from args get into config name'''

def generate_config_string(args):
    '''gets script args and generates config title string'''
    config_list = []
    for arg in vars(args):
        value = getattr(args,arg)
        if type(value) == str and arg in config_filter:
            config_list.append(value)
        elif type(value) == bool and value and arg in config_dict:
            config_list.append(config_dict[arg])
        elif args.ib_flit_sim and arg == 'ar' and not value:
            config_list.append('NoAR')
        config_string = '_'.join(config_list)
    return config_string

def generate_extend_string(args):
    '''gets script args and generates extends string'''
    extend_list = []
    for arg in vars(args):
        value = getattr(args,arg)
        if type(value) == str and arg in config_filter:
            if value in extension_dict:
                extend_list.append(extension_dict[value])
            elif arg == 'topo':
                extend_list.append(value+'_Topo')
            else:
                extend_list.append(value)
        elif type(value) == bool and value and arg in config_dict:
            if config_dict[arg] in extension_dict:
                extend_list.append(extension_dict[config_dict[arg]])
            else:
                extend_list.append(config_dict[arg])
        elif args.ib_flit_sim and arg == 'ar' and not value:
            extend_list.append('NoAR')
        extend_string = ','.join(extend_list)
    return extend_string

def generate_new_config(args, ib_flit_sim_path, network_path):
    '''gets script args and generates config lines list (to be insrted in .ini file)'''
    config_lines = [f'[Config {generate_config_string(args)}]\n', f'extends = {generate_extend_string(args)}\n', '\n']
    if args.ib_flit_sim:
        config_lines.insert(1, f'include {ib_flit_sim_path}/{args.topo}/Main.ini\n')
        config_lines.insert(2, f'include {ib_flit_sim_path}/{args.topo}/neds/{args.topo}_{args.traffic_pat}.ini\n')
        config_lines.insert(3, f'network = {network_path}.{args.topo}.neds.{args.topo}\n')
    return config_lines

def get_sim_string_for_cmd(config_string):
    return config_string.split()[1][:-1]

def index_of_sub_list(test_list, sub_list):
    '''returns index of beginning of sub_list in test_list if sub_list is a sublist of test_list or -1 otherwise'''
    for i in range(len(test_list)-len(sub_list)+1):
        if test_list[i:i+len(sub_list)] == sub_list:
            return i
    return -1

def create_new_config_file(file_name):
    '''creates new .ini file in case it doesn't exist'''
    lines_list = ['include nic.ini\n', 'include traffic.ini\n', 'include omnetpp.ini\n', '\n']
    lines_list = [
        'include nic.ini\n',
        'include traffic.ini\n',
        'include omnetpp.ini\n',
        '\n',
        '[Config PerDest_ManyToOne]\n',
        'extends = ManyToOne,PerDest\n',
        '\n',
        '[Config PerDest_ManyToOne_Light]\n',
        'extends = ManyToOne_Light,PerDest\n',
        '\n',
        '[Config PerDest_AllToAll]\n',
        'sim-time-limit = 1s\n',
        'extends = AllToAll,PerDest\n',
        '\n',
        '[Config PerDest_AllToAll_Light]\n',
        'sim-time-limit = 100ms\n',
        'extends = AllToAll_Light,PerDest\n',
        '\n',
        '[Config PerDest_LongShort]\n',
        'sim-time-limit = 205ms\n',
        'extends=LongShort,PerDest\n'
        '\n'
    ]
    with open(file_name, 'w') as fp:
        fp.writelines(lines_list)

def insert_new_config(config_lines, file_name):
    '''inserts new config into given .ini file (if it doesn't already exist)'''
    new_file = False
    try:
        with open(file_name, 'r') as fp:
            file_lines = fp.readlines()
    except FileNotFoundError:
        create_new_config_file(file_name)
        new_file = True
    if not new_file and index_of_sub_list(file_lines, config_lines) == -1 and config_lines[0] in file_lines:
        raise ValueError(f'config incompatible with already existing config with same name.')
    if new_file or index_of_sub_list(file_lines, config_lines) == -1:
        with open(file_name, 'a') as fp:
            fp.writelines(config_lines)

def delete_new_config(config_lines, file_name):
    '''deletes new config from given .ini file (in case the config falied to run)'''
    with open(file_name, 'r') as fp:
        file_lines = fp.readlines()
    index = index_of_sub_list(file_lines, config_lines)
    del file_lines[index:index+len(config_lines)]
    with open(file_name, 'w') as fp:
        fp.writelines(file_lines)

def is_manual_input_inserted(args):
    if args.algo is None and args.topo is None and args.time_pat is None and args.traffic_pat is None and args.vectors is False and args.number is None and args.ib_flit_sim is False and args.ar is True and args.other is None:
        return False
    else:
        return True

def process_args(args):
    '''raises basic errors in input and prepares args for runnning the simulation(s)'''
    if args.config_file is not None and is_manual_input_inserted(args):
        parser.error("Both manual inputs and config file input are entered. Choose one!")

    if args.ib_flit_sim and args.topo is None:
        parser.error("ib_flit_sim simulation requires topology argument (--topo).")

    if args.config_file == False and (args.algo is None or args.traffic_pat is None or args.number is None):
        parser.error("running without '-c' or '--config-file' requires '--algo', '--traffic-pat' and '-r' arguments.")

    if args.traffic_pat is not None and 'AllToAll' not in args.traffic_pat and args.time_pat is None:
        parser.error("non AllToAll traffic pattern requires '--time-pat' argument.")
    
    if args.ib_flit_sim == False:
        args.ar = False

    if args.time_pat is not None and 'LongShort' in args.time_pat:
        args.time_pat += '_ManyToOne'
        args.traffic_pat = None

def init_args(args, inputs, iter_list):
    '''initializes args with constant parameters'''
    for arg in vars(args):
        if arg == 'config_file':
            setattr(args,arg,False) #to avoid an effect on the rest of the operation
        elif arg not in iter_list:
            setattr(args,arg,inputs[arg])

def ignore_ib_flit_sim_inputs(inputs):
    '''ignores ib_flit_sim related inputs if ib_flit_sim is False'''
    if inputs['ib_flit_sim'] is False:
        inputs['topo'] = None
        inputs['ar'] = False
    return inputs

def run_single_simulation(args, ini_file_name, ib_flit_sim_path, network_path, parallel = False, previous_simulations = None):
    '''inserts new config(s) to file and runs simulation(s), unless simulation is already running in parrallel'''
    config_lines = generate_new_config(args, ib_flit_sim_path, network_path)
    sim_to_run = get_sim_string_for_cmd(config_lines[0])
    try:
        insert_new_config(config_lines, ini_file_name)
    except ValueError as error:
        parser.error(error)
    if args.ib_flit_sim:
        command = f'{ib_flit_sim_path[:-12]}/out/gcc-release/build/ib_flit_sim  {ini_file_name} -c {sim_to_run} -r {args.number} -u Cmdenv'
    else:
        command = f'../out/gcc-release/src/ccsim {ini_file_name} -c {sim_to_run} -r {args.number} -u Cmdenv'
    if parallel:
        if sim_to_run in previous_simulations:
            return None, None
        else:
            return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE), f'{sim_to_run}-{args.number}'
    else:
        subprocess.run(command, shell=True, check=True)

def process_json(args, inputs):
    '''initializes args with constant parameters, returns list of keys which have lists as values and list of lists for iterable parameters'''
    inputs = ignore_ib_flit_sim_inputs(inputs)
    iter_list = [key for key in inputs if type(inputs[key]) is list]
    list_of_lists = [inputs[arg] for arg in vars(args) if arg in inputs and type(inputs[arg]) is list]
    init_args(args, inputs, iter_list)
    return iter_list, list_of_lists

def edit_args(args, iter_list, combination):
    '''edits args for iterable parameters and deletes not needed time-pat in case of running AllToAll'''
    count = 0
    for arg in vars(args):
        if arg in iter_list:
            setattr(args,arg,combination[count])
            count += 1
    deleted_time_pat = None
    if 'AllToAll' in args.traffic_pat:
        deleted_time_pat = getattr(args,'time_pat')
        setattr(args,'time_pat',None)
    return deleted_time_pat

def run_simulations(args, iter_list, list_of_lists, ini_file_name, ib_flit_sim_path, network_path):
    '''inserts new configs to file and runs multiple simulations'''
    simulations = []
    for combination in product(*list_of_lists):
        deleted_time_pat = edit_args(args, iter_list, combination)
        process, sim_name = run_single_simulation(args, ini_file_name, ib_flit_sim_path, network_path, parallel = True, previous_simulations = [item[1] for item in simulations])
        if process is not None:
            simulations.append((process, sim_name))
            print(f'{sim_name} running...')
        if deleted_time_pat:
            setattr(args,'time_pat',deleted_time_pat)
    print('\n')
    count = 0
    percent_list = [0]*len(simulations)
    output_list = ['']*len(simulations)
    done_list = [False]*len(simulations)
    while count < len(simulations):
        time.sleep(2)
        for i, simulation in enumerate(simulations):
            while percent_list[i] < 100:
                temp = simulation[0].stdout.read(1).decode('utf-8')
                if temp:
                    output_list[i] += temp
                    if output_list[i][-9:] == 'completed':
                        percent_list[i] = int(output_list[i][-14:-11])
                        break
                else:
                    break
        for i, simulation in enumerate(simulations):
            process_status = simulation[0].poll()
            if process_status is None:
                print(f'{simulation[1]}: {percent_list[i]}%')
            elif process_status != 0:
                print(f'error in running {simulation[1]}')
            elif done_list[i] is False:
                done_list[i] = True
                count += 1
                print(f'{simulation[1]} finished successfully')
        print('\n')

    # while count < len(simulations):
    #     time.sleep(5)
    #     for simulation in simulations:
    #         if simulation[0].poll() is None:
    #             print(f'{simulation[1]} status update')
    #         elif simulation[0].poll() != 0:
    #             print(f'error in running {simulation[1]}')
    #         else:
    #             print(f'{simulation[1]} finished')

import argparse
import subprocess
import json
from itertools import product 
import time

parser = argparse.ArgumentParser(description='Generate simulation config and run simulation.')
parser.add_argument('--algo', help='a string describing CC algorithm e.g DCTC2')
parser.add_argument('--topo', help='a string describing simulation topology e.g FT_S02_L04_H32')
parser.add_argument('--time-pat', dest='time_pat', help='a string describing simulation timing pattern e.g. Simult')
parser.add_argument('--traffic-pat', dest='traffic_pat', help='a string describing simulation traffic pattern e.g. ManyToOne')
parser.add_argument('-v', '--vectors', action='store_true', help='add Vectors to simulation configuration')
parser.add_argument('-r', '--number', type=int, help='set flag number for simulation scale')#TODO: ask Rotem if this is a good description
parser.add_argument('--ib-flit-sim', dest='ib_flit_sim', action='store_true', help='indicates working with ib-flit-sim')
parser.add_argument('-a', '--no-ar', dest='ar',action='store_false', help='make ib_flit_sim simulation run with NO adaptive routing')
parser.add_argument('-o', '--other', help='manually insert extensions to config')
parser.add_argument('-c', '--config-file', dest='config_file', help='run simulation(s) with config_run_sim.json file instead of input arguments in command line. Please enter dictionary name')

args = parser.parse_args()

if args.ib_flit_sim:
    command = 'export NEDPATH=../../ib_flit_sim/src:../../ib_flit_sim/calibration:../../ib_flit_sim/mpi/mpi_proc:../../ib_flit_sim/simulations/:../FW/Algorithm:../DCTrafficGen/src:../src'
    subprocess.run(command, shell=True, check=True)

process_args(args)

ini_file_name = 'temp.ini'
ib_flit_sim_path = '../../ib_flit_sim/simulations'
network_path = 'ib_model.simulations'

if args.config_file is None:
    run_single_simulation(args, ini_file_name, ib_flit_sim_path, network_path)
else:
    with open('config_run_sim.json','r') as fp:
        inputs = json.load(fp)[args.config_file]
    iter_list, list_of_lists = process_json(args, inputs)
    run_simulations(args, iter_list, list_of_lists, ini_file_name, ib_flit_sim_path, network_path)
