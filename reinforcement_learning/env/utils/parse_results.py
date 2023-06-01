import os

import numpy as np
import pandas as pd

import json

try:
    from .sca_parser import eval_sim_run
except ImportError:
    from sca_parser import eval_sim_run


data_title = ['#Hosts', '#QPs', '#Flows', 'Switch BW','Host BW', 'Goodput BW', 'Fairness [min host bw / max]' ,  'Latency [usec]', 'Packet loss [Gbit/s]', 'Dropped MB']

long_short_data_title = ['#Hosts', '#QPs', '#Flows', 'Long BW]', 'Long Goodput BW', 'Completion Time', 'Packet Loss', 'Dropped MB']
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'nv_ccsim/sim/results')

def parse_many_to_one(data):
    if data is None:
        return None
    try:
        total_bw = np.mean(data['switch_bw'])
        host_bw = np.mean(data['host_bw'])
        fairness = np.min(data['host_bw']) / np.max(data['host_bw'])
        latency = np.mean(data['latency'])
        dropped_bw = np.mean(data['dropped_bw'])
        total_drop = np.mean(data['total_drop'])
        goodput = np.mean(data['goodput'])
        
        print(f'Throughput BW: \t\t {total_bw} [Gbps]')
        print(f'Goodput BW: \t\t {goodput} [Gbps]')
        print(f'Host BW: \t\t {host_bw} [Gbps]')
        print(f'Fairness : \t\t {fairness}')
        print(f'Packet Latency:  \t {latency} [micro sec]')
        print(f'Packet loss: \t\t {dropped_bw} [Gbps]')
        print(f'Total Mb dropped: \t {total_drop} [Mb]')
        return [total_bw, host_bw, goodput, fairness, latency, dropped_bw, total_drop]
    except Exception as e:
        print(str(e))
        return None

def parse_data_all_to_all(data):
    if data is None:
        return None
    try: 
        total_bw = np.mean(data['switch_bw'])
        host_bw = np.mean(data['host_bw'])
        fairness = np.min(data['host_bw']) / np.max(data['host_bw'])
        latency = np.mean(data['latency'])
        dropped_bw = np.mean(data['dropped_bw'])
        total_drop = np.mean(data['total_drop'])
        goodput = np.mean(data['goodput'])
        
        print(f'Throughput BW: \t\t {total_bw} [Gbps]')
        print(f'Goodput BW: \t\t {goodput} [Gbps]')
        print(f'Host BW: \t\t {host_bw} [Gbps]')
        print(f'Fairness : \t\t {fairness}')
        print(f'Packet Latency:  \t {latency} [micro sec]')
        print(f'Packet loss: \t\t {dropped_bw} [Gbps]')
        print(f'Total Mb dropped: \t {total_drop} [Mb]')
        return [total_bw, host_bw, goodput, fairness, latency, dropped_bw, total_drop]
    except Exception as e:
        print(str(e))
        return None


def parse_longshort(data):
    if data is None:
        return None
    long_bw = data['host_bw'][0]
    completion_time = np.mean(data['completion_time'])
    dropped_bw = np.mean(data['dropped_bw'])
    total_drop = np.mean(data['total_drop'])
    goodput = np.mean(data['goodput'])
    
    print(f'Long BW: \t\t {long_bw} [Gbps]')
    print(f'Long Goodput BW: \t {goodput} [Gbps]')
    print(f'Completion Time:  \t {completion_time} [micro sec]')
    print(f'Packet loss: \t\t {dropped_bw} [Gbps]')
    print(f'Total Mb dropped: \t {total_drop} [Mb]')
    # return [total_bw, host_fairness, latency, dropped_bw, data[3], data[4], data[5]]
    return [long_bw, goodput, completion_time, dropped_bw, total_drop]


def parse_results(filename):
    if '.sca' not in filename:
        filename += '.sca'
    with open(os.path.join(os.path.dirname(__file__), 'metrics.json'), 'r') as f:
        relevant_info = json.load(f)

    if 'ManyToOne' in filename:
        relevant_info = relevant_info['many-to-one']
        parse_data_func = parse_many_to_one
    elif 'AllToAll' in filename:
        relevant_info = relevant_info['all-to-all']
        parse_data_func = parse_data_all_to_all
    elif 'LongShort' in filename:
        relevant_info = relevant_info['long-short']
        parse_data_func = parse_longshort
    try:
        eval_info = eval_sim_run(RESULTS_PATH, filename, relevant_info['metrics'], relevant_info['params'])
        test = '{}_{}_{}'.format(eval_info['params']['hosts'], eval_info['params']['qps'], eval_info['flag'])
    except Exception as e:
        print(str(e))
        return 

    hosts, qps = eval_info['params']['hosts'], eval_info['params']['qps']
    if 'LongShort' not in filename:
        print(f'Algorithm: {filename}, {hosts*qps} Flows : {hosts} Hosts, {qps} QPs per Host\n')
    else:
        hosts = hosts - 1
        print(f'Algorithm: {filename}, {hosts*qps} Flows : 1 Long Host and {hosts} Short Hosts, {qps} QPs per host\n')
    scenario_data = parse_data_func(eval_info['measurements'])
    if scenario_data is None:
        print('Failed To get Data')
        return
    scenario_data = [hosts, qps, hosts*qps] + scenario_data
    df = pd.DataFrame([scenario_data], columns= data_title if 'LongShort'  not in filename else long_short_data_title, index=['summary statistics'])
    df.to_csv(os.path.join(RESULTS_PATH, filename.replace('.sca', '.csv')))

def parse_steady_state_files():
    sca_files = [f for f in os.listdir(RESULTS_PATH) if '.sca' in f]
    with open(os.path.join(os.path.dirname(__file__), 'metrics.json'), 'r') as f:
        relevant_info = json.load(f)

    sca_data = {}
    for filename in sca_files:
        if 'ManyToOne' in filename:
            file_info = relevant_info['many-to-one']
            parse_data_func = parse_many_to_one
        elif 'AllToAll' in filename:
            file_info = relevant_info['all-to-all']
            parse_data_func = parse_data_all_to_all
        try:
            eval_info = eval_sim_run(RESULTS_PATH, filename, file_info['metrics'], file_info['params'])
            test = '{}_{}_{}'.format(eval_info['params']['hosts'], eval_info['params']['qps'], eval_info['flag'])
        except Exception as e:
            print(str(e))
            continue

        hosts, qps = eval_info['params']['hosts'], eval_info['params']['qps']
        print(f'Algorithm: {filename}, {hosts*qps} Flows : {hosts} Hosts, {qps} QPs per Host\n')
        scenario_data = parse_data_func(eval_info['measurements'])
        if scenario_data is None:
            print('Failed To get Data')
            continue
        scenario_data = [hosts, qps, hosts*qps] + scenario_data
        sca_data[filename.replace('.sca', '')] = scenario_data

    df = pd.DataFrame(sca_data.values(), columns= data_title, index=sca_data.keys())
    df.to_csv(os.path.join(RESULTS_PATH, 'steady_summary.csv'))



if __name__ == '__main__':
    print("parsing results from all sca files")
    parse_steady_state_files()

