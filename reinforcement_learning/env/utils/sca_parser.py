import re
import numpy as np

class ParsingError(ValueError):
    # Raised when an unexpected pattern is encoutered in simulation result file
    pass

def eval_wrap(str_to_eval):
    if type(str_to_eval) != str:
        raise ParsingError(f'unexpected pattern in simulation result: {str_to_eval}')
    match = re.search('[a-df-zA-DF-Z]', str_to_eval)
    if match:
        raise ParsingError(f'unexpected pattern in simulation result: {str_to_eval}')
    if str_to_eval.count('"') == 2:
        str_to_eval = str_to_eval.split('"')[1]
    return eval(str_to_eval)

def get_sim_name(dir_to_eval, file_name):
    with open(f'{dir_to_eval}/{file_name}','r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('attr configname '):
                sim_name = line.split()[2]
                break
    return sim_name

def get_attributes(dir_to_eval, file_name, params_parse_info):
    attr_flag = False
    with open(f'{dir_to_eval}/{file_name}','r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('attr H '):
                host_num = eval_wrap(line[len('attr H '):])
            elif line.startswith('attr Q '):
                qp_num = eval_wrap(line[len('attr Q '):])
            if line.startswith('attr'):
                    attr_flag = True
            elif attr_flag:
                break
    params = {
        "_hosts": host_num,
        "_qps": qp_num
    }
    for param, string in params_parse_info.items():
        if string is None:
            params[param] = None
        else:
            attr_flag = False
            with open(f'{dir_to_eval}/{file_name}','r') as f:
                lines = f.readlines()
                for line in lines:
                    if string.endswith(' ') and line.startswith(string) or line.startswith(f'{string} '):
                        params[param] = eval_wrap(line[len(string):])
                        break
                    if line.startswith('attr'):
                        attr_flag = True
                    elif attr_flag:
                        params[param] = None
                        break
    return params

def get_index(line, filter, key_word, ancestor_index = None):
    if key_word == 'index':
        return get_index(line, filter, 'ancestorIndex', 0)
    elif key_word == 'parentIndex':
        return get_index(line, filter, 'ancestorIndex', 1)
    elif key_word == 'ancestorIndex':
        split_line = line.split('.')
        filter_position = -1
        for i in range(len(split_line)):
            if filter in split_line[i]:
                filter_position = i
                break
        if filter_position-ancestor_index < 0:
            raise ValueError(f'index out of range')
        return split_line[filter_position-ancestor_index].split('[')[1].split(']')[0]

def insert_index(line, filter, constraint, index, key_word):
    if key_word == 'index':
        constraint = constraint[:index] + str(get_index(line, filter, key_word)) + constraint[index+len(key_word):]
    elif key_word == 'parentIndex':
        constraint = constraint[:index] + str(get_index(line, filter, key_word)) + constraint[index+len(key_word)+len('()'):]
    else: #key_word = 'ancestorIndex'
        ancestor_index_input_str = constraint[index+len('ancestorIndex'):].split('(')[1].split(')')[0]
        constraint = constraint[:index] + str(get_index(line, filter, key_word, int(ancestor_index_input_str))) + constraint[index+len(key_word)+len(f'({ancestor_index_input_str})'):]
    return constraint

def eval_constraint(line, filter, constraint, params):
    for param in sorted(params, reverse=True):
        index = need_replace(constraint, param)
        while index != -1:
            if params[param] is None:
                raise ValueError(f'attribute not found: {param}')
            constraint = constraint[:index] + str(params[param]) + constraint[index+len(param):]
            index = need_replace(constraint, param)
    for key_word in ['index', 'parentIndex', 'ancestorIndex']:
        index = need_replace(constraint, key_word)
        while index != -1:
            try:
                constraint = insert_index(line, filter, constraint, index, key_word)
            except (ValueError, IndexError) as error:
                raise ValueError(f'Conflicting line and constraint:\nline: {line}\nconstraint: {constraint}') from error
            index = need_replace(constraint, key_word)
    try:
        ret_val = eval(constraint)
    except Exception as error:
        raise ValueError(f'Faulty constraint: {constraint}') from error
    return ret_val

def line_has_data(line, param_name, filters, params):
    if param_name in line:
        if filters is not None:
            for filter, constraint in filters.items():
                if eval_constraint(line, filter, constraint, params) is False:
                    return False
        return True
    else:
        return False

def get_metric_from_line(lines, parse_info, i, file_name):
    for j in range(i, len(lines)):
        if parse_info["param_name"] not in lines[j]:
            break
        elif parse_info["field"] is None or parse_info["field"] in lines[j]:
            return eval_wrap(lines[j].split()[3])
    for j in range(i+1, len(lines)):
        if not lines[j].startswith('field'):
            raise ParsingError(f'{parse_info["param_name"]}: field {parse_info["field"]} not found in {file_name}')
        if lines[j].split()[1] == parse_info["field"]:
            return eval_wrap(lines[j].split()[2])

def get_accumulated_metric(result_vec, acc_rule):
    if acc_rule is None:
        return result_vec
    elif acc_rule == 'max':
        return np.max(result_vec)
    elif acc_rule == 'min':
        return np.min(result_vec)
    elif acc_rule == 'avg':
        return np.average(result_vec)
    elif acc_rule == 'stdev':
        return np.std(result_vec)/np.average(result_vec)
    elif acc_rule == 'sum':
        return np.sum(result_vec)
    elif acc_rule.endswith('percentile') and len(acc_rule.split()) == 2 and acc_rule.split()[0].isnumeric():
        return np.percentile(result_vec, int(acc_rule.split()[0]), interpolation='lower')
    else:
        raise ValueError(f'unknown accumulation rule: {acc_rule}')

def get_metric_from_sim_results(dir_to_eval, file_name, parse_info, params):
    with open(f'{dir_to_eval}/{file_name}','r') as f:
        lines = f.readlines()
        result_vec = []
        for i in range(0, len(lines)):
            filters = None
            if 'filters' in parse_info:
                filters = parse_info['filters']
            if line_has_data(lines[i], parse_info['param_name'], filters, params):
                result_vec.append(get_metric_from_line(lines, parse_info, i, file_name))
    return result_vec

def get_measurements(dir_to_eval, file_name, metrics_parse_info, params_parse_info):
    params = get_attributes(dir_to_eval, file_name, params_parse_info)
    measurements = {}
    for metric in metrics_parse_info:
        if metrics_parse_info[metric] is None:
            measurements[metric] = None
        else:
            measurements[metric] = get_metric_from_sim_results(dir_to_eval, file_name, metrics_parse_info[metric], params)
    return measurements, params

def need_replace(op_rule, metric):
    for i in range(len(op_rule)-len(metric)+1):
        if op_rule[i:i+len(metric)] == metric:
            return i
    return -1

def get_edited_metrics(measurements, params, metrics_parse_info, params_parse_info):
    for metric in sorted(metrics_parse_info, reverse=True):
        if metrics_parse_info[metric] is not None and 'operation' in metrics_parse_info[metric] and metrics_parse_info[metric]['operation'] is not None:
            for i in range(len(measurements[metric])):
                temp_op_rule = metrics_parse_info[metric]['operation']
                if params_parse_info is not None:
                    for param in sorted(params_parse_info, reverse=True):
                        index = need_replace(temp_op_rule, param)
                        while index != -1:
                            temp_op_rule = temp_op_rule[:index] + str(params[param]) + temp_op_rule[index+len(param):]
                            index = need_replace(temp_op_rule, param)
                index = need_replace(temp_op_rule, metric)
                while index != -1:
                    temp_op_rule = temp_op_rule[:index] + str(measurements[metric][i]) + temp_op_rule[index+len(metric):]
                    index = need_replace(temp_op_rule, metric)
                measurements[metric][i] = temp_op_rule
            for i in range(len(measurements[metric])):
                measurements[metric][i] = eval(measurements[metric][i])
    return measurements

def process_raw_data(measurements, params, metrics_parse_info, params_parse_info):
    measurements = get_edited_metrics(measurements, params, metrics_parse_info, params_parse_info)
    for metric in metrics_parse_info:
        if metrics_parse_info[metric] is not None and 'accumulation' in metrics_parse_info[metric]:
            measurements[metric] = get_accumulated_metric(measurements[metric], metrics_parse_info[metric]['accumulation'])
    return measurements

def eval_sim_run(dir_to_eval, file_name, metrics_parse_info, params_parse_info):
    sim_name = get_sim_name(dir_to_eval, file_name)
    measurements, params = get_measurements(dir_to_eval, file_name, metrics_parse_info, params_parse_info)
    measurements = process_raw_data(measurements, params, metrics_parse_info, params_parse_info)
    flag = int(file_name.split('-')[1].split('.')[0])
    test_result = {
        "type": sim_name,
        "flag": flag,
        "params": params,
        "measurements": measurements,
    }
    return test_result
