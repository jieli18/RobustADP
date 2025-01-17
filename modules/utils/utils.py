"""


"""
import time

import torch
import torch.nn as nn
import numpy as np

from modules.utils.tensorboard_tools import tb_tags
import random


def get_activation_func(key: str):
    assert isinstance(key, str)

    activation_func = None
    if key == 'relu':
        activation_func = nn.ReLU

    elif key == 'softplus':
        activation_func = nn.Softplus

    elif key == 'elu':
        activation_func = nn.ELU

    elif key == 'selu':
        activation_func = nn.SELU

    elif key == 'gelu':
        activation_func = nn.GELU

    elif key == 'tanh':
        activation_func = nn.Tanh

    elif key == 'linear':
        activation_func = nn.Identity

    if activation_func is None:
        print('input activation name:' + key)
        raise RuntimeError

    return activation_func


def get_apprfunc_dict(key: str, type: str, **kwargs):
    var = dict()
    var['apprfunc'] = kwargs[key + '_func_type']
    var['name'] = kwargs[key + '_func_name']
    var['obs_dim'] = kwargs['obsv_dim']
    var['min_log_std'] = kwargs.get(key + '_min_log_std', float('-inf'))
    var['max_log_std'] = kwargs.get(key + '_max_log_std', float('inf'))
    var['initial_weight'] = kwargs.get('initial_weight', None)

    if type == 'MLP' or type == 'RNN':
        var['hidden_sizes'] = kwargs[key + '_hidden_sizes']
        var['hidden_activation'] = kwargs[key + '_hidden_activation']
        var['output_activation'] = kwargs[key + '_output_activation']
        var['norm_matrix'] = kwargs.get('norm_matrix', [1.0 for i in range(var['obs_dim'])])
        var['zero_constraint'] = kwargs[key + '_zero_constraint'] if kwargs.get(key + '_zero_constraint') else False
    elif type == 'GAUSS':
        var['num_kernel'] = kwargs[key + '_num_kernel']
    elif type == 'CNN':
        var['hidden_activation'] = kwargs[key + '_hidden_activation']
        var['output_activation'] = kwargs[key + '_output_activation']
        var['conv_type'] = kwargs[key + '_conv_type']
    elif type == 'POLY' or type == 'POLYNOMIAL':
        var['polynomial_degree'] = kwargs.get(key + '_polynomial_degree', 2)
        var['norm_matrix'] = kwargs.get('norm_matrix', [1.0 for i in range(var['obs_dim'])])
    else:
        raise NotImplementedError

    if kwargs['action_type'] == 'continu' or kwargs['action_type'] == 'with_adv':
        var['action_high_limit'] = kwargs['action_high_limit']
        var['action_low_limit'] = kwargs['action_low_limit']
        var['act_dim'] = kwargs['action_dim']
        if key == 'adversary' or key == 'p':
            var['action_high_limit'] = np.array(kwargs['adversary_action_high_limit'], dtype=np.float32)
            var['action_low_limit'] = np.array(kwargs['adversary_action_low_limit'], dtype=np.float32)
            var['act_dim'] = kwargs['adversary_dim']
    else:
        var['act_num'] = kwargs['action_num']

    return var


def change_type(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):  # add this line
        return obj.tolist()  # add this line
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = change_type(v)
        return obj
    elif isinstance(obj, list):
        for i, o in enumerate(obj):
            obj[i] = change_type(o)
        return obj
    else:
        return obj


def random_choice_with_index(obj_list):
    obj_len = len(obj_list)
    random_index = random.choice(list(range(obj_len)))
    random_value = obj_list[random_index]
    return random_value, random_index


def array_to_scalar(arrayLike):
    """Convert size-1 array to scalar"""
    return arrayLike if isinstance(arrayLike, (int, float)) else arrayLike.item()


# class Timer(object):
#     def __init__(self, writer, tag=tb_tags['time'], step=None):
#         self.writer = writer
#         self.tag = tag
#         self.step = step
#
#     def __enter__(self):
#         self.start = time.time()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # print(time.time() - self.start)
#         self.writer.add_scalar(self.tag, time.time() - self.start, self.step)
