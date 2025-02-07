#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yao MU
#  Description: Structural definition for approximation function
#
#  Update Date: 2021-05-21, Shengbo Li: revise headline


__all__ = ['DetermPolicy', 'StochaPolicy', 'ActionValue', 'ActionValueDis', 'StateValue']

import numpy as np  # Matrix computation library
import torch
import torch.nn as nn
from modules.utils.utils import get_activation_func


# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Count parameter number of MLP
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# Deterministic policy
class DetermPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.norm_matrix = torch.from_numpy(np.array(kwargs['norm_matrix'], dtype=np.float32))
        self.zero_constraint = kwargs['zero_constraint']
        self.register_buffer('act_high_lim', torch.from_numpy(kwargs['action_high_limit']))
        self.register_buffer('act_low_lim', torch.from_numpy(kwargs['action_low_limit']))

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes,
                      get_activation_func(kwargs['hidden_activation']),
                      get_activation_func(kwargs['output_activation']))
        self.init_weights()

    def forward(self, obs):
        pi = self.pi(torch.mul(obs, self.norm_matrix))
        if self.zero_constraint:
            pi = pi - self.pi(torch.zeros_like(obs)).detach()
        normalized_action = torch.tanh(pi)
        action = (self.act_high_lim - self.act_low_lim)/2 * normalized_action + (self.act_high_lim + self.act_low_lim)/2
        return action

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weight_shape = list(m.weight.data.size())
                fan_in = weight_shape[1]
                fan_out = weight_shape[0]
                w_bound = np.sqrt(6. / (fan_in + fan_out))
                m.weight.data.uniform_(-w_bound, w_bound)
                m.bias.data.fill_(0)


# Stochastic Policy
class StochaPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.mean = mlp(pi_sizes,
                        get_activation_func(kwargs['hidden_activation']),
                        get_activation_func(kwargs['output_activation']))
        self.log_std = mlp(pi_sizes,
                           get_activation_func(kwargs['hidden_activation']),
                           get_activation_func(kwargs['output_activation']))
        self.min_log_std = kwargs['min_log_std']
        self.max_log_std = kwargs['max_log_std']
        self.register_buffer('act_high_lim', torch.from_numpy(kwargs['action_high_limit']))
        self.register_buffer('act_low_lim', torch.from_numpy(kwargs['action_low_limit']))

    def forward(self, obs):
        action_mean = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean(obs)) \
                      + (self.act_high_lim + self.act_low_lim) / 2
        action_std = torch.clamp(self.log_std(obs), self.min_log_std, self.max_log_std).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_num = kwargs['act_num']
        hidden_sizes = kwargs['hidden_sizes']
        print(kwargs['output_activation'])
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_num],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))

    def forward(self, obs):
        return self.q(obs)


class StochaPolicyDis(ActionValueDis):
    pass


class StateValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.norm_matrix = torch.from_numpy(np.array(kwargs['norm_matrix'], dtype=np.float32))
        self.zero_constraint = kwargs['zero_constraint']
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))
        self.init_weights()

    def forward(self, obs):
        v = self.v(torch.mul(obs, self.norm_matrix))
        if self.zero_constraint:
            v = v - self.v(torch.zeros_like(obs)).detach()
        return torch.squeeze(v, -1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weight_shape = list(m.weight.data.size())
                fan_in = weight_shape[1]
                fan_out = weight_shape[0]
                w_bound = np.sqrt(6. / (fan_in + fan_out))
                m.weight.data.uniform_(-w_bound, w_bound)
                m.bias.data.fill_(0)

#
