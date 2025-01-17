#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Jie Li
#  Description: Structural definition for polynomial approximation function

__all__ = ['DetermPolicy', 'StochaPolicy', 'ActionValue', 'ActionValueDis', 'StateValue']


import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from math import factorial
from modules.utils.utils import get_activation_func


def combination(m, n):
    return int(factorial(m) / (factorial(n) * factorial(m - n)))


def make_features(x, degree=2):
    batch = x.shape[0]
    obs_dim = x.shape[1]
    if degree == 2:
        features_dim = combination(degree + obs_dim - 1, degree)
    elif degree == 4:
        features_dim = combination(2 + obs_dim - 1, 2) + combination(degree + obs_dim - 1, degree)
    else:
        raise ValueError("Not set degree properly")
    features = torch.zeros((batch, features_dim))

    if degree == 2:
        k = 0
        for i in range(0, obs_dim):
            for j in range(i, obs_dim):
                features[:, k:k + 1] = torch.mul(x[:, i:i + 1], x[:, j:j + 1])
                k = k + 1
    elif degree == 4:
        if obs_dim == 2:
            k = 0
            for i in range(0, obs_dim):
                for j in range(i, obs_dim):
                    features[:, k:k + 1] = torch.mul(x[:, i:i + 1], x[:, j:j + 1])
                    k = k + 1
            for h in range(0, degree + 1):
                features[:, k:k + 1] = torch.mul(x[:, 0:0 + 1] ** (degree - h), x[:, 1:1 + 1] ** h)
                k = k + 1
        elif obs_dim == 3:
            features[:, 0] = x[:, 0] ** 2
            features[:, 1] = x[:, 1] ** 2
            features[:, 2] = x[:, 2] ** 2
            features[:, 3] = torch.mul(x[:, 0], x[:, 1])
            features[:, 4] = torch.mul(x[:, 0], x[:, 2])
            features[:, 5] = torch.mul(x[:, 1], x[:, 2])
            features[:, 6] = x[:, 0] ** 4
            features[:, 7] = x[:, 1] ** 4
            features[:, 8] = x[:, 2] ** 4
            features[:, 9] = torch.mul(x[:, 0] ** 2, x[:, 1] ** 2)
            features[:, 10] = torch.mul(x[:, 0] ** 2, x[:, 2] ** 2)
            features[:, 11] = torch.mul(x[:, 1] ** 2, x[:, 2] ** 2)
            features[:, 12] = torch.mul(torch.mul(x[:, 0] ** 2, x[:, 1]), x[:, 2])
            features[:, 13] = torch.mul(torch.mul(x[:, 0], x[:, 1] ** 2), x[:, 2])
            features[:, 14] = torch.mul(torch.mul(x[:, 0], x[:, 1]), x[:, 2] ** 2)
            features[:, 15] = torch.mul(x[:, 0] ** 3, x[:, 1])
            features[:, 16] = torch.mul(x[:, 0] ** 3, x[:, 2])
            features[:, 17] = torch.mul(x[:, 1] ** 3, x[:, 0])
            features[:, 18] = torch.mul(x[:, 1] ** 3, x[:, 2])
            features[:, 19] = torch.mul(x[:, 2] ** 3, x[:, 0])
            features[:, 20] = torch.mul(x[:, 2] ** 3, x[:, 1])
        else:
            raise ValueError('Not set obs_dim properly when degree = 4')
    else:
        raise ValueError("Not set degree properly")

    return features


def make_delta_features(x, degree=2):
    batch = x.shape[0]
    obs_dim = x.shape[1]
    if degree == 2:
        features_dim = combination(degree + obs_dim - 1, degree)
    elif degree == 4:
        features_dim = combination(2 + obs_dim - 1, 2) + combination(degree + obs_dim - 1, degree)
    else:
        raise ValueError("Not set degree properly")
    delta_features = torch.zeros((batch, obs_dim, features_dim))

    if degree == 2:
        k = 0
        for i in range(0, obs_dim):
            for j in range(i, obs_dim):
                delta_features[:, i, k] += x[:, j]
                delta_features[:, j, k] += x[:, i]
                k = k + 1
    elif degree == 4:
        if obs_dim == 2:
            delta_features[:, 0, 0] = 2 * x[:, 0]
            delta_features[:, 0, 1] = x[:, 1]
            # delta_features[:, 0, 2] = 0
            delta_features[:, 0, 3] = 4 * x[:, 0] ** 3
            delta_features[:, 0, 4] = 3 * torch.mul(x[:, 0] ** 2, x[:, 1])
            delta_features[:, 0, 5] = 2 * torch.mul(x[:, 0], x[:, 1] ** 2)
            delta_features[:, 0, 6] = x[:, 1] ** 3
            # delta_features[:, 0, 7] = 0
            # delta_features[:, 1, 0] = 0
            delta_features[:, 1, 1] = x[:, 0]
            delta_features[:, 1, 2] = 2 * x[:, 1]
            # delta_features[:, 1, 3] = 0
            delta_features[:, 1, 4] = x[:, 0] ** 3
            delta_features[:, 1, 5] = 2 * torch.mul(x[:, 0] ** 2, x[:, 1])
            delta_features[:, 1, 6] = 3 * torch.mul(x[:, 0], x[:, 1] ** 2)
            delta_features[:, 1, 7] = 4 * x[:, 1] ** 3
        else:
            raise ValueError('Not set obs_dim properly when degree = 4')
    else:
        raise ValueError("Not set degree properly")

    return delta_features


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class DetermPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        self.degree = 2 if kwargs.get('polynomial_degree') is None else kwargs['polynomial_degree']
        self.pi = nn.Linear(obs_dim*self.degree, act_dim)
        action_high_limit = kwargs['action_high_limit']
        action_low_limit = kwargs['action_low_limit']
        self.register_buffer('act_high_lim', torch.from_numpy(action_high_limit))
        self.register_buffer('act_low_lim', torch.from_numpy(action_low_limit))

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        action = (self.act_high_lim-self.act_low_lim)/2 * torch.tanh(self.pi(obs))\
                 + (self.act_high_lim + self.act_low_lim)/2
        return action


class StochaPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        action_high_limit = kwargs['action_high_limit']
        action_low_limit = kwargs['action_low_limit']
        self.min_log_std = kwargs['min_log_std']
        self.max_log_std = kwargs['max_log_std']
        self.degree = 2 if kwargs.get('polynomial_degree') is None else kwargs['polynomial_degree']
        self.mean = nn.Linear(obs_dim * self.degree, act_dim)
        self.log_std = nn.Linear(obs_dim * self.degree, act_dim)
        self.register_buffer('act_high_lim', torch.from_numpy(action_high_limit))
        self.register_buffer('act_low_lim', torch.from_numpy(action_low_limit))

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        action_mean = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean(obs)) \
                      + (self.act_high_lim + self.act_low_lim) / 2
        action_std = torch.clamp(self.log_std(obs), self.min_log_std, self.max_log_std).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        self.degree = 2 if kwargs.get('polynomial_degree') is None else kwargs['polynomial_degree']
        self.q = nn.Linear((obs_dim+act_dim) * self.degree, 1)

    def forward(self, obs, act):
        input = torch.cat([obs, act], dim=-1)
        input = make_features(input, self.degree)
        q = self.q(input)
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim  = kwargs['obs_dim']
        act_num = kwargs['act_num']
        self.degree = 2 if kwargs.get('polynomial_degree') is None else kwargs['polynomial_degree']
        self.q = nn.Linear(obs_dim*self.degree, act_num)

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        return self.q(obs)


class StochaPolicyDis(ActionValueDis):
    pass


class StateValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        self.norm_matrix = torch.from_numpy(np.array(kwargs['norm_matrix'], dtype=np.float32))
        initial_weight = kwargs['initial_weight']
        self.degree = 2 if kwargs.get('polynomial_degree') is None else kwargs['polynomial_degree']
        if self.degree == 2:
            self.num_cell = combination(self.degree + obs_dim - 1, self.degree)
        elif self.degree == 4:
            self.num_cell = combination(2 + obs_dim - 1, 2) + combination(self.degree + obs_dim - 1, self.degree)
        else:
            raise ValueError("Not set degree properly")
        self.v = nn.Linear(self.num_cell, 1, bias=False)
        self.init_weights(initial_weight)

    def forward(self, obs):
        features = make_features(torch.mul(obs, self.norm_matrix), self.degree)
        return self.v(features)

    def delta_sigma(self, obs):
        return make_delta_features(obs, self.degree)

    def init_weights(self, initial_weight):
        if initial_weight is not None:
            # weight initialization
            self.v.weight = Parameter(torch.tensor(initial_weight, dtype=torch.float32), requires_grad=True)
        else:
            # zero initialization
            self.v.weight.data.fill_(0)

            # for m in self.modules():
            #     if isinstance(m, nn.Linear):
            #         m.weight.data.fill_(0)


if __name__ == "__main__":
    # value = StateValue(obs_dim=2, initial_weight=np.array([[13.033586, 1.478648, 18.642330]]))
    # # print(value)
    # # print(value.v.state_dict())
    # print(value.v.weight.detach())
    x = torch.tensor([[2, 3]])
    obs_num = 2
    degree = 4
    print(combination(2 + obs_num - 1, 2) + combination(degree + obs_num - 1, degree))
    print(make_features(x, degree=4))
