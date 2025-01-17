#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Jie Li
#  Description: Simulation of trained policy
#
#  Update Date: 2022-02-12

import datetime
import os

import time

import numpy as np
import torch
from math import sqrt
from modules.create_pkg.create_env import create_env

from modules.utils.action_distributions import GaussDistribution, DiracDistribution, ValueDiracDistribution, CategoricalDistribution


class Simulator():

    def __init__(self, **kwargs):
        self.env = create_env(**kwargs)

        # import algorithm, approx func
        self.algorithm = kwargs['algorithm']
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.render = kwargs['is_render']

        self.num_eval_episode = kwargs['num_eval_episode']
        self.action_type = kwargs['action_type']
        self.policy_func_name = kwargs['policy_func_name']
        self.save_folder = kwargs['save_folder']
        self.eval_save = kwargs.get('eval_save', True)
        self.simulation_step = kwargs['simulation_step']

        if self.action_type == 'continu' or self.action_type == 'with_adv':
            if self.policy_func_name == 'StochaPolicy':
                self.action_distirbution_cls = GaussDistribution
            elif self.policy_func_name == 'DetermPolicy':
                self.action_distirbution_cls = DiracDistribution
        elif self.action_type == 'discret':
            if self.policy_func_name == 'StochaPolicyDis':
                self.action_distirbution_cls = CategoricalDistribution
            elif self.policy_func_name == 'DetermPolicyDis':
                self.action_distirbution_cls = ValueDiracDistribution
        self.print_time = 0
        self.print_iteration = -1

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, gain=None, dynamic_controller=False,
                       without_control=False, dist=False, render=False):
        self.env.reload_para()  # reload uncertain parameters in env_data
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        if dynamic_controller:
            env_A = torch.from_numpy(self.env.A)
            env_B = torch.from_numpy(self.env.B)
            env_D = torch.from_numpy(self.env.D)
            ctr_A = gain['A']
            ctr_B = gain['B']
            ctr_C = gain['C']
            ctr_D = gain['D']
            hidden_state = gain['hidden_state']
            slow_down_time = 10  # simulate a dynamic controller at 10x slower time
            tau = self.env.tau / slow_down_time  # time interval for dynamic controller simulation
        time_list = []
        obs_list = []
        action_list = []
        reward_list = []
        l2_gain_list = []
        self.env.reset()
        obs = self.env.init_obs()
        self.env.env.state = obs
        done = 0
        l2_gain_numerator = 1e-8
        l2_gain_denominator = 1e-4
        # info = {'TimeLimit.truncated': False}
        for i in range(self.simulation_step):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype('float32'))
            if gain is None:
                logits = self.networks.policy(batch_obs)
                if self.action_type == 'with_adv':
                    act = self.networks.policy(batch_obs).detach()
                    adv = self.networks.adversary(batch_obs).detach() * 0.0
                    logits = torch.cat((act, adv), dim=1)
                action_distribution = self.action_distirbution_cls(logits)
                action = action_distribution.mode()
                action = action.detach().numpy()[0]
            else:
                if dynamic_controller:
                    state = batch_obs.t()
                    for j in range(slow_down_time):
                        hidden_state = hidden_state + tau * (torch.mm(ctr_A, hidden_state) + torch.mm(ctr_B, state))
                        hideen_act = torch.mm(ctr_C, hidden_state) + torch.mm(ctr_D, state)
                        if dist:
                            hidden_dist = torch.tensor([[self.env.dist_func(i * self.env.tau + j * tau)[0]]])
                        else:
                            hidden_dist = torch.tensor([[0]])
                        state = state + tau * (torch.mm(env_A, state) + torch.mm(env_B, hideen_act) + torch.mm(env_D, hidden_dist))
                    action = np.hstack(((torch.mm(ctr_C, hidden_state) + torch.mm(ctr_D, state)).numpy()[0], np.array([0])))
                else:
                    action = torch.mm(batch_obs, gain).detach().numpy()[0]
            if without_control:
                action[:self.env.action_dim] = [0]
            if dist:
                action[self.env.action_dim:] = self.env.dist_func(i * self.env.tau)
            else:
                action[self.env.action_dim:] = [0]
            next_obs, reward, done, info = self.env.step(action)
            time_list.append([i * self.env.tau])
            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            # if 'TimeLimit.truncated' not in info.keys():
            #     info['TimeLimit.truncated'] = False
            # draw environment animation
            if render:
                self.env.render()
            reward_list.append(reward)
            if info:
                l2_gain_numerator = l2_gain_numerator + info['reward_positive']
                l2_gain_denominator = l2_gain_denominator + info['reward_negative']
                l2_gain_list.append([sqrt(l2_gain_numerator/l2_gain_denominator)])
            else:
                l2_gain_list.append([0])
        sim_dict = {'reward_list': reward_list, 'action_list': action_list, 'obs_list': obs_list,
                    'time_list': time_list, 'l2_gain_list': l2_gain_list}
        # if self.sim_save:
        #     np.save(self.save_folder + '/simulator/iteration{}_episode{}'.format(iteration, self.print_time), sim_dict)
        # episode_return = sum(reward_list)
        return sim_dict

    def run_n_episodes(self, n, iteration, gain=None):
        sim_dict_list = []
        for _ in range(n):
            sim_dict_list.append(self.run_an_episode(iteration, self.render, gain))
        return sim_dict_list

    def run_simulation(self, iteration, gain=None):
        # return self.run_n_episodes(self.num_sim_episode, iteration, gain)
        return self.run_an_episode(iteration, self.render, gain)

