#  Creator: Jie Li
#  Update Date: 2022-03-30
#  Robust Adaptive Dynamic Programming Algorithm (RADP)


__all__ = ['RADP']

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import time
import math
import warnings

from modules.create_pkg.create_apprfunc import create_apprfunc
from modules.create_pkg.create_env_model import create_env_model
from modules.utils.utils import get_apprfunc_dict
from modules.utils.tensorboard_tools import tb_tags


class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.env_model = create_env_model(**kwargs)

        self.value_func_type = kwargs['value_func_type']
        value_args = get_apprfunc_dict('value', self.value_func_type, **kwargs)
        self.value = create_apprfunc(**value_args)
        self.value_target = deepcopy(self.value)
        self.value_optimizer = SGD(self.value.parameters(), lr=kwargs['value_learning_rate'])  # ML试验表明该优化器慢
        # self.value_optimizer = Adam(self.value.parameters(), lr=kwargs['value_learning_rate'], betas=(0.9, 0.99))
        self.max_gradient_norm = kwargs['max_gradient_norm']

        self.scheduler_lr = kwargs.get('schedule_lr', 'none')
        if self.scheduler_lr == 'exponential':
            start_epoch = kwargs['start_epoch']
            base = kwargs['base']
            lr_lambda = lambda epoch: 1.0 if epoch < start_epoch else base ** (epoch - start_epoch)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda, last_epoch=-1)
        else:
            lr_lambda = lambda epoch: 1.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda, last_epoch=-1)
        self.update_step = 0

    def policy(self, batch_obs):
        batch_obs.requires_grad_(True)
        batch_value_target = self.value_target(batch_obs)
        batch_delta_value_target, = torch.autograd.grad(torch.sum(batch_value_target), batch_obs, create_graph=True)
        batch_obs.requires_grad_(False)
        batch_act = self.env_model.best_act(batch_obs.detach(), batch_delta_value_target)
        batch_adv = self.env_model.worst_adv(batch_obs.detach(), batch_delta_value_target)
        return torch.cat((batch_act, batch_adv), dim=1)

    def update(self, grads):
        # load grad and limit the norm
        for p, grad in zip(self.value.parameters(), grads):
            if self.value_func_type == 'POLY' or self.value_func_type == 'POLYNOMIAL':
                grad_norm = torch.norm(grad).item()
                if grad_norm > self.max_gradient_norm:
                    grad = grad / grad_norm * self.max_gradient_norm
            p.grad = grad

        # perform gradient descent
        self.value_optimizer.step()

        # update the learning rate (if necessary)
        self.scheduler.step(self.update_step)
        self.update_step += 1

        # copy value network
        self.value_target = deepcopy(self.value)


class RADP:
    __has_gpu = torch.cuda.is_available()

    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.value_func_type = kwargs['value_func_type']
        self.value_lr = kwargs['value_learning_rate']
        self.additional_lr = kwargs['additional_term_learning_rate']
        self.max_iteration = kwargs['max_iteration']
        self.prob_intensity = kwargs['prob_intensity']
        self.log_save_interval = kwargs['log_save_interval']
        self.print_interval = kwargs['print_interval']

        self.env_id = kwargs['env_id']
        self.env_model = create_env_model(**kwargs)
        self.enable_cuda = kwargs['enable_cuda']
        self.is_gpu = self.__has_gpu and self.enable_cuda
        # ------------------------------------
        if self.is_gpu:
            self.env_model = self.env_model.cuda()
        # ------------------------------------

        if self.value_func_type == 'POLY' or self.value_func_type == 'POLYNOMIAL':
            self.value_weight = np.ones([math.ceil(self.max_iteration / self.log_save_interval),
                                         self.networks.value_target.num_cell], dtype="float32")
        self.loss_value = np.ones([math.ceil(self.max_iteration / self.log_save_interval), 1], dtype="float32")
        self.state_history = np.ones([math.ceil(self.max_iteration / self.log_save_interval),
                                      self.env_model.state_dim], dtype="float32")
        self.action_history = np.ones([math.ceil(self.max_iteration / self.log_save_interval),
                                       2 * (self.env_model.action_dim + self.env_model.adversary_dim)], dtype="float32")

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "'is not defined in algorithm!"
                warnings.warn(warning_msg)

    def get_parameters(self):
        params = dict()
        params['is_gpu'] = self.is_gpu

        return params

    def compute_gradient(self, data, iteration):

        start_time = time.time()
        self.networks.value.zero_grad()
        loss_value, value = self.compute_loss_value(deepcopy(data))
        loss_value.backward()
        end_time = time.time()

        value_grad = [p.grad for p in self.networks.value.parameters()]

        tb_info = dict()
        tb_info[tb_tags["loss_critic"]] = loss_value.item()
        tb_info[tb_tags["critic_avg_value"]] = value.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        if iteration % self.log_save_interval == 0:
            if self.value_func_type == 'POLY' or self.value_func_type == 'POLYNOMIAL':
                self.value_weight[int(iteration / self.log_save_interval)] = self.networks.value_target.v.weight.detach().numpy().squeeze()
            self.loss_value[int(iteration / self.log_save_interval), 0] = loss_value.item()
            self.state_history[int(iteration / self.log_save_interval)] = data['obs'].detach().numpy()[0]
            act_with_probing = data['act'].detach().numpy()[0]
            advers_with_probing = data['advers'].detach().numpy()[0]
            action_with_probing = np.hstack((act_with_probing, advers_with_probing))
            if abs(self.prob_intensity) < 1e-8:
                probing = 0 * action_with_probing
            else:
                probing = data['probing'].detach().numpy()[0]
            action = action_with_probing - probing
            self.action_history[int(iteration / self.log_save_interval)] = np.hstack((action, probing))

        if iteration % self.print_interval == 0:
            if self.value_func_type == 'POLY' or self.value_func_type == 'POLYNOMIAL':
                print(f'ite: {iteration}, loss value = {loss_value.item():.4f}, '
                      f'weight = {self.networks.value.v.weight.detach()}')
            else:
                print(f'ite: {iteration}, loss value = {loss_value.item():.4f}')

        return value_grad, tb_info

    def compute_loss_value(self, data):
        state = data['obs']

        state.requires_grad_(True)  # torch.Size([batch, 3])
        value = self.networks.value(state)  # torch.Size([batch, 1])
        delta_value, = torch.autograd.grad(torch.sum(value), state, create_graph=True)  # torch.Size([batch, 3])
        state.requires_grad_(False)

        estimated_hamiltonian = self._estimated_hamiltonian(state, delta_value)  # torch.Size([batch, 1])
        clip_stability, derivative_lyapunov = self._derivative_lyapunov(state[-1:, :], delta_value[-1:, :])  # torch.Size([1, 1])
        loss_value = 0.5 * torch.mm(estimated_hamiltonian.t(), estimated_hamiltonian) \
                     + self.additional_lr / self.value_lr * torch.mm(clip_stability.t(), derivative_lyapunov)

        return loss_value, torch.mean(value)

    def _estimated_hamiltonian(self, state, delta_value):

        batch_size = state.size()[0]
        Q = self.env_model.Q
        R = self.env_model.R
        gamma_atte = self.env_model.gamma_atte

        if batch_size > 1:
            Q_x = torch.bmm(torch.mm(state, Q)[:, np.newaxis, :], state[:, :, np.newaxis]).squeeze(-1)
            m_x = torch.bmm(self.env_model.m_x(state, batch_size)[:, np.newaxis, :],
                            self.env_model.m_x(state, batch_size)[:, :, np.newaxis]).squeeze(-1)
            dv_fx = torch.bmm(delta_value[:, np.newaxis, :], self.env_model.f_x(state, batch_size)[:, :, np.newaxis]).squeeze(-1)
            D_d = 1 / gamma_atte ** 2 * torch.bmm(self.env_model.k_x(state, batch_size),
                                                  self.env_model.k_x(state, batch_size).transpose(1, 2)) \
                  + torch.bmm(self.env_model.E_f(state), self.env_model.E_f(state).transpose(1, 2)) \
                  - torch.bmm(torch.matmul(self.env_model.g_x(state, batch_size), torch.inverse(R)),
                              self.env_model.g_x(state, batch_size).transpose(1, 2))
            hamiltonian = Q_x + m_x + dv_fx + 1/4 * torch.bmm(torch.bmm(delta_value[:, np.newaxis, :], D_d),
                                                              delta_value[:, :, np.newaxis]).squeeze(-1)
        else:
            Q_x = torch.mm(torch.mm(state, Q), state.t())
            m_x = torch.mm(self.env_model.f_x(state, 1).t(), self.env_model.f_x(state, 1))  # todo
            dv_fx = torch.mm(delta_value, self.env_model.f_x(state, 1))
            D_d = 1 / gamma_atte ** 2 * torch.mm(self.env_model.k_x(state, 1), self.env_model.k_x(state, 1).t()) \
                  + torch.mm(self.env_model.E_f(state), self.env_model.E_f(state).t()) \
                  - torch.mm(torch.mm(self.env_model.g_x(state, 1), torch.inverse(R)), self.env_model.g_x(state, 1).t())
            hamiltonian = Q_x + m_x + dv_fx + 1/4 * torch.mm(torch.mm(delta_value, D_d), delta_value.t())

        return hamiltonian

    def _derivative_lyapunov(self, state, delta_value):

        batch_size = state.size()[0]
        R = self.env_model.R
        gamma_atte = self.env_model.gamma_atte

        if batch_size > 1:
            D_d = 1 / gamma_atte ** 2 * torch.bmm(self.env_model.k_x(state, batch_size),
                                                  self.env_model.k_x(state, batch_size).transpose(1, 2)) \
                  - torch.bmm(torch.matmul(self.env_model.g_x(state, batch_size), torch.inverse(R)),
                              self.env_model.g_x(state, batch_size).transpose(1, 2))
            delta_state = self.env_model.f_x(state, batch_size) \
                          + 1/2 * torch.bmm(D_d, delta_value[:, :, np.newaxis]).squeeze(-1)  # [64, 2]
            derivative_lyapunov = torch.bmm(state[:, np.newaxis, :],
                                            delta_state[:, :, np.newaxis]).squeeze(-1)  # [64, 1]
        else:
            D_d = 1 / gamma_atte ** 2 * torch.mm(self.env_model.k_x(state, 1), self.env_model.k_x(state, 1).t()) \
                  - torch.mm(torch.mm(self.env_model.g_x(state, 1), torch.inverse(R)), self.env_model.g_x(state, 1).t())
            delta_state = self.env_model.f_x(state, 1) + 1/2 * torch.mm(D_d, delta_value.t())  # [2, 1]
            derivative_lyapunov = torch.mm(state, delta_state)  # [1, 1]

        clip_stability = torch.where(derivative_lyapunov[:, 0] < 0,
                                     torch.zeros(batch_size), torch.ones(batch_size))[:, np.newaxis].detach()

        return clip_stability, derivative_lyapunov

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)


if __name__ == '__main__':
    print('11111')
