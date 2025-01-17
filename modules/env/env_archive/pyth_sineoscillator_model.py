#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Jie Li
#  Description: Sineoscillator Environment
#

import warnings
import torch
import numpy as np

pi = torch.tensor(np.pi, dtype=torch.float32)


class PythSineoscillatorModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        """
        you need to define parameters here
        """
        # define common parameters here
        self.is_adversary = kwargs['is_adversary']
        self.sample_batch_size = kwargs['sample_batch_size']
        self.state_dim = 2
        self.action_dim = 1
        self.adversary_dim = 1
        self.dt = 1 / 200  # seconds between state updates

        # define your custom parameters here
        self.p1_0 = 0.0
        self.p2_0 = 0.0
        self.Ef = torch.eye(self.state_dim)

        # utility information
        self.Q = 2 * torch.eye(self.state_dim)
        self.R = 2 * torch.eye(self.action_dim)
        self.gamma = 1
        self.gamma_atte = kwargs['gamma_atte']

        # state & action space
        self.fixed_initial_state = kwargs['fixed_initial_state']  # for env_data & on_sampler
        self.initial_state_range = kwargs['initial_state_range']  # for env_model
        self.battery_a_initial = self.initial_state_range[0]
        self.battery_b_initial = self.initial_state_range[1]
        self.state_threshold = kwargs['state_threshold']
        self.battery_a_threshold = self.state_threshold[0]
        self.battery_b_threshold = self.state_threshold[1]
        self.max_action = [2.0]
        self.min_action = [-2.0]
        self.max_adv_action = [1.0]
        self.min_adv_action = [-1.0]

        self.lb_state = torch.tensor([-self.battery_a_threshold, -self.battery_b_threshold], dtype=torch.float32)
        self.hb_state = torch.tensor([self.battery_a_threshold, self.battery_b_threshold], dtype=torch.float32)
        self.lb_action = torch.tensor(self.min_action + self.min_adv_action, dtype=torch.float32)  # action & adversary
        self.hb_action = torch.tensor(self.max_action + self.max_adv_action, dtype=torch.float32)

        self.ones_ = torch.ones(self.sample_batch_size)
        self.zeros_ = torch.zeros(self.sample_batch_size)

        # parallel sample
        self.parallel_state = None
        self.lower_step = kwargs['lower_step']
        self.upper_step = kwargs['upper_step']
        self.max_step_per_episode = self.max_step()
        self.step_per_episode = self.initial_step()

    def max_step(self):
        return torch.from_numpy(np.floor(np.random.uniform(self.lower_step, self.upper_step, [self.sample_batch_size])))

    def initial_step(self):
        return torch.zeros(self.sample_batch_size)

    def reset(self):

        battery_a = np.random.uniform(-self.battery_a_initial, self.battery_a_initial, [self.sample_batch_size, 1])
        battery_b = np.random.uniform(-self.battery_b_initial, self.battery_b_initial, [self.sample_batch_size, 1])

        state = np.concatenate([battery_a, battery_b], axis=1)  # concatenate column

        return torch.from_numpy(state).float()

    def step(self, action: torch.Tensor):
        dt = self.dt
        p1_0 = self.p1_0
        p2_0 = self.p2_0
        battery_a, battery_b = self.parallel_state[:, 0], self.parallel_state[:, 1]
        memristor = action[:, 0]  # memristor
        noise = action[:, 1]      # noise

        deri_battery_a = - battery_a + battery_b + p1_0 * torch.mul(battery_b, torch.sin(battery_a))
        deri_battery_b = - 0.5 * battery_a - 0.5 * battery_b \
                         + 0.5 * torch.mul(battery_b, (torch.cos(2 * battery_a) + 2) ** 2) \
                         - 1 / (self.gamma_atte ** 2) * torch.mul(battery_b, (torch.sin(4 * battery_a) + 2) ** 2) \
                         + p2_0 * torch.mul(battery_a, torch.cos(battery_b)) \
                         + torch.mul(torch.cos(2 * battery_a) + 2, memristor) + torch.mul(torch.sin(4 * battery_a) + 2, noise)

        delta_state = torch.stack([deri_battery_a, deri_battery_b], dim=-1)
        self.parallel_state = self.parallel_state + delta_state * dt

        reward = (self.Q[0][0] * battery_a ** 2 + self.Q[1][1] * battery_b ** 2
                  + self.R[0][0] * (memristor ** 2).squeeze(-1) - self.gamma_atte ** 2 * (noise ** 2).squeeze(-1))

        # define the ending condation here the format is just like isdone = l(next_state)
        done = (torch.where(abs(self.parallel_state[:, 0]) > self.battery_a_threshold, self.ones_, self.zeros_).bool()
                | torch.where(abs(self.parallel_state[:, 1]) > self.battery_b_threshold, self.ones_, self.zeros_).bool())

        self.step_per_episode += 1
        info = {'TimeLimit.truncated': torch.where(self.step_per_episode > self.max_step_per_episode,
                                                   self.ones_, self.zeros_).bool()}

        return self.parallel_state, reward, done, info

    def forward(self, state: torch.Tensor, action: torch.Tensor, beyond_done=torch.tensor(1)):
        """
        rollout the model one step, notice this method will not change the value of self.state
        you need to define your own state transition  function here
        notice that all the variables contains the batch dim you need to remember this point
        when constructing your function
        :param state: datatype:torch.Tensor, shape:[batch_size, state_dim]
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :param beyond_done: flag indicate the state is already done which means it will not be calculated by the model
        :return:
                next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                              the state will not change anymore when the corresponding flag done is set to True
                reward:  datatype:torch.Tensor, shape:[batch_size, 1]
                isdone:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        warning_msg = "action out of action space!"
        if not ((action <= self.hb_action).all() and (action >= self.lb_action).all()):
            # warnings.warn(warning_msg)
            action = clip_by_tensor(action, self.lb_action, self.hb_action)

        warning_msg = "state out of state space!"
        if not ((state <= self.hb_state).all() and (state >= self.lb_state).all()):
            # warnings.warn(warning_msg)
            state = clip_by_tensor(state, self.lb_state, self.hb_state)

        dt = self.dt
        p1_0 = self.p1_0
        p2_0 = self.p2_0
        battery_a, battery_b = state[:, 0], state[:, 1]
        memristor = action[:, 0]  # memristor
        noise = action[:, 1]      # noise

        deri_battery_a = - battery_a + battery_b + p1_0 * torch.mul(battery_b, torch.sin(battery_a))
        deri_battery_b = - 0.5 * battery_a - 0.5 * battery_b \
                         + 0.5 * torch.mul(battery_b, (torch.cos(2 * battery_a) + 2) ** 2) \
                         - 1 / (self.gamma_atte ** 2) * torch.mul(battery_b, (torch.sin(4 * battery_a) + 2) ** 2) \
                         + p2_0 * torch.mul(battery_a, torch.cos(battery_b)) \
                         + torch.mul(torch.cos(2 * battery_a) + 2, memristor) + torch.mul(torch.sin(4 * battery_a) + 2, noise)

        delta_state = torch.stack([deri_battery_a, deri_battery_b], dim=-1)
        state_next = state + delta_state * dt
        reward = (self.Q[0][0] * battery_a ** 2 + self.Q[1][1] * battery_b ** 2
                  + self.R[0][0] * (memristor ** 2).squeeze(-1) - self.gamma_atte ** 2 * (noise ** 2).squeeze(-1))
        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = state[:, 0].new_zeros(size=[state.size()[0]], dtype=torch.bool)

        ############################################################################################
        # beyond_done = beyond_done.bool()
        # mask = isdone * beyond_done
        # mask = torch.unsqueeze(mask, -1)
        # state_next = ~mask * state_next + mask * state
        return delta_state, reward, isdone

    def m_x(self, state, batch_size):

        if batch_size > 1:
            mx = torch.zeros((batch_size, self.state_dim))  # [64, 2]
            mx[:, 0] = torch.mul(state[:, 1], torch.sin(state[:, 0]))
            mx[:, 1] = torch.mul(state[:, 0], torch.cos(state[:, 1]))
        else:
            mx = torch.zeros((self.state_dim, 1))  # [2, 1]
            mx[0, 0] = state[0, 1] * torch.sin(state[0, 0])
            mx[1, 0] = state[0, 0] * torch.cos(state[0, 1])

        return mx

    def f_x(self, state, batch_size):

        if batch_size > 1:
            fx = torch.zeros((batch_size, self.state_dim))  # [64, 2]
            fx[:, 0] = - state[:, 0] + state[:, 1]
            fx[:, 1] = - 0.5 * state[:, 0] - 0.5 * state[:, 1] \
                       + 0.5 * torch.mul(state[:, 1], (torch.cos(2 * state[:, 0]) + 2) ** 2) \
                       - 1 / (self.gamma_atte ** 2) * torch.mul(state[:, 1], (torch.sin(4 * state[:, 0]) + 2) ** 2)
        else:
            fx = torch.zeros((self.state_dim, 1))  # [2, 1]
            fx[0, 0] = - state[0, 0] + state[0, 1]
            fx[1, 0] = - 0.5 * state[0, 0] - 0.5 * state[0, 1] \
                       + 0.5 * torch.mul(state[0, 1], (torch.cos(2 * state[0, 0]) + 2) ** 2) \
                       - 1 / (self.gamma_atte ** 2) * torch.mul(state[0, 1], (torch.sin(4 * state[0, 0]) + 2) ** 2)

        return fx

    def g_x(self, state, batch_size):

        if batch_size > 1:
            gx = torch.zeros((batch_size, self.state_dim, self.action_dim))  # [64, 2, 1]
            gx[:, 0, 0] = torch.zeros((batch_size,))
            gx[:, 1, 0] = torch.cos(2 * state[:, 0]) + 2
        else:
            gx = torch.zeros((self.state_dim, self.action_dim))  # [2, 1]
            gx[0, 0] = 0
            gx[1, 0] = torch.cos(2 * state[0, 0]) + 2

        return gx

    def best_act(self, state, delta_value):
        batch_size = state.size()[0]

        if batch_size > 1:
            gx = self.g_x(state, batch_size)  # [64, 2, 1]
            delta_value = delta_value[:, :, np.newaxis]  # [64, 2, 1]
            act = - 0.5 * torch.matmul(self.R.inverse(), torch.bmm(gx.transpose(1, 2), delta_value)).squeeze(-1)  # [64, 1]
        else:
            gx = self.g_x(state, batch_size)  # [2, 1]
            act = - 0.5 * torch.mm(self.R.inverse(), torch.mm(gx.t(), delta_value.t()))

        return act.detach()

    def k_x(self, state, batch_size):

        if batch_size > 1:
            kx = torch.zeros((batch_size, self.state_dim, self.adversary_dim))  # [64, 2, 1]
            kx[:, 0, 0] = torch.zeros((batch_size,))
            kx[:, 1, 0] = torch.sin(4 * state[:, 0]) + 2
        else:
            kx = torch.zeros((self.state_dim, self.adversary_dim))  # [2, 1]
            kx[0, 0] = 0
            kx[1, 0] = torch.sin(4 * state[0, 0]) + 2

        return kx

    def worst_adv(self, state, delta_value):
        batch_size = state.size()[0]

        if batch_size > 1:
            kx = self.k_x(state, batch_size)  # [64, 2, 1]
            delta_value = delta_value[:, :, np.newaxis]  # [64, 2, 1]
            adv = 0.5 / (self.gamma_atte ** 2) * torch.bmm(kx.transpose(1, 2), delta_value).squeeze(-1)  # [64, 1]
        else:
            kx = self.k_x(state, batch_size)  # [2, 1]
            adv = 0.5 / (self.gamma_atte ** 2) * torch.mm(kx.t(), delta_value.t())

        return adv.detach()

    def E_f(self, state):
        batch_size = state.size()[0]

        if batch_size > 1:
            ef = torch.zeros((batch_size, self.state_dim, self.state_dim))  # [64, 2, 2]
            for i in range(batch_size):
                ef[i, :, :] = self.Ef

        else:
            ef = self.Ef

        return ef


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min) * t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max) * t_max
    return result
