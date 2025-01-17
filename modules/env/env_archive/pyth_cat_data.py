#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Jie Li
#  Description: Third-order Nonlinear Environment
#

from math import sin, cos, sqrt, exp, pi
import xlrd
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.wrappers.time_limit import TimeLimit
gym.logger.setLevel(gym.logger.ERROR)

workbook = xlrd.open_workbook(r'E:\GitHub\RobustADP\modules\env\env_archive\white_noise.xls')
sheet = workbook.sheets()[0]
white_noise = sheet.col_values(0)


class _GymCat(gym.Env):
    def __init__(self, **kwargs):
        """
        you need to define parameters here
        """
        # define common parameters here
        self.is_adversary = kwargs['is_adversary']
        self.state_dim = 3
        self.action_dim = 1
        self.adversary_dim = 1
        self.tau = 1 / 200  # seconds between state updates
        self.prob_intensity = kwargs.get('prob_intensity', 1.0)
        self.base_decline = kwargs.get('base_decline', 0.0)
        self.start_decline = kwargs.get('start_decline', 0)
        self.start_cancel = kwargs.get('start_cancel', kwargs['max_iteration'])
        self.dist_func_type = kwargs.get('dist_func_type', 'zero')
        self.initial_obs = kwargs.get('initial_obs', kwargs['fixed_initial_state'])
        self.sample_batch_size = kwargs['sample_batch_size']

        self.time_start_decline = self.start_decline * self.tau * self.sample_batch_size
        self.time_start_cancel = self.start_cancel * self.tau * self.sample_batch_size

        # define your custom parameters here
        self.p_0 = 0.0
        self.bound_delta_p = 1.0
        self.scale_delta_p = 0.0  # [-1, 1]
        self.p = self.p_0 + self.scale_delta_p * self.bound_delta_p

        # utility information
        self.Q = 8 * np.eye(self.state_dim)
        self.R = 5 * np.eye(self.action_dim)
        self.gamma = 1
        self.gamma_atte = kwargs['gamma_atte']

        # state & action space
        self.fixed_initial_state = kwargs['fixed_initial_state']  # for env_data & on_sampler
        self.initial_state_range = kwargs['initial_state_range']  # for env_model
        self.x1_initial = self.initial_state_range[0]
        self.x2_initial = self.initial_state_range[1]
        self.x3_initial = self.initial_state_range[2]
        self.state_threshold = kwargs['state_threshold']
        self.x1_threshold = self.state_threshold[0]
        self.x2_threshold = self.state_threshold[1]
        self.x3_threshold = self.state_threshold[2]
        self.max_action = [2.0]
        self.min_action = [-2.0]
        self.max_adv_action = [1.0]
        self.min_adv_action = [-1.0]

        self.observation_space = spaces.Box(low=np.array([-self.x1_threshold, -self.x2_threshold, -self.x3_threshold]),
                                            high=np.array([self.x1_threshold, self.x2_threshold, self.x3_threshold]),
                                            shape=(3,)
                                            )
        # self.action_space = spaces.Box(low=np.array(self.min_action + self.min_adv_action),
        #                                high=np.array(self.max_action + self.max_adv_action),
        #                                shape=(2,)
        #                                )
        self.action_space = spaces.Box(low=np.array(self.min_action),
                                       high=np.array(self.max_action),
                                       shape=(1,)
                                       )

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.max_episode_steps = kwargs['max_episode_steps']  # original = 200
        self.steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reload_para(self):  # reload uncertain parameters in env_data
        self.p = self.p_0 + self.scale_delta_p * self.bound_delta_p

    def stepPhysics(self, action, adv_action):

        tau = self.tau
        # print(f'p = {self.p}')
        x1, x2, x3 = self.state
        control = action[0]
        disturbance = adv_action[0]

        x1_dot = - x1 + x2 + 0.5 * self.p * x1 * sin(x2) * cos(x3)
        x2_dot = 0.1 * x1 - x2 - x1 * x3 + control + disturbance
        x3_dot = x1 * x2 - x3

        next_x1 = x1_dot * tau + x1
        next_x2 = x2_dot * tau + x2
        next_x3 = x3_dot * tau + x3
        return next_x1, next_x2, next_x3

    def step(self, inputs):
        action = inputs[:self.action_dim]
        adv_action = inputs[self.action_dim:]
        if adv_action is None:
            adv_action = 0

        x1, x2, x3 = self.state
        control = action[0]
        disturbance = adv_action[0]
        self.state = self.stepPhysics(action, adv_action)
        next_x1, next_x2, next_x3 = self.state
        done = next_x1 < -self.x1_threshold or next_x1 > self.x1_threshold \
               or next_x2 < -self.x2_threshold or next_x2 > self.x2_threshold \
               or next_x3 < -self.x3_threshold or next_x3 > self.x3_threshold
        done = bool(done)

        # -----------------
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            done = True
        # ---------------

        if not done:
            reward = self.Q[0][0] * x1 ** 2 + self.Q[1][1] * x2 ** 2 + self.Q[2][2] * x3 ** 2 \
                     + self.R[0][0] * control ** 2 - self.gamma_atte ** 2 * (disturbance ** 2)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = self.Q[0][0] * x1 ** 2 + self.Q[1][1] * x2 ** 2 + self.Q[2][2] * x3 ** 2 \
                     + self.R[0][0] * control ** 2 - self.gamma_atte ** 2 * (disturbance ** 2)
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        reward_positive = self.Q[0][0] * x1 ** 2 + self.Q[1][1] * x2 ** 2 + self.Q[2][2] * x3 ** 2 \
                          + self.R[0][0] * control ** 2
        reward_negative = disturbance ** 2

        return np.array(self.state), reward, done, {'reward_positive': reward_positive, 'reward_negative': reward_negative}

    def exploration_noise(self, time):
        n = sin(time) ** 2 * cos(time) + sin(2 * time) ** 2 * cos(0.1 * time) + sin(1.2 * time) ** 2 * cos(0.5 * time) \
            + sin(time) ** 5 + sin(1.12 * time) ** 2 + sin(2.4 * time) ** 3 * cos(2.4 * time)
        if time > self.time_start_cancel:
            self.prob_intensity = 0
        if time < self.time_start_decline:
            final_prob_intensity = self.prob_intensity
        else:
            final_prob_intensity = self.prob_intensity * exp(self.base_decline * (time - self.time_start_decline))
        return np.array([final_prob_intensity * n, 0])

    def reset(self):  # for on_sampler
        self.state = self.fixed_initial_state
        self.steps_beyond_done = None
        self.steps = 0
        return np.array(self.state)

    def init_obs(self):
        return np.array(self.initial_obs, dtype="float32")  # [0.1, 0.0]

    def dist_func(self, time):
        if self.dist_func_type == 'sine':
            return dist_func_sine_noise(time)
        elif self.dist_func_type == 'white':
            return dist_func_white_noise(int(time / self.tau))
        elif self.dist_func_type == 'zero':
            return dist_func_zero_noise(time)
        else:
            raise ValueError("Not set dist_func_type properly")

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()


def dist_func_sine_noise(time):
    # No.1
    t0 = 0.0
    dist = [0.5 * sin(pi / 5 * (time - t0))]  # 0.5
    # # No.2
    # te = 4 * pi / (2 * sqrt(self.k_0 / self.m))
    # dist = [0.5 * sin(2 * sqrt(self.k_0 / self.m) * time)] if time < te else [0]
    # # No.3
    # dist = [0.5 * exp(-0.1 * time) * sin(2 * sqrt(self.k_0 / self.m) * time)]
    return dist


def dist_func_white_noise(step):
    dist = [3 * white_noise[step]]
    return dist


def dist_func_zero_noise(time):
    dist = [0]
    return dist


def env_creator(**kwargs):
    return TimeLimit(_GymCat(**kwargs), _GymCat(**kwargs).max_episode_steps)  # original = 200


if __name__ == '__main__':
    scale = 0.5
    scale_list = np.arange(-1, 1 + scale, scale)
    print(scale_list)
    c = {}
    print(bool(c))
