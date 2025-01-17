#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Jie Li
#  Description: Sineoscillator Environment
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


class _GymSineoscillator(gym.Env):
    def __init__(self, **kwargs):
        """
        you need to define parameters here
        """
        # define common parameters here
        self.is_adversary = kwargs['is_adversary']
        self.state_dim = 2
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
        self.p1_0 = 0.0
        self.p2_0 = 0.0
        self.bound_delta_p1 = 1.0
        self.bound_delta_p2 = 1.0
        self.scale_delta_p1 = 0.  # [-1, 1]
        self.scale_delta_p2 = 0.  # [-1, 1]
        self.p1 = self.p1_0 + self.scale_delta_p1 * self.bound_delta_p1
        self.p2 = self.p2_0 + self.scale_delta_p2 * self.bound_delta_p2

        # utility information
        self.Q = 2 * np.eye(self.state_dim)
        self.R = 2 * np.eye(self.action_dim)
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

        self.observation_space = spaces.Box(low=np.array([-self.battery_a_threshold, -self.battery_b_threshold]),
                                            high=np.array([self.battery_a_threshold, self.battery_b_threshold]),
                                            shape=(2,)
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
        self.p1 = self.p1_0 + self.scale_delta_p1 * self.bound_delta_p1
        self.p2 = self.p2_0 + self.scale_delta_p2 * self.bound_delta_p2

    def stepPhysics(self, action, adv_action):

        tau = self.tau
        # print(f'p1 = {self.p1}, p2 = {self.p2}')
        battery_a, battery_b = self.state
        memristor = action[0]  # memritor
        noise = adv_action[0]  # noise

        battery_a_dot = - battery_a + battery_b + self.p1 * battery_b * sin(battery_a)
        battery_b_dot = - 0.5 * battery_a - 0.5 * battery_b \
                        + 0.5 * battery_b * (cos(2 * battery_a) + 2) ** 2 \
                        - 1 / (self.gamma_atte ** 2) * battery_b * (sin(4 * battery_a) + 2) ** 2 \
                        + self.p2 * battery_a * cos(battery_b) \
                        + (cos(2 * battery_a) + 2) * memristor + (sin(4 * battery_a) + 2) * noise

        next_battery_a = battery_a_dot * tau + battery_a
        next_battery_b = battery_b_dot * tau + battery_b
        return next_battery_a, next_battery_b

    def step(self, inputs):
        action = inputs[:self.action_dim]
        adv_action = inputs[self.action_dim:]
        if adv_action is None:
            adv_action = 0

        battery_a, battery_b = self.state
        memristor = action[0]  # memristor
        noise = adv_action[0]  # noise
        self.state = self.stepPhysics(action, adv_action)
        next_battery_a, next_battery_b = self.state
        done = next_battery_a < -self.battery_a_threshold or next_battery_a > self.battery_a_threshold \
               or next_battery_b < -self.battery_b_threshold or next_battery_b > self.battery_b_threshold
        done = bool(done)

        # -----------------
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            done = True
        # ---------------

        if not done:
            reward = self.Q[0][0] * battery_a ** 2 + self.Q[1][1] * battery_b ** 2 \
                     + self.R[0][0] * memristor ** 2 - self.gamma_atte ** 2 * (noise ** 2)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = self.Q[0][0] * battery_a ** 2 + self.Q[1][1] * battery_b ** 2 \
                     + self.R[0][0] * memristor ** 2 - self.gamma_atte ** 2 * (noise ** 2)
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        reward_positive = self.Q[0][0] * battery_a ** 2 + self.Q[1][1] * battery_b ** 2 + self.R[0][0] * memristor ** 2
        reward_negative = noise ** 2

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
    dist = [0.5 * sin(2 * sqrt(2 / 3) * (time - t0))]  # 0.5
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
    return TimeLimit(_GymSineoscillator(**kwargs), _GymSineoscillator(**kwargs).max_episode_steps)  # original = 200


if __name__ == '__main__':
    scale = 0.5
    scale_list = np.arange(-1, 1 + scale, scale)
    print(scale_list)
    c = {}
    print(bool(c))

    # class Env:
    #     def __init__(self):
    #         self.position_random_walk = 0
    #
    #     def random_walk(self, time=0):
    #         probability_up = 0.5 if self.position_random_walk <= 0 else 0.4
    #         step = 0.005 if random.random() < probability_up else -0.005
    #         self.position_random_walk = self.position_random_walk + step
    #         return self.position_random_walk
    #
    # env = Env()
    # position = []
    # for i in range(10000):
    #     position.append(env.random_walk(i))
    #
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()  # 生成窗口
    # ax = fig.add_subplot(111)  # 返回一个axes对象，里面的参数abc表示在一个figure窗口中，有a行b列个小窗口，然后本次plot在第c个窗口中
    # ax.plot(position)
    # plt.show()

    workbook = xlrd.open_workbook(r'E:\GitHub\RobustADP\results\RADP\0419-135111\effect\white_noise.xlsx')
    sheet = workbook.sheets()[0]
    white_noise = sheet.col_values(0)

