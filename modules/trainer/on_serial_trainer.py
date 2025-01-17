#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yang GUAN
#  Description: Serial trainer for RL algorithms
#
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update Date: 2021-05-21, Shengbo LI: Format Revise


__all__ = ['OnSerialTrainer']

import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modules.utils.tensorboard_tools import add_scalars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from modules.utils.tensorboard_tools import tb_tags
import time


class OnSerialTrainer():
    def __init__(self, alg, sampler, evaluator, **kwargs):
        self.alg = alg

        self.sampler = sampler
        self.evaluator = evaluator

        # Import algorithm, appr func, sampler & buffer
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.sampler_name = kwargs['sampler_name']
        self.iteration = 0
        self.max_iteration = kwargs.get('max_iteration')
        self.batch_size = kwargs['sample_batch_size']
        self.ini_network_dir = kwargs['ini_network_dir']
        self.num_epoch = kwargs['num_epoch']

        # initialize the networks
        if self.ini_network_dir is not None:
            self.networks.load_state_dict(torch.load(self.ini_network_dir))

        self.save_folder = kwargs['save_folder']
        self.log_save_interval = kwargs['log_save_interval']
        self.apprfunc_save_interval = kwargs['apprfunc_save_interval']
        self.eval_interval = kwargs['eval_interval']
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.writer.add_scalar(tb_tags['alg_time'], 0, 0)
        self.writer.add_scalar(tb_tags['sampler_time'], 0, 0)

        self.writer.flush()
        # setattr(self.alg, "writer", self.evaluator.writer)
        self.state_history = np.ones([int(self.max_iteration / self.log_save_interval),
                                      self.alg.env_model.state_dim], dtype="float32")
        self.action_history = np.ones([int(self.max_iteration / self.log_save_interval),
                                       self.alg.env_model.action_dim + 1], dtype="float32")
        self.start_time = time.time()
        self.latest_sample_mean_step = None
        self.total_avg_return = np.ones([int(self.max_iteration / self.eval_interval), 1], dtype="float32")
        self.total_step = np.ones([int(self.max_iteration / self.eval_interval), 1], dtype="float32")
        self.sample_mean_step = np.ones([int(self.max_iteration / 200), 1], dtype="float32")

    def step(self):
        # sampling
        self.sampler.networks.load_state_dict(self.networks.state_dict())
        samples_with_replay_format, sampler_tb_dict = self.sampler.sample_with_replay_format()
        alg_tb_dict = {}
        for _ in range(self.num_epoch):
            # learning
            self.alg.networks.load_state_dict(self.networks.state_dict())
            grads, alg_tb_dict = self.alg.compute_gradient(samples_with_replay_format, self.iteration)

            # apply grad
            self.networks.update(grads)
            self.iteration += 1

        # log
        if self.iteration % self.log_save_interval == 0:
            # print('Iter = ', self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)
            data = samples_with_replay_format
            self.state_history[int(self.iteration - 1 / self.log_save_interval)] \
                = data['obs'].detach().numpy()[0]
            self.action_history[int(self.iteration - 1 / self.log_save_interval)] \
                = np.hstack((data['act'].detach().numpy()[0], data['advers'].detach().numpy()[0]))
        # evaluate
        if self.iteration % self.eval_interval == 0:
            self.evaluator.networks.load_state_dict(self.networks.state_dict())
            total_avg_return, total_step = self.evaluator.run_evaluation(self.iteration)
            # self.writer.add_scalar(tb_tags['TAR of RL iteration'],
            #                        total_avg_return,
            #                        self.iteration)
            # self.writer.add_scalar(tb_tags['TAR of total time'],
            #                        total_avg_return,
            #                        int(time.time() - self.start_time))
            # self.writer.add_scalar(tb_tags['TAR of collected samples'],
            #                        total_avg_return,
            #                        self.sampler.get_total_sample_number())
            self.total_avg_return[int(self.iteration / self.eval_interval) - 1, 0] = total_avg_return
            self.total_step[int(self.iteration / self.eval_interval) - 1, 0] = total_step
            print(f'ite = {self.iteration}, total_avg_return = {total_avg_return:.4f}, total_step = {total_step}')
        # print
        if (self.iteration - 1) % 200 == 0:
            if self.sampler_name == 'parallel_sampler':
                sorted_step_per_episode, _ = torch.sort(self.sampler.env_model.step_per_episode, dim=0, descending=True)
                print(f'the furthest 10 agents = {sorted_step_per_episode[:10]}')
                self.latest_sample_mean_step = torch.mean(self.sampler.env_model.step_per_episode).item()
                print(f'average step of {self.batch_size} agents: {self.latest_sample_mean_step:.2f}')
                self.sample_mean_step[int((self.iteration - 1) / 200), 0] = self.latest_sample_mean_step
        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            torch.save(self.networks.state_dict(),
                       self.save_folder + '/apprfunc/apprfunc_{}.pkl'.format(self.iteration))


    def train(self):
        while self.iteration < self.max_iteration:
            self.step()

        self.writer.flush()
