import argparse
import copy
import datetime
import json
import os
import math
import numpy as np

from modules.create_pkg.create_alg import create_alg
from modules.create_pkg.create_buffer import create_buffer
from modules.create_pkg.create_env import create_env
from modules.create_pkg.create_evaluator import create_evaluator
from modules.create_pkg.create_sampler import create_sampler
from modules.create_pkg.create_trainer import create_trainer
from modules.utils.utils import change_type
from modules.utils.init_args import init_args
from modules.utils.plot import plot_all, my_plot
from modules.utils.tensorboard_tools import start_tensorboard, save_tb_to_csv, save_data

os.environ["OMP_NUM_THREADS"] = "1"
eps = 1e-8

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for Users
    parser.add_argument('--env_id', type=str, default='pyth_vibration', help='')
    parser.add_argument('--algorithm', type=str, default='ADP', help='')
    parser.add_argument('--enable_cuda', default=False, help='Disable CUDA')
    parser.add_argument("--mode", type=str, default="training")  # training testing debug

    ################################################
    # 1. Parameters for environment
    parser.add_argument('--obsv_dim', type=int, default=None, help='')
    parser.add_argument('--action_dim', type=int, default=None, help='')
    parser.add_argument('--action_high_limit', type=list, default=None, help='')
    parser.add_argument('--action_low_limit', type=list, default=None, help='')
    parser.add_argument('--action_type', type=str, default='continu', help='')
    parser.add_argument('--is_render', type=bool, default=False, help='')
    parser.add_argument('--is_adversary', type=bool, default=True, help='Adversary training')
    parser.add_argument('--is_constrained', type=bool, default=False, help='Constrained training')

    ################################################
    # 2.1 Parameters of value approximate function
    # Options: StateValue/ActionValue/ActionValueDis
    parser.add_argument('--value_func_name', type=str, default='StateValue')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--value_func_type', type=str, default='POLYNOMIAL')
    value_func_type = parser.parse_args().value_func_type
    if value_func_type == 'POLY':
        pass

    # 2.2 Parameters of policy approximate function
    # Options: None/DetermPolicy/StochaPolicy
    parser.add_argument('--policy_func_name', type=str, default='DetermPolicy')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--policy_func_type', type=str, default='POLYNOMIAL')
    policy_func_type = parser.parse_args().policy_func_type
    if policy_func_type == 'POLY':
        pass

    ################################################
    # 3. Parameters for algorithm
    parser.add_argument('--value_learning_rate', type=float, default=0.25, help='0.5')
    parser.add_argument('--additional_term_learning_rate', type=float, default=0.2, help='0.2')
    parser.add_argument('--max_gradient_norm', type=float, default=3.0, help='3.0')
    parser.add_argument('--gamma_atte', type=float, default=0.5, help='0.32')
    parser.add_argument('--initial_weight', type=np.array, default=np.array([[0.0, 0.0, 0.0]]),
                        help='[[1.0, 0.0, 1.0]], good for batch = 64, lr_c = 0.5, lc_s = 0.2, '
                             '[[0.0, 0.0, 0.0]], good for batch = 64, lr_c = 0.25, lc_s = 0.2')

    ################################################
    # 4. Parameters for trainer
    # Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer
    parser.add_argument('--trainer', type=str, default='on_serial_trainer')
    # Maximum iteration number
    parser.add_argument('--max_iteration', type=int, default=1e3, help='1e4')
    parser.add_argument('--ini_network_dir', type=str, default=None)
    # 4.1. Parameters for on_serial_trainer
    parser.add_argument('--num_epoch', type=int, default=1, help='')

    ################################################
    # 5. Parameters for sampler
    parser.add_argument('--sampler_name', type=str, default='on_sampler')
    # Batch size of sampler for buffer store
    parser.add_argument('--sample_batch_size', type=int, default=64,
                        help='Batch size of sampler for buffer store = 64')
    # Add noise to actions for better exploration
    parser.add_argument('--noise_params', type=dict, default=None, help='add noise to actions for exploration')
    parser.add_argument('--probing_noise', type=bool, default=True, help='the persistency of excitation (PE) condition')
    parser.add_argument('--prob_intensity', type=float, default=1.0, help='the intensity of probing noise')
    parser.add_argument('--base_decline', type=float, default=0.0, help='the decline of probing noise')
    # Initial state
    parser.add_argument('--fixed_initial_state', type=list, default=[0.5, -0.5], help='for env_data [0.5, -0.5]')
    parser.add_argument('--initial_state_range', type=list, default=[0.5, 0.5], help='for env_model')
    # State threshold
    parser.add_argument('--state_threshold', type=list, default=[2.0, 2.0])
    # Rollout steps
    parser.add_argument('--lower_step', type=int, default=500, help='for env_model')
    parser.add_argument('--upper_step', type=int, default=1000, help='for env_model')
    parser.add_argument('--max_episode_steps', type=int, default=1e16, help='for env_data')

    ################################################
    # 6. Parameters for buffer
    parser.add_argument('--buffer_name', type=str, default='replay_buffer')
    parser.add_argument('--buffer_warm_size', type=int, default=parser.parse_args().sample_batch_size)
    parser.add_argument('--buffer_max_size', type=int, default=parser.parse_args().sample_batch_size)

    ################################################
    # 7. Parameters for evaluator
    parser.add_argument('--evaluator_name', type=str, default='evaluator')
    parser.add_argument('--num_eval_episode', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1e16)
    parser.add_argument('--print_interval', type=int, default=20)

    ################################################
    # 8. Data savings
    parser.add_argument('--save_folder', type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument('--apprfunc_save_interval', type=int, default=1e16,
                        help='Save value/policy every N updates')
    # Save key info every N updates
    parser.add_argument('--log_save_interval', type=int, default=1,
                        help='Save gradient time/critic loss/actor loss/average value every N updates')

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    # start_tensorboard(args['save_folder'])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # alg.set_parameters({'gamma': 0.995, 'loss_coefficient_value': 0.5, 'loss_coefficient_entropy': 0.01})
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    # Start training ... ...
    trainer.train()
    print('Training is finished!')

    xlabel = 'time [s]'
    time_scale = args['log_save_interval'] * env.tau * args['sample_batch_size']

    data_value_weight = trainer.alg.value_weight
    num_data = data_value_weight.shape[0]
    num_line = data_value_weight.shape[1]
    save_data(data=data_value_weight, row=num_data, column=num_line,
              save_file=args['save_folder'], xls_name='/value_weight_{:d}'.format(num_data))
    gt = np.array([[5.6289, 1.1623, 5.1563]])  # gamma_atte = 0.5
    gt_value_weight = gt.repeat(num_data, axis=0)
    my_plot(data=data_value_weight, gt=gt_value_weight, time=np.arange(num_data)[:, np.newaxis] * time_scale,
            figure_size_scalar=1,
            color_list=None, label_list=[r'$\mathregular{\omega_' + str(i + 1) + '}$' for i in range(num_line)],
            loc_legend='lower right', ncol=1, style_legend='italic',
            xlim=(0, num_data * time_scale), ylim=None,
            xtick=None, ytick=None,
            xlabel=xlabel, ylabel='weights of value network',
            xline=None, yline=None,
            pad=None,
            figure_name=args['save_folder'] + '/value_weight_{:d}'.format(num_data), figure_type='png',
            display=False)

    data_state_history = trainer.alg.state_history
    num_data = data_state_history.shape[0]
    num_line = data_state_history.shape[1]
    my_plot(data=data_state_history, time=np.arange(num_data)[:, np.newaxis] * time_scale,
            figure_size_scalar=1,
            color_list=None, label_list=[r'$\mathregular{x_' + str(i + 1) + '}$' for i in range(num_line)],
            loc_legend='lower right', ncol=1, style_legend='italic',
            xlim=(0, num_data * time_scale), ylim=None,
            xtick=None, ytick=None,
            xlabel=xlabel, ylabel='state',
            xline=None, yline=None,
            pad=None,
            figure_name=args['save_folder'] + '/state_history_{:d}'.format(num_data), figure_type='png',
            display=False)

    data_action_history = trainer.alg.action_history
    num_data = data_action_history.shape[0]
    num_line = data_action_history.shape[1]
    my_plot(data=data_action_history, time=np.arange(num_data)[:, np.newaxis] * time_scale,
            figure_size_scalar=1,
            color_list=None, label_list=['control', 'disturbance', 'probing_control', 'probing_disturbance'],
            loc_legend='lower right', ncol=1, style_legend='italic',
            xlim=(0, num_data * time_scale), ylim=None,
            xtick=None, ytick=None,
            xlabel=xlabel, ylabel='action',
            xline=None, yline=None,
            pad=None,
            figure_name=args['save_folder'] + '/action_history_{:d}'.format(num_data), figure_type='png',
            display=False)

    accuracy_value_weight = np.zeros((num_data, 1))
    for i in range(num_data):
        accuracy_value_weight[i, 0] = math.log10(
            np.linalg.norm(data_value_weight[i, :] - gt[0, :]) / np.linalg.norm(gt[0, :]) + eps)
    num_data = accuracy_value_weight.shape[0]
    num_line = accuracy_value_weight.shape[1]
    save_data(data=accuracy_value_weight, row=num_data, column=num_line,
              save_file=args['save_folder'], xls_name='/weight_error_{:d}'.format(num_data))
    my_plot(data=accuracy_value_weight, gt=None, time=np.arange(num_data)[:, np.newaxis] * time_scale,
            figure_size_scalar=1,
            color_list=['#DE869E'], label_list=None,
            loc_legend='lower right', ncol=1,
            xlim=(0, num_data * time_scale), ylim=None,
            xtick=None, ytick=None,
            xlabel=xlabel, ylabel='logarithm of error',
            xline=None, yline=None,
            pad=None,
            figure_name=args['save_folder'] + '/weight_error_{:d}'.format(num_data), figure_type='png',
            display=True)
