import argparse
import copy
import datetime
import json
import os
import xlrd
import numpy as np
import torch

from modules.create_pkg.create_alg import create_alg
from modules.create_pkg.create_buffer import create_buffer
from modules.create_pkg.create_env import create_env
from modules.create_pkg.create_evaluator import create_evaluator
from modules.create_pkg.create_simulator import create_simulator
from modules.create_pkg.create_sampler import create_sampler
from modules.create_pkg.create_trainer import create_trainer
from modules.utils.utils import change_type
from modules.utils.init_args import init_args
from modules.utils.plot import plot_all, my_plot, single_raincloud, double_raincloud, present_raincloud
from modules.utils.tensorboard_tools import start_tensorboard, save_tb_to_csv, save_data

os.environ["OMP_NUM_THREADS"] = "1"
eps = 1e-8


def build_parser():
    # Parameters Setup
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train")  # train test plot present
    mode = parser.parse_args().mode
    if mode != "train":
        time_now = datetime.datetime.now().strftime("%m%d-%H%M%S")

        test_log_dir = r'E:\GitHub\RobustADP\results\RADP\250112-181239'
        network_number = '3000'
        dist_func_type = 'sine'
        plot_data = {'sine': r'\test_sine_3000-0518-135158',
                     'white': r'\test_white_3000-0518-141633'}
        scale = 0.25  # 0.25
        test_network = r'\apprfunc\apprfunc_{}.pkl'.format(network_number)
        test_network_dir = test_log_dir + test_network

        data_dir_list = []
        if mode == "test":
            data_dir = test_log_dir
            save_folder = test_log_dir + r'\test_{}_{}-{}'.format(dist_func_type, network_number, time_now)
        elif mode == 'plot':
            data_dir = test_log_dir + plot_data[dist_func_type]
            save_folder = data_dir + r'\plot-{}'.format(time_now)
        elif mode == 'present':
            data_dir = test_log_dir
            data_dir_list = [test_log_dir + r'\test_sine_3000-0514-205648',
                             test_log_dir + r'\test_white_3000-0514-215431']
            save_folder = data_dir + r'\present-{}'.format(time_now)
        else:
            raise ValueError("Error mode!")

        if dist_func_type == 'sine':
            simulation_step = 10001
        elif dist_func_type == 'white':
            simulation_step = 50001
        else:
            raise ValueError("Error dist_func_type!")

        os.makedirs(save_folder, exist_ok=True)
        print(f'test_network = {test_network}')

        params = json.loads(open(test_log_dir + "/config.json").read())
        params.update(dict(
            env_id='pyth_sineoscillator',
            simulator_name='simulator',
            simulation_step=simulation_step,
            dist_func_type=dist_func_type,
            scale=scale,
            initial_obs=[0.0, 0.0],
            test_log_dir=test_log_dir,
            test_network_dir=test_network_dir,
            data_dir=data_dir,
            data_dir_list=data_dir_list,
            save_folder=save_folder,
        ))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    ################################################
    # Key Parameters for Users
    parser.add_argument('--env_id', type=str, default='pyth_sineoscillator', help='')
    parser.add_argument('--algorithm', type=str, default='RADP', help='')
    parser.add_argument('--enable_cuda', default=False, help='Disable CUDA')

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

    # 2.2 Parameters of policy approximate function
    # Options: None/DetermPolicy/StochaPolicy
    parser.add_argument('--policy_func_name', type=str, default='DetermPolicy')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--policy_func_type', type=str, default='POLYNOMIAL')

    ################################################
    # 3. Parameters for algorithm
    parser.add_argument('--value_learning_rate', type=float, default=0.003, help='0.01')
    parser.add_argument('--additional_term_learning_rate', type=float, default=0.03, help='0.03')
    parser.add_argument('--max_gradient_norm', type=float, default=1.0, help='1.0')
    parser.add_argument('--gamma_atte', type=float, default=2.0, help='2.0')
    parser.add_argument('--initial_weight', type=np.array, default=np.array([[1.0, 0.0, 2.0]]))

    ################################################
    # 4. Parameters for trainer
    # Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer
    parser.add_argument('--trainer', type=str, default='off_serial_trainer')
    # Maximum iteration number
    parser.add_argument('--max_iteration', type=int, default=3000, help='3000')
    parser.add_argument('--ini_network_dir', type=str, default=None)
    # 4.1. Parameters for on_serial_trainer
    parser.add_argument('--num_epoch', type=int, default=1, help='')

    ################################################
    # 5. Parameters for sampler
    parser.add_argument('--sampler_name', type=str, default='off_sampler')
    # Batch size of sampler for buffer store
    parser.add_argument('--sample_batch_size', type=int, default=1)
    # Add noise to actions for better exploration
    parser.add_argument('--noise_params', type=dict, default=None, help='add noise to actions for exploration')
    parser.add_argument('--probing_noise', type=bool, default=False, help='the persistency of excitation (PE) condition')
    parser.add_argument('--prob_intensity', type=float, default=0.0, help='the intensity of probing noise = 0.0')
    parser.add_argument('--base_decline', type=float, default=-1e-2, help='the decline of probing noise = -1e-2')
    # Initial state
    parser.add_argument('--fixed_initial_state', type=list, default=[0.3, 0.3], help='for env_data [0.5, -0.5]')
    parser.add_argument('--initial_state_range', type=list, default=[0.5, 0.5], help='for env_model')
    # State threshold
    parser.add_argument('--state_threshold', type=list, default=[5.0, 5.0])
    # Rollout steps
    parser.add_argument('--lower_step', type=int, default=500, help='for env_model')
    parser.add_argument('--upper_step', type=int, default=1000, help='for env_model')
    parser.add_argument('--max_episode_steps', type=int, default=1e16, help='for env_data')

    ################################################
    # 6. Parameters for buffer
    parser.add_argument('--buffer_name', type=str, default='replay_buffer')
    # Size of collected samples before training
    parser.add_argument('--buffer_warm_size', type=int, default=0)
    # Max size of reply buffer
    parser.add_argument('--buffer_max_size', type=int, default=10000)
    # Batch size of replay samples from buffer
    parser.add_argument('--replay_batch_size', type=int, default=64)
    # Period of sync central policy of each sampler
    parser.add_argument('--sampler_sync_interval', type=int, default=1)

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
    parser.add_argument('--apprfunc_save_interval', type=int, default=1000, help='Save value/policy every N updates')
    # Save key info every N updates
    parser.add_argument('--log_save_interval', type=int, default=1, help='Save every N updates')

    return parser.parse_args()


if __name__ == "__main__":
    # Get parameter dictionary
    args = vars(build_parser())
    env = create_env(**args)
    args = init_args(env, **args)

    mode = args['mode']
    if mode == 'train':
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
        my_plot(data=data_value_weight, gt=None, time=np.arange(num_data)[:, np.newaxis] * time_scale,
                figure_size_scalar=1,
                color_list=None, label_list=[r'$\mathregular{\omega_' + str(i + 1) + '}$' for i in range(num_line)],
                loc_legend='center right', ncol=1, style_legend='italic',
                xlim=(0, num_data * time_scale), ylim=None,
                xtick=None, ytick=None,
                xlabel=xlabel, ylabel='weights of value network',
                xline=None, yline=None,
                pad=None,
                figure_name=args['save_folder'] + '/value_weight_{:d}'.format(num_data), figure_type='png',
                display=False)

        data_state_history = trainer.state_history
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

        data_action_history = trainer.action_history
        num_data = data_action_history.shape[0]
        num_line = data_action_history.shape[1]
        my_plot(data=data_action_history, time=np.arange(num_data)[:, np.newaxis] * time_scale,
                figure_size_scalar=1,
                color_list=None, label_list=['control', 'disturbance'],
                loc_legend='lower right', ncol=1, style_legend='normal',
                xlim=(0, num_data * time_scale), ylim=None,
                xtick=None, ytick=None,
                xlabel=xlabel, ylabel='action',
                xline=None, yline=None,
                pad=None,
                figure_name=args['save_folder'] + '/action_history_{:d}'.format(num_data), figure_type='png',
                display=True)

        # accuracy_value_weight = np.zeros((num_data, 1))
        # for i in range(num_data):
        #     accuracy_value_weight[i, 0] = np.linalg.norm(data_value_weight[i, :] - gt[0, :]) / np.linalg.norm(gt[0, :])
        # num_data = accuracy_value_weight.shape[0]
        # num_line = accuracy_value_weight.shape[1]
        # save_data(data=accuracy_value_weight, row=num_data, column=num_line,
        #           save_file=args['save_folder'], xls_name='/weight_error_{:d}'.format(num_data))
        # my_plot(data=accuracy_value_weight, gt=None, time=np.arange(num_data)[:, np.newaxis] * time_scale,
        #         figure_size_scalar=1,
        #         color_list=['#DE869E'], label_list=None,
        #         loc_legend='lower right', ncol=1,
        #         xlim=(0, num_data * time_scale), ylim=None, set_yscale_log=True,
        #         xtick=None, ytick=None,
        #         xlabel=xlabel, ylabel='weight error',
        #         xline=None, yline=None,
        #         pad=None,
        #         figure_name=args['save_folder'] + '/weight_error_{:d}'.format(num_data), figure_type='png',
        #         display=True)

    else:
        # Step 4: create evaluator/simulator in trainer
        simulator = create_simulator(**args)
        scale = args['scale']
        scale_list = np.arange(-1.0, 1.0 + scale, scale)
        scale_num = len(scale_list) ** 2
        x1_game = []
        x2_game = []
        control_game = []
        attenuation_game = []
        x1_radp = []
        x2_radp = []
        control_radp = []
        attenuation_radp = []
        color_list = []
        label_list = ['OLA', 'RADP']
        color_dict = {'OLA': 'tab:orange',
                      'RADP': 'tab:green'}
        rgb_dict = {'green_cloud': tuple(np.array([85, 182, 152]) / 255),
                    'orange_cloud': tuple(np.array([233, 129, 88]) / 255),
                    'blue_cloud': tuple(np.array([112, 137, 195]) / 255),
                    'pink_cloud': tuple(np.array([219, 112, 177]) / 255)}
        cloud_dict = {'lqr_cloud': rgb_dict['green_cloud'],
                      'lmi_cloud': rgb_dict['orange_cloud'],
                      'wo_ucc_cloud': rgb_dict['orange_cloud'],
                      'game_cloud': rgb_dict['blue_cloud'],
                      'radp_cloud': rgb_dict['pink_cloud']}

        num_data = args['simulation_step']
        num_line = scale_num
        mode = args['mode']
        if mode == 'test':
            # game
            simulator.networks.load_state_dict(
                torch.load(r'E:\GitHub\RobustADP\results\ADP\250112-164507\apprfunc\apprfunc_1000.pkl'))
            for i in range(len(scale_list)):
                for j in range(len(scale_list)):
                    print(f'simulation for game: {i + 1} / {len(scale_list)}, {j + 1} / {len(scale_list)}')
                    simulator.env.env.scale_delta_p1 = scale_list[i]
                    simulator.env.env.scale_delta_p2 = scale_list[j]
                    sim_dict_list_game = simulator.run_an_episode(args['max_iteration'], gain=None, dist=True)
                    obs_game = np.stack(sim_dict_list_game['obs_list'], axis=0)
                    act_game = np.stack(sim_dict_list_game['action_list'], axis=0)
                    l2_gain_game = np.stack(sim_dict_list_game['l2_gain_list'], axis=0)
                    x1_game.append(obs_game[:, 0])
                    x2_game.append(obs_game[:, 1])
                    control_game.append(act_game[:, 0])
                    attenuation_game.append(l2_gain_game[:, 0])
                    color_list.append('tab:orange')

            # radp
            simulator.networks.load_state_dict(torch.load(args['test_network_dir']))
            for i in range(len(scale_list)):
                for j in range(len(scale_list)):
                    print(f'simulation for radp: {i + 1} / {len(scale_list)}, {j + 1} / {len(scale_list)}')
                    simulator.env.env.scale_delta_p1 = scale_list[i]
                    simulator.env.env.scale_delta_p2 = scale_list[j]
                    sim_dict_list_radp = simulator.run_an_episode(args['max_iteration'], gain=None, dist=True)
                    obs_radp = np.stack(sim_dict_list_radp['obs_list'], axis=0)
                    act_radp = np.stack(sim_dict_list_radp['action_list'], axis=0)
                    l2_gain_radp = np.stack(sim_dict_list_radp['l2_gain_list'], axis=0)
                    x1_radp.append(obs_radp[:, 0])
                    x2_radp.append(obs_radp[:, 1])
                    control_radp.append(act_radp[:, 0])
                    attenuation_radp.append(l2_gain_radp[:, 0])
                    color_list.append('tab:green')

            time = np.stack(sim_dict_list_radp['time_list'], axis=0)

            save_data(data=np.stack(attenuation_game, axis=1), row=num_data, column=num_line,
                      save_file=args['save_folder'], xls_name='/eg2_attenuation_game_' + args['dist_func_type'])
            save_data(data=np.stack(attenuation_radp, axis=1), row=num_data, column=num_line,
                      save_file=args['save_folder'], xls_name='/eg2_attenuation_radp_' + args['dist_func_type'])
            save_data(data=time, row=num_data, column=1,
                      save_file=args['save_folder'], xls_name='/eg2_time_' + args['dist_func_type'])

        elif mode == 'plot':
            data_dir = args['data_dir']

            workbook = xlrd.open_workbook(data_dir + '/eg2_attenuation_game_' + args['dist_func_type'] + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_game = np.ones((num_data, 1))
                for i in range(0, num_data):
                    l2_gain_game[i, 0] = sheet.cell_value(i, j)
                attenuation_game.append(l2_gain_game[:, 0])
                color_list.append(color_dict['OLA'])

            workbook = xlrd.open_workbook(data_dir + '/eg2_attenuation_radp_' + args['dist_func_type'] + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_radp = np.ones((num_data, 1))
                for i in range(0, num_data):
                    l2_gain_radp[i, 0] = sheet.cell_value(i, j)
                attenuation_radp.append(l2_gain_radp[:, 0])
                color_list.append(color_dict['RADP'])

            workbook = xlrd.open_workbook(data_dir + '/eg2_time_' + args['dist_func_type'] + '.xls')
            sheet = workbook.sheets()[0]
            time = np.ones((num_data, 1))
            for i in range(0, num_data):
                time[i, 0] = sheet.cell_value(i, 0)

        elif mode == 'present':
            data_dir_list = args['data_dir_list']
            num_data_list = [10001, 50001]
            ratios = [1.0, 1.0]

            dist_func_type = 'sine'
            attenuation_game_sine = []
            attenuation_radp_sine = []
            data_dir = data_dir_list[0]
            num_data = num_data_list[0] - 1

            workbook = xlrd.open_workbook(data_dir + '/eg2_attenuation_game_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_game_sine = np.ones((1, 1))
                l2_gain_game_sine[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_game_sine.append(l2_gain_game_sine[:, 0])
                color_list.append(color_dict['OLA'])

            workbook = xlrd.open_workbook(data_dir + '/eg2_attenuation_radp_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_radp_sine = np.ones((1, 1))
                l2_gain_radp_sine[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_radp_sine.append(l2_gain_radp_sine[:, 0])
                color_list.append(color_dict['RADP'])

            dist_func_type = 'white'
            attenuation_game_white = []
            attenuation_radp_white = []
            data_dir = data_dir_list[1]
            num_data = num_data_list[1] - 1

            workbook = xlrd.open_workbook(data_dir + '/eg2_attenuation_game_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_game_white = np.ones((1, 1))
                l2_gain_game_white[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_game_white.append(l2_gain_game_white[:, 0])
                color_list.append(color_dict['OLA'])

            workbook = xlrd.open_workbook(data_dir + '/eg2_attenuation_radp_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_radp_white = np.ones((1, 1))
                l2_gain_radp_white[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_radp_white.append(l2_gain_radp_white[:, 0])
                color_list.append(color_dict['RADP'])

            present_raincloud(
                datas=[np.stack(attenuation_game_sine + attenuation_radp_sine, axis=1),
                       np.stack(attenuation_game_white + attenuation_radp_white, axis=1)],
                column=num_line, methods=label_list, titles=['(a) Sinusoidal ', '(b) White noise'],
                ylim=[(1.135, 1.535), (1.03, 1.61)], point_size=2.0,
                pad=0.2, wspace=0.2,
                color=[cloud_dict['game_cloud'], cloud_dict['radp_cloud']],
                figure_name=args['save_folder'] + '/eg2_raincloud_present',
                display=True)

        else:
            raise ValueError("Error mode!")

        # my_plot(data=np.stack(x1_game + x1_radp, axis=1), time=time,
        #         figure_size_scalar=1,
        #         color_list=color_list, label_list=label_list,
        #         loc_legend='upper right', ncol=1,
        #         xlim=(0, time[-1, 0]), ylim=None,
        #         xtick=None, ytick=None,
        #         xlabel='time [s]', ylabel='x1',
        #         xline=None, yline=None,
        #         pad=None,
        #         figure_name=args['save_folder'] + '/eg2_x1_' + args['dist_func_type'],
        #         figure_type='png',
        #         display=False)
        # my_plot(data=np.stack(x2_game + x2_radp, axis=1), time=time,
        #         figure_size_scalar=1,
        #         color_list=color_list, label_list=label_list,
        #         loc_legend='upper right', ncol=1,
        #         xlim=(0, time[-1, 0]), ylim=None,
        #         xtick=None, ytick=None,
        #         xlabel='time [s]', ylabel='x2',
        #         xline=None, yline=None,
        #         pad=None,
        #         figure_name=args['save_folder'] + '/eg2_x2_' + args['dist_func_type'],
        #         figure_type='png',
        #         display=False)
        #
        # my_plot(data=np.stack(control_game + control_radp, axis=1), time=time,
        #         figure_size_scalar=1,
        #         color_list=color_list, label_list=label_list,
        #         loc_legend='upper right', ncol=1,
        #         xlim=(0, time[-1, 0]), ylim=None,
        #         xtick=None, ytick=None,
        #         xlabel='time [s]', ylabel='control',
        #         xline=None, yline=None,
        #         pad=None,
        #         figure_name=args['save_folder'] + '/eg2_control_' + args['dist_func_type'],
        #         figure_type='png',
        #         display=False)

        if args['dist_func_type'] == 'sine':
            ylim = (1.135, 1.535)
            ratio = 1.0
        elif args['dist_func_type'] == 'white':
            ylim = (1.03, 1.61)
            ratio = 1.0
        else:
            raise ValueError("error dist_func_type")

        single_raincloud(data=np.stack(attenuation_game + attenuation_radp, axis=1),
                         row=num_data, column=num_line, methods=label_list,
                         ylim=ylim, ratio=ratio, point_size=2.0,
                         color=[cloud_dict['game_cloud'], cloud_dict['radp_cloud']],
                         figure_name=args['save_folder'] + '/eg2_raincloud_' + args['dist_func_type'])

        if args['dist_func_type'] == 'sine':
            ylim = (1.135, 1.535)
            ratio = 0.2
        elif args['dist_func_type'] == 'white':
            ylim = (1.03, 1.57)
            ratio = 0.2
        else:
            raise ValueError("error dist_func_type")

        double_raincloud(data=np.stack(attenuation_game + attenuation_radp, axis=1),
                         row=num_data, column=num_line, methods=label_list,
                         ylim=ylim, ratio=ratio, point_size=2.0,
                         color=[cloud_dict['game_cloud'], cloud_dict['radp_cloud']],
                         figure_name=args['save_folder'] + '/eg2_double_raincloud_' + args['dist_func_type'])
        my_plot(data=np.stack(attenuation_game + attenuation_radp, axis=1), time=time,
                figure_size_scalar=1,
                color_list=color_list, label_list=label_list,
                width_thick_line=1.5, width_thin_line=0.2,
                loc_legend='lower right', ncol=1,
                xlim=(0, time[-1, 0]), ylim=None,
                xtick=None, ytick=None,
                xlabel='time [s]', ylabel='attenuation',
                xline=None, yline=None,
                pad=None,
                figure_name=args['save_folder'] + '/eg2_attenuation_' + args['dist_func_type'],
                figure_type='png',
                display=True)
