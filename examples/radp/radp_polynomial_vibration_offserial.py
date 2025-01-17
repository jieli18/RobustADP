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

        test_log_dir = r'E:\GitHub\RobustADP\results\RADP\250112-145133'
        network_number = '2000'
        dist_func_type = 'sine'
        plot_data = {'sine': r'\test_sine_2000-0112-155507',
                     'white': r'\test_white_2000-0112-155612'}
        scale = 1.0  # 0.25
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
            data_dir_list = [test_log_dir + r'\test_sine_2000-0514-215514',
                             test_log_dir + r'\test_white_2000-0514-220131']
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
            env_id='pyth_vibration',
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
    parser.add_argument('--env_id', type=str, default='pyth_vibration', help='')
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
    parser.add_argument('--value_learning_rate', type=float, default=0.01, help='0.01')
    parser.add_argument('--additional_term_learning_rate', type=float, default=0.5, help='0.5')
    parser.add_argument('--max_gradient_norm', type=float, default=3.0)
    parser.add_argument('--gamma_atte', type=float, default=0.5)
    parser.add_argument('--initial_weight', type=np.array, default=np.array([[0.0, 0.0, 0.0]]))

    ################################################
    # 4. Parameters for trainer
    # Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer
    parser.add_argument('--trainer', type=str, default='off_serial_trainer', help='off_serial_trainer')
    # Maximum iteration number
    parser.add_argument('--max_iteration', type=int, default=2000, help='2000')
    parser.add_argument('--ini_network_dir', type=str, default=None)
    # 4.1. Parameters for on_serial_trainer
    parser.add_argument('--num_epoch', type=int, default=1, help='')

    ################################################
    # 5. Parameters for sampler
    parser.add_argument('--sampler_name', type=str, default='off_sampler', help='off_sampler')
    # Batch size of sampler for buffer store
    parser.add_argument('--sample_batch_size', type=int, default=1)
    # Add noise to actions for better exploration
    parser.add_argument('--noise_params', type=dict, default=None, help='add noise to actions for exploration')
    parser.add_argument('--probing_noise', type=bool, default=False, help='the persistency of excitation (PE) condition')
    parser.add_argument('--prob_intensity', type=float, default=0.0, help='the intensity of probing noise = 0.0')
    parser.add_argument('--base_decline', type=float, default=0.0, help='the decline of probing noise')
    # Initial state
    parser.add_argument('--fixed_initial_state', type=list, default=[0.7, -0.3])
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
        gt = np.array([[8.98808751, 1.74772708, 11.38678365]])
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
                display=False)

        accuracy_value_weight = np.zeros((num_data, 2))
        for i in range(num_data):
            accuracy_value_weight[i, 0] = i * time_scale
            accuracy_value_weight[i, 1] = np.linalg.norm(data_value_weight[i, :] - gt[0, :]) / np.linalg.norm(gt[0, :])
        num_data = accuracy_value_weight.shape[0]
        num_line = accuracy_value_weight.shape[1]
        save_data(data=accuracy_value_weight, row=num_data, column=num_line,
                  save_file=args['save_folder'],
                  xls_name='/weight_error_{:d}_{:d}'.format(num_data, args['replay_batch_size']))
        my_plot(data=accuracy_value_weight[:, 1:2], gt=None, time=np.arange(num_data)[:, np.newaxis] * time_scale,
                figure_size_scalar=1,
                color_list=['#DE869E'], label_list=None,
                loc_legend='lower right', ncol=1,
                xlim=(0, num_data * time_scale), ylim=None, set_yscale_log=True,
                xtick=None, ytick=None,
                xlabel=xlabel, ylabel='weight error',
                xline=None, yline=None,
                pad=None,
                figure_name=args['save_folder'] + '/weight_error_{:d}'.format(num_data), figure_type='png',
                display=True)

    else:
        # Step 4: create evaluator/simulator in trainer
        simulator = create_simulator(**args)
        scale = args['scale']
        scale_list = np.arange(-1.0, 1.0 + scale, scale)
        scale_num = len(scale_list) ** 2
        position_lqr = []
        velocity_lqr = []
        force_lqr = []
        attenuation_lqr = []
        position_lmi = []
        velocity_lmi = []
        force_lmi = []
        attenuation_lmi = []
        position_game = []
        velocity_game = []
        force_game = []
        attenuation_game = []
        position_radp = []
        velocity_radp = []
        force_radp = []
        attenuation_radp = []
        position_mu = []
        velocity_mu = []
        force_mu = []
        attenuation_mu = []
        color_list = []
        label_list = ['LQR', 'LMI', 'OLA', 'RADP']
        color_dict = {'LQR': 'tab:red',
                      'LMI': 'tab:blue',
                      'OLA': 'tab:orange',
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
            # lqr
            gain_lqr = torch.tensor([[-1.7417, 0.0], [-13.5413, 0.0]], dtype=torch.float32)
            for i in range(len(scale_list)):
                for j in range(len(scale_list)):
                    print(f'simulation for lqr: {i + 1} / {len(scale_list)}, {j + 1} / {len(scale_list)}')
                    simulator.env.env.scale_delta_b = scale_list[i]
                    simulator.env.env.scale_delta_k = scale_list[j]
                    sim_dict_list_lqr = simulator.run_an_episode(args['max_iteration'], gain=gain_lqr, dist=True)
                    obs_lqr = np.stack(sim_dict_list_lqr['obs_list'], axis=0)
                    act_lqr = np.stack(sim_dict_list_lqr['action_list'], axis=0)
                    l2_gain_lqr = np.stack(sim_dict_list_lqr['l2_gain_list'], axis=0)
                    position_lqr.append(obs_lqr[:, 0])
                    velocity_lqr.append(obs_lqr[:, 1])
                    force_lqr.append(act_lqr[:, 0])
                    attenuation_lqr.append(l2_gain_lqr[:, 0])
                    color_list.append('tab:red')

            # lmi
            gain_lmi = torch.tensor([[-6.0119, 0.0], [-22.7811, 0.0]], dtype=torch.float32)  # gamma < 0.5, 4 LMIs + 1 LMI
            for i in range(len(scale_list)):
                for j in range(len(scale_list)):
                    print(f'simulation for lmi: {i + 1} / {len(scale_list)}, {j + 1} / {len(scale_list)}')
                    simulator.env.env.scale_delta_b = scale_list[i]
                    simulator.env.env.scale_delta_k = scale_list[j]
                    sim_dict_list_lmi = simulator.run_an_episode(args['max_iteration'], gain=gain_lmi, dist=True)
                    obs_lmi = np.stack(sim_dict_list_lmi['obs_list'], axis=0)
                    act_lmi = np.stack(sim_dict_list_lmi['action_list'], axis=0)
                    l2_gain_lmi = np.stack(sim_dict_list_lmi['l2_gain_list'], axis=0)
                    position_lmi.append(obs_lmi[:, 0])
                    velocity_lmi.append(obs_lmi[:, 1])
                    force_lmi.append(act_lmi[:, 0])
                    attenuation_lmi.append(l2_gain_lmi[:, 0])
                    color_list.append('tab:blue')

            # game
            gain_game = torch.tensor([[-1.9371, 0.0], [-17.1876, 0.0]], dtype=torch.float32)
            for i in range(len(scale_list)):
                for j in range(len(scale_list)):
                    print(f'simulation for game: {i + 1} / {len(scale_list)}, {j + 1} / {len(scale_list)}')
                    simulator.env.env.scale_delta_b = scale_list[i]
                    simulator.env.env.scale_delta_k = scale_list[j]
                    sim_dict_list_game = simulator.run_an_episode(args['max_iteration'], gain=gain_game, dist=True)
                    obs_game = np.stack(sim_dict_list_game['obs_list'], axis=0)
                    act_game = np.stack(sim_dict_list_game['action_list'], axis=0)
                    l2_gain_game = np.stack(sim_dict_list_game['l2_gain_list'], axis=0)
                    position_game.append(obs_game[:, 0])
                    velocity_game.append(obs_game[:, 1])
                    force_game.append(act_game[:, 0])
                    attenuation_game.append(l2_gain_game[:, 0])
                    color_list.append('tab:orange')

            # radp
            # simulator.networks.load_state_dict(torch.load(args['test_network_dir']))
            gain_radp = torch.tensor([[-2.9129, 0.0], [-37.9559, 0.0]], dtype=torch.float32)
            for i in range(len(scale_list)):
                for j in range(len(scale_list)):
                    print(f'simulation for radp: {i + 1} / {len(scale_list)}, {j + 1} / {len(scale_list)}')
                    simulator.env.env.scale_delta_b = scale_list[i]
                    simulator.env.env.scale_delta_k = scale_list[j]
                    sim_dict_list_radp = simulator.run_an_episode(args['max_iteration'], gain=gain_radp, dist=True)
                    obs_radp = np.stack(sim_dict_list_radp['obs_list'], axis=0)
                    act_radp = np.stack(sim_dict_list_radp['action_list'], axis=0)
                    l2_gain_radp = np.stack(sim_dict_list_radp['l2_gain_list'], axis=0)
                    position_radp.append(obs_radp[:, 0])
                    velocity_radp.append(obs_radp[:, 1])
                    force_radp.append(act_radp[:, 0])
                    attenuation_radp.append(l2_gain_radp[:, 0])
                    color_list.append('tab:green')

            time = np.stack(sim_dict_list_radp['time_list'], axis=0)

            save_data(data=np.stack(attenuation_lqr, axis=1), row=num_data, column=num_line,
                      save_file=args['save_folder'], xls_name='/eg1_attenuation_lqr_' + args['dist_func_type'])
            save_data(data=np.stack(attenuation_lmi, axis=1), row=num_data, column=num_line,
                      save_file=args['save_folder'], xls_name='/eg1_attenuation_lmi_' + args['dist_func_type'])
            save_data(data=np.stack(attenuation_game, axis=1), row=num_data, column=num_line,
                      save_file=args['save_folder'], xls_name='/eg1_attenuation_game_' + args['dist_func_type'])
            save_data(data=np.stack(attenuation_radp, axis=1), row=num_data, column=num_line,
                      save_file=args['save_folder'], xls_name='/eg1_attenuation_radp_' + args['dist_func_type'])
            save_data(data=time, row=num_data, column=1,
                      save_file=args['save_folder'], xls_name='/eg1_time_' + args['dist_func_type'])

        elif mode == 'plot':
            data_dir = args['data_dir']

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_lqr_' + args['dist_func_type'] + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_lqr = np.ones((num_data, 1))
                for i in range(0, num_data):
                    l2_gain_lqr[i, 0] = sheet.cell_value(i, j)
                attenuation_lqr.append(l2_gain_lqr[:, 0])
                color_list.append(color_dict['LQR'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_lmi_' + args['dist_func_type'] + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_lmi = np.ones((num_data, 1))
                for i in range(0, num_data):
                    l2_gain_lmi[i, 0] = sheet.cell_value(i, j)
                attenuation_lmi.append(l2_gain_lmi[:, 0])
                color_list.append(color_dict['LMI'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_game_' + args['dist_func_type'] + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_game = np.ones((num_data, 1))
                for i in range(0, num_data):
                    l2_gain_game[i, 0] = sheet.cell_value(i, j)
                attenuation_game.append(l2_gain_game[:, 0])
                color_list.append(color_dict['OLA'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_radp_' + args['dist_func_type'] + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_radp = np.ones((num_data, 1))
                for i in range(0, num_data):
                    l2_gain_radp[i, 0] = sheet.cell_value(i, j)
                attenuation_radp.append(l2_gain_radp[:, 0])
                color_list.append(color_dict['RADP'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_time_' + args['dist_func_type'] + '.xls')
            sheet = workbook.sheets()[0]
            time = np.ones((num_data, 1))
            for i in range(0, num_data):
                time[i, 0] = sheet.cell_value(i, 0)

        elif mode == 'present':
            data_dir_list = args['data_dir_list']
            num_data_list = [10001, 50001]
            ratios = [1.0, 1.0]

            dist_func_type = 'sine'
            attenuation_lqr_sine = []
            attenuation_lmi_sine = []
            attenuation_game_sine = []
            attenuation_radp_sine = []
            data_dir = data_dir_list[0]
            num_data = num_data_list[0] - 1

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_lqr_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_lqr_sine = np.ones((1, 1))
                l2_gain_lqr_sine[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_lqr_sine.append(l2_gain_lqr_sine[:, 0])
                color_list.append(color_dict['LQR'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_lmi_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_lmi_sine = np.ones((1, 1))
                l2_gain_lmi_sine[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_lmi_sine.append(l2_gain_lmi_sine[:, 0])
                color_list.append(color_dict['LMI'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_game_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_game_sine = np.ones((1, 1))
                l2_gain_game_sine[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_game_sine.append(l2_gain_game_sine[:, 0])
                color_list.append(color_dict['OLA'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_radp_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_radp_sine = np.ones((1, 1))
                l2_gain_radp_sine[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_radp_sine.append(l2_gain_radp_sine[:, 0])
                color_list.append(color_dict['RADP'])

            dist_func_type = 'white'
            attenuation_lqr_white = []
            attenuation_lmi_white = []
            attenuation_game_white = []
            attenuation_radp_white = []
            data_dir = data_dir_list[1]
            num_data = num_data_list[1] - 1

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_lqr_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_lqr_white = np.ones((1, 1))
                l2_gain_lqr_white[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_lqr_white.append(l2_gain_lqr_white[:, 0])
                color_list.append(color_dict['LQR'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_lmi_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_lmi_white = np.ones((1, 1))
                l2_gain_lmi_white[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_lmi_white.append(l2_gain_lmi_white[:, 0])
                color_list.append(color_dict['LMI'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_game_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_game_white = np.ones((1, 1))
                l2_gain_game_white[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_game_white.append(l2_gain_game_white[:, 0])
                color_list.append(color_dict['OLA'])

            workbook = xlrd.open_workbook(data_dir + '/eg1_attenuation_radp_' + dist_func_type + '.xls')
            sheet = workbook.sheets()[0]
            for j in range(0, num_line):
                l2_gain_radp_white = np.ones((1, 1))
                l2_gain_radp_white[0, 0] = sheet.cell_value(int(ratios[0] * num_data), j)
                attenuation_radp_white.append(l2_gain_radp_white[:, 0])
                color_list.append(color_dict['RADP'])

            present_raincloud(
                datas=[np.stack(attenuation_lqr_sine + attenuation_lmi_sine + attenuation_game_sine + attenuation_radp_sine, axis=1),
                       np.stack(attenuation_lqr_white + attenuation_lmi_white + attenuation_game_white + attenuation_radp_white, axis=1)],
                column=num_line, methods=label_list, titles=['(a) Sinusoidal ', '(b) White noise'],
                ylim=[(0.31, 0.45), (0.285, 0.435)], point_size=1.5,
                pad=0.2, wspace=0.24,
                color=[cloud_dict['lqr_cloud'], cloud_dict['lmi_cloud'], cloud_dict['game_cloud'],
                       cloud_dict['radp_cloud']],
                figure_name=args['save_folder'] + '/eg1_raincloud_present',
                display=True)

        else:
            raise ValueError("Error mode!")

        # my_plot(data=np.stack(position_lqr + position_lmi + position_game + position_radp, axis=1), time=time,
        #         figure_size_scalar=1,
        #         color_list=color_list, label_list=label_list,
        #         loc_legend='upper right', ncol=1,
        #         xlim=(0, time[-1, 0]), ylim=(-0.035, 0.045),  # None, (-0.035, 0.045)
        #         xtick=None, ytick=None,
        #         xlabel='time [s]', ylabel='position [m]',
        #         xline=None, yline=None,
        #         pad=None,
        #         figure_name=args['save_folder'] + '/eg1_position_' + args['dist_func_type'],
        #         figure_type='png',
        #         display=False)
        # my_plot(data=np.stack(velocity_lqr + velocity_lmi + velocity_game + velocity_radp, axis=1), time=time,
        #         figure_size_scalar=1,
        #         color_list=color_list, label_list=label_list,
        #         loc_legend='upper right', ncol=1,
        #         xlim=(0, time[-1, 0]), ylim=None,
        #         xtick=None, ytick=None,
        #         xlabel='time [s]', ylabel='velocity [m/s]',
        #         xline=None, yline=None,
        #         pad=None,
        #         figure_name=args['save_folder'] + '/eg1_velocity_' + args['dist_func_type'],
        #         figure_type='png',
        #         display=False)
        #
        # my_plot(data=np.stack(force_lqr + force_lmi + force_game + force_radp, axis=1), time=time,
        #         figure_size_scalar=1,
        #         color_list=color_list, label_list=label_list,
        #         loc_legend='upper right', ncol=1,
        #         xlim=(0, time[-1, 0]), ylim=None,
        #         xtick=None, ytick=None,
        #         xlabel='time [s]', ylabel='control [N]',
        #         xline=None, yline=None,
        #         pad=None,
        #         figure_name=args['save_folder'] + '/eg1_control_' + args['dist_func_type'],
        #         figure_type='png',
        #         display=False)

        if args['dist_func_type'] == 'sine':
            ylim = (0.31, 0.45)
            ratio = 1.0
        elif args['dist_func_type'] == 'white':
            ylim = (0.285, 0.435)
            ratio = 1.0
        else:
            raise ValueError("error dist_func_type")

        single_raincloud(data=np.stack(attenuation_lqr + attenuation_lmi + attenuation_game + attenuation_radp, axis=1),
                         row=num_data, column=num_line, methods=label_list,
                         ylim=ylim, ratio=ratio, point_size=1.5,
                         color=[cloud_dict['lqr_cloud'], cloud_dict['lmi_cloud'], cloud_dict['game_cloud'], cloud_dict['radp_cloud']],
                         figure_name=args['save_folder'] + '/eg1_raincloud_' + args['dist_func_type'])

        if args['dist_func_type'] == 'sine':
            ylim = (0.31, 0.45)
            ratio = 0.2
        elif args['dist_func_type'] == 'white':
            ylim = (0.285, 0.435)
            ratio = 0.2
        else:
            raise ValueError("error dist_func_type")

        double_raincloud(data=np.stack(attenuation_lqr + attenuation_lmi + attenuation_game + attenuation_radp, axis=1),
                         row=num_data, column=num_line, methods=label_list,
                         ylim=ylim, ratio=ratio, point_size=1.5,
                         color=[cloud_dict['lqr_cloud'], cloud_dict['lmi_cloud'], cloud_dict['game_cloud'], cloud_dict['radp_cloud']],
                         figure_name=args['save_folder'] + '/eg1_double_raincloud_' + args['dist_func_type'])
        my_plot(data=np.stack(attenuation_lqr + attenuation_lmi + attenuation_game + attenuation_radp, axis=1), time=time,
                figure_size_scalar=1,
                color_list=color_list, label_list=label_list,
                width_thick_line=1.5, width_thin_line=0.2,
                loc_legend='lower right', ncol=1,  # upper right
                xlim=(0, time[-1, 0]), ylim=None,  # (0.05, 0.25), (0.08, 0.28)
                xtick=None, ytick=None,
                xlabel='time [s]', ylabel='attenuation',
                xline=None, yline=None,
                pad=None,
                figure_name=args['save_folder'] + '/eg1_attenuation_' + args['dist_func_type'],
                figure_type='png',
                display=True)
