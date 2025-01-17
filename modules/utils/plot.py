#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Zhang Yuhang
#  Description: plot_figure, load_tensorboard_file

#  General Optimal control Problem Solver (GOPS)

import os
import xlrd
import string

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
from matplotlib import cm
from matplotlib.pylab import MultipleLocator
from itertools import cycle
from modules.utils.tensorboard_tools import read_tensorboard
from modules.utils.raincloud import RainCloud
import numpy as np


def self_plot(data,
              fname=None,
              xlabel=None,
              ylabel=None,
              legend=None,
              legend_loc="best",
              color_list=None,
              xlim=None,
              ylim=None,
              xtick=None,
              ytick=None,
              yline=None,
              xline=None,
              ncol=1,
              figsize_scalar=1,
              display=True,
             ):
    """
    plot a single figure containing several curves.
    """
    default_cfg = dict()

    default_cfg['fig_size'] = (8.5, 6.5)
    default_cfg['dpi'] = 300
    default_cfg['pad'] = 0.2

    default_cfg['tick_size'] = 8
    default_cfg['tick_label_font'] = 'Times New Roman'
    default_cfg['legend_font'] = {'family': 'Times New Roman', 'size': '8', 'weight': 'normal'}
    default_cfg['label_font'] = {'family': 'Times New Roman', 'size': '9', 'weight': 'normal'}


    # pre-process
    assert isinstance(data, (dict, list, tuple))

    if isinstance(data, dict):
        data = [data]
    num_data = len(data)

    fig_size = (default_cfg['fig_size'] * figsize_scalar, default_cfg['fig_size'] * figsize_scalar)
    _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg['dpi'])


    # color list
    if (color_list is None) or len(color_list) < num_data:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_list = [next(tableau_colors) for _ in range(num_data)]

    # plot figure
    for (i, d) in enumerate(data):
        plt.plot(d["x"], d["y"], color=color_list[i])

    # legend
    plt.tick_params(labelsize=default_cfg['tick_size'])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg['tick_label_font']) for label in labels]

    if legend is not None:
        plt.legend(legend, loc=legend_loc, ncol=ncol, prop=default_cfg['legend_font'])

    #  label
    plt.xlabel(xlabel, default_cfg['label_font'])
    plt.ylabel(ylabel, default_cfg['label_font'])

    if yline is not None:
        plt.axhline(yline, ls=":", c="grey")
    if xline is not None:
        plt.axvline(xline, ls=":", c="grey")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xtick is not None:
        plt.xticks(xtick)
    if ytick is not None:
        plt.yticks(ytick)
    plt.tight_layout(pad=default_cfg['pad'])

    if fname is None:
        pass
    else:
        plt.savefig(fname)
    
    if display:
        plt.show()


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_all(path):
    data = read_tensorboard(path)
    for (key, values) in data.items():
        self_plot(values,
                  os.path.join(path, str_edit(key) + ".tiff"),
                  xlabel='Iteration Steps',
                  ylabel=str_edit(key))


def str_edit(str_):
    str_ = str_.replace('\\', '/')
    if '/' in str_:
        str_ = str_.split('/')
        str_ = str_[-1]
    return string.capwords(str_, '_')


def my_plot(data,
            gt=None,
            time=None,
            figure_size_scalar=1,
            color_list=None,
            label_list=None,
            width_thick_line=1.5,
            width_thin_line=0.5,
            loc_legend=None,
            ncol=1,
            style_legend='normal',
            font_size_legend=12,
            size_line_legend=1.5,
            xlim=None,
            ylim=None,
            set_xscale_log=False,
            set_yscale_log=False,
            xtick=None,
            ytick=None,
            xlabel=None,
            ylabel=None,
            xline=None,
            yline=None,
            pad=None,
            figure_name=None,
            figure_type='svg',
            display=False,
            ):
    """
    plot a single figure containing several curves.
    """

    # pre-process
    # assert isinstance(data, (dict, list, tuple))
    # if isinstance(data, dict):
    #     data = [data]
    num_data = data.shape[0]
    num_line = data.shape[1]
    num_legend = len(label_list) if label_list is not None else 1
    default_cfg = dict()

    # figure size
    default_cfg['figure_size'] = (3.6, 2.7)  # PPT

    # font size
    default_cfg['font_size_legend'] = font_size_legend  # 图例
    default_cfg['font_size_label'] = 14   # 坐标名称
    default_cfg['font_size_title'] = 14   # 标题

    # font name
    default_cfg['font_name'] = 'Times New Roman'

    # font name
    default_cfg['font_legend'] = {'family': default_cfg['font_name'],
                                  'size': default_cfg['font_size_legend'],
                                  'weight': 'normal',
                                  'style': style_legend,
                                  }
    default_cfg['font_label'] = {'family': default_cfg['font_name'],
                                 'size': default_cfg['font_size_label'],
                                 'weight': 'normal',
                                 }
    default_cfg['font_title'] = {'family': default_cfg['font_name'],
                                 'size': default_cfg['font_size_title'],
                                 'weight': 'normal',
                                 }

    # size of line
    default_cfg['size_line'] = width_thick_line if num_line == num_legend else width_thin_line  # continuous line
    default_cfg['size_s'] = 2       # rare discrete point

    # tick interval
    default_cfg['yaw_major_locator'] = MultipleLocator(0.4)
    default_cfg['lateral_major_locator'] = MultipleLocator(0.01)
    default_cfg['slope_major_locator'] = MultipleLocator(5)

    # dpi
    default_cfg['dpi'] = 300
    # default_cfg['pad'] = 0.2

    # color list
    if (color_list is None) or len(color_list) < num_line:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_list = [next(tableau_colors) for _ in range(num_line)]

    # plot
    fig = plt.figure(figsize=default_cfg['figure_size'] * figure_size_scalar)
    # subplot
    ax1 = fig.add_subplot(111)

    # plot ground-truth
    if gt is not None:
        if time is None:
            for i in range(num_line):
                plt.plot(list(np.arange(num_data)), gt[:, i], 'k-.',
                         linewidth=default_cfg['size_line'])
        else:
            for i in range(num_line):
                plt.plot(time[:, 0], gt[:, i], 'k-.',
                         linewidth=default_cfg['size_line'])

    # plot data, color: gbgrbr
    # c = b--blue, c--cyan, g--green, k--black, m--magenta, r--red, w--white, y--yellow
    if time is None:
        if num_line > 1:
            for i in range(num_line):
                plt.plot(list(np.arange(num_data)), data[:, i], color=color_list[i],
                         linewidth=default_cfg['size_line'], label=label_list[i])
        else:
            plt.plot(list(np.arange(num_data)), data[:, 0], color=color_list[0],
                     linewidth=default_cfg['size_line'])
    else:
        if num_line > num_legend:
            headlines = []
            for i in range(num_line):
                if (i % (num_line/num_legend)) == 0:
                    headline, = plt.plot(time[:, 0], data[:, i], color=color_list[i],
                                         linewidth=default_cfg['size_line'])
                    headlines.append(headline)
                else:
                    plt.plot(time[:, 0], data[:, i], color=color_list[i],
                             linewidth=default_cfg['size_line'])
        else:
            if num_line > 1:
                if time.shape[1] > 1:
                    for i in range(num_line):
                        plt.plot(time[:, i], data[:, i], color=color_list[i],
                                 linewidth=default_cfg['size_line'], label=label_list[i])
                else:
                    for i in range(num_line):
                        plt.plot(time[:, 0], data[:, i], color=color_list[i],
                                 linewidth=default_cfg['size_line'], label=label_list[i])
            else:
                plt.plot(time[:, 0], data[:, 0], color=color_list[0],
                         linewidth=default_cfg['size_line'])

    # set legend
    if num_line > num_legend:
        leg = plt.legend(headlines, label_list, loc=loc_legend, ncol=ncol, prop=default_cfg['font_legend'])
        # set the line width of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(size_line_legend)
    else:
        if num_line > 1:
            plt.legend(loc=loc_legend, ncol=ncol, prop=default_cfg['font_legend'])

    # set coordinates
    # the range of coordinates
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if set_xscale_log:
        plt.gca().set_xscale('log')
    if set_yscale_log:
        plt.gca().set_yscale('log')
    # set ticks
    if xtick is not None:
        plt.xticks(xtick)
    if ytick is not None:
        plt.yticks(ytick)
    # the font of the coordinates
    plt.tick_params(labelsize=default_cfg['font_size_label'])
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # the name of coordinates
    plt.xlabel(xlabel, default_cfg['font_label'])
    plt.ylabel(ylabel, default_cfg['font_label'])
    ax1.set_xlabel(xlabel=xlabel, fontdict={'family': 'Times New Roman'})
    ax1.set_ylabel(ylabel=ylabel, fontdict={'family': 'Times New Roman'})
    if xline is not None:
        plt.axvline(xline, ls=":", c="grey")
    if yline is not None:
        plt.axhline(yline, ls=":", c="grey")

    # adjust padding between and around subplots
    if pad is not None:
        plt.tight_layout(pad=pad)

    # save figure
    if figure_name is not None:
        plt.savefig(figure_name + '.' + figure_type, format=figure_type, dpi=default_cfg['dpi'], bbox_inches='tight')
        plt.savefig(figure_name + '.' + 'svg', format='svg', dpi=default_cfg['dpi'], bbox_inches='tight')
        plt.savefig(figure_name + '.' + 'pdf', format='pdf', dpi=default_cfg['dpi'], bbox_inches='tight')

    if display:
        plt.show()


def plot_range(data,
               gt=None,
               time=None,
               figure_size_scalar=1,
               color_list=None,
               label_list=None,
               loc_legend=None,
               ncol=1,
               style_legend='normal',
               xlim=None,
               ylim=None,
               set_yscale_log=False,
               xtick=None,
               ytick=None,
               xlabel=None,
               ylabel=None,
               xline=None,
               yline=None,
               pad=None,
               figure_name=None,
               figure_type='svg',
               display=False,
               ):
    """
    plot a single figure containing several curves.
    """
    default_cfg = dict()

    # figure size
    default_cfg['figure_size'] = (3.6, 2.7)  # PPT

    # font size
    default_cfg['font_size_legend'] = 11  # 图例
    default_cfg['font_size_tick'] = 12  # 坐标名称
    default_cfg['font_size_label'] = 13   # 坐标名称
    default_cfg['font_size_title'] = 14   # 标题

    # font name
    default_cfg['font_name'] = 'Times New Roman'

    # font name
    default_cfg['font_legend'] = {'family': default_cfg['font_name'],
                                  'size': default_cfg['font_size_legend'],
                                  'weight': 'normal',
                                  'style': style_legend,
                                  }
    default_cfg['font_tick'] = {'family': default_cfg['font_name'],
                                'size': default_cfg['font_size_tick'],
                                'weight': 'normal',
                                }
    default_cfg['font_label'] = {'family': default_cfg['font_name'],
                                 'size': default_cfg['font_size_label'],
                                 'weight': 'normal',
                                 }
    default_cfg['font_title'] = {'family': default_cfg['font_name'],
                                 'size': default_cfg['font_size_title'],
                                 'weight': 'normal',
                                 }

    # size of line
    default_cfg['size_line'] = 2.0   # continuous line
    default_cfg['size_s'] = 2        # rare discrete point
    default_cfg['size_alpha'] = 0.3  # transparency

    # tick interval
    default_cfg['yaw_major_locator'] = MultipleLocator(0.4)
    default_cfg['lateral_major_locator'] = MultipleLocator(0.01)
    default_cfg['slope_major_locator'] = MultipleLocator(5)

    # dpi
    default_cfg['dpi'] = 300
    # default_cfg['pad'] = 0.2

    # pre-process
    # assert isinstance(data, (dict, list, tuple))
    # if isinstance(data, dict):
    #     data = [data]
    num_large_train = data.shape[0]
    num_data = data.shape[1]
    num_line = data.shape[2]
    num_legend = len(label_list) if label_list is not None else 1

    # color list
    if (color_list is None) or len(color_list) < num_line:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_list = [next(tableau_colors) for _ in range(num_line)]

    # plot
    fig = plt.figure(figsize=default_cfg['figure_size'] * figure_size_scalar)
    # subplot
    ax1 = fig.add_subplot(111)

    # plot ground-truth
    if gt is not None:
        if time is None:
            for i in range(num_line):
                plt.plot(list(np.arange(num_data)), gt[:, i], 'k-.',
                         linewidth=default_cfg['size_line'])
        else:
            for i in range(num_line):
                plt.plot(time[:, 0], gt[:, i], 'k-.',
                         linewidth=default_cfg['size_line'])

    # plot data, color: gbgrbr
    # c = b--blue, c--cyan, g--green, k--black, m--magenta, r--red, w--white, y--yellow
    if time is None:
        if num_line > 1:
            for i in range(num_line):
                plt.fill_between(list(np.arange(num_data)), np.min(data, 0)[:, i].tolist(),
                                 np.max(data, 0)[:, i].tolist(), color=color_list[i], alpha=default_cfg['size_alpha'])
                plt.plot(list(np.arange(num_data)), np.mean(data, 0)[:, i], color=color_list[i],
                         linewidth=default_cfg['size_line'], label=label_list[i])
        else:
            plt.fill_between(list(np.arange(num_data)), np.min(data, 0)[:, 0].tolist(),
                             np.max(data, 0)[:, 0].tolist(), color=color_list[0], alpha=default_cfg['size_alpha'])
            plt.plot(list(np.arange(num_data)), np.mean(data, 0)[:, 0], color=color_list[0],
                     linewidth=default_cfg['size_line'])
    else:
        if num_line > 1:
            for i in range(num_line):
                plt.fill_between(time[:, 0], np.min(data, 0)[:, i].tolist(),
                                 np.max(data, 0)[:, i].tolist(), color=color_list[i], alpha=default_cfg['size_alpha'])
                plt.plot(time[:, 0], np.mean(data, 0)[:, i], color=color_list[i],
                         linewidth=default_cfg['size_line'], label=label_list[i])
        else:
            plt.fill_between(time[:, 0], np.min(data, 0)[:, 0].tolist(),
                             np.max(data, 0)[:, 0].tolist(), color=color_list[0], alpha=default_cfg['size_alpha'])
            plt.plot(time[:, 0], np.mean(data, 0)[:, 0], color=color_list[0],
                     linewidth=default_cfg['size_line'])

    # set legend
    if num_line > 1:
        legend = plt.legend(loc=loc_legend, ncol=ncol, prop=default_cfg['font_legend'])
        legend.set_title('松弛参数', prop=default_cfg['font_legend'])

    # set coordinates
    # the range of coordinates
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if set_yscale_log:
        plt.gca().set_yscale('log')
    # set ticks
    if xtick is not None:
        plt.xticks(xtick)
    if ytick is not None:
        plt.yticks(ytick)
    # the font of the coordinates
    plt.tick_params(labelsize=default_cfg['font_size_tick'])
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # the name of coordinates
    plt.xlabel(xlabel, default_cfg['font_label'])
    plt.ylabel(ylabel, default_cfg['font_label'])
    ax1.set_xlabel(xlabel=xlabel, fontdict={'family': 'Times New Roman'})
    ax1.set_ylabel(ylabel=ylabel, fontdict={'family': 'Times New Roman'})
    if xline is not None:
        plt.axvline(xline, ls=":", c="grey")
    if yline is not None:
        plt.axhline(yline, ls=":", c="grey")

    # adjust padding between and around subplots
    if pad is not None:
        plt.tight_layout(pad=pad)

    # save figure
    if figure_name is not None:
        plt.savefig(figure_name + '.' + figure_type, format=figure_type, dpi=default_cfg['dpi'], bbox_inches='tight')
        plt.savefig(figure_name + '.' + 'svg', format='svg', dpi=default_cfg['dpi'], bbox_inches='tight')
        plt.savefig(figure_name + '.' + 'pdf', format='pdf', dpi=default_cfg['dpi'], bbox_inches='tight')

    if display:
        plt.show()


def plot_3d(data_x,
            data_y,
            data_z,
            figure_size_scalar=1,
            color_list=None,
            label_list=None,
            loc_legend=None,
            ncol=1,
            style_legend='normal',
            xlim=None,
            ylim=None,
            zlim=None,
            set_zscale_log=False,
            xtick=None,
            ytick=None,
            ztick=None,
            xlabel=None,
            ylabel=None,
            zlabel=None,
            pad=None,
            figure_name=None,
            figure_type='svg',
            display=False,
            ):
    """
    plot a single figure containing several curves.
    """
    default_cfg = dict()

    # figure size
    default_cfg['figure_size'] = (3.6, 2.7)  # PPT

    # font size
    default_cfg['font_size_legend'] = 12  # 图例
    default_cfg['font_size_label'] = 8   # 坐标名称
    default_cfg['font_size_title'] = 14   # 标题

    # font name
    default_cfg['font_name'] = 'Times New Roman'

    # font name
    default_cfg['font_legend'] = {'family': default_cfg['font_name'],
                                  'size': default_cfg['font_size_legend'],
                                  'weight': 'normal',
                                  'style': style_legend,
                                  }
    default_cfg['font_label'] = {'family': default_cfg['font_name'],
                                 'size': default_cfg['font_size_label'],
                                 'weight': 'normal',
                                 }
    default_cfg['font_title'] = {'family': default_cfg['font_name'],
                                 'size': default_cfg['font_size_title'],
                                 'weight': 'normal',
                                 }

    # size of line
    default_cfg['size_line'] = 2.0   # continuous line
    default_cfg['size_s'] = 2        # rare discrete point
    default_cfg['size_alpha'] = 0.3  # transparency

    # tick interval
    default_cfg['yaw_major_locator'] = MultipleLocator(0.4)
    default_cfg['lateral_major_locator'] = MultipleLocator(0.01)
    default_cfg['slope_major_locator'] = MultipleLocator(5)

    # dpi
    default_cfg['dpi'] = 300
    # default_cfg['pad'] = 0.2

    # plot
    fig9 = plt.figure(figsize=default_cfg['figure_size'] * figure_size_scalar)
    # subplot
    ax1 = fig9.gca(projection='3d')
    ax1.plot_surface(data_x, data_y, data_z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # set coordinates
    # the range of coordinates
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if zlim is not None:
        # plt.zlim(zlim)
        ax1.set_zlim(zlim)
    if set_zscale_log:
        plt.gca().set_zscale('log')
    # set ticks
    if xtick is not None:
        plt.xticks(xtick)
    if ytick is not None:
        plt.yticks(ytick)
    if ztick is not None:
        # plt.zticks(ztick)
        ax1.set_zticks(ztick)
    # the font of the coordinates
    plt.tick_params(axis='both', labelsize=default_cfg['font_size_label'])  # axis='both'?
    labels = ax1.get_xticklabels() + ax1.get_yticklabels() + ax1.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # the name of coordinates
    # plt.xlabel(xlabel, default_cfg['font_label'])
    # plt.ylabel(ylabel, default_cfg['font_label'])
    # plt.zlabel(zlabel, default_cfg['font_label'])
    ax1.set_xlabel(xlabel, default_cfg['font_label'])
    ax1.set_ylabel(ylabel, default_cfg['font_label'])
    ax1.set_zlabel(zlabel, default_cfg['font_label'])

    # adjust padding between and around subplots
    if pad is not None:
        plt.tight_layout(pad=pad)

    # save figure
    if figure_name is not None:
        plt.savefig(figure_name + '.' + figure_type, format=figure_type, dpi=default_cfg['dpi'], bbox_inches='tight')
        plt.savefig(figure_name + '.' + 'svg', format='svg', dpi=default_cfg['dpi'], bbox_inches='tight')

    if display:
        plt.show()
    return ax1


def single_raincloud(data,
                     row,
                     column,
                     methods,
                     ylim,
                     figure_name,
                     ratio=1.0,
                     point_size=1.5,
                     color=None,
                     palette=None,
                     dpi=300,
                     display=False):
    sns.set(style="whitegrid", font_scale=2)

    data_all = []
    count_method = 0
    for method in methods:
        for col in range(0, column):
            data_temp = [data[int(ratio * (row - 1)), col + count_method * column], method]
            data_all.append(data_temp)
        count_method += 1

    df = pd.DataFrame(data_all, columns=['attenuation', 'method'])
    df.head()
    average_attenuation_by_method = df.groupby('method')['attenuation'].mean()
    print(average_attenuation_by_method)
    range_attenuation = df.groupby('method')['attenuation'].agg(['min', 'max']).assign(range=lambda x: x['max'] - x['min'])
    print(1 - range_attenuation.loc['RADP', 'range'] / range_attenuation.loc['OLA', 'range'])

    plt.figure(figsize=(2, 3))
    # subplot
    ax1 = plt.subplot(111)
    RainCloud(x='method', y='attenuation', hue=None, data=df,
              ax=ax1, orient='v',
              width_viol=.45, offset=-.2,
              width_box=.15,
              point_size=point_size, move=.0,
              color=color, palette=palette,
              )
    ax1.set_xlabel(xlabel=None)
    plt.ylim(ylim)

    plt.savefig(figure_name + '.' + 'png', format='png', dpi=dpi, bbox_inches='tight')
    plt.savefig(figure_name + '.' + 'svg', format='svg', dpi=dpi, bbox_inches='tight')
    plt.savefig(figure_name + '.' + 'pdf', format='pdf', dpi=dpi, bbox_inches='tight')

    if display:
        plt.show()


def double_raincloud(data,
                     row,
                     column,
                     methods,
                     ylim,
                     figure_name,
                     ratio=0.2,
                     point_size=1.5,
                     color=None,
                     palette=None,
                     dt=0.005,
                     dpi=300,
                     display=False):
    sns.set(style="whitegrid", font_scale=2)

    data_all_10 = []
    data_all_50 = []
    count_method = 0
    for method in methods:
        nrows = row - 1
        for col in range(0, column):
            data_temp = [data[int(ratio * nrows), col + count_method * column], method]
            data_all_10.append(data_temp)
            data_temp = [data[nrows, col + count_method * column], method]
            data_all_50.append(data_temp)
        count_method += 1

    df_10 = pd.DataFrame(data_all_10, columns=['attenuation', 'method'])
    df_10.head()
    df_50 = pd.DataFrame(data_all_50, columns=['attenuation', 'method'])
    df_50.head()

    plt.figure(figsize=(4, 3))
    font_name = 'Times New Roman'
    font_title = {'family': font_name,
                  'size': 12,
                  'weight': 'normal',
                  }
    # subplot
    ax1 = plt.subplot(121)
    RainCloud(x='method', y='attenuation', hue=None, data=df_10,
              ax=ax1, orient='v',
              width_viol=.45, offset=-.2,
              width_box=.15,
              point_size=point_size, move=.0,
              color=color, palette=palette,
              )
    # ax1.set_title(label='t = 50 s', fontdict=font_title)
    ax1.set_xlabel(xlabel='t = {} s'.format(int(ratio * nrows * dt)), fontdict=font_title)
    plt.ylim(ylim)
    # subplot
    ax2 = plt.subplot(122)
    RainCloud(x='method', y='attenuation', hue=None, data=df_50,
              ax=ax2, orient='v',
              width_viol=.45, offset=-.2,
              width_box=.15,
              point_size=point_size, move=.0,
              color=color, palette=palette,
              )
    # ax2.set_title(label='t = 250 s', fontdict=font_title)
    ax2.set_xlabel(xlabel='t = {} s'.format(int(nrows * dt)), fontdict=font_title)
    ax2.set_ylabel(ylabel=None)
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    plt.ylim(ylim)

    plt.subplots_adjust(left=0.17, bottom=0.17, right=0.97, top=0.97, wspace=0.07)

    plt.savefig(figure_name + '.' + 'png', format='png', dpi=dpi, bbox_inches='tight')
    plt.savefig(figure_name + '.' + 'svg', format='svg', dpi=dpi, bbox_inches='tight')

    if display:
        plt.show()


def present_raincloud(datas,
                      column,
                      methods,
                      titles,
                      ylim,
                      figure_name,
                      point_size=1.5,
                      pad=0.5,
                      wspace=0.2,
                      color=None,
                      palette=None,
                      dpi=300,
                      display=False):
    sns.set(style="whitegrid", font_scale=2)

    data_all = []
    for data_id in range(len(titles)):
        data_all_temp = []
        count_method = 0
        for method in methods:
            for col in range(0, column):
                data_temp = [datas[data_id][0, col + count_method * column], method]
                data_all_temp.append(data_temp)
            count_method += 1
        data_all.append(data_all_temp)

    data_sine = data_all[0]
    data_white = data_all[1]

    df_sine = pd.DataFrame(data_sine, columns=['attenuation', 'method'])
    df_sine.head()
    df_white = pd.DataFrame(data_white, columns=['attenuation', 'method'])
    df_white.head()

    plt.figure(figsize=(4.5, 3))
    font_name = 'Times New Roman'
    font_title = {'family': font_name,
                  'size': 12,
                  'weight': 'normal',
                  }
    # subplot
    ax1 = plt.subplot(121)
    RainCloud(x='method', y='attenuation', hue=None, data=df_sine,
              ax=ax1, orient='v',
              width_viol=.45, offset=-.2,
              width_box=.15,
              point_size=point_size, move=.0,
              color=color, palette=palette,
              )
    # ax1.set_title(label='t = 50 s', fontdict=font_title)
    ax1.set_xlabel(xlabel=titles[0], fontdict=font_title)
    plt.ylim(ylim[0])
    plt.tick_params(axis='y', pad=pad)
    # subplot
    ax2 = plt.subplot(122)
    RainCloud(x='method', y='attenuation', hue=None, data=df_white,
              ax=ax2, orient='v',
              width_viol=.45, offset=-.2,
              width_box=.15,
              point_size=point_size, move=.0,
              color=color, palette=palette,
              )
    # ax2.set_title(label='t = 250 s', fontdict=font_title)
    ax2.set_xlabel(xlabel=titles[1], fontdict=font_title)
    ax2.set_ylabel(ylabel=None)
    # ax2.yaxis.set_major_formatter(plt.NullFormatter())
    plt.ylim(ylim[1])
    plt.tick_params(axis='y', pad=pad)

    plt.subplots_adjust(left=0.13, bottom=0.17, right=0.97, top=0.97, wspace=wspace)

    plt.savefig(figure_name + '.' + 'png', format='png', dpi=dpi, bbox_inches='tight')
    plt.savefig(figure_name + '.' + 'svg', format='svg', dpi=dpi, bbox_inches='tight')

    if display:
        plt.show()


if __name__ == '__main__':
    import numpy as np

    # s = "Total_average_return"
    # print(str_edit(s))
    a = np.array([[1, 2, 3]])
    print(a)
    # print(np.expand_dims(a, 0).repeat(5, axis=0))
    print(a.repeat(5, axis=0))

