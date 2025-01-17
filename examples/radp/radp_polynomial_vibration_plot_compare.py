import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pylab import MultipleLocator
from itertools import cycle


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = cur_dir + r'\results\RADP\250112-152314\compare'

    workbook = xlrd.open_workbook(data_dir + '/weight_error_3125_64.xls')  # 'on_serial_trainer' + 'on_sampler'
    sheet = workbook.sheets()[0]
    num_data = sheet.nrows
    data_online = np.ones((num_data, 2))
    for i in range(0, num_data):
        data_online[i, 0] = sheet.cell_value(i, 0)
        data_online[i, 1] = sheet.cell_value(i, 1)

    workbook = xlrd.open_workbook(data_dir + '/weight_error_10000_1.xls')
    sheet = workbook.sheets()[0]
    num_data = sheet.nrows
    data_offline_1 = np.ones((num_data, 2))
    for i in range(0, num_data):
        data_offline_1[i, 0] = sheet.cell_value(i, 0)
        data_offline_1[i, 1] = sheet.cell_value(i, 1)

    workbook = xlrd.open_workbook(data_dir + '/weight_error_10000_2.xls')
    sheet = workbook.sheets()[0]
    num_data = sheet.nrows
    data_offline_2 = np.ones((num_data, 2))
    for i in range(0, num_data):
        data_offline_2[i, 0] = sheet.cell_value(i, 0)
        data_offline_2[i, 1] = sheet.cell_value(i, 1)

    workbook = xlrd.open_workbook(data_dir + '/weight_error_10000_4.xls')
    sheet = workbook.sheets()[0]
    num_data = sheet.nrows
    data_offline_4 = np.ones((num_data, 2))
    for i in range(0, num_data):
        data_offline_4[i, 0] = sheet.cell_value(i, 0)
        data_offline_4[i, 1] = sheet.cell_value(i, 1)

    workbook = xlrd.open_workbook(data_dir + '/weight_error_10000_8.xls')
    sheet = workbook.sheets()[0]
    num_data = sheet.nrows
    data_offline_8 = np.ones((num_data, 2))
    for i in range(0, num_data):
        data_offline_8[i, 0] = sheet.cell_value(i, 0)
        data_offline_8[i, 1] = sheet.cell_value(i, 1)

    workbook = xlrd.open_workbook(data_dir + '/weight_error_10000_16.xls')
    sheet = workbook.sheets()[0]
    num_data = sheet.nrows
    data_offline_16 = np.ones((num_data, 2))
    for i in range(0, num_data):
        data_offline_16[i, 0] = sheet.cell_value(i, 0)
        data_offline_16[i, 1] = sheet.cell_value(i, 1)

    workbook = xlrd.open_workbook(data_dir + '/weight_error_10000_32.xls')
    sheet = workbook.sheets()[0]
    num_data = sheet.nrows
    data_offline_32 = np.ones((num_data, 2))
    for i in range(0, num_data):
        data_offline_32[i, 0] = sheet.cell_value(i, 0)
        data_offline_32[i, 1] = sheet.cell_value(i, 1)

    figure_size_scalar = 1
    color_list = None
    # label_list = ['w/o ER',
    #               r'$\mathregular{N_b = 1}$', r'$\mathregular{N_b = 2}$', r'$\mathregular{N_b = 4}$',
    #               r'$\mathregular{N_b = 8}$', r'$\mathregular{N_b = 16}$', r'$\mathregular{N_b = 32}$']
    label_list = ['w/o', '1', '2', '4', '8', '16', '32']
    width_thick_line = 1.5
    width_thin_line = 0.5
    loc_legend = 'lower right'
    ncol = 1
    style_legend = 'normal'
    font_size_legend = 12
    size_line_legend = 1.5
    xlim = (0, 50)
    ylim = None
    set_xscale_log = False
    set_yscale_log = True
    xtick = np.arange(0, 60, 10)
    ytick = None
    xlabel = 'time [s]'
    ylabel = 'weight error'
    xline = None
    yline = None
    pad = None
    figure_name = data_dir + '/comparison'
    figure_type = 'png'
    display = True
    """
    plot a single figure containing several curves.
    """

    # pre-process
    num_line = len(label_list)
    num_legend = len(label_list) if label_list is not None else 1
    default_cfg = dict()

    # figure size
    default_cfg['figure_size'] = (3.6, 2.7)  # PPT

    # font size
    default_cfg['font_size_legend'] = font_size_legend  # 图例
    default_cfg['font_size_label'] = 14  # 坐标名称
    default_cfg['font_size_title'] = 14  # 标题

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
    default_cfg['size_s'] = 2  # rare discrete point

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

    # plot data, color: gbgrbr
    # c = b--blue, c--cyan, g--green, k--black, m--magenta, r--red, w--white, y--yellow
    plt.plot(data_online[:, 0], data_online[:, 1],
             color=color_list[0], linewidth=default_cfg['size_line'], label=label_list[0])
    plt.plot(data_offline_1[:, 0], data_offline_1[:, 1],
             color=color_list[1], linewidth=default_cfg['size_line'], label=label_list[1])
    plt.plot(data_offline_2[:, 0], data_offline_2[:, 1],
             color=color_list[2], linewidth=default_cfg['size_line'], label=label_list[2])
    plt.plot(data_offline_4[:, 0], data_offline_4[:, 1],
             color=color_list[3], linewidth=default_cfg['size_line'], label=label_list[3])
    plt.plot(data_offline_8[:, 0], data_offline_8[:, 1],
             color=color_list[4], linewidth=default_cfg['size_line'], label=label_list[4])
    plt.plot(data_offline_16[:, 0], data_offline_16[:, 1],
             color=color_list[5], linewidth=default_cfg['size_line'], label=label_list[5])
    plt.plot(data_offline_32[:, 0], data_offline_32[:, 1],
             color=color_list[6], linewidth=default_cfg['size_line'], label=label_list[6])

    # set legend
    # plt.legend(loc=loc_legend, ncol=ncol, title=r'$\mathregular{N_b}$', title_fontsize=10,
    #            prop=default_cfg['font_legend'])
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
