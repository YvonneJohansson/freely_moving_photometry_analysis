import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')
from scipy.interpolate import interp1d
from utils.plotting import calculate_error_bars
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import os
from matplotlib.lines import Line2D
import math
import pandas as pd

def make_change_over_time_plot(mice, ax, window_for_binning=40):
    data_root = r'W:\photometry_2AC\processed_data\peak_analysis'
    colours = sns.color_palette("pastel")
    interp_x = []
    interp_y = []
    for mouse_num, mouse in enumerate(mice):
        saving_folder = os.path.join(data_root, mouse)
        filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks_new.npz'
        save_filename = os.path.join(saving_folder, filename)
        rolling_mean_data = np.load(save_filename)
        rolling_mean_x = rolling_mean_data['rolling_mean_x']
        rolling_mean_peaks = rolling_mean_data['rolling_mean_peaks']
        rolling_mean_traces = rolling_mean_data['rolling_mean_traces']
        peak_trace_inds = rolling_mean_data['peak_trace_inds']
        f = interp1d(rolling_mean_x, rolling_mean_peaks)
        xnew = np.arange(int(np.min(rolling_mean_x)) + 1, int(np.max(rolling_mean_x)) - 1)
        ynew = f(xnew)
        interp_x.append(xnew)
        interp_y.append(ynew)
        # ax.plot(rolling_mean_x, rolling_mean_peaks, color=colours[mouse_num])
    min_x = min([np.min(i) for i in interp_x])
    max_x = max([np.max(i) for i in interp_x])
    size_of_ys = max_x + 1
    all_ys = np.empty((len(interp_y), size_of_ys))
    all_ys[:] = np.NaN
    for mouse_num, mouse_data in enumerate(interp_y):
        xs = interp_x[mouse_num]
        all_ys[mouse_num, xs] = mouse_data
    mean_y = np.mean(all_ys, axis=0)
    error_bar_lower, error_bar_upper = calculate_error_bars(mean_y, all_ys, error_bar_method='sem')
    ax.plot(np.arange(0, size_of_ys), mean_y, color='k', lw=2)
    ax.fill_between(np.arange(0, size_of_ys), error_bar_lower, error_bar_upper, alpha=0.5,
                    facecolor='#7FB5B5', linewidth=0)
    ax.set_ylabel('Peak size (z-score)')
    ax.set_xlabel('Trial number')


def make_example_traces_plot(mouse, ax, window_for_binning=50):
    data_root = r'W:\photometry_2AC\processed_data\peak_analysis'
    saving_folder = os.path.join(data_root, mouse)
    filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks.npz'
    save_filename = os.path.join(saving_folder, filename)
    rolling_mean_data = np.load(save_filename)
    rolling_mean_x = rolling_mean_data['rolling_mean_x']
    rolling_mean_peaks = rolling_mean_data['rolling_mean_peaks']
    rolling_mean_traces = rolling_mean_data['rolling_mean_trace']
    peak_trace_inds = (rolling_mean_data['peak_trace_inds'] - 8000) / 1000
    num_bins = len(rolling_mean_traces)
    colours = cm.viridis(np.linspace(0, 0.8, num_bins))
    for trace_num, trace in enumerate(rolling_mean_traces):
        x_vals = np.linspace(-8, 8, np.shape(trace)[0])
        ax.plot(x_vals, trace, color=colours[trace_num], alpha=0.8, lw=2)
        # ax.scatter(peak_trace_inds[trace_num], rolling_mean_peaks[trace_num], color=colours[trace_num])
    ax.set_xlim([-0.5, 1.2])
    custom_lines = [Line2D([0], [0], color=colours[0], lw=4),
                    Line2D([0], [0], color=colours[int(num_bins / 2)], lw=4),
                    Line2D([0], [0], color=colours[-1], lw=4)]

    ax.legend(custom_lines, ['Early', 'Middle', 'Late'], frameon=False, prop={'size': 6})
    ax.set_ylabel('Peak size (z-score)')
    ax.set_xlabel('Time (s)')


def make_change_over_time_plot_general(mice, file_tag, ax, window_for_binning=40, color='k'):
    data_root = r'W:\photometry_2AC\processed_data\peak_analysis'
    colours = sns.color_palette("pastel")
    interp_x = []
    interp_y = []
    for mouse_num, mouse in enumerate(mice):
        saving_folder = os.path.join(data_root, mouse)
        filename = mouse + '_binned_' + str(window_for_binning) + file_tag
        save_filename = os.path.join(saving_folder, filename)
        rolling_mean_data = np.load(save_filename)
        rolling_mean_x = rolling_mean_data['rolling_mean_x']
        rolling_mean_peaks = rolling_mean_data['rolling_mean_peaks']
        rolling_mean_traces = rolling_mean_data['rolling_mean_traces']
        peak_trace_inds = rolling_mean_data['peak_trace_inds']
        # if rolling_mean_x.shape[0] >10:
        #     print(mouse)
        #     f = interp1d(rolling_mean_x, rolling_mean_peaks)
        #     xnew = np.arange(int(np.min(rolling_mean_x)) + 1, int(np.max(rolling_mean_x)) - 1)
        #     ynew = f(xnew)
        #     interp_x.append(xnew)
        #     interp_y.append(ynew)
        ax.plot(rolling_mean_x, rolling_mean_peaks, color=color)
    # min_x = min([np.min(i) for i in interp_x])
    # max_x = max([np.max(i) for i in interp_x])
    # size_of_ys = max_x + 1
    # all_ys = np.empty((len(interp_y), size_of_ys))
    # all_ys[:] = np.NaN
    # for mouse_num, mouse_data in enumerate(interp_y):
    #     xs = interp_x[mouse_num]
    #     all_ys[mouse_num, xs] = mouse_data
    # mean_y = np.mean(all_ys, axis=0)
    # error_bar_lower, error_bar_upper = calculate_error_bars(mean_y, all_ys, error_bar_method='sem')
    # ax.plot(np.arange(0, size_of_ys), mean_y, color=color, lw=2)
    #ax.fill_between(np.arange(0, size_of_ys), error_bar_lower, error_bar_upper, alpha=0.5,
    #                facecolor=color, linewidth=0)
    ax.set_ylabel('Peak size (z-score)')
    ax.set_xlabel('Trial number')

def make_example_traces_general_plot(mouse, file_tag, ax, window_for_binning=50):
    data_root = r'W:\photometry_2AC\processed_data\peak_analysis'
    saving_folder = os.path.join(data_root, mouse)
    filename = mouse + '_binned_' + str(window_for_binning) + file_tag
    save_filename = os.path.join(saving_folder, filename)
    rolling_mean_data = np.load(save_filename)
    rolling_mean_x = rolling_mean_data['rolling_mean_x']
    rolling_mean_peaks = rolling_mean_data['rolling_mean_peaks']
    rolling_mean_traces = rolling_mean_data['rolling_mean_traces']
    peak_trace_inds = (rolling_mean_data['peak_trace_inds'] - 8000) / 1000
    num_bins = len(rolling_mean_traces)
    colours = cm.viridis(np.linspace(0, 0.8, num_bins))
    for trace_num, trace in enumerate(rolling_mean_traces):
        x_vals = np.linspace(-8, 8, np.shape(trace)[0])
        #ax.plot(x_vals, trace, color=colours[trace_num], alpha=0.8, lw=2)
        ax.scatter(peak_trace_inds[trace_num], rolling_mean_peaks[trace_num], color=colours[trace_num])
    ax.set_xlim([-0.5, 1.2])
    #custom_lines = [Line2D([0], [0], color=colours[0], lw=4),
    #                Line2D([0], [0], color=colours[int(num_bins / 2)], lw=4),
    #                Line2D([0], [0], color=colours[-1], lw=4)]

    #ax.legend(custom_lines, ['Early', 'Middle', 'Late'], frameon=False, prop={'size': 6})
    ax.set_ylabel('Peak size (z-score)')
    ax.set_xlabel('Time (s)')


def make_change_over_time_plot_per_mouse(mice,  window_for_binning=40):
    data_root = r'W:\photometry_2AC\processed_data\peak_analysis'
    colours = sns.color_palette("pastel")
    interp_x = []
    interp_y = []
    file_tag = '_average_then_peaks_correct_trials.npz'
    color= 'green'
    fig, axs = plt.subplots(math.ceil(len(mice) / 4), 4, sharey=True, sharex=True)
    axs = axs.ravel()
    window_size = 10
    for mouse_num, mouse in enumerate(mice):
        saving_folder = os.path.join(data_root, mouse)
        filename = mouse + '_binned_' + str(window_for_binning) + file_tag
        save_filename = os.path.join(saving_folder, filename)
        rolling_mean_data = np.load(save_filename)
        valid_trial_nums = rolling_mean_data['rolling_mean_x']
        valid_peaks = rolling_mean_data['rolling_mean_peaks']
        rolling_mean_traces = rolling_mean_data['rolling_mean_traces']
        peak_trace_inds = rolling_mean_data['peak_trace_inds']
        rolling_mean_peak = []
        x_vals = []
        for window_num in range(int(np.shape(valid_peaks)[0] / window_size)):
            rolling_mean_peak.append(np.nanmean(valid_peaks[window_num * window_size: (window_num + 1) * window_size]))
            x_vals.append(np.mean(valid_trial_nums[window_num * window_size: (window_num + 1) * window_size]))

        axs[mouse_num].scatter(x_vals, rolling_mean_peak, color=color, s=2)

        axs[mouse_num].set_ylabel('Peak size (z-score)')
        axs[mouse_num].set_xlabel('Trial number')
    file_tag = '_average_then_peaks_incorrect_trials.npz'
    color= 'red'
    for mouse_num, mouse in enumerate(mice):
        saving_folder = os.path.join(data_root, mouse)
        filename = mouse + '_binned_' + str(window_for_binning) + file_tag
        save_filename = os.path.join(saving_folder, filename)
        rolling_mean_data = np.load(save_filename)
        valid_trial_nums = rolling_mean_data['rolling_mean_x']
        valid_peaks = rolling_mean_data['rolling_mean_peaks']
        rolling_mean_traces = rolling_mean_data['rolling_mean_traces']
        peak_trace_inds = rolling_mean_data['peak_trace_inds']
        rolling_mean_peak = []
        x_vals = []
        for window_num in range(int(np.shape(valid_peaks)[0] / window_size)):
            rolling_mean_peak.append(np.nanmean(valid_peaks[window_num * window_size: (window_num + 1) * window_size]))
            x_vals.append(np.mean(valid_trial_nums[window_num * window_size: (window_num + 1) * window_size]))

        axs[mouse_num].scatter(x_vals, rolling_mean_peak, color=color, s=2)

        axs[mouse_num].set_ylabel('Peak size (z-score)')
        axs[mouse_num].set_xlabel('Trial number')