import numpy as np
import pandas as pd
import matplotlib as matplotlib
from matplotlib import colors, cm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import decimate


def extract_relevant_trials(fibre_side, session_bpod_data, params):

    state_of_int = session_bpod_data[(session_bpod_data['State type'] == params['state_type_of_interest'])]
    if params['outcome'] == 1:
        state_of_int = state_of_int[(state_of_int['First choice correct'] == 1)]
    elif params['outcome'] == 0:
        state_of_int = state_of_int[(state_of_int['First choice correct'] == 0)]
    elif params['outcome'] == 2:
        pass

    if params['no_repeats'] == 0:
        if params['instance'] == 1:
            state_of_int = state_of_int.groupby('Trial num').first()
        elif params['instance'] == -1:
            state_of_int = state_of_int.groupby('Trial num').last()
    elif params['no_repeats'] == 1:
        state_of_int = state_of_int[(state_of_int['Max times in state'] == 1)]

    all_trials = state_of_int

    if fibre_side == 'left':
        ipsi_trials = all_trials[(all_trials['Trial type'] == 1)]
        contra_trials = all_trials[(all_trials['Trial type'] == 7)]
    elif fibre_side == 'right':
        contra_trials = all_trials[(all_trials['Trial type'] == 1)]
        ipsi_trials = all_trials[(all_trials['Trial type'] == 7)]

    return all_trials, contra_trials, ipsi_trials

def extract_relevant_trials_psychometric(session_bpod_data, params):

    state_of_int = session_bpod_data[(session_bpod_data['State type'] == params['state_type_of_interest'])]
    if params['outcome'] == 1:
        state_of_int = state_of_int[(state_of_int['First choice correct'] == 1)]
    elif params['outcome'] == 0:
        state_of_int = state_of_int[(state_of_int['First choice correct'] == 0)]
    elif params['outcome'] == 2:
        pass

    if params['no_repeats'] == 0:
        if params['instance'] == 1:
            state_of_int = state_of_int.groupby('Trial num').first()
        elif params['instance'] == -1:
            state_of_int = state_of_int.groupby('Trial num').last()
    elif params['no_repeats'] == 1:
        state_of_int = state_of_int[(state_of_int['Max times in state'] == 1)]

    all_trials = state_of_int

    trials_98 = all_trials[(all_trials['Trial type'] == 1)]
    trials_82 = all_trials[(all_trials['Trial type'] == 2)]
    trials_66 = all_trials[(all_trials['Trial type'] == 3)]
    trials_50 = all_trials[(all_trials['Trial type'] == 4)]
    trials_34 = all_trials[(all_trials['Trial type'] == 5)]
    trials_18 = all_trials[(all_trials['Trial type'] == 6)]
    trials_2 = all_trials[(all_trials['Trial type'] == 7)]

    return all_trials, trials_98, trials_82, trials_66, trials_50, trials_34, trials_18, trials_2


def get_z_scored_traces(relevant_df, photometry_trace, params, pre_window=8, post_window=8):

    z_scored_traces = []

    timestamps = relevant_df[params['align_to']]
    timestamps = list(timestamps * 10000)

    relevant_traces = np.zeros((len(timestamps), (pre_window + post_window)*10000))
    for n, timestamp in enumerate(timestamps):
        trace = photometry_trace[(int(timestamp) - pre_window*10000):(int(timestamp) + post_window*10000)]
        relevant_traces[n, :] = stats.zscore(trace)
    z_scored_traces.append(relevant_traces)
    z_scored_traces = z_scored_traces[0]

    time_points = np.linspace(-pre_window, post_window, z_scored_traces.shape[1])

    return z_scored_traces, time_points

class photometry_data:

    def __init__(self, fibre_side, session_bpod_data, params, photometry_trace):
        self.fibre_side = fibre_side
        self.photometry_trace = photometry_trace
        self.params = params
        self.all_trials, self.contra_trials, self.ipsi_trials = extract_relevant_trials(fibre_side, session_bpod_data, params)

        self.all_trials = z_scored_traces(self.all_trials, self.photometry_trace, self.params)
        self.contra_trials = z_scored_traces(self.contra_trials, self.photometry_trace, self.params)
        self.ipsi_trials = z_scored_traces(self.ipsi_trials, self.photometry_trace, self.params)


class photometry_data_psychometric:

    def __init__(self, fibre_side, session_bpod_data, params, photometry_trace):
        self.fibre_side = fibre_side
        self.photometry_trace = photometry_trace
        self.params = params
        self.all_trials, self.trials_98, self.trials_82, self.trials_66, self.trials_50, self.trials_34, self.trials_18, self.trials_2 = extract_relevant_trials_psychometric(session_bpod_data, params)

        self.all_trials = z_scored_traces(self.all_trials, self.photometry_trace, self.params)
        self.trials_98 = z_scored_traces(self.trials_98, self.photometry_trace, self.params)
        self.trials_82 = z_scored_traces(self.trials_82, self.photometry_trace, self.params)
        self.trials_66 = z_scored_traces(self.trials_66, self.photometry_trace, self.params)
        self.trials_50 = z_scored_traces(self.trials_50, self.photometry_trace, self.params)
        self.trials_34 = z_scored_traces(self.trials_34, self.photometry_trace, self.params)
        self.trials_18 = z_scored_traces(self.trials_18, self.photometry_trace, self.params)
        self.trials_2 = z_scored_traces(self.trials_2, self.photometry_trace, self.params)


class z_scored_traces(object):

    def __init__(self, relevant_df, photometry_trace, params):
        self.params = params
        self.df = relevant_df
        self.z_scored_traces, self.time_points = get_z_scored_traces(relevant_df, photometry_trace, self.params, pre_window=8, post_window=8)
        self.mean_trace = np.mean(self.z_scored_traces, axis=0)
        self.min = np.min(self.z_scored_traces)
        self.max = np.max(self.z_scored_traces)


def make_y_lims_same(ylim_ipsi, ylim_contra):
    ylim_min = min(ylim_ipsi[0], ylim_contra[0])
    ylim_max = max(ylim_ipsi[1], ylim_contra[1])
    return ylim_min, ylim_max

def calculate_error_bars(mean_trace, data, error_bar_method='sem'):
    if error_bar_method == 'sem':
        sem = stats.sem(data, axis=0)
        lower_bound = mean_trace - sem
        upper_bound = mean_trace + sem
    elif error_bar_method == 'ci':
        lower_bound, upper_bound = bootstrap(data, n_boot=1000, ci=68)
    return lower_bound, upper_bound

def plot_one_side(one_side_data, fig,  ax1, ax2, dff_range=None, error_bar_method='sem', sort=False, white_dot='default', cue_vline=0):
    mean_trace = decimate(one_side_data.mean_trace, 10)
    time_points = decimate(one_side_data.time_points, 10)
    traces = decimate(one_side_data.z_scored_traces, 10)
    ax1.plot(time_points, mean_trace, lw=1.5, color='#3F888F')

    if error_bar_method is not None:
        error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                traces,
                                                                error_bar_method=error_bar_method)
        ax1.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                            facecolor='#7FB5B5', linewidth=0)


    ax1.axvline(0, color='k', linewidth=1)
    if cue_vline != 0:
        ax1.axvline(cue_vline, color='k', linewidth=1, ls='--')
    else:
        pass
    ax1.set_xlim(one_side_data.params['plot_range'])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('z-score')

    '''if white_dot == 'reward':
        white_dot_point = one_side_data.outcome_times
    else:
        white_dot_point = one_side_data.reaction_times
    if sort:
        arr1inds = white_dot_point.argsort()
        one_side_data.reaction_times = one_side_data.reaction_times[arr1inds[::-1]]
        one_side_data.outcome_times = one_side_data.outcome_times[arr1inds[::-1]]
        one_side_data.sorted_traces = one_side_data.sorted_traces[arr1inds[::-1]]
        one_side_data.sorted_next_poke = one_side_data.sorted_next_poke[arr1inds[::-1]]'''

    heat_im = ax2.imshow(one_side_data.z_scored_traces, aspect='auto',
                            extent=[-10, 10, one_side_data.z_scored_traces.shape[0], 0], cmap='viridis')

    ax2.axvline(0, color='w', linewidth=1)
    '''if white_dot == 'reward':
        ax2.scatter(one_side_data.outcome_times,
                       np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=1)
    else:
        ax2.scatter(one_side_data.reaction_times,
                    np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=1)
    ax2.scatter(one_side_data.sorted_next_poke,
                   np.arange(one_side_data.sorted_next_poke.shape[0]) + 0.5, color='k', s=1)'''
    ax2.tick_params(labelsize=10)
    ax2.set_xlim(one_side_data.params['plot_range'])
    ax2.set_ylim([one_side_data.z_scored_traces.shape[0], 0])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Trial (sorted)')
    if dff_range:
        vmin = dff_range[0]
        vmax = dff_range[1]
        edge = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        heat_im.set_norm(norm)
    return heat_im

def line_plot_dff(x_vals, y_vals, ind_traces, ax, x_range, cue_vline=0, error_bar_method='sem'):
    ax.plot(x_vals, y_vals, color='#3F888F', lw=2)

    error_bar_lower, error_bar_upper = calculate_error_bars(y_vals,
                                                            ind_traces,
                                                            error_bar_method=error_bar_method)
    ax.fill_between(x_vals, error_bar_lower, error_bar_upper, alpha=0.5,
                     facecolor='#7FB5B5', linewidth=0)

    ax.axvline(cue_vline, color='k', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z-score')
    ax.set_xlim(x_range)

def adjust_label_distances(ax, x_space, y_space):
    ax.yaxis.set_label_coords(-y_space, 0.5)
    ax.xaxis.set_label_coords(0.5, -x_space)

def make_plot_and_heatmap(photometry_data, *mean_data, error_bar_method='sem', mean_across_mice=False, xlims=[-2, 2], cue_vline=0):

    if mean_across_mice:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 4))
        fig.tight_layout(pad=1.3)
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5.5, 5.5))
        fig.tight_layout(pad=2.1)

    font = {'size': 10}
    matplotlib.rc('font', **font)

    heatmap_min, heatmap_max = make_y_lims_same((photometry_data.ipsi_trials.min, photometry_data.ipsi_trials.max),
                                                (photometry_data.contra_trials.min, photometry_data.contra_trials.max))
    dff_range = (heatmap_min, heatmap_max)
    ipsi_heatmap = plot_one_side(photometry_data.ipsi_trials, fig, axs[1, 0], axs[1, 1], dff_range, error_bar_method=error_bar_method, cue_vline=cue_vline)
    contra_heatmap = plot_one_side(photometry_data.contra_trials, fig, axs[0, 0], axs[0, 1], dff_range, error_bar_method=error_bar_method, cue_vline=cue_vline)
    ylim_ipsi = axs[1, 0].get_ylim()
    ylim_contra = axs[0, 0].get_ylim()
    ylim_min, ylim_max = make_y_lims_same(ylim_ipsi, ylim_contra)
    axs[0, 0].set_ylim([ylim_min, ylim_max])
    axs[1, 0].set_ylim([ylim_min, ylim_max])
    axs[0, 0].set_xlim(xlims)
    axs[1, 0].set_xlim(xlims)
    axs[1, 1].set_xlim(xlims)
    axs[0, 1].set_xlim(xlims)
    axs[0, 0].set_ylabel('z-score')
    axs[1, 0].set_ylabel('z-score')

    cb_ipsi = fig.colorbar(ipsi_heatmap, ax=axs[1, 1], orientation='vertical', fraction=.1)
    cb_contra = fig.colorbar(contra_heatmap, ax=axs[0, 1], orientation='vertical', fraction=.1)
    cb_ipsi.ax.set_title('z-score', fontsize=9, pad=2)
    cb_contra.ax.set_title('z-score', fontsize=9, pad=2)

    if mean_across_mice:
        x_range = axs[0, 0].get_xlim()
        ipsi_data = mean_data[0]
        contra_data = mean_data[1]
        mean_ipsi_data = mean_data[0].mean(axis=0)
        mean_contra_data = mean_data[1].mean(axis=0)
        line_plot_dff(photometry_data.ipsi_trials.time_points, mean_ipsi_data, ipsi_data, axs[1, 2], x_range, error_bar_method=error_bar_method)
        line_plot_dff(photometry_data.contra_trials.time_points, mean_contra_data, contra_data, axs[0, 2], x_range, error_bar_method=error_bar_method)
        ylim_ipsi = axs[1, 2].get_ylim()
        ylim_contra = axs[0, 2].get_ylim()
        ylim_min, ylim_max = make_y_lims_same(ylim_ipsi, ylim_contra)
        axs[0, 2].set_ylim([ylim_min, ylim_max])
        axs[1, 2].set_ylim([ylim_min, ylim_max])


        for ax in [axs[0, 0], axs[1, 0]]:
            adjust_label_distances(ax, x_space=0.2, y_space=0.12)
        for ax in [axs[0, 1], axs[1, 1], axs[0, 2], axs[1, 2]]:
            adjust_label_distances(ax, x_space=0.2, y_space=0.2)

    return fig

