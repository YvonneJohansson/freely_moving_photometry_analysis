import numpy as np
import pandas as pd
import matplotlib as matplotlib
from matplotlib import colors, cm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import decimate

animalID = 'SNL_photo58'
date = '20211206'

inputDir = '/mnt/winstor/swc/sjones/users/Matt/photometry_2AC/processed_data/' + animalID

fibre_side = 'left'

params = {'state_type_of_interest': 3, # 5.5 = first incorrect choice
    'outcome': 1, # correct or incorrect: 0 = incorrect, 1 = correct, 2 = both
    'last_outcome': 0,  # NOT USED CURRENTLY
    'no_repeats' : 0, # 0 = dont care, 1 = state only entered once,
    'last_response': 0, # trial before: 0 = dont care. 1 = left, 2 = right
    'align_to' : 'Time start', # time end or time start
    'instance': -1, # only for no repeats = 0, -1 = last instance, 1 = first instance
    'plot_range': [-6, 6],
    'first_choice_correct': 1, # useful for non-punished trials 0 = dont care, 1 = only correct trials, (-1 = incorrect trials)
    'cue': None}

'''Plot parameters'''
error_bars= 'sem'
xlims = [-2, 3]
cue_vline = 0



session_bpod_data = pd.read_pickle(inputDir + '/' + animalID + '_' + date + '_restructured_data.pkl')
photometry_trace = np.load(inputDir + '/' + animalID + '_' + date + '_smoothed_signal.npy')

'''DATA MANIPULATION SECTION
 the following section defines function and classes used for manipulating and organising data in preparation for plotting'''



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

def get_z_scored_traces(relevant_df, photometry_trace, pre_window=8, post_window=8):

    z_scored_traces = []

    timestamps = relevant_df[params['align_to']]
    timestamps = list(timestamps * 10000)

    relevant_traces = np.zeros((len(timestamps), (pre_window + post_window)*10000))
    for n, timestamp in enumerate(timestamps):
        trace = photometry_trace[int(timestamp - pre_window*10000):int(timestamp + post_window*10000)]
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

class z_scored_traces(object):

    def __init__(self, relevant_df, photometry_trace, params):
        self.params = params
        self.df = relevant_df
        self.z_scored_traces, self.time_points = get_z_scored_traces(relevant_df, photometry_trace, pre_window=8, post_window=8)
        self.mean_trace = np.mean(self.z_scored_traces, axis=0)
        self.min = np.min(self.z_scored_traces)
        self.max = np.max(self.z_scored_traces)

photometry_data = photometry_data(fibre_side, session_bpod_data, params, photometry_trace)


'''PLOTTING SECTION
the following section defines function and classes used for plotting data'''

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

def make_plot_and_heatmap(photometry_data, error_bar_method='sem', xlims=[-2, 2], cue_vline=0):

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

    '''if mean_across_mice:
        x_range = axs[0, 0].get_xlim()
        ipsi_data = mean_data[0]
        contra_data = mean_data[1]
        line_plot_dff(aligned_session_data.ipsi_data.time_points, ipsi_data, axs[1, 2], x_range)
        line_plot_dff(aligned_session_data.ipsi_data.time_points, contra_data, axs[0, 2], x_range)
        ylim_ipsi = axs[1, 2].get_ylim()
        ylim_contra = axs[0, 2].get_ylim()
        ylim_min, ylim_max = make_y_lims_same(ylim_ipsi, ylim_contra)
        axs[0, 2].set_ylim([ylim_min, ylim_max])
        axs[1, 2].set_ylim([ylim_min, ylim_max])

        for ax in [axs[0, 0], axs[1, 0]]:
            adjust_label_distances(ax, x_space=0.2, y_space=0.12)
        for ax in [axs[0, 1], axs[1, 1], axs[0, 2], axs[1, 2]]:
            adjust_label_distances(ax, x_space=0.2, y_space=0.2)'''

    return fig


'''***PLOT IS CREATED FROM FUNCTION BELOW***'''

make_plot_and_heatmap(photometry_data, error_bar_method=error_bars, xlims=xlims, cue_vline=cue_vline)









