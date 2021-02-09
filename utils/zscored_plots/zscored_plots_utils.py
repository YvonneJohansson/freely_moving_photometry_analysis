import pickle
import numpy as np
from utils.individual_trial_analysis_utils import SessionData
import matplotlib.pyplot as plt
from scipy.signal import decimate
import seaborn as sns
from matplotlib.colors import ListedColormap
from utils.plotting import calculate_error_bars
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_data_for_figure(recording_site):
    if recording_site == 'VS':
        example_mouse = 'SNL_photo35'
        example_date = '20201119'
    elif recording_site == 'TS':
        example_mouse = 'SNL_photo26'
        example_date = '20200812'
    saving_folder = 'W:\\photometry_2AC\\processed_data\\for_figure\\' + example_mouse + '\\'
    aligned_filename = example_mouse + '_' + example_date + '_' + 'aligned_traces_for_fig.p'
    save_filename = saving_folder + aligned_filename
    example_session_data = pickle.load(open(save_filename, "rb"))
    return example_session_data


def get_correct_data_for_plot(session_data, plot_type):
    if plot_type == 'ipsi':
        return session_data.choice_data.ipsi_data, 'event end'
    elif plot_type == 'contra':
        return session_data.choice_data.contra_data, 'event end'
    elif plot_type == 'rewarded':
        return session_data.outcome_data.reward_data, 'next trial'
    elif plot_type == 'unrewarded':
        return session_data.outcome_data.no_reward_data, 'next trial'
    else:
        raise ValueError('Unknown type of plot specified.')
    
    
def get_data_for_recording_site(recording_site, ax):
    aligned_session_data = get_data_for_figure(recording_site)
    all_data = []
    all_white_dot_points = []
    ymins = []
    ymaxs = []
    axes = []
    for ax_type, ax in ax.items():
        data, sort_by = get_correct_data_for_plot(aligned_session_data, ax_type)

        if sort_by == 'event end':
            white_dot_point = data.reaction_times
        elif sort_by == 'next trial':
            white_dot_point = data.sorted_next_poke
        else:
            raise ValueError('Unknown method of sorting trials')
        all_data.append(data)
        all_white_dot_points.append(white_dot_point)
        ymin, ymax = get_min_and_max(data)
        ymins.append(ymin)
        ymaxs.append(ymax)
        axes.append(ax[0])
        plot_average_trace(ax[1], data)
        ax[1].set_xlim([-1.5, 1.5])
    return axes, all_data, all_white_dot_points, ymins, ymaxs


def plot_all_heatmaps_same_scale(fig, axes, all_data, all_white_dot_points, cb_range):
    for ax_num, ax_id in enumerate(axes):
        heat_map = plot_heat_map(ax_id, all_data[ax_num], all_white_dot_points[ax_num], dff_range=cb_range)
        ax_id.set_xlim([-1.5, 1.5])
        divider = make_axes_locatable(ax_id)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(heat_map, cax=cax)
        cb.ax.set_title('z-score', fontsize=8, pad=0.05)
    return heat_map



def get_min_and_max(data):
    ymin = np.min(data.sorted_traces)
    ymax = np.max(data.sorted_traces)
    return ymax, ymin

def plot_average_trace(ax, data, error_bar_method='sem'):
    mean_trace = decimate(data.mean_trace, 10)
    time_points = decimate(data.time_points, 10)
    traces = decimate(data.sorted_traces, 10)
    ax.plot(time_points, mean_trace, lw=1, color='navy')

    if error_bar_method is not None:
        error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                traces,
                                                                error_bar_method=error_bar_method)
        ax.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                            facecolor='navy', linewidth=0)


    ax.axvline(0, color='k', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z-score')
        

def plot_heat_map(ax, data, white_dot_point, dff_range=None):
    data.sorted_next_poke[-1] = np.nan
    arr1inds = white_dot_point.argsort()
    data.reaction_times = data.reaction_times[arr1inds]
    data.outcome_times = data.outcome_times[arr1inds]
    data.sorted_traces = data.sorted_traces[arr1inds]
    data.sorted_next_poke = data.sorted_next_poke[arr1inds]

    my_cmap = ListedColormap(sns.color_palette("YlGnBu_r",256))
    heat_im = ax.imshow(data.sorted_traces, aspect='auto',
                        extent=[-10, 10, data.sorted_traces.shape[0], 0], cmap='viridis')


    ax.axvline(0, color='w', linewidth=1)

    ax.scatter(data.reaction_times,
               np.arange(data.reaction_times.shape[0]) + 0.5, color='w', s=0.5)
    ax.scatter(data.sorted_next_poke,
               np.arange(data.sorted_next_poke.shape[0]) + 0.5, color='k', s=0.5)
    ax.tick_params(labelsize=8)
    ax.set_xlim(data.params.plot_range)
    ax.set_ylim([data.sorted_traces.shape[0], 0])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial (sorted)')
    if dff_range:
        vmin = dff_range[0]
        vmax = dff_range[1]
        edge = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        heat_im.set_norm(norm)
    return heat_im



def make_y_lims_same_heat_map(ymins, ymaxs):
    ylim_min = min(ymins)
    ylim_max = max(ymaxs)
    return ylim_min, ylim_max