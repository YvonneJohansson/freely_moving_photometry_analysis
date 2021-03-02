import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis' )

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.ndimage.interpolation import shift
from scipy.signal import decimate
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import sem
import pandas as pd
import pickle
from matplotlib import colors, cm
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.regression.linear_regression_utils import *
import os
import seaborn as sns


def get_regression_data_for_plot(recording_site='tail'):

    experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
    experiment_record['date'] = experiment_record['date'].astype(str)

    if recording_site == 'tail':
        mouse_ids = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26']
    elif recording_site == 'Nacc':
        mouse_ids = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33',
                     'SNL_photo34', 'SNL_photo35']

    good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    clean_experiments = remove_bad_recordings(good_experiments)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == recording_site)].reset_index(drop=True)
    experiments_to_process = get_first_x_sessions(all_experiments_to_process).reset_index(drop=True)
    exp_numbers = []
    mice = []
    for index, experiment in experiments_to_process.iterrows():
        mouse = experiment['mouse_id']
        date = experiment['date']
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
        save_filename = mouse + '_' + date + '_' + 'linear_regression_kernels_different_shifts.p'

        fiber_side = experiment['fiber_side']
        num_recordings = experiments_to_process[experiments_to_process['mouse_id'] == mouse].shape
        sorted_exps = pd.to_datetime(
            experiments_to_process[experiments_to_process['mouse_id'] == mouse]['date']).sort_values(ignore_index=True)
        date_as_dt = pd.to_datetime(date)
        exp_number = sorted_exps[sorted_exps == date_as_dt].index[0]
        exp_numbers.append(exp_number)
        mice.append(mouse)
        if fiber_side == 'left':
            ipsi_cue = 'high cues'
            contra_cue = 'low cues'
        else:
            ipsi_cue = 'low cues'
            contra_cue = 'high cues'
        with open(saving_folder + save_filename, "rb") as f:
            session_kernels = pickle.load(f)
            print(mouse, date)
            if index == 0:
                ipsi_choice_kernel = session_kernels['kernels']['ipsi choices']
                contra_choice_kernel = session_kernels['kernels']['contra choices']
                ipsi_cue_kernel = session_kernels['kernels'][ipsi_cue]
                contra_cue_kernel = session_kernels['kernels'][contra_cue]
                reward_kernel = session_kernels['kernels']['rewards']
                no_reward_kernel = session_kernels['kernels']['rewards']
                #time_stamps = session_kernels['time_stamps']
            else:
                ipsi_choice_kernel = np.vstack([ipsi_choice_kernel, session_kernels['kernels']['ipsi choices']])
                contra_choice_kernel = np.vstack([contra_choice_kernel, session_kernels['kernels']['contra choices']])
                ipsi_cue_kernel = np.vstack([ipsi_cue_kernel, session_kernels['kernels'][ipsi_cue]])
                contra_cue_kernel = np.vstack([contra_cue_kernel, session_kernels['kernels'][contra_cue]])
                reward_kernel = np.vstack([reward_kernel, session_kernels['kernels']['rewards']])
                no_reward_kernel = np.vstack([no_reward_kernel, session_kernels['kernels']['no rewards']])
    time_stamps = {}

    time_stamps['ipsi choices'] = session_kernels['shifts']['ipsi choices'] / 10000 * 100
    time_stamps['contra choices'] = session_kernels['shifts']['contra choices']  / 10000 * 100
    time_stamps['ipsi cues'] = session_kernels['shifts']['high cues']  / 10000 * 100
    time_stamps['contra cues'] = session_kernels['shifts']['low cues']  / 10000 * 100
    time_stamps['rewards'] = session_kernels['shifts']['rewards']  / 10000 * 100
    time_stamps['no rewards'] = session_kernels['shifts']['no rewards']  / 10000 * 100
    means, sems = organise_data_means(ipsi_choice_kernel, contra_choice_kernel, ipsi_cue_kernel, contra_cue_kernel, reward_kernel, no_reward_kernel)
    return time_stamps, means, sems

def make_example_figure(ax1, ax2):
    axs= [ax1, ax2]
    mice_dates = pd.DataFrame(
        {'mouse': ['SNL_photo17', 'SNL_photo35', ], 'date': ['20200206', '20201121', ], 'site': ['tail', 'Nacc'],
         'inds': [np.arange(39100, 39300), np.arange(112450, 112650)]})
    for ind, mouse_date in mice_dates.iterrows():
        mouse = mouse_date['mouse']
        date = mouse_date['date']
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
        save_filename = saving_folder + mouse + '_' + date + '_'
        save_out_filename = saving_folder + mouse + '_' + date + '_example.npz'
        mouse_data = np.load(save_out_filename)
        time_stamps = {}
        time_stamps['rewards_ind'] = np.where(mouse_data['rewards'] == 1)
        time_stamps['high_cues_ind'] = np.where(mouse_data['high_cues'] == 1)
        time_stamps['contra_ind'] = np.where(mouse_data['contra_choices'] == 1)
        time_stamps['low_cues_ind'] = np.where(mouse_data['low_cues'] == 1)
        axs[ind].plot(mouse_data['dff'], color='gray', label='trace')
        axs[ind].plot(mouse_data['choice_pred'], label='choice kernel', color='#b7094c')
        axs[ind].plot(mouse_data['cue_pred'], label='cue kernel', color='#90be6d')
        axs[ind].plot(mouse_data['outcome_pred'], label='outcome kernel', color='#89c2d9')
        axs[ind].legend()
        axs[ind].axvline(time_stamps['rewards_ind'][0], color='k', lw=0.8)
        axs[ind].text(time_stamps['rewards_ind'][0], -0.1, 'reward', transform=axs[ind].get_xaxis_transform(), size=6)
        axs[ind].axvline(time_stamps['contra_ind'][0], color='k', lw=0.8)
        axs[ind].text(time_stamps['contra_ind'][0], -0.25, 'contra\nchoice', transform=axs[ind].get_xaxis_transform(), size=6)
        if mouse == 'SNL_photo35':
            axs[ind].axvline(time_stamps['high_cues_ind'][0], color='k', lw=0.8)
            axs[ind].text(time_stamps['high_cues_ind'][0], -0.1, 'cue', transform=axs[ind].get_xaxis_transform(), size=6)
        else:
            axs[ind].axvline(time_stamps['low_cues_ind'][0], color='k', lw=0.8)
            axs[ind].text(time_stamps['low_cues_ind'][0], -0.1, 'cue', transform=axs[ind].get_xaxis_transform(), size=6)
        axs[ind].axis('off')

        axs[ind].legend(loc='lower left', bbox_to_anchor=(0.71, 0.8),
                   borderaxespad=0, frameon=False, prop={'size': 6})

def organise_data_means(ipsi_choice_kernel, contra_choice_kernel, ipsi_cue_kernel, contra_cue_kernel, reward_kernel, no_reward_kernel):
    means = {}
    sems = {}
    means, sems = calculate_mean_and_sem(ipsi_choice_kernel, 'ipsi choices', means, sems)
    means, sems = calculate_mean_and_sem(contra_choice_kernel, 'contra choices', means, sems)
    means, sems = calculate_mean_and_sem(ipsi_cue_kernel, 'ipsi cues', means, sems)
    means, sems = calculate_mean_and_sem(contra_cue_kernel, 'contra cues', means, sems)
    means, sems = calculate_mean_and_sem(reward_kernel, 'rewards', means, sems)
    means, sems = calculate_mean_and_sem(no_reward_kernel, 'no rewards', means, sems)
    return means, sems


def calculate_mean_and_sem(kernel, param_name, plotting_dict_means, plotting_dict_sems):
    plotting_dict_means[param_name] = np.mean(kernel, axis=0)
    plotting_dict_sems[param_name] = sem(kernel, axis=0)
    return plotting_dict_means, plotting_dict_sems


def plot_kernels(axs, param_name, means_dict, sems_dict, time_stamps, colour='#7FB5B5', legend=False):
    param_kernel = means_dict[param_name]

    axs.plot(time_stamps[param_name], param_kernel, label=param_name, color=colour)
    axs.axvline(0, color='k')
    axs.fill_between(time_stamps[param_name], param_kernel - sems_dict[param_name],  param_kernel + sems_dict[param_name], alpha=0.5,
                            facecolor=colour, linewidth=0)
    axs.set_ylabel('regression coefficient')
    axs.set_xlabel('time (s)')
    if legend:
        axs.legend(loc='lower left', bbox_to_anchor=(0.6, 0.8),
            borderaxespad=0, frameon=False,prop={'size': 6 })


def plot_kernels_different_shifts(parameter_names, coefs, all_shifts, shift_window_sizes):
    fig, axs = plt.subplots(nrows=1, ncols=len(parameter_names), sharey=True, figsize=(15,8))
    axs[0].set_ylabel('Regression coefficient')
    for param_num, param_name in enumerate(parameter_names):
        shifts = all_shifts[param_num]
        shift_window_size = shift_window_sizes[param_num]
        starting_ind = int(np.sum(shift_window_sizes[:param_num]))
        param_kernel = coefs[starting_ind: starting_ind + shift_window_size]
        axs[param_num].plot(shifts*100/10000, param_kernel, label=param_name)
        axs[param_num].set_title(param_name)
        axs[param_num].axvline([0])
        axs[param_num].set_xlabel('Time (s)')

def plot_kernels_for_site(move_axs, cue_axs, reward_axs, means, sems, time_stamps, legend=False):
    plot_kernels(move_axs, 'ipsi choices', means, sems, time_stamps, colour='#415a77', legend=legend)
    plot_kernels(move_axs, 'contra choices', means, sems, time_stamps, colour='#b7094c', legend=legend)
    plot_kernels(cue_axs, 'ipsi cues', means, sems, time_stamps, colour='#4281a4', legend=legend)
    plot_kernels(cue_axs, 'contra cues', means, sems, time_stamps, colour='#90be6d', legend=legend)
    plot_kernels(reward_axs, 'rewards', means, sems, time_stamps, colour='#89c2d9', legend=legend)
    plot_kernels(reward_axs, 'no rewards', means, sems, time_stamps, colour='#f08080', legend=legend)


def load_exp_var_data_for_site(site):
    if site == 'tail':
        mice = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26']
    elif site == 'Nacc':
        mice = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33',
                     'SNL_photo34', 'SNL_photo35']
    file_name = site + '_explained_variances.p'
    processed_data_dir = os.path.join('W:\\photometry_2AC\\processed_data\\linear_regression_data\\')
    saving_filename = os.path.join('W:\\photometry_2AC\\processed_data\\linear_regression_data\\', file_name)

    reg_stats = pickle.load(open(saving_filename, 'rb'))
    reg_stats = reg_stats[reg_stats['mouse_id'].isin(mice)]
    mean_stats = reg_stats.groupby(['mouse_id'])[ ['cue explained variance', 'choice explained variance', 'outcome explained variance', 'full model explained variance']].apply(np.mean)
    types = []
    variances = []
    for ind, row in mean_stats.iterrows():
        types.append('cue')
        types.append('choice')
        types.append('outcome')
        types.append('full')
        variances.append(row['cue explained variance'])
        variances.append(row['choice explained variance'])
        variances.append(row['outcome explained variance'])
        variances.append(row['full model explained variance'])
    stats_dict = {'predictor': types, 'explained variance': variances}
    reshaped_stats = pd.DataFrame(stats_dict)
    if site == 'Nacc':
        label = 'VS'
    elif site == 'tail':
        label = 'AudS'
    reshaped_stats['site'] = label
    return reshaped_stats


def get_data_both_sites_for_predictor(nacc_data, tail_data, predictor):
    df = pd.concat([nacc_data[nacc_data['predictor']==predictor], tail_data[tail_data['predictor']==predictor]])
    return df


def make_box_plot(df, fig_ax,  dx ='site', dy = 'explained variance', ort = "v", pal = "Set2", set_ylims=False, label=None):
    sns.stripplot( x = dx, y = dy, data = df, palette = pal, edgecolor = "white",
                     size = 5, jitter = 1, zorder = 0, orient = ort, ax=fig_ax)
    sns.boxplot( x = dx, y = dy, data = df, color = "black", width = .5, zorder = 10,linewidth=0.5, \
                showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                showfliers=False, whiskerprops = {'linewidth':0.5, "zorder":10},\
                   saturation = 1, orient = ort, ax=fig_ax)
    fig_ax.set_xlim([-0.5, 1.5])
    if set_ylims:
        fig_ax.set_ylim([-2, np.max(df[dy]) + 2])
    if label:
        fig_ax.text(0.5, 1, label, transform=fig_ax.get_xaxis_transform(), size=8, ha='center')