
import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')
from scipy.interpolate import interp1d
from utils.plotting import calculate_error_bars
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import os
import matplotlib
from matplotlib.lines import Line2D
from utils.plotting_visuals import makes_plots_pretty
import pickle
import peakutils
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.regression.linear_regression_utils import get_first_x_sessions
import pandas as pd
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
from utils.plotting import calculate_error_bars


def get_correct_incorrect_trials(experiments_to_process):
    exp_numbers = []
    mice = []
    for index, experiment in experiments_to_process.iterrows():
        mouse = experiment['mouse_id']
        date = experiment['date']
        dates = experiments_to_process[experiments_to_process['mouse_id'] == mouse]['date'].values
        session_starts = get_bpod_trial_nums_per_session(mouse, dates)
        session_ind = np.where(dates == date)[0][0]
        session_start_trial = session_starts[session_ind]
        saving_folder = 'W:\\photometry_2AC\\processed_data\\for_figure\\' + mouse + '\\'
        save_filename = mouse + '_' + date + '_' + 'aligned_traces_correct_incorrect.p'

        sorted_exps = pd.to_datetime(
            experiments_to_process[experiments_to_process['mouse_id'] == mouse]['date']).sort_values(ignore_index=True)
        date_as_dt = pd.to_datetime(date)
        exp_number = sorted_exps[sorted_exps == date_as_dt].index[0]
        exp_numbers.append(exp_number)
        with open(saving_folder + save_filename, "rb") as f:
            content = pickle.load(f)
        print(mouse, date)
        if index == 0:
            correct = content.choice_data.contra_correct_data.sorted_traces[:,
                      int(160000 / 2 - 20000):int(160000 / 2 + 20000)]
            incorrect = content.choice_data.contra_incorrect_data.sorted_traces[:,
                        int(160000 / 2 - 20000):int(160000 / 2 + 20000)]
            time_stamps = content.choice_data.contra_correct_data.time_points[
                          int(160000 / 2 - 20000):int(160000 / 2 + 20000)]
            correct_trial_nums = content.choice_data.contra_correct_data.trial_nums + session_start_trial
            incorrect_trial_nums = content.choice_data.contra_incorrect_data.trial_nums + session_start_trial
            reaction_times = content.choice_data.contra_incorrect_data.reaction_times
            correct_reaction_times = content.choice_data.contra_correct_data.reaction_times
        else:
            correct = np.vstack([correct, content.choice_data.contra_correct_data.sorted_traces[:,
                                          int(160000 / 2 - 20000):int(160000 / 2 + 20000)]])
            incorrect = np.vstack([incorrect, content.choice_data.contra_incorrect_data.sorted_traces[:,
                                              int(160000 / 2 - 20000):int(160000 / 2 + 20000)]])
            correct_trial_nums = np.concatenate(
                (correct_trial_nums, content.choice_data.contra_correct_data.trial_nums + session_start_trial))
            incorrect_trial_nums = np.concatenate(
                (incorrect_trial_nums, content.choice_data.contra_incorrect_data.trial_nums + session_start_trial))
            reaction_times = np.concatenate((reaction_times, content.choice_data.contra_incorrect_data.reaction_times))
            correct_reaction_times = np.concatenate(
                (correct_reaction_times, content.choice_data.contra_correct_data.reaction_times))
    return correct, incorrect, correct_trial_nums, incorrect_trial_nums, correct_reaction_times, reaction_times, time_stamps


def find_nearest_trials(target_trials, other_trials):
    differences = (target_trials.reshape(1,-1) - other_trials.reshape(-1,1))
    indices = np.abs(differences).argmin(axis=0)
    residual = np.diagonal(differences[indices,])
    return indices


def get_traces_and_mean(all_traces, inds):
    traces = all_traces[inds, :]
    mean_trace = np.mean(traces, axis=0)
    return traces, mean_trace


def plot_trace(all_traces, inds, time_stamps, ax, color='green'):
    traces, mean_trace = get_traces_and_mean(all_traces, inds)

    ax.plot(time_stamps, mean_trace, color=color)
    error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                            traces,
                                                            error_bar_method='sem')
    ax.fill_between(time_stamps, error_bar_lower, error_bar_upper, alpha=0.5,
                     facecolor=color, linewidth=0)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('z-scored fluorescence')


def get_peak(all_traces, inds, median_reaction_time):
    traces, mean_trace = get_traces_and_mean(all_traces, inds)
    half_way = int(traces.shape[1]/2)
    trace_from_event = mean_trace[half_way:half_way + int(1000*(median_reaction_time))]
    trial_peak_inds = peakutils.indexes(trace_from_event.flatten('F'))
    if trial_peak_inds.shape[0] > 0 or len(trial_peak_inds > 1):
        trial_peak_inds = trial_peak_inds[0]
        trial_peaks = trace_from_event.flatten('F')[trial_peak_inds]
    else:
        trial_peak_inds = np.argmax(trace_from_event)
        trial_peaks = np.max(trace_from_event)
    trial_peak_inds += half_way
    return trial_peaks, trial_peak_inds,


mouse_ids = ['SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo17', 'SNL_photo16', 'SNL_photo18']
site = 'tail'
experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[
    (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
    drop=True)
experiments_to_process = remove_bad_recordings(all_experiments_to_process).reset_index(drop=True)
for mouse_ind, mouse in enumerate(mouse_ids):
    mouse_experiments = experiments_to_process[experiments_to_process['mouse_id'] == mouse].reset_index(drop=True)
    correct, incorrect, correct_trial_nums, incorrect_trial_nums, correct_reaction_times, incorrect_reaction_times, time_stamps = get_correct_incorrect_trials(mouse_experiments)
    upper_quartile = np.quantile(correct_reaction_times, 0.75)
    lower_quartile = np.quantile(correct_reaction_times, 0.25)
    incorrect_reaction_times_valid_inds = np.where(incorrect_reaction_times <= upper_quartile)
    correct_reaction_times_valid_inds = np.where(correct_reaction_times <= upper_quartile)
    correct_reaction_times = correct_reaction_times[correct_reaction_times_valid_inds]
    incorrect_reaction_times = incorrect_reaction_times[incorrect_reaction_times_valid_inds]
    incorrect_trial_nums = incorrect_trial_nums[incorrect_reaction_times_valid_inds]
    incorrect = incorrect[incorrect_reaction_times_valid_inds]
    correct_trial_nums = correct_trial_nums[correct_reaction_times_valid_inds]
    correct = correct[correct_reaction_times_valid_inds]

    max_trials = np.max(np.concatenate((incorrect_trial_nums, correct_trial_nums)))
    early_incorrect_trials = incorrect_trial_nums[incorrect_trial_nums < int(max_trials/3)]
    mid_incorrect_trials = incorrect_trial_nums[np.logical_and(incorrect_trial_nums < int(max_trials/3)*2, incorrect_trial_nums > int(max_trials/3))]
    late_incorrect_trials = incorrect_trial_nums[np.logical_and(incorrect_trial_nums <= max_trials, incorrect_trial_nums > int(max_trials/3)*2)]

    early_incorrect_inds = np.nonzero(np.in1d(incorrect_trial_nums, early_incorrect_trials))[0]
    mid_incorrect_inds = np.nonzero(np.in1d(incorrect_trial_nums, mid_incorrect_trials))[0]
    late_incorrect_inds = np.nonzero(np.in1d(incorrect_trial_nums, late_incorrect_trials))[0]
    early_correct_inds = find_nearest_trials(early_incorrect_trials, correct_trial_nums)
    mid_correct_inds = find_nearest_trials(mid_incorrect_trials, correct_trial_nums)
    late_correct_inds = find_nearest_trials(late_incorrect_trials, correct_trial_nums)

    #early_correct_trials = correct_trial_nums[correct_trial_nums < int(max_trials / 3)]
    #mid_correct_trials = correct_trial_nums[
    #    np.logical_and(correct_trial_nums < int(max_trials / 3) * 2, correct_trial_nums > int(max_trials / 3))]
    #late_correct_trials = correct_trial_nums[
    #    np.logical_and(correct_trial_nums <= max_trials, correct_trial_nums > int(max_trials / 3) * 2)]
    #early_correct_inds = np.nonzero(np.in1d(correct_trial_nums, early_correct_trials))[0]
    #mid_correct_inds = np.nonzero(np.in1d(correct_trial_nums, mid_correct_trials))[0]
    #late_correct_inds = np.nonzero(np.in1d(correct_trial_nums, late_correct_trials))[0]
    all_reaction_times = np.concatenate([incorrect_reaction_times, correct_reaction_times])
    median_reaction_time = np.median(all_reaction_times)
    sd_reaction_times = np.std(all_reaction_times)
    limit = 2 * sd_reaction_times + median_reaction_time
    early_correct_trial_peaks, trial_peak_inds = get_peak(correct, early_correct_inds, limit)
    early_incorrect_trial_peaks, trial_peak_inds = get_peak(incorrect, early_incorrect_inds, limit)
    mid_correct_trial_peaks, trial_peak_inds = get_peak(correct, mid_correct_inds, limit)
    mid_incorrect_trial_peaks, trial_peak_inds = get_peak(incorrect, mid_incorrect_inds, limit)
    late_correct_trial_peaks, trial_peak_inds = get_peak(correct, late_correct_inds, limit)
    late_incorrect_trial_peaks, trial_peak_inds = get_peak(incorrect, late_incorrect_inds, limit)
    if mouse_ind == 0:
        mice = np.full([6, 1], mouse)
        trial_types = np.expand_dims(np.array(['correct', 'incorrect', 'correct', 'incorrect', 'correct', 'incorrect']), axis=1)
        time_point = np.expand_dims(np.array(['early', 'early', 'mid', 'mid', 'late', 'late']), axis=1)
        response_sizes = np.expand_dims(np.array([early_correct_trial_peaks, early_incorrect_trial_peaks, mid_correct_trial_peaks, mid_incorrect_trial_peaks, late_correct_trial_peaks, late_incorrect_trial_peaks]), axis=1)
    else:
        mice = np.vstack([mice, np.full([6, 1], mouse)])
        trial_types = np.vstack([trial_types, np.expand_dims(np.array(['correct', 'incorrect', 'correct', 'incorrect', 'correct', 'incorrect']), axis=1)])
        time_point = np.vstack([time_point, np.expand_dims(np.array(['early', 'early', 'mid', 'mid', 'late', 'late']), axis=1)])
        response_sizes = np.vstack([response_sizes, np.expand_dims(np.array([early_correct_trial_peaks, early_incorrect_trial_peaks, mid_correct_trial_peaks, mid_incorrect_trial_peaks, late_correct_trial_peaks, late_incorrect_trial_peaks]), axis=1)])

data = pd.DataFrame({'mouse': np.squeeze(mice), 'trial outcome': np.squeeze(trial_types), 'learning phase': np.squeeze(time_point), 'response size': np.squeeze(response_sizes)})
correct_data = data[data['trial outcome'] == 'correct'].reset_index(drop=True).copy(deep=True)
diff_data = correct_data[['mouse', 'learning phase']]
diff_data['difference'] = data[data['trial outcome'] == 'incorrect']['response size'].reset_index(drop=True).copy(deep=True) - correct_data['response size']
fig, axs = plt.subplots(1,1)
for mouse in mouse_ids:
    data_for_plot = data[data['mouse'] == mouse].copy(deep=True)
    responses = data_for_plot.loc[:, ['response size']]
    data_for_plot['normalised response size'] = responses/data_for_plot.loc[(data_for_plot['learning phase']== 'early') & (data_for_plot['trial outcome'] == 'correct')]['response size'].values[0]
    sns.pointplot(x='learning phase', y='response size', data=data_for_plot, hue='trial outcome', palette=['green', 'red'], ax=axs, legend=False)
    axs.get_legend().remove()

fig, axs = plt.subplots(1,3, sharey=True)
sns.pointplot(x='trial outcome', y='response size', data=data[data['learning phase'] == 'early'], hue='mouse', ax=axs[0], legend=False)
axs[0].get_legend().remove()
sns.pointplot(x='trial outcome', y='response size', data=data[data['learning phase'] == 'mid'], hue='mouse', ax=axs[1], legend=False)
axs[1].get_legend().remove()
sns.pointplot(x='trial outcome', y='response size', data=data[data['learning phase'] == 'late'], hue='mouse', ax=axs[2], legend=False)

fig, axs = plt.subplots(1,1)
sns.pointplot(x='learning phase', y='difference', data=diff_data, ax=axs, legend=False, hue='mouse')
plt.show()
print('a')