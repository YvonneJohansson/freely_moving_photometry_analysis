
import pickle
import datetime
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import math
import pandas as pd
import scipy as scipy
from scipy import optimize
import peakutils
import os
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
from scipy.signal import decimate


def get_valid_traces(mouse, dates, window_around_mean=0.2, recording_site='tail', side='contra', window_size=40):
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    data_root = r'W:\photometry_2AC\processed_data\peak_analysis'
    all_bins = []
    all_reaction_times =[]
    all_trial_numbers = []
    all_actual_trial_numbers = []
    all_traces = []
    # if I can get the trial numbers ever that things belong to, then we are in business
    for date_num, date in enumerate(dates):
        print(date)
        peaks_saving_folder = os.path.join(data_root, mouse)
        aligned_filename = saving_folder + mouse + '_' + date + '_' + 'aligned_traces.p'
        #filename = mouse + '_' + date + '_' + 'peaks.p'
        #aligned_filename = os.path.join(peaks_saving_folder, filename)
        #mean_and_sem_filename = saving_folder + mouse + '_' + date + '_' + 'peaks_' + type_of_session + '_data.p'
        with open(aligned_filename, 'rb') as f:
            data = pickle.load(f)
        if recording_site == 'tail':
            recording_site_data = data.choice_data.contra_data
            all_reaction_times.append(recording_site_data.reaction_times)
        elif recording_site == 'Nacc':
            recording_site_data = data.cue_data.contra_data
            all_reaction_times.append(recording_site_data.outcome_times)
        actual_trial_numbers = recording_site_data.trial_nums + session_starts[date_num]
        all_traces.append(decimate(recording_site_data.sorted_traces, 10))
        all_actual_trial_numbers.append(actual_trial_numbers)
        all_trial_numbers.append(len(recording_site_data.reaction_times))
        all_bins.append(np.arange(start=min(recording_site_data.reaction_times),
                                  stop=max(recording_site_data.reaction_times) + 0.1, step=0.1))

    flattened_actual_trial_nums = [item for sublist in all_actual_trial_numbers for item in sublist]
    flattened_reaction_times = [item for sublist in all_reaction_times for item in sublist]
    trials_per_session = [i.shape[0] for i in all_traces]
    num_trials = np.sum(trials_per_session)
    trials_per_session.insert(0, 0)
    flattened_traces = np.zeros([num_trials, all_traces[0].shape[1]])
    cum_sum = np.cumsum(trials_per_session)
    for i, trace in enumerate(all_traces):
        flattened_traces[cum_sum[i]:cum_sum[i + 1], :] = trace
    median_reaction_time = np.median(flattened_reaction_times)
    valid_trials = np.where(
        np.logical_and(np.greater_equal(flattened_reaction_times, median_reaction_time - window_around_mean),
                       np.less_equal(flattened_reaction_times, median_reaction_time + window_around_mean)))
    valid_flattened_traces = flattened_traces[valid_trials[0], :]
    valid_trial_nums = np.array(flattened_actual_trial_nums)[valid_trials]
    rolling_mean_trace = []
    rolling_mean_x = []
    rolling_mean_peak = []
    all_peak_trace_inds = []
    num_bins = int(np.shape(valid_flattened_traces)[0]/window_size)
    colours = cm.viridis(np.linspace(0, 0.8, num_bins))
    #fig, ax = plt.subplots(1, 2)
    for window_num in range(num_bins):
        rolling_mean_trace.append(np.nanmean(valid_flattened_traces[window_num * window_size: (window_num + 1) * window_size, :], axis=0))
        rolling_mean_x.append(np.nanmean(valid_trial_nums[window_num * window_size: (window_num + 1) * window_size]))
        #ax[0].plot(rolling_mean_trace[window_num], color=colours[window_num])
        half_way = int(valid_flattened_traces.shape[1]/2)
        trace_from_event = rolling_mean_trace[window_num][half_way:half_way + int(1000*(median_reaction_time))]
        trial_peak_inds = peakutils.indexes(trace_from_event.flatten('F'))
        if trial_peak_inds.shape[0] > 0 or len(trial_peak_inds > 1):
            trial_peak_inds = trial_peak_inds[0]
            trial_peaks = trace_from_event.flatten('F')[trial_peak_inds]
        else:
            trial_peak_inds = np.argmax(trace_from_event)
            trial_peaks = np.max(trace_from_event)
            print(window_num)
        rolling_mean_peak.append(trial_peaks)
        peak_trace_inds = trial_peak_inds + half_way
        all_peak_trace_inds.append(peak_trace_inds)
        #ax[0].scatter(peak_trace_inds, trial_peaks, color=colours[window_num])
        #ax[1].scatter(rolling_mean_x, rolling_mean_peak)
    valid_reaction_times = np.array(flattened_reaction_times)[valid_trials]
    return rolling_mean_x, rolling_mean_peak, all_peak_trace_inds, rolling_mean_trace

class OverTimeData(object):
    def __init__(self):
        self.all_bins = []
        self.reaction_times = []
        self.all_trial_numbers = []
        self.all_actual_trial_numbers = []
        self.traces = []

    def add_data(self, data, session_start):
        self.reaction_times.append(data.reaction_times)
        actual_trial_numbers = data.trial_nums + session_start
        self.traces.append(decimate(data.sorted_traces, 10))
        self.all_actual_trial_numbers.append(actual_trial_numbers)
        self.all_trial_numbers.append(len(data.reaction_times))
        self.all_bins.append(np.arange(start=min(data.reaction_times),
                                  stop=max(data.reaction_times) + 0.1, step=0.1))

    def get_rolling_mean(self, time_limit, window_size=50):
        flattened_actual_trial_nums = [item for sublist in self.all_actual_trial_numbers for item in sublist]
        flattened_reaction_times = [item for sublist in self.reaction_times for item in sublist]
        trials_per_session = [i.shape[0] for i in self.traces]
        num_trials = np.sum(trials_per_session)
        trials_per_session.insert(0, 0)
        flattened_traces = np.zeros([num_trials, self.traces[0].shape[1]])
        cum_sum = np.cumsum(trials_per_session)
        for i, trace in enumerate(self.traces):
            flattened_traces[cum_sum[i]:cum_sum[i + 1], :] = trace

        valid_trials = np.where(
            np.logical_and(np.greater_equal(flattened_reaction_times,0),
                           np.less_equal(flattened_reaction_times, time_limit)))
        valid_flattened_traces = flattened_traces[valid_trials[0], :]
        valid_trial_nums = np.array(flattened_actual_trial_nums)[valid_trials]
        rolling_mean_trace = []
        rolling_mean_x = []
        rolling_mean_peak = []
        all_peak_trace_inds = []
        num_bins = int(np.shape(valid_flattened_traces)[0] / window_size)
        colours = cm.viridis(np.linspace(0, 0.8, num_bins))
        # fig, ax = plt.subplots(1, 2)
        for window_num in range(num_bins):
            rolling_mean_trace.append(
                np.nanmean(valid_flattened_traces[window_num * window_size: (window_num + 1) * window_size, :], axis=0))
            rolling_mean_x.append(
                np.nanmean(valid_trial_nums[window_num * window_size: (window_num + 1) * window_size]))
            # ax[0].plot(rolling_mean_trace[window_num], color=colours[window_num])
            half_way = int(valid_flattened_traces.shape[1] / 2)
            trace_from_event = rolling_mean_trace[window_num][half_way:half_way + int(1000 * (time_limit))]
            trial_peak_inds = peakutils.indexes(trace_from_event.flatten('F'))
            if trial_peak_inds.shape[0] > 0 or len(trial_peak_inds > 1):
                trial_peak_inds = trial_peak_inds[0]
                trial_peaks = trace_from_event.flatten('F')[trial_peak_inds]
            else:
                trial_peak_inds = np.argmax(trace_from_event)
                trial_peaks = np.max(trace_from_event)
                print(window_num)
            rolling_mean_peak.append(trial_peaks)
            peak_trace_inds = trial_peak_inds + half_way
            all_peak_trace_inds.append(peak_trace_inds)
        valid_reaction_times = np.array(flattened_reaction_times)[valid_trials]
        return rolling_mean_x, rolling_mean_peak, all_peak_trace_inds, rolling_mean_trace


def get_correct_incorrect_traces(mouse, dates, window_around_mean=0.2, recording_site='tail', side='contra', window_size=40):
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    saving_folder = 'W:\\photometry_2AC\\processed_data\\for_figure\\' + mouse + '\\'
    data_root = r'W:\photometry_2AC\processed_data\peak_analysis'
    correct_data = OverTimeData()
    incorrect_data = OverTimeData()
    for date_num, date in enumerate(dates):
        print(date)
        peaks_saving_folder = os.path.join(data_root, mouse)
        aligned_filename = mouse + '_' + date + '_' + 'aligned_traces_correct_incorrect.p'
        with open(saving_folder + aligned_filename, 'rb') as f:
            data = pickle.load(f)
        if recording_site == 'tail':
            correct = data.choice_data.contra_correct_data
            correct_data.add_data(correct, session_starts[date_num])
            incorrect = data.choice_data.contra_incorrect_data
            incorrect_data.add_data(incorrect, session_starts[date_num])
    correct_reaction_times = [item for sublist in correct_data.reaction_times for item in sublist]
    reaction_time_limit = np.quantile(correct_reaction_times, 0.75)
    correct_rolling_means = {}
    incorrect_rolling_means = {}
    rolling_mean_x, rolling_mean_peaks, peak_trace_inds, rolling_mean_traces = correct_data.get_rolling_mean(reaction_time_limit, window_size=window_size)
    correct_filename = mouse + '_binned_' + str(window_size) + '_average_then_peaks_correct_trials.npz'
    np.savez(os.path.join(peaks_saving_folder, correct_filename), rolling_mean_x=rolling_mean_x, rolling_mean_peaks=rolling_mean_peaks,
             rolling_mean_traces=rolling_mean_traces, peak_trace_inds=peak_trace_inds)

    i_rolling_mean_x, i_rolling_mean_peaks, i_peak_trace_inds, i_rolling_mean_traces = incorrect_data.get_rolling_mean(reaction_time_limit, window_size=window_size)
    incorrect_filename = mouse + '_binned_' + str(window_size) + '_average_then_peaks_incorrect_trials.npz'
    np.savez(os.path.join(peaks_saving_folder, incorrect_filename), rolling_mean_x=i_rolling_mean_x, rolling_mean_peaks=i_rolling_mean_peaks,
             rolling_mean_traces=i_rolling_mean_traces, peak_trace_inds=i_peak_trace_inds)
