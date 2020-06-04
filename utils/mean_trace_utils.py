import pickle
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import peakutils
import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
from utils.plotting import correctData

class mouseDates(object):
    def __init__(self, mouse_id, dates):
        self.mouse = mouse_id
        self.dates = dates

def make_dates_pretty(inputDates):
    # assumes input style YYYYMMDD
    outputDates = []
    for date in inputDates:
        x = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]))
        outputDates.append(x.strftime("%b%d"))
    return (outputDates)


def percentage_correct_correlation(mouse, dates, peaks):
    BpodProtocol = '/Two_Alternative_Choice/'
    GeneralDirectory = 'W:/photometry_2AC/bpod_data/'
    DFfile = GeneralDirectory + mouse + BpodProtocol + 'Data_Analysis/' + mouse + '_dataframe.pkl'
    behavioural_stats = pd.read_pickle(DFfile)
    reformatted_dates = make_dates_pretty(dates)
    percentage_correct = []
    for date_num, date in enumerate(reformatted_dates):
        points_for_day = behavioural_stats[behavioural_stats['SessionTime'].str.contains(date)]
        percentage_correct.append(100 * np.sum(points_for_day['FirstPokeCorrect']) / len(points_for_day))

    num_types = len(dates)
    colours = cm.viridis(np.linspace(0, 0.8, num_types))
    fig, axs = plt.subplots(1, ncols=1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    fig.suptitle('Exit centre poke', fontsize=16)
    fig.text(0.06, 0.02, mouse, fontsize=12)
    axs.title.set_text('Contralateral choice peak activity')
    axs.scatter(percentage_correct, peaks, color=colours)
    axs.set_xlabel('Percentage correct')
    axs.set_ylabel('Peak size (z-score)')
    return (peaks, percentage_correct)


def num_rewards_correlation(mouse, dates, peaks):
    BpodProtocol = '/Two_Alternative_Choice/'
    GeneralDirectory = 'W:/photometry_2AC/bpod_data/'
    DFfile = GeneralDirectory + mouse + BpodProtocol + 'Data_Analysis/' + mouse + '_dataframe.pkl'
    behavioural_stats = pd.read_pickle(DFfile)
    reformatted_dates = make_dates_pretty(dates)
    num_rewards = []
    for date_num, date in enumerate(reformatted_dates):
        points_for_day = behavioural_stats[behavioural_stats['SessionTime'].str.contains(date)]
        num_rewards.append(np.sum(points_for_day['FirstPokeCorrect']))
    cum_num_rewards = np.cumsum(num_rewards)
    num_types = len(dates)
    colours = cm.viridis(np.linspace(0, 0.8, num_types))
    fig, axs = plt.subplots(1, ncols=1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    fig.suptitle('Exit centre poke', fontsize=16)
    fig.text(0.06, 0.02, mouse, fontsize=12)
    axs.title.set_text('Contralateral choice peak activity')
    axs.scatter(cum_num_rewards, peaks, color=colours)
    axs.set_xlabel('Number or rewards ever')
    axs.set_ylabel('Peak size (z-score)')
    return (peaks, cum_num_rewards)


def multi_day_peaks(mouse, dates):
    reformatted_dates = []
    for date in dates:
        year = int(date[0:4])
        month = int(date[4:6])
        day = int(date[6:])
        reformatted_dates.append(datetime.date(year, month, day))

    reform_dates = np.array(reformatted_dates, dtype='datetime64')
    days_since_last_recording = np.concatenate([np.array([0], dtype='timedelta64[D]'), np.diff(reform_dates)]).astype(
        int)
    day_of_training = np.cumsum(days_since_last_recording)

    num_types = len(dates)
    colours = cm.viridis(np.linspace(0, 0.8, num_types))
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    fig, axs = plt.subplots(1, ncols=1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    fig.suptitle('Exit centre poke', fontsize=16)
    fig.text(0.06, 0.02, mouse, fontsize=12)
    all_peaks = []
    all_days = []
    for date_num, date in enumerate(dates):
        mean_and_sem_filename = saving_folder + mouse + '_' + date + '_' + 'mean_correct_data.p'
        correct_data = pickle.load(open(mean_and_sem_filename, "rb"))
        num_samples = correct_data.contra_mean_y_vals.shape[0]
        peaks = peakutils.indexes(correct_data.contra_mean_y_vals, thres=0.3)
        first_peak_ind = peaks[np.where(peaks > num_samples / 2)]
        first_peak_ind = first_peak_ind[np.where(first_peak_ind < int(0.66 * num_samples))][0]
        first_peak_val = correct_data.contra_mean_y_vals[first_peak_ind]
        all_peaks.append(first_peak_val)
        all_days.append(day_of_training[date_num])
        axs.title.set_text('Contralateral choice peak activity')
        axs.scatter(day_of_training[date_num], first_peak_val, color=colours[date_num])
        axs.legend(dates, frameon=False)

    X = np.array(all_days).reshape(-1, 1)
    Y = np.array(all_peaks).reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    r_sq = linear_regressor.score(X, Y)
    plt.plot(X, Y_pred, lw=1, color='#746D69')
    axs.set_xlabel('Days since start of training')
    axs.set_ylabel('Session mean z-scored response')
    print(r_sq)
    return (all_peaks)


def peaks_correlations(mouse, dates, peaks):
    reformatted_dates = []
    for date in dates:
        year = int(date[0:4])
        month = int(date[4:6])
        day = int(date[6:])
        reformatted_dates.append(datetime.date(year, month, day))

    reform_dates = np.array(reformatted_dates, dtype='datetime64')
    days_since_last_recording = np.concatenate([np.array([0], dtype='timedelta64[D]'), np.diff(reform_dates)]).astype(
        int)

    num_types = len(dates)
    colours = cm.viridis(np.linspace(0, 0.8, num_types))
    fig, axs = plt.subplots(1, ncols=1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    fig.suptitle('Exit centre poke', fontsize=16)
    fig.text(0.06, 0.02, mouse, fontsize=12)
    axs.title.set_text('Contralateral choice peak activity')
    axs.scatter(days_since_last_recording, peaks, color=colours)
    axs.set_xlabel('Days since last recording')
    axs.set_ylabel('Peak size (z-score)')


def plot_multiple_days(mouse, dates, type_of_session='correct', plotting_style='overlayed'):
    file_tag = 'mean_' + type_of_session + '_data.p'
    num_types = len(dates)
    colours = cm.viridis(np.linspace(0, 0.8, num_types))
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    if plotting_style == 'overlayed':
        fig, axs = plt.subplots(1, ncols=2, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        fig.suptitle(type_of_session + ' response', fontsize=16)
        fig.text(0.06, 0.02, mouse, fontsize=12)
    elif plotting_style == 'wrapped':
        fig1, axs1 = plt.subplots(nrows=2,ncols=math.ceil(num_types / 2), figsize=(10, 8), sharex=True, sharey=True)
        fig1.subplots_adjust(hspace=0.5, wspace=0.2)
        axs1 = axs1.flatten()
        fig1.suptitle(type_of_session + ', contralateral response', fontsize=16)
        fig1.text(0.06, 0.02, mouse, fontsize=12)

        fig2, axs2 = plt.subplots(nrows=2,ncols=math.ceil(num_types / 2), figsize=(10,8), sharex=True, sharey=True)
        fig2.subplots_adjust(hspace=0.5, wspace=0.2)
        axs2 = axs2.flatten()
        fig2.suptitle(type_of_session + ', ipsilateral response', fontsize=16)
        fig2.text(0.06, 0.02, mouse, fontsize=12)

    for date_num, date in enumerate(dates):
        mean_and_sem_filename = saving_folder + mouse + '_' + date + '_' + file_tag
        correct_data = pickle.load(open(mean_and_sem_filename, "rb"))
        num_samples = correct_data.contra_mean_y_vals.shape[0]
        peaks = peakutils.indexes(correct_data.contra_mean_y_vals, thres=0.4)
        first_peak_ind = peaks[np.where(peaks > num_samples / 2)]
        first_peak_ind = first_peak_ind[np.where(first_peak_ind < int(0.66 * num_samples))][0]
        first_peak_val = correct_data.contra_mean_y_vals[first_peak_ind]
        first_peak_val = correct_data.contra_mean_y_vals[first_peak_ind]

        if plotting_style == 'overlayed':
            axs[0].title.set_text('Contralateral choice')
            axs[0].plot(correct_data.contra_mean_x_vals, correct_data.contra_mean_y_vals, color=colours[date_num], lw=3)
            axs[0].fill_between(correct_data.contra_mean_x_vals, correct_data.contra_CI[0], correct_data.contra_CI[1],
                                color=colours[date_num], alpha=0.3, linewidth=0)
            axs[0].legend(dates, frameon=False)
            # axs[0].scatter(correct_data.contra_mean_x_vals[first_peak_ind],correct_data.contra_mean_y_vals[first_peak_ind], color='k', marker='x' )

            axs[1].title.set_text('Ipsilateral choice')
            axs[1].plot(correct_data.ipsi_mean_x_vals, correct_data.ipsi_mean_y_vals, color=colours[date_num], lw=3)
            axs[1].fill_between(correct_data.ipsi_mean_x_vals, correct_data.ipsi_CI[0], correct_data.ipsi_CI[1],
                                color=colours[date_num], alpha=0.3, linewidth=0)
            axs[1].axvline(0, color='k', linewidth=2)

        elif plotting_style == 'wrapped':
            axs1[date_num].title.set_text(date)
            axs1[date_num].plot(correct_data.contra_mean_x_vals, correct_data.contra_mean_y_vals, color=colours[date_num], lw=3)
            axs1[date_num].fill_between(correct_data.contra_mean_x_vals, correct_data.contra_CI[0], correct_data.contra_CI[1],
                                color=colours[date_num], alpha=0.3, linewidth=0)
            axs1[date_num].axvline(0, color='k', linewidth=2)
            axs1[date_num].set_xlim([-3, 3])
            axs1[date_num].set_ylim([-1.2, 2])
            axs1[date_num].set_xlabel('Time (s)')
            axs1[date_num].set_ylabel('z-scored fluorescence')

            axs2[date_num].title.set_text(date)
            axs2[date_num].plot(correct_data.ipsi_mean_x_vals, correct_data.ipsi_mean_y_vals,
                                color=colours[date_num], lw=3)
            axs2[date_num].fill_between(correct_data.ipsi_mean_x_vals, correct_data.ipsi_CI[0],
                                        correct_data.ipsi_CI[1],
                                        color=colours[date_num], alpha=0.3, linewidth=0)
            axs2[date_num].axvline(0, color='k', linewidth=2)
            axs2[date_num].set_xlim([-3, 3])
            axs2[date_num].set_ylim([-1.2, 2])
            axs2[date_num].set_xlabel('Time (s)')
            axs2[date_num].set_ylabel('z-scored fluorescence')

    if plotting_style == 'overlayed':
        axs[0].axvline(0, color='k', linewidth=2)
        axs[0].set_xlim([-3, 3])
        axs[0].set_ylim([-1.2, 2])
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('z-scored fluorescence')
        axs[1].axvline(0, color='k', linewidth=2)
        axs[1].set_xlim([-3, 3])
        axs[1].set_ylim([-1.2, 2])
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('z-scored fluorescence')



def peaks_correlations_multi_mice(mice_dates, ipsi_or_contra='contra'):
    fig, axs = plt.subplots(1, ncols=1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    fig.suptitle('Exit centre poke', fontsize=16)
    all_mice_peaks = []
    all_mice_days = []
    for mouse_dates in mice_dates:
        mouse = mouse_dates.mouse
        dates = mouse_dates.dates
        reformatted_dates = []
        for date in dates:
            year = int(date[0:4])
            month = int(date[4:6])
            day = int(date[6:])
            reformatted_dates.append(datetime.date(year, month, day))

        reform_dates = np.array(reformatted_dates, dtype='datetime64')
        days_since_last_recording = np.concatenate(
            [np.array([0], dtype='timedelta64[D]'), np.diff(reform_dates)]).astype(int)
        day_of_training = np.cumsum(days_since_last_recording)

        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
        days_of_recording = []
        all_peaks = []
        for date_num, date in enumerate(dates):
            mean_and_sem_filename = saving_folder + mouse + '_' + date + '_' + 'mean_correct_data.p'
            correct_data = pickle.load(open(mean_and_sem_filename, "rb"))
            num_samples = correct_data.contra_mean_y_vals.shape[0]
            if ipsi_or_contra == 'contra':
                peaks = peakutils.indexes(correct_data.contra_mean_y_vals, thres=0.3)
            elif ipsi_or_contra == 'ipsi':
                peaks = peakutils.indexes(correct_data.ipsi_mean_y_vals, thres=0.3)

            first_peak_ind = peaks[np.where(peaks > num_samples / 2)]
            first_peak_ind = first_peak_ind[np.where(first_peak_ind < int(0.66 * num_samples))][0]
            first_peak_val = correct_data.contra_mean_y_vals[first_peak_ind]
            all_peaks.append(first_peak_val)
            days_of_recording.append(day_of_training[date_num])
        normalised_peaks = all_peaks / all_peaks[0] * 100
        all_mice_peaks.append(normalised_peaks)
        all_mice_days.append(days_of_recording)
    all_mice_days_flat = [item for sublist in all_mice_days for item in sublist]
    all_mice_peaks_flat = [item for sublist in all_mice_peaks for item in sublist]
    X = np.array(all_mice_days_flat).reshape(-1, 1)
    Y = np.array(all_mice_peaks_flat).reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    r_sq = linear_regressor.score(X, Y)
    plt.plot(X, Y_pred, lw=1, color='#746D69')
    axs.scatter(X, Y, color='#3F888F', alpha=0.8)
    print(r_sq)

    axs.title.set_text('Contralateral choice peak activity')
    axs.set_xlabel('Days since start of training')
    axs.set_ylabel('Percentage change since first day of recording')


def percentage_correct_correlations_multi_mice(mice_dates, ipsi_or_contra='contra'):
    fig, axs = plt.subplots(1, ncols=1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    fig.suptitle('Exit centre poke', fontsize=16)
    all_mice_peaks = []
    all_mice_pc = []
    for mouse_dates in mice_dates:
        mouse = mouse_dates.mouse
        dates = mouse_dates.dates
        reformatted_dates = []
        for date in dates:
            year = int(date[0:4])
            month = int(date[4:6])
            day = int(date[6:])
            reformatted_dates.append(datetime.date(year, month, day))

        reform_dates = np.array(reformatted_dates, dtype='datetime64')
        days_since_last_recording = np.concatenate(
            [np.array([0], dtype='timedelta64[D]'), np.diff(reform_dates)]).astype(int)
        day_of_training = np.cumsum(days_since_last_recording)
        percentage_correct = []
        BpodProtocol = '/Two_Alternative_Choice/'
        GeneralDirectory = 'W:/photometry_2AC/bpod_data/'
        DFfile = GeneralDirectory + mouse + BpodProtocol + 'Data_Analysis/' + mouse + '_dataframe.pkl'
        behavioural_stats = pd.read_pickle(DFfile)
        reformatted_dates = make_dates_pretty(dates)

        for date_num, date in enumerate(reformatted_dates):
            points_for_day = behavioural_stats[behavioural_stats['SessionTime'].str.contains(date)]
            percentage_correct.append(100 * np.sum(points_for_day['FirstPokeCorrect']) / len(points_for_day))

        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
        days_of_recording = []
        all_peaks = []
        for date_num, date in enumerate(dates):

            mean_and_sem_filename = saving_folder + mouse + '_' + date + '_' + 'mean_correct_data.p'
            correct_data = pickle.load(open(mean_and_sem_filename, "rb"))
            num_samples = correct_data.contra_mean_y_vals.shape[0]
            if ipsi_or_contra == 'contra':
                peaks = peakutils.indexes(correct_data.contra_mean_y_vals, thres=0.3)
            elif ipsi_or_contra == 'ipsi':
                peaks = peakutils.indexes(correct_data.ipsi_mean_y_vals, thres=0.3)

            first_peak_ind = peaks[np.where(peaks > num_samples / 2)]
            first_peak_ind = first_peak_ind[np.where(first_peak_ind < int(0.66 * num_samples))][0]
            first_peak_val = correct_data.contra_mean_y_vals[first_peak_ind]
            all_peaks.append(first_peak_val)
            days_of_recording.append(day_of_training[date_num])
        normalised_peaks = all_peaks / all_peaks[0] * 100
        all_mice_peaks.append(normalised_peaks)
        all_mice_pc.append(percentage_correct)

    all_mice_pc_flat = [item for sublist in all_mice_pc for item in sublist]
    all_mice_peaks_flat = [item for sublist in all_mice_peaks for item in sublist]
    X = np.array(all_mice_pc_flat).reshape(-1, 1)
    Y = np.array(all_mice_peaks_flat).reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    r_sq = linear_regressor.score(X, Y)
    plt.plot(X, Y_pred, lw=1, color='#746D69')
    axs.scatter(X, Y, color='#3F888F', alpha=0.8)
    print(r_sq)

    axs.title.set_text('Contralateral choice peak activity')
    axs.set_xlabel('Percentage correct')
    axs.set_ylabel('Percentage change since first day of recording')


def plot_average_mouse(mice_dates, ipsi_or_contra='contra'):
    contra_mean_traces = []
    ipsi_mean_traces = []

    for mouse_dates in mice_dates:
        mouse = mouse_dates.mouse
        date = mouse_dates.dates

        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
        mean_and_sem_filename = saving_folder + mouse + '_' + date + '_' + 'mean_correct_data.p'
        correct_data = pickle.load(open(mean_and_sem_filename, "rb"))
        num_samples = correct_data.contra_mean_y_vals.shape[0]
        x_vals = correct_data.contra_mean_x_vals
        contra_mean_traces.append(correct_data.contra_mean_y_vals)
        ipsi_mean_traces.append(correct_data.ipsi_mean_y_vals)

    fig, axs = plt.subplots(1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    axs.set_ylim([-0.6, 1.8])
    axs.set_xlim([-6, 6])
    axs.set_xlabel('Time from leaving centre poke (s)')
    axs.set_ylabel('z-scored fluorescence')
    axs.axvline(0, color='k', linewidth=2)

    if ipsi_or_contra == 'ipsi':
        axs.title.set_text('ipsilateral choice mean accross mice')
        axs.plot(x_vals, np.mean(np.array(ipsi_mean_traces), axis=0), color='#3F888F', lw=3)
    elif ipsi_or_contra == 'contra':
        axs.title.set_text('contralateral choice mean accross mice')
        axs.plot(x_vals, np.mean(np.array(contra_mean_traces), axis=0), color='#3F888F', lw=3)

    plt.show()