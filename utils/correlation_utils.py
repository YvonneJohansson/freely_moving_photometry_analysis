import pickle
import datetime
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import math
import pandas as pd
import scipy as scipy
from scipy import optimize
from utils.mean_trace_utils import mouseDates
from utils.reaction_time_utils import get_valid_trials

def plot_all_valid_trials_over_time(session_starts, valid_peaks, valid_trial_nums):
    num_sessions = len(session_starts)
    colours = cm.viridis(np.linspace(0, 0.8, num_sessions))

    fig, ax = plt.subplots(1, ncols=1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)

    all_mean_peaks = []
    for session_num, session_start in enumerate(session_starts):
        if session_start != session_starts[-1]:
            session_inds = np.where(np.logical_and(np.greater_equal(valid_trial_nums, session_start),
                                                   np.less(valid_trial_nums, session_starts[session_num + 1])))
        else:
            session_inds = np.where(valid_trial_nums > session_start)
        session_trials = valid_trial_nums[session_inds]
        session_peaks = valid_peaks[session_inds]
        session_mean_peak = np.nanmedian(session_peaks)
        all_mean_peaks.append(session_mean_peak)
        ax.scatter(session_trials, session_peaks, color=colours[session_num], s=2)
        ax.axvline(session_start, color=colours[session_num])
        ax.set_xlabel('trial number')
        ax.set_ylabel('z-scored peak')
    plt.show()

def plot_binned_valid_trials(valid_peaks, valid_trial_nums, window_size=50, fit_line='exponential decay', plotting=True):
    def exponential(x, a, k, b):
        return (a * np.exp(x * k)) + b
    def logarithm(x, a, k, b):
        return (a * np.log(x * k)) + b
    def linear(x, m, c):
        return ( (m * x) + c)

    if fit_line == 'exponential decay':
        fit_equation = exponential
        starting_params = [1.8, -0.0003, 0.5]
        legend = "y= %0.5f$e^{%0.5fx}$ + %0.5f"
    elif fit_line == 'logarithmic growth':
        fit_equation = logarithm
        starting_params = [1.8, 0.0003, 0.5]
        legend = "y= %0.5f$log(%0.5fx)$ + %0.5f"
    elif fit_line == 'linear decay':
        fit_equation = linear
        starting_params = [-0.0003, 0.5]
        legend = "y= %0.5f$x$ + %0.5f"
    else:
        fit_equation = None

    rolling_mean_peak = []
    x_vals = []
    for window_num in range(int(np.shape(valid_peaks)[0] / window_size)):
        rolling_mean_peak.append(np.nanmean(valid_peaks[window_num * window_size: (window_num + 1) * window_size]))
        x_vals.append(np.mean(valid_trial_nums[window_num * window_size: (window_num + 1) * window_size]))
    norm_rolling_mean_peak = np.array(rolling_mean_peak)/ 1.0 # np.max(rolling_mean_peak)

    if fit_line != 'None':
        popt_exponential, pcov_exponential = scipy.optimize.curve_fit(fit_equation, np.array(x_vals), np.array(norm_rolling_mean_peak), p0=starting_params)
        perr_exponential = np.sqrt(np.diag(pcov_exponential))

        print("pre-exponential factor = %0.2f (+/-) %0.2f" % (popt_exponential[0], perr_exponential[0]))
        print("rate constant = %0.5f (+/-) %0.5f" % (popt_exponential[1], perr_exponential[1]))

        x_vals_fit = np.linspace(np.min(x_vals), np.max(x_vals), 1000)
        if fit_line == 'linear decay':
            y_vals_fit = fit_equation(x_vals_fit, popt_exponential[0], popt_exponential[1])
            residuals = norm_rolling_mean_peak - fit_equation(np.array(x_vals), popt_exponential[0], popt_exponential[1])
        else:
            residuals = norm_rolling_mean_peak - fit_equation(np.array(x_vals), popt_exponential[0], popt_exponential[1], popt_exponential[2])

            y_vals_fit = fit_equation(x_vals_fit, popt_exponential[0], popt_exponential[1], popt_exponential[2])

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((norm_rolling_mean_peak - np.mean(norm_rolling_mean_peak)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print('r-squared value: ', r_squared)
    else:
        x_vals_fit = None
        y_vals_fit = None

    if plotting:
        fig, ax = plt.subplots(1, ncols=1, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        ax.scatter(x_vals, norm_rolling_mean_peak)
        if fit_line != 'None':
            ax.plot( x_vals_fit, y_vals_fit, color='grey')
        ax.set_xlabel('bin (trials binned in groups of ' + str(window_size)+')')
        ax.set_xlim([0,13000])
        ax.set_ylabel('z-scored peak normalised to max size')

    plt.show()
    return(x_vals, norm_rolling_mean_peak, x_vals_fit, y_vals_fit)

def multi_animal_scatter_and_fit(mice_dates, window_size=30, fit_type='exponential decay'):
    fig, ax = plt.subplots(1, ncols=1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    num_types = len(mice_dates)
    colours = cm.viridis(np.linspace(0, 0.8, num_types))
    for mouse_num, mouse_dates in enumerate(mice_dates):
        mouse = mouse_dates.mouse
        dates = mouse_dates.dates
        session_starts, valid_trials, valid_reaction_times, valid_peaks, valid_trial_nums = get_valid_trials(mouse,
                                                                                                             dates,
                                                                                                             window_around_mean=2)
        x_vals, norm_rolling_mean_peak, x_vals_fit, y_vals_fit = plot_binned_valid_trials(valid_peaks, valid_trial_nums, window_size=window_size, fit_line=fit_type, plotting=False)
        ax.scatter(x_vals, norm_rolling_mean_peak, color=colours[mouse_num], label=mouse)
        if fit_type != 'None':
            ax.plot(x_vals_fit, y_vals_fit, color=colours[mouse_num])
    ax.set_xlabel('bin (trials binned in groups of ' + str(30)+')')
    ax.set_xlim([0,13000])
    ax.set_ylabel('z-scored peak')
    ax.legend(loc='best')
    plt.show()
