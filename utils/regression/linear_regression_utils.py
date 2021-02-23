import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.ndimage.interpolation import shift
from scipy.signal import decimate
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import pickle


def rolling_zscore(x, window=10*10000):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z


def turn_timestamps_into_continuous(num_samples, *behavioural_events):
    continuous_parameters = []
    for event_type_timestamps in behavioural_events:
        continuous_time_version = np.zeros([num_samples])
        continuous_time_version[event_type_timestamps] = 1
        continuous_parameters.append(continuous_time_version)
    return continuous_parameters


def convert_behavioural_timestamps_into_samples(time_stamps, window_to_remove, sample_rate=10000, decimate_factor=100):
    adjusted_stamps = (time_stamps - window_to_remove)*sample_rate/decimate_factor
    adjusted_stamps = np.round(np.vstack(adjusted_stamps).astype(np.float)).astype(int)
    return adjusted_stamps


def make_design_matrix(parameters, window_min=-1*10000/100, window_max=1.5*10000/100):
    num_parameters = len(parameters)
    shifts = np.arange(window_min, window_max + 1)
    shift_window_size = shifts.shape[0]
    X = np.zeros([parameters[0].shape[0], shift_window_size*num_parameters])
    all_param_indices = []
    for shift_num, shift_val in  enumerate(shifts):
        for param_num, param in enumerate(parameters):
            param_indices = range(param_num*shift_window_size, param_num*shift_window_size + shift_window_size)
            all_param_indices.append(param_indices)
            shifted_param = shift(param, shift_val, cval=0)
            X[:, param_indices[shift_num]] = shifted_param
    return(all_param_indices, X)


def plot_kernels(parameter_names, regression_results, window_min=-1 * 10000 / 100, window_max=1.5 * 10000 / 100):
    fig, axs = plt.subplots(nrows=1, ncols=len(parameter_names), sharey=True, figsize=(15, 8))
    axs[0].set_ylabel('Regression coefficient')
    shifts = np.arange(window_min, window_max + 1) / 100
    shift_window_size = shifts.shape[0]
    for param_num, param_name in enumerate(parameter_names):
        param_kernel = regression_results.coef_[param_num * shift_window_size:(param_num + 1) * shift_window_size]
        axs[param_num].plot(shifts, param_kernel, label=param_name)
        axs[param_num].set_title(param_name)
        axs[param_num].set_xlabel('Time (s)')


def save_kernels(save_filename, parameter_names, regression_results, downsampled_dff, X, window_min=-1 * 10000 / 100, window_max=1.5 * 10000/ 100):
    shifts = np.arange(window_min, window_max + 1) / 100
    shift_window_size = shifts.shape[0]
    param_kernels = {}
    for param_num, param_name in enumerate(parameter_names):
        kernel_name = parameter_names[param_num]
        param_kernels[kernel_name] = regression_results.coef_[param_num * shift_window_size:(param_num + 1) * shift_window_size]
    session_kernels = {}
    session_kernels['kernels'] = param_kernels
    session_kernels['time_stamps'] = shifts
    session_kernels['intercept'] = regression_results.intercept_

    kernel_filename = save_filename + 'linear_regression_kernels_no_repeated_cues_both_cues.p'
    inputs_X_filename = save_filename + 'linear_regression_X.p'
    inputs_y_filename = save_filename + 'linear_regression_y.p'
    with open(kernel_filename, "wb") as f:
        pickle.dump(session_kernels, f)
    with open(inputs_X_filename, "wb") as f:
        pickle.dump(X, f)
    with open(inputs_y_filename, "wb") as f:
        pickle.dump(downsampled_dff, f)


def get_first_x_sessions(sorted_experiment_record, x=3):
    i = []
    for mouse in np.unique(sorted_experiment_record['mouse_id']):
        i.append(sorted_experiment_record[sorted_experiment_record['mouse_id'] == mouse][0:3].index)
    flattened_i = [val for sublist in i for val in sublist]
    return (sorted_experiment_record.loc[flattened_i])
