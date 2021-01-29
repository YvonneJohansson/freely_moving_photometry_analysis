import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.ndimage.interpolation import shift
from scipy.signal import decimate
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import pickle
from utils.linear_regression_utils import *
import gc
from linear_regression.get_time_stamps_for_regression import remove_manipulation_days
mouse_ids = ['SNL_photo26']
site = 'tail'

experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_manipulation_days(experiment_record)
experiments_to_process = clean_experiments[(clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)]

for index, experiment in experiments_to_process.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    print('proccessing' + mouse + date)
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(saving_folder + dff_trace_filename)

    window_size_seconds = 10
    sample_rate = 10000
    decimate_factor = 100

    rolling_zscored_dff = rolling_zscore(pd.Series(dff), window=window_size_seconds * sample_rate)
    downsampled_zscored_dff = decimate(
        decimate(rolling_zscored_dff[window_size_seconds * sample_rate:], int(decimate_factor / 10)),
        int(decimate_factor / 10))

    num_samples = downsampled_zscored_dff.shape[0]
    aligned_filename = mouse + '_' + date + '_' + 'behavioural_events.p'
    save_filename = saving_folder + aligned_filename
    example_session_data = pickle.load(open(save_filename, "rb"))

    ipsi_choices = convert_behavioural_timestamps_into_samples(example_session_data.choice_data.ipsi_data.event_times,
                                                               window_size_seconds)
    contra_choices = convert_behavioural_timestamps_into_samples(
        example_session_data.choice_data.contra_data.event_times, window_size_seconds)
    ipsi_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.ipsi_data.event_times,
                                                            window_size_seconds)
    contra_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.contra_data.event_times,
                                                              window_size_seconds)
    ipsi_rewards = convert_behavioural_timestamps_into_samples(example_session_data.reward_data.ipsi_data.event_times,
                                                               window_size_seconds)
    contra_rewards = convert_behavioural_timestamps_into_samples(
        example_session_data.reward_data.contra_data.event_times, window_size_seconds)

    all_cues = np.concatenate([ipsi_cues, contra_cues])
    all_rewards = np.concatenate([ipsi_rewards, contra_rewards])

    parameters = turn_timestamps_into_continuous(num_samples, ipsi_choices, contra_choices, all_cues, all_rewards)
    all_param_indices, X = make_design_matrix(parameters)
    results = LinearRegression().fit(X, downsampled_zscored_dff)
    param_names = ['ipsi choices', 'contra choices', 'cues', 'rewards']
    window_min = -1 * 10000 / 100
    window_max = 2 * 10000 / 100

    shifts = np.arange(window_min, window_max + 1) / 100

    save_filename = mouse + '_' + date + '_' + 'linear_regresion_kernels.p'
    save_kernels(saving_folder + save_filename, param_names, results)

    gc.collect()