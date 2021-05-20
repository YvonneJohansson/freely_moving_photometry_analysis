import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import numpy as np
import pickle
from utils.individual_trial_analysis_utils import SessionEvents
import pandas as pd
import os
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.regression.linear_regression_utils import get_first_x_sessions

def get_all_experimental_records():
    experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record


def add_timestamps_to_aligned_data(experiments_to_add):
    for index, experiment in experiments_to_add.iterrows():
        data_folder = 'W:\\photometry_2AC\\processed_data\\' + experiment['mouse_id'] + '\\'
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + experiment['mouse_id'] + '\\linear_regression\\'
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
        #trial_data = pd.read_pickle(data_folder + restructured_data_filename)
        dff_trace_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'smoothed_signal.npy'
        #dff = np.load(data_folder + dff_trace_filename)

        session_events = SessionEvents(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
        session_events.get_choice_events()
        session_events.get_cue_events()
        session_events.get_reward_events()
        aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'behavioural_events_clean_cues.p' #'behavioural_events_no_repeated_cues.p'
        save_filename = saving_folder + aligned_filename
        pickle.dump(session_events, open(save_filename, "wb"))


def remove_manipulation_days(experiments):
    exemption_list = ['psychometric', 'state change medium cloud', 'value blocks', 'state change white noise', 'omissions and large rewards']
    exemptions = '|'.join(exemption_list)
    index_to_remove = experiments[np.logical_xor(experiments['include'] == 'no', experiments['experiment_notes'].str.contains(exemptions, na=False))].index
    cleaned_experiments = experiments.drop(index=index_to_remove)
    return cleaned_experiments



if __name__ == '__main__':
    mouse_ids = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
    site = 'Nacc'
    experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
    experiment_record['date'] = experiment_record['date'].astype(str)
    good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    clean_experiments = remove_bad_recordings(good_experiments)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
        drop=True)
    experiments_to_process = get_first_x_sessions(all_experiments_to_process)
    add_timestamps_to_aligned_data(experiments_to_process)

    # for mouse_id in mouse_ids:
    #     date = '20201121'
    #     all_experiments = get_all_experimental_records()
    #     clean_experiments = remove_exps_after_manipulations(all_experiments, [mouse_id])
    #     site = 'Nacc'
    #     index_to_remove = clean_experiments[clean_experiments['recording_site'] != site].index
    #     cleaned_experiments = clean_experiments.drop(index=index_to_remove)
    #
    #
    #     if (mouse_id =='all') & (date == 'all'):
    #         experiments_to_process = cleaned_experiments
    #     elif (mouse_id == 'all') & (date != 'all'):
    #         experiments_to_process = cleaned_experiments[cleaned_experiments['date'] == date]
    #     elif (mouse_id != 'all') & (date == 'all'):
    #         experiments_to_process = cleaned_experiments[cleaned_experiments['mouse_id'] == mouse_id]
    #     elif (mouse_id != 'all') & (date != 'all'):
    #         experiments_to_process = cleaned_experiments[(cleaned_experiments['date'] == date) & (cleaned_experiments['mouse_id'] == mouse_id)]
    #     add_timestamps_to_aligned_data(experiments_to_process)

