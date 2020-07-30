import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import numpy as np
import pickle
from utils.individual_trial_analysis_utils import SessionData
import pandas as pd

def get_all_experimental_records():
    experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record


def add_experiment_to_aligned_data(experiments_to_add):
    for index, experiment in experiments_to_add.iterrows():
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + experiment['mouse_id'] + '\\'
        restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
        session_traces.get_choice_responses()
        session_traces.get_cue_responses()
        session_traces.get_reward_responses()
        aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'aligned_traces.p'
        save_filename = saving_folder + aligned_filename
        pickle.dump(session_traces, open(save_filename, "wb"))


if __name__ == '__main__':
    mouse_id = 'SNL_photo19'
    date = 'all'
    all_experiments = get_all_experimental_records()

    if mouse_id =='all' & date == 'all':
        experiments_to_process = all_experiments
    elif mouse_id == 'all' & date!= 'all':
        experiments_to_process = all_experiments[all_experiments['date'] == date]
    elif mouse_id != 'all' & date == 'all':
        experiments_to_process = all_experiments[all_experiments['mouse_id'] == mouse_id]
    elif mouse_id != 'all' & date != 'all':
        experiments_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]


