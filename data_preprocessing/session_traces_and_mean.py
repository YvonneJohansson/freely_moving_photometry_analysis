import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import numpy as np
import pickle
from utils.individual_trial_analysis_utils import SessionData
import pandas as pd
import os

def get_all_experimental_records():
    experiment_record = pd.read_csv('/mnt/winstor/swc/sjones/users/Matt/photometry_2AC/experimental_record _matt.csv')
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record


def add_experiment_to_aligned_data(experiments_to_add):
    data_root = r'W:\photometry_2AC\processed_data'
    for index, experiment in experiments_to_add.iterrows():
        saving_folder = os.path.join(data_root, experiment['mouse_id'])
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
        session_traces.get_choice_responses()
        session_traces.get_cue_responses()
        session_traces.get_reward_responses()
        aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'aligned_traces.p'
        save_filename = os.path.join(saving_folder, aligned_filename)
        pickle.dump(session_traces, open(save_filename, "wb"))

def remove_experiments(experiments, ones_to_remove):
    for mouse in ones_to_remove.keys():
        for date in ones_to_remove[mouse]:
            index_to_remove = experiments[(experiments['mouse_id'] == mouse) & (experiments['date'] == date)].index[0]
            experiments = experiments.drop(index=index_to_remove)
    return experiments


if __name__ == '__main__':
    mouse_ids = ['SNL_photo47']
    date = '20210713'
    for mouse_id in mouse_ids:
        all_experiments = get_all_experimental_records()

        if (mouse_id =='all') & (date == 'all'):
            experiments_to_process = all_experiments
        elif (mouse_id == 'all') & (date != 'all'):
            experiments_to_process = all_experiments[all_experiments['date'] == date]
        elif (mouse_id != 'all') & (date == 'all'):
            experiments_to_process = all_experiments[all_experiments['mouse_id'] == mouse_id]
        elif (mouse_id != 'all') & (date != 'all'):
            experiments_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
        add_experiment_to_aligned_data(experiments_to_process)


