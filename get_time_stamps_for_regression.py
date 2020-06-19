import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import numpy as np
import pickle
from utils.individual_trial_analysis_utils import SessionEvents
import pandas as pd


experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
experiment_record['date'] = experiment_record['date'].astype(str)

for index, experiment in experiment_record.iterrows():
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + experiment['mouse_id'] + '\\'
    restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
    dff_trace_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'smoothed_signal.npy'
    dff = np.load(saving_folder + dff_trace_filename)

    session_events = SessionEvents(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
    session_events.get_choice_events()
    session_events.get_cue_events()
    session_events.get_reward_events()
    aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'behavioural_events.p'
    save_filename = saving_folder + aligned_filename
    pickle.dump(session_events, open(save_filename, "wb"))