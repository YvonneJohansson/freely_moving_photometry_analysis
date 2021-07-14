import os
import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
import numpy as np
import pandas as pd
from utils.post_processing_utils import open_experiment, CustomAlignedData, get_all_experimental_records
from utils.large_reward_omission_utils import get_traces_and_reward_types

exp_name = 'large_rewards_omissions'
processed_data_dir = os.path.join('W:\photometry_2AC\processed_data', 'large_rewards_omissions_data')
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

all_experiments = get_all_experimental_records()
mice = ['SNL_photo37', 'SNL_photo43', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26']#['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo32', 'SNL_photo34', 'SNL_photo35']

block_data_file = os.path.join(processed_data_dir, 'all_tail_reward_change_data_new.csv')

if os.path.isfile(block_data_file):
    all_reward_block_data = pd.read_pickle(block_data_file)
else:
    for mouse_num, mouse_id in enumerate(mice):
        sessions = all_experiments[
            (all_experiments['mouse_id'] == mouse_id) & (all_experiments['experiment_notes'] == 'omissions and large rewards')][
            'date'].values
        for session_idx, date in enumerate(sessions):
            experiment_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
            session_data, behavioural_data = open_experiment(experiment_to_process)
            params = {'state_type_of_interest': 5,
                      'outcome': 2,
                      'last_outcome': 0,  # NOT USED CURRENTLY
                      'no_repeats': 0,
                      'last_response': 0,
                      'align_to': 'Time end',
                      'instance': -1,
                      'plot_range': [-6, 6],
                      'first_choice_correct': 0,
                      'cue': 'None'}
            data = CustomAlignedData(session_data, params, peak_quantification=False)
            session_reward_type_data = get_traces_and_reward_types(data, behavioural_data)
            session_reward_type_data['mouse'] = mouse_id
            session_reward_type_data['session'] = date
            if (session_idx > 0) or (mouse_num > 0):
                all_reward_type_data = pd.concat([all_reward_type_data, session_reward_type_data], ignore_index=True)
            else:
                all_reward_type_data = session_reward_type_data

    all_reward_type_data.to_pickle(block_data_file)

