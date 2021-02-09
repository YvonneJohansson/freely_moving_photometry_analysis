import pickle
import os
from utils.individual_trial_analysis_utils import SessionData
from data_preprocessing.session_traces_and_mean import get_all_experimental_records

def add_experiment_to_aligned_data(experiments_to_add):
    data_root = r'W:\photometry_2AC\processed_data\for_figure'
    for index, experiment in experiments_to_add.iterrows():
        saving_folder = os.path.join(data_root, experiment['mouse_id'])
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
        session_traces.get_choice_responses()
        session_traces.get_cue_responses()
        session_traces.get_reward_responses()
        session_traces.get_outcome_responses()
        aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'aligned_traces_for_fig.p'
        save_filename = os.path.join(saving_folder, aligned_filename)
        pickle.dump(session_traces, open(save_filename, "wb"))


if __name__ == '__main__':
    mouse_ids = ['SNL_photo26']
    date = '20200812'
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

