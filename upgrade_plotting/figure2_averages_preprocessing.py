import pickle
import pandas as pd
import numpy as np
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.regression.linear_regression_utils import get_first_x_sessions
import matplotlib.pyplot as plt

def get_all_mice_data(experiments_to_process):
    exp_numbers = []
    mice = []
    for index, experiment in experiments_to_process.iterrows():
        mouse = experiment['mouse_id']
        date = experiment['date']
        saving_folder = 'W:\\photometry_2AC\\processed_data\\for_figure\\' + mouse + '\\'
        save_filename = mouse + '_' + date + '_' + 'aligned_traces_for_fig.p'

        sorted_exps = pd.to_datetime(
            experiments_to_process[experiments_to_process['mouse_id'] == mouse]['date']).sort_values(ignore_index=True)
        date_as_dt = pd.to_datetime(date)
        exp_number = sorted_exps[sorted_exps == date_as_dt].index[0]
        exp_numbers.append(exp_number)
        with open(saving_folder + save_filename, "rb") as f:
            session_data = pickle.load(f)
            print(mouse, date)
            if index == 0:
                ipsi_choice = session_data.choice_data.ipsi_data.mean_trace
                contra_choice = session_data.choice_data.contra_data.mean_trace
                reward = session_data.outcome_data.reward_data.mean_trace
                no_reward = session_data.outcome_data.no_reward_data.mean_trace
                time_stamps = session_data.choice_data.contra_data.time_points
            else:
                ipsi_choice = np.vstack([ipsi_choice, session_data.choice_data.ipsi_data.mean_trace])
                contra_choice = np.vstack([contra_choice, session_data.choice_data.contra_data.mean_trace])
                reward = np.vstack([reward, session_data.outcome_data.reward_data.mean_trace])
                no_reward = np.vstack([no_reward, session_data.outcome_data.no_reward_data.mean_trace])
    return ipsi_choice, contra_choice, reward, no_reward, time_stamps


mouse_ids = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
site = 'Nacc'

experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[
    (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
    drop=True)
experiments_to_process = get_first_x_sessions(all_experiments_to_process).reset_index(
    drop=True)
ipsi_choice, contra_choice, reward, no_reward, time_stamps = get_all_mice_data(experiments_to_process)
plt.plot(np.mean(contra_choice, axis=0))
plt.show()
dir = 'W:\\photometry_2AC\\processed_data\\for_figure\\'
file_name = 'group_data_' + site +'.npz'
np.savez(dir + file_name, ipsi_choice=ipsi_choice, contra_choice=contra_choice, reward=reward, no_reward=no_reward, time_stamps=time_stamps)

