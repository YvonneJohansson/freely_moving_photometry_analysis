
import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import numpy as np
import pandas as pd
import pickle
from utils.nacc_individual_trial_analysis_utils import rewardTracesZScored
# NAcc recordings
mouse = 'SNL_photo19'
dates = ['20200222', '20200225', '20200227', '20200302', '20200304', '20200306', '20200311', '20200317']
fiber_side = 'left'
for date in dates:
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(saving_folder + dff_trace_filename)

    behavioural_params = rewardTracesZScored( fiber_side, mouse, date)
    behavioural_params.get_peaks(dff, trial_data)
    mean_and_sem_filename = mouse + '_' + date + '_' + 'peaks_reward_data.p'
    save_filename = saving_folder + mean_and_sem_filename
    pickle.dump(behavioural_params, open(save_filename, "wb"))