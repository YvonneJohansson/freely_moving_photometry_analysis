import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import numpy as np
import pandas as pd
import pickle
from utils.individual_trial_analysis_utils import SessionData
mouse = 'SNL_photo20'
dates = ['20200225', '20200302', '20200304', '20200306', '20200311', '20200313', '20200317']
fiber_side = 'left'
for date in dates:
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(saving_folder + dff_trace_filename)

    session_traces = SessionData(fiber_side, mouse, date)
    session_traces.get_choice_responses()
    choice_aligned_filename = mouse + '_' + date + '_' + 'choice_aligned_traces.p'
    save_filename = saving_folder + choice_aligned_filename
    pickle.dump(session_traces, open(save_filename, "wb"))