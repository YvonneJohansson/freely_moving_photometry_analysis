import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')
import peakutils
from matplotlib import colors, cm
from scipy.signal import decimate
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from utils.plotting import HeatMapParams
from utils.plotting import heat_map_and_mean
from utils.plotting import get_photometry_around_event
from scipy import stats
from utils.psychometric_utils import *
from utils.post_processing_utils import get_all_experimental_records


mouse_id = 'SNL_photo26'
site = 'tail'
all_experiments = get_all_experimental_records()
experiment_to_process = all_experiments[(all_experiments['recording_site'] == site) & (all_experiments['mouse_id'] == mouse_id) & (all_experiments['experiment_notes'] == 'psychometric')].reset_index(drop=True)


for ind, experiment in experiment_to_process.iterrows():
    session_data = open_experiment(experiment)
    if ind == 0:
        params = {'state_type_of_interest': 5,
                  'outcome': 2,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 1,
                  'last_response': 0,
                  'align_to': 'Time start',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': 0,
                  'cue': 'None'}
        aligned_data = CustomAlignedDataPsychometric(session_data, params)
    else:
        aligned_data.contra_data.add_trials(session_data,)
