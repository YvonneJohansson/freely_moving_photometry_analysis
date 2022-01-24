sys.path.append( '/home/matthew/Documents/code/photometry_analysis_code/freely_moving_photometry_analysis/post_processing(matt)/utils' )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lineplot_and_heatmap_utils import *

generalDir = '/mnt/winstor/swc/sjones/users/francesca/photometry_2AC/processed_data/'

animalIDs = ['SNL_photo68', 'SNL_photo69', 'SNL_photo70', 'SNL_photo71', 'SNL_photo72', 'SNL_photo73', 'SNL_photo74']
fibre_sides = ['right'] * 7
dates = ['20220117']

example_animalID = 'SNL_photo72'

params = {'state_type_of_interest': 3, # 5.5 = first incorrect choice
    'outcome': 1, # correct or incorrect: 0 = incorrect, 1 = correct, 2 = both
    'last_outcome': 0,  # NOT USED CURRENTLY
    'no_repeats' : 0, # 0 = dont care, 1 = state only entered once,
    'last_response': 0, # trial before: 0 = dont care. 1 = left, 2 = right
    'align_to' : 'Time start', # time end or time start
    'instance': -1, # only for no repeats = 0, -1 = last instance, 1 = first instance
    'plot_range': [-6, 6],
    'first_choice_correct': 1, # useful for non-punished trials 0 = dont care, 1 = only correct trials, (-1 = incorrect trials)
    'cue': None}

'''Plot parameters'''
error_bars= 'sem'
xlims = [-2, 3]
cue_vline = 0



all_animals = []

for animalID, fibre_side in zip(animalIDs, fibre_sides):

    inputDir = generalDir + animalID
    dates_dict = {}

    for date in dates:
        session_bpod_data = pd.read_pickle(inputDir + '/' + animalID + '_' + date + '_restructured_data.pkl')
        photometry_trace = np.load(inputDir + '/' + animalID + '_' + date + '_smoothed_signal.npy')

        dates_dict[date] = photometry_data(fibre_side, session_bpod_data, params, photometry_trace)

    all_animals.append(dates_dict)

animals_dict = {animalID: dict for animalID, dict in zip(animalIDs, all_animals)}


contra_mean_traces = []
ipsi_mean_traces = []

for animalID in animalIDs:
    for date in dates:
        contra_mean_traces.append(animals_dict[animalID][date].contra_trials.mean_trace)
        ipsi_mean_traces.append(animals_dict[animalID][date].ipsi_trials.mean_trace)

'''contra_mean_trace = np.stack(contra_mean_traces, axis=0).mean(axis=0)
ipsi_mean_trace = np.stack(ipsi_mean_traces, axis=0).mean(axis=0)

mean_traces = [ipsi_mean_trace, contra_mean_trace]'''

contra_mean_traces = np.stack(contra_mean_traces, axis=0)
ipsi_mean_traces = np.stack(ipsi_mean_traces, axis=0)

mean_traces = [ipsi_mean_traces, contra_mean_traces]

'''***PLOT IS CREATED FROM FUNCTION BELOW***'''

make_plot_and_heatmap(animals_dict[example_animalID][dates[0]], *mean_traces, error_bar_method=error_bars, mean_across_mice=True, xlims=xlims, cue_vline=cue_vline)



#
#

#

#










