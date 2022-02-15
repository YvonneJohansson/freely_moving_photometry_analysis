import sys
sys.path.append('/post_processing(matt)/utils')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lineplot_and_heatmap_utils import *

generalDir = '/mnt/winstor/swc/sjones/users/francesca/photometry_2AC/processed_data/'

animalIDs = ['SNL_photo71'] #'SNL_photo68', 'SNL_photo69', 'SNL_photo70', 'SNL_photo71', 'SNL_photo72', 'SNL_photo73', 'SNL_photo74'
fibre_sides = ['right'] * len(animalIDs)
recording_site = 'tail'
dates = ['20220210', '20220211', '20220214']

#outputDir = '/home/matthew/Documents/figures/SNL_photo/' + 'mean_' + 'SNL_photo' + '_'.join([animalID[9:] for animalID in animalIDs])
#if not os.path.isdir(outputDir):
#    os.mkdir(outputDir)

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

xlims_conv = [(xlim+8)*100 for xlim in xlims]
plot_len = abs(xlims_conv[0]-xlims_conv[1])

all_animals = []

for animalID, fibre_side in zip(animalIDs, fibre_sides):

    inputDir = generalDir + animalID
    dates_dict = {}

    for date in dates:
        session_bpod_data = pd.read_pickle(inputDir + '/' + animalID + '_' + date + '_restructured_data.pkl')
        photometry_trace = np.load(inputDir + '/' + animalID + '_' + date + '_smoothed_signal.npy')

        dates_dict[date] = photometry_data_psychometric(fibre_side, session_bpod_data, params, photometry_trace)

    all_animals.append(dates_dict)

animals_dict = {animalID: dict for animalID, dict in zip(animalIDs, all_animals)}



combined_sessions = {}

for animalID in animalIDs:

    all_trials_98 = []
    all_trials_82 = []
    all_trials_66 = []
    all_trials_50 = []
    all_trials_34 = []
    all_trials_18 = []
    all_trials_2 = []

    for date in dates:
        all_trials_98.append(animals_dict[animalID][date].trials_98.z_scored_traces)
        all_trials_82.append(animals_dict[animalID][date].trials_82.z_scored_traces)
        all_trials_66.append(animals_dict[animalID][date].trials_66.z_scored_traces)
        all_trials_50.append(animals_dict[animalID][date].trials_50.z_scored_traces)
        all_trials_34.append(animals_dict[animalID][date].trials_34.z_scored_traces)
        all_trials_18.append(animals_dict[animalID][date].trials_18.z_scored_traces)
        all_trials_2.append(animals_dict[animalID][date].trials_2.z_scored_traces)

    all_trials = {'trials_98': all_trials_98,
                 'trials_82': all_trials_82,
                 'trials_66': all_trials_66,
                 'trials_50': all_trials_50,
                 'trials_34': all_trials_34,
                 'trials_18': all_trials_18,
                 'trials_2': all_trials_2}

    for k, v in all_trials.items():
        all_trials[k] = np.vstack(v)
        print(all_trials[k].shape[0])
        all_trials[k] = all_trials[k][:, ::100]
        all_trials[k] = all_trials[k][:, xlims_conv[0]:xlims_conv[1]]

    combined_sessions[animalID] = all_trials

if params['state_type_of_interest'] == 3:
    alignment= 'cue onset'
elif params['state_type_of_interest'] == 5:
    if params['align_to'] == 'Time start':
        alignment= 'movement to side port'
    elif params['align_to'] == 'Time end':
        alignment= 'reward delivery'

#fig, axs = plt.subplots(nrows=len(animalIDs), ncols=1, figsize=(5, 7.5))

for animalID in animalIDs:
    x_ticks = np.linspace(0, plot_len, abs(xlims[0] - xlims[1]) + 1)
    x_ticks_labels = np.linspace(xlims[0], xlims[1], abs(xlims[0] - xlims[1]) + 1)
    color_weights = np.logspace(0.2, 1, num=7)[::-1]/10
    x_zero = abs(xlims[0]) * 100

    for n, (k, v) in enumerate(combined_sessions[animalID].items()):
        label = k.split('_')[1] + '%'
        plt.plot(v.mean(axis=0), label=label, color='blue', alpha= color_weights[n])

    plt.axvline(x_zero, ls='--', color='black', alpha=0.5)
    plt.xticks(x_ticks, labels=x_ticks_labels)
    plt.xlabel('Time from ' + alignment + ' (s)')
    plt.ylabel('z-score')
    plt.legend()

plt.show()