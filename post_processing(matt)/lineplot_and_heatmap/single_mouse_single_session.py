import sys

sys.path.append( '/home/matthew/Documents/code/photometry_analysis_code/freely_moving_photometry_analysis/post_processing(matt)/utils' )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from lineplot_and_heatmap_utils import *

# Do you want to save the figures created?
save_figs = True
reference_csv = pd.read_csv('/mnt/winstor/swc/sjones/users/francesca/photometry_2AC/experimental_record_matt.csv')
generalDir = '/mnt/winstor/swc/sjones/users/francesca/photometry_2AC/processed_data/'
GenOutputDir = '/home/matthew/Documents/figures/SNL_photo/'
# list mice you want to create plots for
animalIDs = ['SNL_photo71']# 'SNL_photo68', 'SNL_photo69', 'SNL_photo70', 'SNL_photo71', 'SNL_photo72', 'SNL_photo73', 'SNL_photo74', 'SNL_photo77'
# list dates you want to create plots for
# NOTE: plots will be created individually for each date, NOT multiple dates combined into a plot
dates = ['20220208']

params = {'state_type_of_interest': 3, # 5.5 = first incorrect choice
    'outcome': 1, # correct or incorrect: 0 = incorrect, 1 = correct, 2 = both
    'last_outcome': 0,  # NOT USED CURRENTLY
    'no_repeats' : 0, # 0 = dont care, 1 = state only entered once,
    'last_response': 0, # trial before: 0 = dont care. 1 = left, 2 = right
    'align_to' : 'Time start', # time end or time start
    'instance': -1, # only for no repeats = 0, -1 = last instance, 1 = first instance
    'plot_range': [-6, 6],
    'first_choice_correct': 0, # useful for non-punished trials 0 = dont care, 1 = only correct trials, (-1 = incorrect trials)
    'cue': None}
'''Plot parameters'''
error_bars= 'sem'
xlims = [-2, 3]
cue_vline = 0

all_animals = []

for animalID in animalIDs:

    inputDir = generalDir + animalID
    dates_dict = {}

    for date in dates:
        session_bpod_data = pd.read_pickle(inputDir + '/' + animalID + '_' + date + '_restructured_data.pkl')
        photometry_trace = np.load(inputDir + '/' + animalID + '_' + date + '_smoothed_signal.npy')
        fibre_side = reference_csv[(reference_csv['mouse_id'] == animalID) &
              (reference_csv['date'] == date)]['fiber_side'].values[0]

        dates_dict[date] = photometry_data(fibre_side, session_bpod_data, params, photometry_trace)

    all_animals.append(dates_dict)

animals_dict = {animalID: dict for animalID, dict in zip(animalIDs, all_animals)}


for animalID in animalIDs:
    for date in dates:

        recording_site = reference_csv[(reference_csv['mouse_id'] == animalID) &
              (reference_csv['date'] == date)]['recording_site'].values[0]

        # Make plot
        make_plot_and_heatmap(animals_dict[animalID][date], error_bar_method=error_bars,
                              mean_across_mice=False, xlims=xlims, cue_vline=cue_vline)
        plt.suptitle(animalID + '_' + date)

        # Save fig
        if save_figs == True:

            outputDir = GenOutputDir + animalID
            if not os.path.isdir(outputDir):
                os.mkdir(outputDir)

            if params['state_type_of_interest'] == 3:
                plt.savefig(
                    outputDir + '/' + animalID + '_' + date + '_' + recording_site + '_' + 'aligned_cue' + '.png')
            elif params['state_type_of_interest'] == 5:
                if params['align_to'] == 'Time end':
                    plt.savefig(
                        outputDir + '/' + animalID + '_' + date + '_' + recording_site + '_' + 'aligned_reward' + '.png')
                elif params['align_to'] == 'Time start':
                    plt.savefig(
                        outputDir + '/' + animalID + '_' + date + '_' + recording_site + '_' + 'aligned_movement' + '.png')



