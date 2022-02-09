sys.path.append( '/home/matthew/Documents/code/photometry_analysis_code/freely_moving_photometry_analysis/post_processing(matt)/utils' )

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lineplot_and_heatmap_utils import *

generalDir = '/mnt/winstor/swc/sjones/users/francesca/photometry_2AC/processed_data/'

animalIDs = ['SNL_photo68', 'SNL_photo69', 'SNL_photo70', 'SNL_photo71', 'SNL_photo72', 'SNL_photo73', 'SNL_photo74']
fibre_sides = ['right'] * len(animalIDs)
recording_site = 'tail'
dates = ['20220113'] *len(animalIDs) #'20220121', '20220121', '20220121', '20220121', '20220124', '20220121', '20220121'

outputDir = '/home/matthew/Documents/figures/SNL_photo/' + 'mean_' + 'SNL_photo' + '_'.join([animalID[9:] for animalID in animalIDs])
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

params = {'state_type_of_interest': 5, # 5.5 = first incorrect choice
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

for animalID, fibre_side, date in zip(animalIDs, fibre_sides, dates):

    inputDir = generalDir + animalID
    dates_dict = {}
    session_bpod_data = pd.read_pickle(inputDir + '/' + animalID + '_' + date + '_restructured_data.pkl')
    photometry_trace = np.load(inputDir + '/' + animalID + '_' + date + '_smoothed_signal.npy')

    dates_dict[date] = photometry_data(fibre_side, session_bpod_data, params, photometry_trace)
    '''for date in dates:
        session_bpod_data = pd.read_pickle(inputDir + '/' + animalID + '_' + date + '_restructured_data.pkl')
        photometry_trace = np.load(inputDir + '/' + animalID + '_' + date + '_smoothed_signal.npy')

        dates_dict[date] = photometry_data(fibre_side, session_bpod_data, params, photometry_trace)
    '''
    all_animals.append(dates_dict)

animals_dict = {animalID: dict for animalID, dict in zip(animalIDs, all_animals)}


contra_mean_traces = []
ipsi_mean_traces = []

for animalID, date in zip(animalIDs, dates):
    contra_mean_traces.append(animals_dict[animalID][date].contra_trials.mean_trace)
    ipsi_mean_traces.append(animals_dict[animalID][date].ipsi_trials.mean_trace)

    '''for date in dates:
        contra_mean_traces.append(animals_dict[animalID][date].contra_trials.mean_trace)
        ipsi_mean_traces.append(animals_dict[animalID][date].ipsi_trials.mean_trace)'''

'''contra_mean_trace = np.stack(contra_mean_traces, axis=0).mean(axis=0)
ipsi_mean_trace = np.stack(ipsi_mean_traces, axis=0).mean(axis=0)

mean_traces = [ipsi_mean_trace, contra_mean_trace]'''

if params['state_type_of_interest'] == 3:
    alignment= 'cue onset'
elif params['state_type_of_interest'] == 5:
    if params['align_to'] == 'Time start':
        alignment= 'movement to side port'
    elif params['align_to'] == 'Time end':
        alignment= 'reward delivery'


contra_mean_traces = np.stack(contra_mean_traces, axis=0)
contra_mean_traces = decimate(contra_mean_traces, 10)
ipsi_mean_traces = np.stack(ipsi_mean_traces, axis=0)
ipsi_mean_traces = decimate(ipsi_mean_traces, 10)

xlims_conv = [(xlim+8)*1000 for xlim in xlims]
plot_len = abs(xlims_conv[0]-xlims_conv[1])

contra_mean_traces = contra_mean_traces[:, xlims_conv[0]:xlims_conv[1]]
ipsi_mean_traces = ipsi_mean_traces[:, xlims_conv[0]:xlims_conv[1]]

contra_ymax = contra_mean_traces.max()
contra_ymin = contra_mean_traces.min()
ipsi_ymax = ipsi_mean_traces.max()
ipsi_ymin = ipsi_mean_traces.min()
ylim_max = max(ipsi_ymax, contra_ymax)
ylim_min = min(ipsi_ymin, contra_ymin)
ylims = [ylim_min, ylim_max]

mean_traces = [contra_mean_traces, ipsi_mean_traces]

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 7.5))

for ax, relevant_mean_traces in zip(axs, mean_traces):

    sem = stats.sem(relevant_mean_traces, axis=0)
    lower_bound = relevant_mean_traces.mean(axis=0) - sem
    upper_bound = relevant_mean_traces.mean(axis=0) + sem
    x_ticks = np.linspace(0, plot_len, abs(xlims[0] - xlims[1]) + 1)
    x_ticks_labels = np.linspace(xlims[0], xlims[1], abs(xlims[0] - xlims[1]) + 1)
    x_zero = abs(xlims[0])*1000
    mean_trace = relevant_mean_traces.mean(axis=0)

    for trace in relevant_mean_traces:
        ax.plot(trace, color= 'grey', alpha=0.4, linewidth= 1.5)
    ax.plot(mean_trace, color= 'blue', alpha=1, linewidth=2)

    ax.fill_between(np.linspace(0,(plot_len-1),plot_len), lower_bound, upper_bound, alpha=0.3,
                                facecolor='blue', linewidth=0)

    ax.axvline(x_zero, ls='--', color='black', alpha=0.5)
    ax.axhline(0, ls='--', color='black', alpha=0.3)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_ylim(ylims)
    ax.set_xlabel('Time from ' + alignment + ' (s)')
    ax.set_ylabel('z-score')

fig.tight_layout()
plt.show()

plt.savefig(outputDir + '/' + 'mean_' + dates[0] + '_' + recording_site + '_' + alignment + '_' + '_'.join([animalID[9:] for animalID in animalIDs]))




# make_plot_and_heatmap(animals_dict[example_animalID][dates[0]], *mean_traces, error_bar_method=error_bars, mean_across_mice=True, xlims=xlims, cue_vline=cue_vline)