import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')
from data_preprocessing.session_traces_and_mean import get_all_experimental_records
from utils.reaction_time_utils import plot_reaction_times, plot_reaction_times_overlayed, get_valid_trials
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.correlation_utils import plot_all_valid_trials_over_time, plot_binned_valid_trials, multi_animal_scatter_and_fit
from utils.change_over_time_utils import get_valid_traces
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
data_root = r'W:\photometry_2AC\processed_data\peak_analysis'

mice = ['SNL_photo17']
recording_site = 'tail'
window_for_binning = 40
for mouse_num, mouse in enumerate(mice):
    all_experiments = get_all_experimental_records()
    all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
    all_experiments = remove_bad_recordings(all_experiments)
    # all_experiments = remove_experiments(all_experiments, experiments_to_remove)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]
    dates = experiments_to_process['date'].values
    rolling_mean_x, rolling_mean_peaks, peak_trace_inds, rolling_mean_traces = get_valid_traces(mouse, dates, window_around_mean=0.2, recording_site=recording_site, side='contra', window_size=window_for_binning)
    saving_folder = os.path.join(data_root, mouse)
    filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks_new.npz'
    save_filename = os.path.join(saving_folder, filename)
    np.savez(save_filename, rolling_mean_x=rolling_mean_x, rolling_mean_peaks=rolling_mean_peaks, rolling_mean_traces=rolling_mean_traces, peak_trace_inds=peak_trace_inds)

# mice = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
# recording_site = 'Nacc'
# window_for_binning = 40
# colours = sns.color_palette("pastel")
# for mouse_num, mouse in enumerate(mice):
#     saving_folder = os.path.join(data_root, mouse)
#     filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks.npz'
#     save_filename = os.path.join(saving_folder, filename)
#     rolling_mean_data = np.load(save_filename)
#     rolling_mean_x = rolling_mean_data['rolling_mean_x']
#     rolling_mean_peaks = rolling_mean_data['rolling_mean_peaks']
#     rolling_mean_traces = rolling_mean_data['rolling_mean_trace']
#     peak_trace_inds = rolling_mean_data['peak_trace_inds']
#

