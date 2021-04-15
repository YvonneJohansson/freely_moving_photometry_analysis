import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')
from data_preprocessing.session_traces_and_mean import get_all_experimental_records
from utils.reaction_time_utils import plot_reaction_times, plot_reaction_times_overlayed, get_valid_trials
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.correlation_utils import plot_all_valid_trials_over_time, plot_binned_valid_trials, multi_animal_scatter_and_fit

# mouse = 'SNL_photo30'
# recording_site = 'Nacc'
# all_experiments = get_all_experimental_records()
# all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
# all_experiments = remove_bad_recordings(all_experiments)
# experiments_to_process = all_experiments[(all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]
# dates = experiments_to_process['date'].values
# session_starts, valid_trials, valid_reaction_times, valid_peaks, valid_trial_nums = get_valid_trials(mouse, dates, window_around_mean=2, recording_site=recording_site, side='contra')
#
# plot_all_valid_trials_over_time(session_starts, valid_peaks, valid_trial_nums)
# a = plot_binned_valid_trials(valid_peaks, valid_trial_nums, window_size=50, fit_line=None)
mice = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
multi_animal_scatter_and_fit(mice, recording_site='Nacc', window_size=100, fit_type=None)