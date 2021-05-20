import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

from utils.regression.linear_regression_utils import *
import gc
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
mouse_ids = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26']
site = 'tail'

experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
experiment_record['date'] = experiment_record['date'].astype(str)
good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
clean_experiments = remove_bad_recordings(good_experiments)
all_experiments_to_process = clean_experiments[(clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(drop=True)
experiments_to_process = get_first_x_sessions(all_experiments_to_process)
for index, experiment in experiments_to_process.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    print('proccessing' + mouse + date)
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    events_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\linear_regression\\'
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(saving_folder + dff_trace_filename)

    window_size_seconds = 10
    sample_rate = 10000
    decimate_factor = 100
    window_min = -0.5 * 10000 / 100
    window_max = 1.5 * 10000 / 100

    rolling_zscored_dff = rolling_zscore(pd.Series(dff), window=window_size_seconds * sample_rate)
    downsampled_zscored_dff = decimate(
        decimate(rolling_zscored_dff[window_size_seconds * sample_rate:], int(decimate_factor / 10)),
        int(decimate_factor / 10))

    num_samples = downsampled_zscored_dff.shape[0]
    aligned_filename = mouse + '_' + date + '_' + 'behavioural_events_clean_cues.p'
    save_filename = events_folder + aligned_filename
    example_session_data = pickle.load(open(save_filename, "rb"))

    ipsi_choices = convert_behavioural_timestamps_into_samples(example_session_data.choice_data.ipsi_data.event_times,
                                                               window_size_seconds)
    contra_choices = convert_behavioural_timestamps_into_samples(
        example_session_data.choice_data.contra_data.event_times, window_size_seconds)
    high_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.high_cue_data.event_times,
                                                            window_size_seconds)
    low_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.low_cue_data.event_times,
                                                           window_size_seconds)
    rewards = convert_behavioural_timestamps_into_samples(example_session_data.reward_data.reward_data.event_times,
                                                          window_size_seconds)
    no_rewards = convert_behavioural_timestamps_into_samples(
        example_session_data.reward_data.no_reward_data.event_times, window_size_seconds)

    parameters = turn_timestamps_into_continuous(num_samples, high_cues, low_cues, ipsi_choices, contra_choices,
                                                 rewards, no_rewards)
    all_cues = np.concatenate(
        [example_session_data.cue_data.high_cue_data.trial_nums, example_session_data.cue_data.low_cue_data.trial_nums])
    all_choices = np.concatenate([example_session_data.choice_data.contra_data.trial_nums,
                                  example_session_data.choice_data.ipsi_data.trial_nums])
    all_outcomes = np.concatenate([example_session_data.reward_data.reward_data.trial_nums,
                                   example_session_data.reward_data.no_reward_data.trial_nums])
    temp = np.intersect1d(all_cues, all_choices)
    intersect_trial_nums = np.intersect1d(temp, all_outcomes)

    _, high_cue_inds, _ = np.intersect1d(example_session_data.cue_data.high_cue_data.trial_nums, intersect_trial_nums,
                                         return_indices=True)
    _, low_cue_inds, _ = np.intersect1d(example_session_data.cue_data.low_cue_data.trial_nums, intersect_trial_nums,
                                        return_indices=True)
    _, contra_choice_inds, _ = np.intersect1d(example_session_data.choice_data.contra_data.trial_nums,
                                              intersect_trial_nums, return_indices=True)
    _, ipsi_choice_inds, _ = np.intersect1d(example_session_data.choice_data.ipsi_data.trial_nums, intersect_trial_nums,
                                            return_indices=True)
    _, reward_inds, _ = np.intersect1d(example_session_data.reward_data.reward_data.trial_nums, intersect_trial_nums,
                                       return_indices=True)
    _, no_reward_inds, _ = np.intersect1d(example_session_data.reward_data.no_reward_data.trial_nums,
                                          intersect_trial_nums, return_indices=True)
    high_cue_trial_starts = (example_session_data.cue_data.high_cue_data.trial_starts[high_cue_inds])
    low_cue_trial_starts = (example_session_data.cue_data.low_cue_data.trial_starts[low_cue_inds])
    contra_trial_starts = (example_session_data.choice_data.contra_data.trial_starts[contra_choice_inds])
    ipsi_trial_starts = (example_session_data.choice_data.ipsi_data.trial_starts[ipsi_choice_inds])
    reward_trial_starts = (example_session_data.reward_data.reward_data.trial_starts[reward_inds])
    no_reward_trial_starts = (example_session_data.reward_data.no_reward_data.trial_starts[no_reward_inds])

    high_cue_trial_ends = (example_session_data.cue_data.high_cue_data.sorted_next_poke[high_cue_inds])
    low_cue_trial_ends = (example_session_data.cue_data.low_cue_data.sorted_next_poke[low_cue_inds])
    contra_trial_ends = (example_session_data.choice_data.contra_data.sorted_next_poke[contra_choice_inds])
    ipsi_trial_ends = (example_session_data.choice_data.ipsi_data.sorted_next_poke[ipsi_choice_inds])
    reward_trial_ends = (example_session_data.reward_data.reward_data.sorted_next_poke[reward_inds])
    no_reward_trial_ends = (example_session_data.reward_data.no_reward_data.sorted_next_poke[no_reward_inds])

    all_trial_starts = np.unique(np.concatenate(
        [high_cue_trial_starts, low_cue_trial_starts, contra_trial_starts, ipsi_trial_starts, reward_trial_starts,
         no_reward_trial_starts]))
    all_trial_ends = np.unique(np.concatenate(
        [high_cue_trial_ends, low_cue_trial_ends, contra_trial_ends, ipsi_trial_ends, reward_trial_ends,
         no_reward_trial_ends]))

    all_trial_ends = all_trial_ends[np.where(all_trial_ends != 0)[0]]

    trial_starts_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_starts, window_size_seconds))
    trial_ends_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_ends, window_size_seconds))

    trial_durations = all_trial_ends - all_trial_starts
    trials_to_include = pd.DataFrame(
        {'trial starts': trial_starts_samps, 'trial ends': trial_ends_samps, 'durations': trial_durations})
    trials_to_remove = trials_to_include[
        trials_to_include['durations'] > np.mean(trial_durations) + 2 * np.std(trial_durations)].reset_index(drop=True)

    trials_to_use = trials_to_include[
        trials_to_include['durations'] < np.mean(trial_durations) + 2 * np.std(trial_durations)].reset_index(drop=True)

    trace_for_regression = np.array([])
    for ind, trial in trials_to_use.iterrows():
        trial_start = int(trial['trial starts'])
        trial_end = int(trial['trial ends'])
        trace_for_regression = np.append(trace_for_regression, downsampled_zscored_dff[trial_start:trial_end])

    params_for_regression = []
    for param in parameters:
        param_new = np.array([])
        for ind, trial in trials_to_use.iterrows():
            trial_start = int(trial['trial starts'])
            trial_end = int(trial['trial ends'])
            param_new = np.append(param_new, param[trial_start:trial_end])
        params_for_regression.append(param_new)


    param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards']
    shifts, windows = make_shifts_for_params(param_names)
    param_inds, X = make_design_matrix_different_shifts(params_for_regression, shifts, windows)
    results = LinearRegression().fit(X, trace_for_regression)
    print(results.score(X, trace_for_regression))

    save_filename = mouse + '_' + date + '_only_trials_'
    save_kernels_different_shifts(saving_folder + save_filename, param_names, results, trace_for_regression, X.astype(int), shifts, windows,)
    gc.collect()