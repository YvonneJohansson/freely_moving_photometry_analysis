import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

from utils.regression.linear_regression_utils import *
import gc
from utils.post_processing_utils import remove_exps_after_manipulations
mouse_ids = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26'] #['SNL_photo28', 'SNL_photo29', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
site = 'tail'

experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[(clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)]
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

    rolling_zscored_dff = rolling_zscore(pd.Series(dff), window=window_size_seconds * sample_rate)
    downsampled_zscored_dff = decimate(
        decimate(rolling_zscored_dff[window_size_seconds * sample_rate:], int(decimate_factor / 10)),
        int(decimate_factor / 10))

    num_samples = downsampled_zscored_dff.shape[0]
    aligned_filename = mouse + '_' + date + '_' + 'behavioural_events_with_no_rewards_added.py'
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

    all_trial_starts = np.unique(np.concatenate([example_session_data.cue_data.high_cue_data.trial_starts,
                                                 example_session_data.cue_data.low_cue_data.trial_starts,
                                                 example_session_data.choice_data.contra_data.trial_starts,
                                                 example_session_data.choice_data.ipsi_data.trial_starts,
                                                 example_session_data.reward_data.no_reward_data.trial_starts,
                                                 example_session_data.reward_data.reward_data.trial_starts]))
    all_trial_ends = np.unique(np.concatenate(
        [example_session_data.cue_data.high_cue_data.trial_ends, example_session_data.cue_data.low_cue_data.trial_ends,
         example_session_data.choice_data.contra_data.trial_ends, example_session_data.choice_data.ipsi_data.trial_ends,
         example_session_data.reward_data.no_reward_data.trial_ends,
         example_session_data.reward_data.reward_data.trial_ends]))

    trial_starts_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_starts, window_size_seconds))
    trial_ends_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_ends, window_size_seconds))

    trials_to_include = pd.DataFrame({'trial starts': trial_starts_samps, 'trial ends': trial_ends_samps})

    trace_for_regression = np.array([])
    for ind, trial in trials_to_include.iterrows():
        trial_start = trial['trial starts']
        trial_end = trial['trial ends']
        trace_for_regression = np.append(trace_for_regression, downsampled_zscored_dff[trial_start:trial_end])

    params_for_regression = []
    for param in parameters:
        param_new = np.array([])
        for ind, trial in trials_to_include.iterrows():
            trial_start = trial['trial starts']
            trial_end = trial['trial ends']
            param_new = np.append(param_new, param[trial_start:trial_end])
        params_for_regression.append(param_new)

    all_param_indices, X = make_design_matrix(params_for_regression)
    results = LinearRegression().fit(X, trace_for_regression)
    param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards']
    window_min = -1 * 10000 / 100
    window_max = 1.5 * 10000 / 100

    shifts = np.arange(window_min, window_max + 1) / 100

    save_filename = mouse + '_' + date + '_'
    save_kernels(saving_folder + save_filename, param_names, results, trace_for_regression, X.astype(bool))

    gc.collect()