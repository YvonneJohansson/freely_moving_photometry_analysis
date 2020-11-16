import numpy as np
import pandas as pd
from scipy.signal import decimate
import os
from utils.plotting import calculate_error_bars
from matplotlib import cm


def find_contra_and_ipsi_traces_for_criterion(pre_trials, post_trials, contra_data, ipsi_data):
    _, pre_contra_trials, _ = np.intersect1d(contra_data.trial_nums, pre_trials, return_indices=True)
    _, post_contra_trials, _ = np.intersect1d(contra_data.trial_nums, post_trials, return_indices=True)
    _, pre_ipsi_trials, _ = np.intersect1d(ipsi_data.trial_nums, pre_trials, return_indices=True)
    _, post_ipsi_trials, _ = np.intersect1d(ipsi_data.trial_nums, post_trials, return_indices=True)
    pre_contra_traces = contra_data.sorted_traces[pre_contra_trials, :]
    post_contra_traces = contra_data.sorted_traces[post_contra_trials, :]
    pre_ipsi_traces = contra_data.sorted_traces[pre_ipsi_trials, :]
    post_ipsi_traces = contra_data.sorted_traces[post_ipsi_trials, :]
    pre_both_traces = np.concatenate([pre_contra_traces, pre_ipsi_traces])
    post_both_traces = np.concatenate([post_contra_traces, post_ipsi_traces])
    return pre_contra_traces, post_contra_traces, pre_ipsi_traces, post_ipsi_traces, pre_both_traces, post_both_traces


def get_traces_and_reward_types(photometry_data, trial_data):
    omission_trials = trial_data.loc[trial_data['State type'] == 10]['Trial num'].values
    left_large_reward_trials = trial_data.loc[trial_data['State type'] == 12]['Trial num'].values
    right_large_reward_trials = trial_data.loc[trial_data['State type'] == 13]['Trial num'].values
    all_large_reward_trials = np.concatenate([left_large_reward_trials, right_large_reward_trials])
    normal_left_reward_trials = trial_data.loc[trial_data['State type'] == 6]['Trial num'].values
    normal_right_reward_trials = trial_data.loc[trial_data['State type'] == 7]['Trial num'].values
    normal_all_reward_trials = np.concatenate([normal_left_reward_trials, normal_right_reward_trials])
    pre_trials = normal_all_reward_trials
    post_trials = all_large_reward_trials
    contra_data = photometry_data.contra_data
    ipsi_data = photometry_data.ipsi_data
    pre_contra_trial_nums, pre_contra_trials, _ = np.intersect1d(contra_data.trial_nums, pre_trials,
                                                                 return_indices=True)
    post_contra_trial_nums, post_contra_trials, _ = np.intersect1d(contra_data.trial_nums, post_trials,
                                                                   return_indices=True)
    omission_contra_trial_nums, omission_contra_trials, _ = np.intersect1d(contra_data.trial_nums, omission_trials,
                                                                           return_indices=True)
    pre_ipsi_trial_nums, pre_ipsi_trials, _ = np.intersect1d(ipsi_data.trial_nums, pre_trials, return_indices=True)
    post_ipsi_trial_nums, post_ipsi_trials, _ = np.intersect1d(ipsi_data.trial_nums, post_trials, return_indices=True)
    omission_ipsi_trial_nums, omission_ipsi_trials, _ = np.intersect1d(ipsi_data.trial_nums, omission_trials,
                                                                       return_indices=True)
    pre_contra_traces = contra_data.sorted_traces[pre_contra_trials, :]
    post_contra_traces = contra_data.sorted_traces[post_contra_trials, :]
    omission_contra_traces = contra_data.sorted_traces[omission_contra_trials, :]
    pre_ipsi_traces = ipsi_data.sorted_traces[pre_ipsi_trials, :]
    post_ipsi_traces = ipsi_data.sorted_traces[post_ipsi_trials, :]
    omission_ipsi_traces = ipsi_data.sorted_traces[omission_ipsi_trials, :]
    pre_both_traces = np.concatenate([pre_contra_traces, pre_ipsi_traces])
    post_both_traces = np.concatenate([post_contra_traces, post_ipsi_traces])
    all_contra_trial_nums = np.concatenate([pre_contra_trial_nums, post_contra_trial_nums])
    all_ipsi_trial_nums = np.concatenate([pre_ipsi_trial_nums, post_ipsi_trial_nums])
    all_pre_trial_nums = np.concatenate([pre_ipsi_trial_nums, pre_contra_trial_nums])
    all_post_trial_nums = np.concatenate([post_contra_trial_nums, post_ipsi_trial_nums])
    all_trial_nums = np.concatenate([all_contra_trial_nums, all_ipsi_trial_nums])
    all_contra_traces = np.concatenate([pre_contra_traces, post_contra_traces])
    all_ipsi_traces = np.concatenate([pre_ipsi_traces, post_ipsi_traces])
    all_traces = np.concatenate([all_contra_traces, all_ipsi_traces])
    all_omission_traces = np.concatenate([omission_contra_traces, omission_ipsi_traces])
    all_omission_trial_nums = np.concatenate([omission_contra_trial_nums, omission_ipsi_trial_nums])
    list_traces = [all_traces[i, :] for i in range(all_traces.shape[0])]
    list_omission_traces = [all_omission_traces[i, :] for i in range(all_omission_traces.shape[0])]
    omission_data = {}
    omission_data['trial number'] = all_omission_trial_nums
    omission_data['side'] = np.where(np.isin(all_omission_trial_nums, omission_ipsi_trial_nums), 'ipsi', 'contra')
    omission_dataf = pd.DataFrame(omission_data)
    omission_dataf['traces'] = pd.Series(list_omission_traces, index=omission_dataf.index)
    omission_dataf['reward'] = 'omission'
    one_session_data = {}
    ipsi_contra_labels = np.where(np.isin(all_trial_nums, all_ipsi_trial_nums), 'ipsi', 'contra')
    label = 'large reward'
    reward_size_labels = np.where(np.isin(all_trial_nums, all_post_trial_nums), label, 'normal')
    one_session_data['reward'] = reward_size_labels
    one_session_data['trial number'] = all_trial_nums
    one_session_data['side'] = ipsi_contra_labels
    one_session_dataf = pd.DataFrame(one_session_data)
    one_session_dataf['traces'] = pd.Series(list_traces, index=one_session_dataf.index)
    all_reward_type_data = pd.concat([one_session_dataf, omission_dataf])
    all_reward_type_data['time points'] = pd.Series([photometry_data.contra_data.time_points] *
                                                    (len(list_traces) + len(all_omission_traces)),
                                                    index=all_reward_type_data.index)
    return all_reward_type_data


def plot_mean_trace_for_condition(ax, block_change_info, time_points, key, error_bar_method=None, save_location=None):
    mouse = block_change_info['mouse'].iloc[0]
    if key == 'reward ipsi':
        condition = 'reward'
        leg_title = condition
    elif key == 'side':
        condition = 'side'
        leg_title = condition
    elif key == 'reward contra':
        condition = 'reward'
        leg_title = condition
    elif key == 'reward':
        condition = 'reward'
        leg_title = condition
    else:
        raise ValueError('Condition not recognised')

    reward_amounts = np.sort(block_change_info[condition].unique())
    colours = cm.inferno(np.linspace(0, 0.8, reward_amounts.shape[0]))
    time_points = decimate(time_points, 10)
    for reward_indx, reward_amount in enumerate(reward_amounts):
        rows = block_change_info[(block_change_info[condition] == reward_amount)]
        traces = rows['traces'].values
        flat_traces = np.zeros([traces.shape[0], traces[0].shape[0]])
        for idx, trace in enumerate(traces):
            flat_traces[idx, :] = trace
        mean_trace = decimate(np.mean(flat_traces, axis=0), 10)
        ax.plot(time_points, mean_trace, lw=1.5, color=colours[reward_indx], label=reward_amount)
        if error_bar_method is not None:
            # bootstrapping takes a long time. calculate once and save:
            filename = 'errors_{}_{}_{}.npz'.format(mouse, key, reward_amount)
            if not os.path.isfile(os.path.join(save_location, filename)):
                error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                    decimate(flat_traces, 10),
                                                                    error_bar_method=error_bar_method)
                np.savez(os.path.join(save_location, filename), error_bar_lower=error_bar_lower,
                         error_bar_upper=error_bar_upper)
            else:
                error_info = np.load(os.path.join(save_location, filename))
                error_bar_lower = error_info['error_bar_lower']
                error_bar_upper = error_info['error_bar_upper']
            ax.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                             facecolor=colours[reward_indx], linewidth=0)

    ax.axvline(0, color='k')
    ax.set_xlim([-2, 2])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('z-scored fluorescence')
    #ax.set_title(condition)
    #lg = ax.legend(title=leg_title, bbox_to_anchor=(1., 1.), fontsize=14)
    #lg.get_title().set_fontsize(14)