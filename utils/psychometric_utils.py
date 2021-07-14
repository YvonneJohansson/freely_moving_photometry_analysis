import numpy as np
import pandas as pd
from scipy import stats

from utils.individual_trial_analysis_utils import get_next_centre_poke, get_photometry_around_event, get_first_poke, get_next_reward_time, HeatMapParams, get_peak_each_trial, SessionData

def open_experiment(experiment):
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + experiment['mouse_id'] + '\\'
    restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
    dff_trace_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'smoothed_signal.npy'
    dff = np.load(saving_folder + dff_trace_filename)
    session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
    return session_traces

def find_and_z_score_traces(trial_data, demod_signal, params, norm_window=8, sort=False, get_photometry_data=True):
    response_names = ['both left and right', 'left', 'right']
    outcome_names = ['incorrect', 'correct', 'both correct and incorrect']

    if params.state == 10 or params.state == 12 or params.state == 13:  # omissions, large rewards left and right
        if params.state == 10:
            omission_events = trial_data.loc[(trial_data['State type'] == params.state)]
        else:
            left_large_reward_events = trial_data.loc[(trial_data['State type'] == 12)]
            right_large_reward_events = trial_data.loc[(trial_data['State type'] == 13)]
            omission_events = pd.concat([left_large_reward_events, right_large_reward_events])

        trials_of_int = omission_events['Trial num'].values
        omission_trials_all_states = trial_data.loc[(trial_data['Trial num'].isin(trials_of_int))]
        events_of_int = omission_trials_all_states.loc[
            (omission_trials_all_states['State type'] == 5)]  # get the action aligned trace
    elif params.state == 5.5:
        events_of_int = trial_data.loc[
            np.logical_or((trial_data['State type'] == 5.5), (trial_data['State type'] == 5))]
        trial_nums = events_of_int['Trial num'].unique()
        for trial in trial_nums:
            events = events_of_int[events_of_int['Trial num'] == trial]
            if events.shape[0] > 1:
                second_event = pd.to_numeric(events['Time end']).idxmax()
                if events['State type'][second_event] == 5.5:
                    events_of_int = events_of_int.drop(second_event)

    else:
        events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]

    if params.response != 0:
        if params.state == 5.5:
            correct_choices = np.logical_and(events_of_int['Response'] == params.response,
                                             events_of_int['State type'] == 5)
            incorrect_first_choices = np.logical_and(events_of_int['First response'] == params.response,
                                                     events_of_int['State type'] == 5.5)
            events_of_int = events_of_int.loc[np.logical_or(correct_choices, incorrect_first_choices)]
        else:
            events_of_int = events_of_int.loc[events_of_int['Response'] == params.response]
    if params.first_choice != 0:
        events_of_int = events_of_int.loc[events_of_int['First response'] == params.first_choice]
    if params.last_response != 0:
        events_of_int = events_of_int.loc[events_of_int['Last response'] == params.last_response]
        title = ' last response: ' + response_names[params.last_response]
    else:
        title = response_names[params.response]
    if not params.outcome == 2: # if you don't care about the reward or not
        events_of_int = events_of_int.loc[events_of_int['Trial outcome'] == params.outcome]
    #events_of_int = events_of_int.loc[events_of_int['Last outcome'] == 0]

    if params.cue == 'high':
        events_of_int = events_of_int.loc[events_of_int['Trial type'] == 7]
    elif params.cue == 'low':
        events_of_int = events_of_int.loc[events_of_int['Trial type'] == 1]

    if params.state == 10:
        title = title + ' ' + 'omission'
    else:
        title = title + ' ' + outcome_names[params.outcome]

    if params.instance == -1:
        events_of_int = events_of_int.loc[
            (events_of_int['Instance in state'] / events_of_int['Max times in state'] == 1)]
    elif params.instance == 1:
        events_of_int = events_of_int.loc[(events_of_int['Instance in state'] == 1)]
        if params.no_repeats == 1:
            events_of_int = events_of_int.loc[events_of_int['Max times in state'] == 1]
    elif params.instance == 0:
        events_of_int = events_of_int

    if params.first_choice_correct == 1:
        events_of_int = events_of_int.loc[
            (events_of_int['First choice correct'] == 1)]
    elif params.first_choice_correct == -1:
        events_of_int = events_of_int.loc[np.logical_or(
            (events_of_int['First choice correct'] == 0), (events_of_int['First choice correct'].isnull()))]
        if events_of_int['State type'].isin([5.5]).any():
            events_of_int = events_of_int.loc[events_of_int['First choice correct'].isnull()]

    events_of_int_reset = events_of_int.reset_index(drop=True)
    if events_of_int['State type'].isin([5.5]).any():
        event_times = np.zeros(events_of_int_reset.shape[0])
        for i, event in events_of_int_reset.iterrows():
            if event['State type'] == 5.5:
                align_to = 'Time start'
                event_times[i] = event[align_to]
            else:
                event_times[i] = event[params.align_to]
    else:
        event_times = events_of_int[params.align_to].values
    trial_nums = events_of_int['Trial num'].values
    #trial_starts = events_of_int['Trial start'].values
    trial_ends = events_of_int['Trial end'].values
    trial_types = events_of_int['Trial type'].values

    if params.state == 5.5:

        other_event = np.zeros([events_of_int_reset.shape[0]])
        for i, event in events_of_int_reset.iterrows():
            if event['State type'] == 5.5:
                trial_num = event['Trial num']
                this_trial_data = trial_data[trial_data['Trial num'] == trial_num]
                out_of_centre = this_trial_data[this_trial_data['State type'] == 4].tail(1)['Time end'].values[0]
                other_event[i] = out_of_centre - np.squeeze(event[params.align_to])
            else:
                other_event[i] = np.squeeze(event[params.other_time_point]) - np.squeeze(event[params.align_to])
    else:
        other_event = np.asarray(
            np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(events_of_int[params.align_to].values))
    if params.state == 12 or params.state == 13:
        state_name = 'LargeReward'
    else:
        state_name = events_of_int['State name'].values[0]

    last_trial = np.max(trial_data['Trial num'])
    last_trial_num = events_of_int['Trial num'].unique()[-1]
    events_reset_indx = events_of_int.reset_index(drop=True)
    last_trial_event_indx = events_reset_indx.loc[(events_reset_indx['Trial num'] == last_trial_num)].index
    next_centre_poke = get_next_centre_poke(trial_data, events_of_int, last_trial_num==last_trial)
    trial_starts = get_first_poke(trial_data, events_of_int)
    outcome_times = get_next_reward_time(trial_data, events_of_int)
    outcome_times = outcome_times - event_times

    #print(events_of_int.shape)
    # this all deals with getting photometry data
    if get_photometry_data == True:
        next_centre_poke[last_trial_event_indx] = events_reset_indx[params.align_to].values[last_trial_event_indx]
        next_centre_poke_norm = next_centre_poke - event_times

        event_photo_traces = get_photometry_around_event(event_times, demod_signal, pre_window=norm_window,
                                                         post_window=norm_window)
        norm_traces = stats.zscore(event_photo_traces.T, axis=0)

        if len(other_event) != norm_traces.shape[1]:
            other_event = other_event[:norm_traces.shape[1]]
        if sort:
            arr1inds = other_event.argsort()
            sorted_other_event = other_event[arr1inds[::-1]]
            sorted_traces = norm_traces.T[arr1inds[::-1]]
            sorted_next_poke = next_centre_poke_norm[arr1inds[::-1]]
        else:
            sorted_other_event = other_event
            sorted_traces = norm_traces.T
            sorted_next_poke = next_centre_poke_norm

        time_points = np.linspace(-norm_window, norm_window, norm_traces.shape[0], endpoint=True, retstep=False, dtype=None,
                             axis=0)
        mean_trace = np.mean(sorted_traces, axis=0)

        return time_points, mean_trace, sorted_traces, sorted_other_event, state_name, title, sorted_next_poke, trial_nums, event_times, outcome_times
    else:
        if sort:
            arr1inds = other_event.argsort()
            sorted_other_event = other_event[arr1inds[::-1]]
            sorted_next_poke = next_centre_poke[arr1inds[::-1]]
        else:
            sorted_other_event = other_event
            sorted_next_poke = next_centre_poke
        return sorted_other_event, state_name, sorted_next_poke, trial_nums, event_times, trial_starts, trial_ends, trial_types


class CustomAlignedDataPsychometric(object):
    def __init__(self, session_data, params):
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        self.contra_data = ZScoredTracesPsychometric(trial_data, dff,params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.ipsi_data = ZScoredTracesPsychometric(trial_data, dff,params, fiber_side_numeric, fiber_side_numeric)

    def add_experiment(self, session_data, params):
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        self.contra_data.add_trials(trial_data, dff)
        self.ipsi_data.add_trials(trial_data, dff)

    def get_peaks(self):
        self.ipsi_data.get_peaks()
        self.contra_data.get_peaks()


class ZScoredTracesPsychometric(object):
    def __init__(self,  trial_data, dff, params, response, first_choice):
        self.trial_peaks = None
        self.params = HeatMapParams(params, response, first_choice)
        self.time_points, self.mean_trace, self.sorted_traces, self.reaction_times, self.state_name, self.sorted_next_poke, self.trial_nums, self.event_times, self.outcome_times, self.trial_types = find_and_z_score_traces(
            trial_data, dff, self.params)

    def add_trials(self, trial_data, dff):
        time_points, mean_trace, sorted_traces, reaction_times, state_name, sorted_next_poke, trial_nums, event_times, outcome_times, trial_types = find_and_z_score_traces(
            trial_data, dff, self.params)
        self.time_points = np.concatenate((self.time_points, time_points))
        self.sorted_traces = np.concatenate((self.sorted_traces, sorted_traces), axis=0)
        self.reaction_times = np.concatenate((self.reaction_times, reaction_times))
        self.sorted_next_poke = np.concatenate((self.sorted_next_poke, sorted_next_poke))
        self.trial_nums = np.concatenate((self.trial_nums, trial_nums))
        self.event_times = np.concatenate((self.event_times, event_times))
        self.outcome_times = np.concatenate((self.outcome_times, outcome_times))
        self.trial_types = np.concatenate((self.trial_types, trial_types))

    def get_peaks(self, save_traces=True):
        if self.params.align_to == 'Time start':
            other_time_point = self.outcome_times
        else: # for reward or non reward aligned data
            other_time_point = self.sorted_next_poke
        self.trial_peaks = get_peak_each_trial(self.sorted_traces, self.time_points, other_time_point)
        if not save_traces:
            self.sorted_traces = None


