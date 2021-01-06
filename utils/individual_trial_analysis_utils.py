import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import peakutils
import pandas as pd


class HeatMapParams(object):
    def __init__(self, params, response, first_choice):
        self.state = params['state_type_of_interest']
        self.outcome = params['outcome']
        #self.last_outcome = params['last_outcome']
        self.response = response
        self.last_response = params['last_response']
        self.align_to = params['align_to']
        self.other_time_point = np.array(['Time start', 'Time end'])[np.where(np.array(['Time start', 'Time end']) != params['align_to'])]
        self.instance = params['instance']
        self.plot_range = params['plot_range']
        self.no_repeats = params['no_repeats']
        self.first_choice_correct = params['first_choice_correct']
        self.first_choice = first_choice
        self.cue = params['cue']


def get_photometry_around_event(all_trial_event_times, demodulated_trace, pre_window=5, post_window=5, sample_rate=10000):
    num_events = len(all_trial_event_times)
    event_photo_traces = np.zeros((num_events, sample_rate*(pre_window + post_window)))
    for event_num, event_time in enumerate(all_trial_event_times):
        plot_start = int(round(event_time*sample_rate)) - pre_window*sample_rate
        plot_end = int(round(event_time*sample_rate)) + post_window*sample_rate
        if plot_end - plot_start != sample_rate*(pre_window + post_window):
            print(event_time)
            plot_start = plot_start + 1
            print(plot_end - plot_start)
        event_photo_traces[event_num, :] = demodulated_trace[plot_start:plot_end]
        #except:
        #   event_photo_traces = event_photo_traces[:event_num,:]
    print(event_photo_traces.shape)
    return event_photo_traces


def get_next_centre_poke(trial_data, events_of_int):
    trial_numbers = events_of_int['Trial num'].unique()[:-1]
    next_centre_poke_times = np.zeros(events_of_int.shape[0])
    events_of_int = events_of_int.reset_index(drop=True)
    for event_trial_num in trial_numbers:
        trial_num = event_trial_num
        event_indx_for_that_trial = events_of_int.loc[(events_of_int['Trial num'] == trial_num)].index
        next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]
        wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)]
        next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
        next_centre_poke_times[event_indx_for_that_trial] = next_wait_for_poke['Time end'].values[0]
    return next_centre_poke_times

def get_next_reward_time(trial_data, events_of_int):
    trial_numbers = events_of_int['Trial num'].values
    next_reward_times = []
    for event_trial_num in range(len(trial_numbers)):
        trial_num = trial_numbers[event_trial_num]
        other_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num)]
        choices = other_trial_events.loc[(other_trial_events['State type'] == 5)]
        max_times_in_state_choices = choices['Max times in state'].unique()
        choice = choices.loc[(choices['Instance in state'] == max_times_in_state_choices)]
        next_reward_times.append(choice['Time end'].values[0])
    return next_reward_times

def find_and_z_score_traces(trial_data, demod_signal, params, norm_window=8, sort=False, get_photometry_data=True):
    response_names = ['both left and right', 'left', 'right']
    outcome_names = ['incorrect', 'correct', 'both correct and incorrect']
    events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]
    if params.state == 10 or params.state == 12 or params.state == 13: # omissions, large rewards left and right
        if params.state == 10:
            omission_events = trial_data.loc[(trial_data['State type'] == params.state)]
        else:
            left_large_reward_events = trial_data.loc[(trial_data['State type'] == 12)]
            right_large_reward_events =  trial_data.loc[(trial_data['State type'] == 13)]
            omission_events = pd.concat([left_large_reward_events, right_large_reward_events])

        trials_of_int = omission_events['Trial num'].values
        omission_trials_all_states = trial_data.loc[(trial_data['Trial num'].isin(trials_of_int))]
        events_of_int = omission_trials_all_states.loc[(omission_trials_all_states['State type'] == 5)] # get the action aligned trace
    else:
        events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]
    if params.response != 0:
        events_of_int = events_of_int.loc[events_of_int['Response'] == params.response]
    print(events_of_int.shape)
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
        events_of_int = events_of_int.loc[
            (events_of_int['First choice correct'] == 0)]

    event_times = events_of_int[params.align_to].values
    trial_nums = events_of_int['Trial num'].values
    if params.state == 12 or params.state == 13:
        state_name = 'LargeReward'
    else:
        state_name = events_of_int['State name'].values[0]
    other_event = np.asarray(
        np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(events_of_int[params.align_to].values))
    next_centre_poke = get_next_centre_poke(trial_data, events_of_int)
    outcome_times = get_next_reward_time(trial_data, events_of_int)
    outcome_times = outcome_times - event_times

    last_trial_num = events_of_int['Trial num'].unique()[-1]
    events_reset_indx = events_of_int.reset_index(drop=True)
    last_trial_event_indx = events_reset_indx.loc[(events_reset_indx['Trial num'] == last_trial_num)].index
    next_centre_poke[last_trial_event_indx] = events_reset_indx[params.align_to].values[last_trial_event_indx]
    next_centre_poke_norm = next_centre_poke - event_times

    # this all deals with getting photometry data
    if get_photometry_data == True:
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
            sorted_next_poke = next_centre_poke_norm[arr1inds[::-1]]
        else:
            sorted_other_event = other_event
            sorted_next_poke = next_centre_poke_norm
        return sorted_other_event, state_name, title, sorted_next_poke, trial_nums, event_times




def get_peak_each_trial(sorted_traces, time_points, sorted_other_events):
    all_trials_peaks = []
    for trial_num in range(0, len(sorted_other_events)):
        indices_to_integrate = np.where(np.logical_and(np.greater_equal(time_points, 0), np.less_equal(time_points, sorted_other_events[trial_num])))
        trial_trace = (sorted_traces[trial_num, indices_to_integrate]).T
        trial_trace = trial_trace # - trial_trace[0]s
        trial_peak_inds = peakutils.indexes(trial_trace.flatten('F'))
        if len(trial_peak_inds>1):
            trial_peak_inds = trial_peak_inds[0]
        trial_peaks = trial_trace.flatten('F')[trial_peak_inds]
        all_trials_peaks.append(trial_peaks)
        #plt.plot(trial_trace)
        #plt.scatter(trial_peak_inds, trial_peaks)
    flat_peaks = all_trials_peaks
    #plt.show()
    return flat_peaks

def get_peak_each_trial_psychometric(sorted_traces, time_points, sorted_other_events):
    all_trials_peaks = []
    #plt.figure()
    for trial_num in range(0, len(sorted_other_events)):
        indices_to_integrate = np.where(np.logical_and(np.greater_equal(time_points, 0),
                                                       np.less_equal(time_points, sorted_other_events[trial_num])))
        trial_trace = (sorted_traces[trial_num, indices_to_integrate]).T
        trial_trace = trial_trace  #- trial_trace[0]
        trial_peak_inds = peakutils.indexes(trial_trace.flatten('F'), thres=0.3)
        if trial_peak_inds.shape[0] > 0:
            if len(trial_peak_inds > 1):
                trial_peak_inds = trial_peak_inds[0]

            trial_peaks = trial_trace.flatten('F')[trial_peak_inds]
        else:
            trial_peaks = np.nan
        all_trials_peaks.append(trial_peaks)
            #plt.plot(trial_trace)
           # plt.scatter(trial_peak_inds, trial_peaks)
    flat_peaks = all_trials_peaks
    #plt.show()
   # plt.figure()
    #plt.plot(np.mean(sorted_traces, axis=0))
    return flat_peaks

def get_max_each_trial(sorted_traces, time_points, sorted_other_events):
    all_trials_peaks = []
    plt.figure()
    for trial_num in range(0, len(sorted_other_events)):
        indices_to_integrate = np.where(np.logical_and(np.greater_equal(time_points, 0),
                                                       np.less_equal(time_points, sorted_other_events[trial_num])))
        trial_trace = (sorted_traces[trial_num, indices_to_integrate]).T
        trial_trace = trial_trace  - trial_trace[0]
        all_trials_peaks.append(np.max(trial_trace))
    flat_peaks = all_trials_peaks
    return flat_peaks


class SessionData(object):
    def __init__(self, fiber_side, recording_site, mouse_id, date):
        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.recording_site = recording_site
        self.date = date
        self.choice_data = None
        self.cue_data = None
        self.reward_data = None

    def get_choice_responses(self):
        self.choice_data = ChoiceAlignedData(self)

    def get_cue_responses(self):
        self.cue_data = CueAlignedData(self)

    def get_reward_responses(self):
        self.reward_data = RewardAlignedData(self)


class SessionEvents(object):
    def __init__(self, fiber_side, recording_site, mouse_id, date):
        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.recording_site = recording_site
        self.date = date
        self.choice_data = None
        self.cue_data = None
        self.reward_data = None

    def get_reaction_times(self, dff, trial_data):
        self.ipsi_reaction_times, state_name, title, ipsi_sorted_next_poke, self.ipsi_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)
        self.contra_reaction_times, state_name, title, contra_sorted_next_poke, self.contra_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)

    def get_choice_events(self):
        self.choice_data = ChoiceAlignedEvents(self)

    def get_cue_events(self):
        self.cue_data = CueAlignedEvents(self)

    def get_reward_events(self):
        self.reward_data = RewardAlignedEvents(self)


class ZScoredTraces(object):
    def __init__(self,  trial_data, dff, params, response, first_choice):
        self.trial_peaks = None
        self.params = HeatMapParams(params, response, first_choice)
        self.time_points, self.mean_trace, self.sorted_traces, self.reaction_times, self.state_name, title, self.sorted_next_poke, self.trial_nums, self.event_times, self.outcome_times = find_and_z_score_traces(
            trial_data, dff, self.params, sort=False)

    def get_peaks(self):
        self.trial_peaks = get_peak_each_trial(self.sorted_traces, self.time_points, self.outcome_times)


class BehaviouralEvents(object):
    def __init__(self,  trial_data, dff, params, response, first_choice):
        self.params = HeatMapParams(params, response, first_choice)
        self.reaction_times, self.state_name, title, self.sorted_next_poke, self.trial_nums, self.event_times = find_and_z_score_traces(
            trial_data, dff, self.params, sort=False,  get_photometry_data=False)


class ChoiceAlignedData(object):
    def __init__(self, session_data):
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 5,
            'outcome': 2,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 1,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': -1,
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
            'cue': None}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.ipsi_data.get_peaks()

        self.contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.contra_data.get_peaks()


class ChoiceAlignedEvents(object):
    def __init__(self, session_data):
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 5,
            'outcome': 2,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 1,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': -1,
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
            'cue': None}


        self.ipsi_data = BehaviouralEvents(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.contra_data = BehaviouralEvents(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)

class CueAlignedData(object):
    def __init__(self, session_data):
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 3,
            'outcome': 2,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 0,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': -1,
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
            'cue': None}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.ipsi_data.get_peaks()
        self.contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.contra_data.get_peaks()

        params['cue'] = 'high'
        self.high_cue_data =  ZScoredTraces(trial_data, dff, params, 0, 0)
        params['cue'] = 'low'
        self.low_cue_data = ZScoredTraces(trial_data, dff, params, 0, 0)

class CueAlignedEvents(object):
    def __init__(self, session_data):
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 3,
            'outcome': 2,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 1,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': 1,
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
             'cue': None}

        self.ipsi_data = BehaviouralEvents(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.contra_data = BehaviouralEvents(trial_data, dff,params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        params['cue'] = 'high'
        self.high_cue_data = BehaviouralEvents(trial_data, dff, params, 0, 0)
        params['cue'] = 'low'
        self.low_cue_data = BehaviouralEvents(trial_data, dff, params, 0, 0)

class RewardAlignedData(object):
    def __init__(self, session_data):
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 5,
                  'outcome': 1,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': 1,
                  'cue': 'None'}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)

        self.contra_data = ZScoredTraces(trial_data, dff,params, contra_fiber_side_numeric, contra_fiber_side_numeric)




class RewardAlignedEvents(object):
    def __init__(self, session_data):
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 5,
            'outcome': 1,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 0,
            'last_response': 0,
            'align_to' : 'Time end',
            'instance': -1,
            'plot_range': [-6, 6],
            'first_choice_correct': 1,
            'cue': None}

        self.ipsi_data = BehaviouralEvents(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.contra_data = BehaviouralEvents(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)



