import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from tqdm import tqdm
import peakutils


class HeatMapParams(object):
    def __init__(self, state_type_of_interest, response, first_choice, last_response, outcome, last_outcome,first_choice_correct, align_to, instance, no_repeats, plot_range):

        self.state = state_type_of_interest
        self.outcome = outcome
        #self.last_outcome = last_outcome
        self.response = response
        self.last_response = last_response
        self.align_to = align_to
        self.other_time_point = np.array(['Time start', 'Time end'])[np.where(np.array(['Time start', 'Time end']) != align_to)]
        self.instance = instance
        self.plot_range = plot_range
        self.no_repeats = no_repeats
        self.first_choice_correct = first_choice_correct
        self.first_choice = first_choice

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
    trial_numbers = events_of_int['Trial num'].values
    next_centre_poke_times = []
    for event_trial_num in range(len(trial_numbers) - 1):
        trial_num = trial_numbers[event_trial_num]
        next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]
        wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)]
        next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
        next_centre_poke_times.append(next_wait_for_poke['Time end'].values[0])
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
    outcome_names = ['incorrect', 'correct']
    events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]
    if params.response != 0:
        events_of_int = events_of_int.loc[events_of_int['Response'] == params.response]
    if params.first_choice != 0:
        events_of_int = events_of_int.loc[events_of_int['First response'] == params.first_choice]
    if params.last_response != 0:
        events_of_int = events_of_int.loc[events_of_int['Last response'] == params.last_response]
        title = ' last response: ' + response_names[params.last_response]
    else:
        title = response_names[params.response]
    events_of_int = events_of_int.loc[events_of_int['Trial outcome'] == params.outcome]
    # events_of_int = events_of_int.loc[events_of_int['Last outcome'] == 0]

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
    if params.first_choice_correct:
        events_of_int = events_of_int.loc[
            (events_of_int['First choice correct'] == 1)]

    event_times = events_of_int[params.align_to].values
    trial_nums = events_of_int['Trial num'].values
    state_name = events_of_int['State name'].values[0]
    other_event = np.asarray(
        np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(events_of_int[params.align_to].values))
    next_centre_poke = get_next_centre_poke(trial_data, events_of_int)
    next_centre_poke.append(event_times[-1])
    next_centre_poke_norm = next_centre_poke - event_times

    if params.state == 3:
        other_event = get_next_reward_time(trial_data, events_of_int) - event_times
    if params.state == 5 and params.align_to == 'Time end':
        other_event = np.ones(np.shape(next_centre_poke_norm))*0.8
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

        return time_points, mean_trace, sorted_traces, sorted_other_event, state_name, title, sorted_next_poke, trial_nums
    else:
        if sort:
            arr1inds = other_event.argsort()
            sorted_other_event = other_event[arr1inds[::-1]]
            sorted_next_poke = next_centre_poke_norm[arr1inds[::-1]]
        else:
            sorted_other_event = other_event
            sorted_next_poke = next_centre_poke_norm
        return sorted_other_event, state_name, title, sorted_next_poke, trial_nums


def get_peak_each_trial(sorted_traces, time_points, sorted_other_events, ipsi_or_contra):
    all_trials_peaks = []
    for trial_num in range(0, len(sorted_other_events)):
        indices_to_integrate = np.where(np.logical_and(np.greater_equal(time_points, 0), np.less_equal(time_points, sorted_other_events[trial_num])))
        trial_trace = (sorted_traces[trial_num, indices_to_integrate]).T
        trial_trace = trial_trace # - trial_trace[0]
        trial_peak_inds = peakutils.indexes(trial_trace.flatten('F'), thres=0.3)
        if len(trial_peak_inds>1):
            trial_peak_inds = trial_peak_inds[0]
        trial_peaks = trial_trace.flatten('F')[trial_peak_inds]
        all_trials_peaks.append(trial_peaks)
        plt.plot(trial_trace)
        plt.scatter(trial_peak_inds, trial_peaks)
    plt.title(ipsi_or_contra)
    flat_peaks =  all_trials_peaks
    plt.show()
    return flat_peaks

class RawTracesZScored(object):
    def __init__(self, fiber_side, mouse_id, date):

        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.date = date

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]

        state_type_of_interest = 5
        outcome = 1
        last_outcome = 0 # NOT USED CURRENLY
        no_repeats = 1
        last_response = 0
        align_to = 'Time start'
        instance = -1
        plot_range = [-2,3]
        first_choice_correct = 1

        response = fiber_side_numeric
        first_choice = fiber_side_numeric
        self.ipsi_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                         last_outcome, first_choice_correct, align_to, instance, no_repeats, plot_range)
        contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]
        response = contra_fiber_side_numeric
        first_choice = contra_fiber_side_numeric
        self.contra_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                           last_outcome, first_choice_correct, align_to, instance, no_repeats,
                                           plot_range)

    def get_reaction_times(self, dff, trial_data):
        self.ipsi_reaction_times, state_name, title, ipsi_sorted_next_poke, self.ipsi_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)
        self.contra_reaction_times, state_name, title, contra_sorted_next_poke, self.contra_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)

    def get_peaks(self, dff, trial_data):
        time_points, ipsi_mean_trace, ipsi_sorted_traces, self.ipsi_reaction_times, state_name, title, ipsi_sorted_next_poke, self.ipsi_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=False)
        ipsi_trials_peaks = get_peak_each_trial(ipsi_sorted_traces, time_points,  self.ipsi_reaction_times, 'ipsi')
        self.ipsi_trials_peaks = ipsi_trials_peaks
        time_points, contra_mean_trace, contra_sorted_traces, self.contra_reaction_times, state_name, title, contra_sorted_next_poke, self.contra_trial_nums  = find_and_z_score_traces(
        trial_data, dff, self.contra_params, sort=False)
        contra_trials_peaks = get_peak_each_trial(contra_sorted_traces, time_points, self.contra_reaction_times, 'contra')
        self.contra_trials_peaks = contra_trials_peaks

class cueTracesZScored(object):
    def __init__(self, fiber_side, mouse_id, date):

        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.date = date

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]

        state_type_of_interest = 3
        outcome = 1
        last_outcome = 0  # NOT USED CURRENLY
        no_repeats = 1
        last_response = 0
        align_to = 'Time start'
        instance = 1
        plot_range = [-2, 3]
        first_choice_correct = 1

        response = fiber_side_numeric
        first_choice = fiber_side_numeric
        self.ipsi_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                         last_outcome, first_choice_correct, align_to, instance, no_repeats, plot_range)
        contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]
        response = contra_fiber_side_numeric
        first_choice = contra_fiber_side_numeric
        self.contra_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                           last_outcome, first_choice_correct, align_to, instance, no_repeats,
                                           plot_range)

    def get_reaction_times(self, dff, trial_data):
        self.ipsi_reaction_times, state_name, title, ipsi_sorted_next_poke, self.ipsi_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)
        self.contra_reaction_times, state_name, title, contra_sorted_next_poke, self.contra_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)

    def get_peaks(self, dff, trial_data):
        time_points, ipsi_mean_trace, ipsi_sorted_traces, self.ipsi_reaction_times, state_name, title, ipsi_sorted_next_poke, self.ipsi_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=False)
        ipsi_trials_peaks = get_peak_each_trial(ipsi_sorted_traces, time_points,  self.ipsi_reaction_times, 'ipsi')
        self.ipsi_trials_peaks = ipsi_trials_peaks
        time_points, contra_mean_trace, contra_sorted_traces, self.contra_reaction_times, state_name, title, contra_sorted_next_poke, self.contra_trial_nums  = find_and_z_score_traces(
        trial_data, dff, self.contra_params, sort=False)
        contra_trials_peaks = get_peak_each_trial(contra_sorted_traces, time_points, self.contra_reaction_times, 'contra')
        self.contra_trials_peaks = contra_trials_peaks



class rewardTracesZScored(object):
    def __init__(self, fiber_side, mouse_id, date):

        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.date = date

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]

        state_type_of_interest = 5
        outcome = 1
        last_outcome = 0  # NOT USED CURRENLY
        no_repeats = 1
        last_response = 0
        align_to = 'Time end'
        instance = -1
        plot_range = [-2, 3]
        first_choice_correct = 1

        response = fiber_side_numeric
        first_choice = fiber_side_numeric
        self.ipsi_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                         last_outcome, first_choice_correct, align_to, instance, no_repeats, plot_range)
        contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]
        response = contra_fiber_side_numeric
        first_choice = contra_fiber_side_numeric
        self.contra_params = HeatMapParams(state_type_of_interest, response, first_choice, last_response, outcome,
                                           last_outcome, first_choice_correct, align_to, instance, no_repeats,
                                           plot_range)

    def get_reaction_times(self, dff, trial_data):
        self.ipsi_reaction_times, state_name, title, ipsi_sorted_next_poke, self.ipsi_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)
        self.contra_reaction_times, state_name, title, contra_sorted_next_poke, self.contra_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)

    def get_peaks(self, dff, trial_data):
        time_points, ipsi_mean_trace, ipsi_sorted_traces, self.ipsi_reaction_times, state_name, title, ipsi_sorted_next_poke, self.ipsi_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=False)
        ipsi_trials_peaks = get_peak_each_trial(ipsi_sorted_traces, time_points,  self.ipsi_reaction_times, 'ipsi')
        self.ipsi_trials_peaks = ipsi_trials_peaks
        time_points, contra_mean_trace, contra_sorted_traces, self.contra_reaction_times, state_name, title, contra_sorted_next_poke, self.contra_trial_nums  = find_and_z_score_traces(
        trial_data, dff, self.contra_params, sort=False)
        contra_trials_peaks = get_peak_each_trial(contra_sorted_traces, time_points, self.contra_reaction_times, 'contra')
        self.contra_trials_peaks = contra_trials_peaks



