import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import os
import bpod_open_ephys_analysis.utils.load_nested_structs as load_ns
import numpy as np
import pandas as pd

def find_bpod_file(mouse, date, protocol):
    Bpod_data_path = 'W:\\photometry_2AC\\bpod_data\\' + mouse + '\\{}\\Session Data\\'.format(protocol)
    Bpod_file_search_tool = mouse + '_{}_'.format(protocol) + date
    files_in_bpod_path = os.listdir(Bpod_data_path)
    files_on_that_day = [s for s in files_in_bpod_path if Bpod_file_search_tool in s]
    mat_files_on_that_day = [s for s in files_on_that_day if '.mat' in s]

    if len(mat_files_on_that_day) == 2:
        no_extension_files = [os.path.splitext(filename)[0] for filename in mat_files_on_that_day]
        file_times = [filename.split('_')[-1] for filename in no_extension_files]
        main_session_file = Bpod_data_path + mat_files_on_that_day[file_times.index(max(file_times))]
        return main_session_file
    elif len(mat_files_on_that_day) == 1:
        no_extension_files = [os.path.splitext(filename)[0] for filename in mat_files_on_that_day]
        file_times = [filename.split('_')[-1] for filename in no_extension_files]
        main_session_file = Bpod_data_path +  mat_files_on_that_day[file_times.index(max(file_times))]
        return main_session_file
    else:
        print('0 or more than 2 sessions that day!')

def find_daq_file(mouse, date):
    daq_data_path = 'W:\\photometry_2AC\\freely_moving_photometry_data\\' + mouse +  '\\'
    folders_in_photo_path = os.listdir(daq_data_path)
    folders_on_that_day = [s for s in folders_in_photo_path if date in s]

    if len(folders_on_that_day) == 2:
        main_session_file = daq_data_path + '\\' + folders_on_that_day[-1] + '\\' + 'AI.tdms'
        return main_session_file
        print('2 sessions that day')
    elif len(folders_on_that_day) == 1:
        main_session_file = daq_data_path + '\\' + folders_on_that_day[-1] + '\\' + 'AI.tdms'
        return main_session_file
    else:
        print('0 or more than 2 sessions that day!')


def is_empty(any_structure):
    if any_structure:
        return False
    else:
        return True


def load_bpod_file(main_session_file):
    # gets the Bpod data out of MATLAB struct and into python-friendly format
    loaded_bpod_file = load_ns.loadmat(main_session_file)

    # as RawEvents.Trial is a cell array of structs in MATLAB, we have to loop through the array and convert the structs to dicts
    trial_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']

    for trial_num, trial in enumerate(trial_raw_events):
        trial_raw_events[trial_num] = load_ns._todict(trial)

    loaded_bpod_file['SessionData']['RawEvents']['Trial'] = trial_raw_events
    first_trial = trial_raw_events[0]
    return loaded_bpod_file, trial_raw_events


def find_num_times_in_state(trial_states):
    unique_states = np.unique(trial_states)
    state_occurences = np.zeros(trial_states.shape)
    max_occurences = np.zeros(trial_states.shape)
    for state in unique_states:
        total_occurences = np.where(trial_states==state)[0].shape[0]
        num_occurences = 0
        for idx, val in enumerate(trial_states):
            if val==state:
                num_occurences+=1
                state_occurences[idx] = num_occurences
                max_occurences[idx] = total_occurences
    return state_occurences, max_occurences


def restructure_bpod_timestamps_cue_reward(loaded_bpod_file, trial_start_ttls_daq):
    original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
    original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps']
    daq_trials_start_ttls = trial_start_ttls_daq
    original_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']
    # loops through all the trials and pulls out all the states
    for trial, state_timestamps in enumerate(original_state_timestamps_all_trials):
        state_info = {}
        event_info = {}
        trial_states = original_state_data_all_trials[trial]
        num_states = (len(trial_states))
        state_info['Trial num'] = np.ones((num_states)) * trial
        state_info['State type'] = trial_states
        num_times_in_state = find_num_times_in_state(trial_states)
        state_info['Instance in state'] = num_times_in_state[0]
        state_info['Max times in state'] = num_times_in_state[1]
        state_info['State name'] = loaded_bpod_file['SessionData']['RawData']['OriginalStateNamesByNumber'][0][
            trial_states - 1]
        state_info['Time start'] = state_timestamps[0:-1] + daq_trials_start_ttls[trial]
        state_info['Time end'] = state_timestamps[1:] + daq_trials_start_ttls[trial]

        state_info['Trial start'] = np.ones((num_states)) * daq_trials_start_ttls[trial]
        state_info['Trial end'] = np.ones((num_states)) * (state_timestamps[-1] + daq_trials_start_ttls[trial])
        trial_data = pd.DataFrame(state_info)

        trial_events = original_raw_events[trial]['Events']
        reward_time = state_timestamps[5]
        lick_times = trial_events.get('Port4In', -1)
        licks_pre_reward = np.where(lick_times < reward_time)
        lick_times = lick_times[licks_pre_reward]
        lick_intervals = np.concatenate((np.array([1]), np.diff(lick_times)))
        lick_bout_starts = np.where(lick_intervals >= 0.2)[0][1:]
        if  lick_bout_starts.size > 0 and lick_times.size > 0:
            print(lick_bout_starts)
            lick_bout_start_times = lick_times[lick_bout_starts]
            lick_times = lick_bout_start_times
        else:
            lick_times = np.empty([0])
            lick_bout_start_times = np.empty([0])
        if lick_times.size > 0:
            num_licks = len(lick_times)
            event_info['Time start'] = lick_times + daq_trials_start_ttls[trial]
            licks_off = trial_events['Port4Out']
            licks_off_pre_reward = np.where(licks_off < reward_time)
            event_info['Time end'] = lick_times
            event_info['Trial num'] = np.ones((num_licks)) * trial
            event_info['State type'] = np.ones((num_licks)) * 0
            event_info['Instance in state'] = np.arange(0, num_licks)
            event_info['Max times in state'] = np.ones((num_licks)) * num_licks
            event_info['State name'] = np.asarray(['lick'] * num_licks)

            event_info['Trial start'] = np.ones((num_licks)) * daq_trials_start_ttls[trial]
            event_info['Trial end'] = np.ones((num_licks)) * (state_timestamps[-1] + daq_trials_start_ttls[trial])

        if trial == 0:
            restructured_data = trial_data
            if lick_times.size > 0:
                event_data = pd.DataFrame(event_info)
                restructured_data = pd.concat([restructured_data, event_data], ignore_index=True)
        else:
            restructured_data = pd.concat([restructured_data, trial_data], ignore_index=True)
            if lick_times.size > 0:
                event_data = pd.DataFrame(event_info)
                restructured_data = pd.concat([restructured_data, event_data], ignore_index=True)
    return restructured_data

def restructure_bpod_timestamps_opto_stim(loaded_bpod_file, trial_start_ttls_daq):
    original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
    original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps']
    daq_trials_start_ttls = trial_start_ttls_daq
    original_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']
    # loops through all the trials and pulls out all the states
    for trial, state_timestamps in enumerate(original_state_timestamps_all_trials):
        state_info = {}
        event_info = {}
        trial_states = original_state_data_all_trials[trial]
        num_states = (len(trial_states))
        state_info['Trial num'] = np.ones((num_states)) * trial
        state_info['State type'] = trial_states
        num_times_in_state = find_num_times_in_state(trial_states)
        state_info['Instance in state'] = num_times_in_state[0]
        state_info['Max times in state'] = num_times_in_state[1]
        state_info['State name'] = loaded_bpod_file['SessionData']['RawData']['OriginalStateNamesByNumber'][0][
            trial_states - 1]
        state_info['Time start'] = state_timestamps[0:-1] + daq_trials_start_ttls[trial]
        state_info['Time end'] = state_timestamps[1:] + daq_trials_start_ttls[trial]

        state_info['Trial start'] = np.ones((num_states)) * daq_trials_start_ttls[trial]
        state_info['Trial end'] = np.ones((num_states)) * (state_timestamps[-1] + daq_trials_start_ttls[trial])
        trial_data = pd.DataFrame(state_info)
        if trial == 0:
            restructured_data = trial_data
        else:
            restructured_data = pd.concat([restructured_data, trial_data], ignore_index=True)
    return restructured_data