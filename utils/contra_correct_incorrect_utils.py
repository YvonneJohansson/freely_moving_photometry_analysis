from utils.individual_trial_analysis_utils import get_photometry_around_event, HeatMapParams, get_next_centre_poke, get_first_poke, get_next_reward_time, find_and_z_score_traces, get_peak_each_trial,  get_max_each_trial, ZScoredTraces
import pandas as pd
import numpy as np

class SessionData(object):
    def __init__(self, fiber_side, recording_site, mouse_id, date):
        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.recording_site = recording_site
        self.date = date
        self.choice_data = None
        self.cue_data = None
        self.reward_data = None
        self.outcome_data = None

    def get_choice_responses(self, save_traces=True):
        self.choice_data = ContraCorrectIncorrect(self, save_traces=save_traces)


class ContraCorrectIncorrect(object):
    def __init__(self, session_data, save_traces=True):
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
            'no_repeats' : 1,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': -1,
            'plot_range': [-6, 6],
            'first_choice_correct': 1,
            'cue': None}


        self.contra_correct_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.contra_correct_data.get_peaks(save_traces=save_traces)

        params = {'state_type_of_interest': 5,
                  'outcome': 2,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 1,
                  'last_response': 0,
                  'align_to': 'Time start',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': -1,
                  'cue': None}

        self.contra_incorrect_data = ZScoredTraces(trial_data, dff, params, 0, contra_fiber_side_numeric)
        self.contra_incorrect_data.get_peaks(save_traces=save_traces)
