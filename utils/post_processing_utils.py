import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
import numpy as np
import pandas as pd
from utils.individual_trial_analysis_utils import ZScoredTraces, SessionData, CueAlignedData


class CustomAlignedData(object):
    def __init__(self, session_data, params, peak_quantification=True):
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + session_data.mouse + '\\'
        #saving_folder = 'C:\\Users\\francescag\\Documents\\PhD_Project\\SNL_photo_photometry\\processed_data' + \
        #                session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.contra_data = ZScoredTraces(trial_data, dff,params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        if peak_quantification:
            self.contra_data.get_peaks()
            self.ipsi_data.get_peaks()


def get_all_experimental_records():
    experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record

def find_manipulation_days(experiment_records, mice):
    experiments = experiment_records[experiment_records['mouse_id'].isin(mice)]
    exemption_list = ['psychometric', 'state change medium cloud', 'value blocks', 'state change white noise',
                      'omissions and large rewards', 'contingency switch', 'ph3', 'saturation', 'centre port hold']
    exemptions = '|'.join(exemption_list)
    index_to_remove = experiments[experiments['experiment_notes'].str.contains(exemptions,na=False)].index
    mouse_dates = experiments.loc[index_to_remove][['mouse_id', 'date']].reset_index(drop=True)
    reformatted_dates = pd.to_datetime(mouse_dates['date'])
    mouse_dates['date'] = reformatted_dates
    return mouse_dates


def remove_exps_after_manipulations(experiments, mice):
    manipulation_days = find_manipulation_days(experiments, mice)
    for mouse in manipulation_days['mouse_id'].unique():
        all_manipulation_days = manipulation_days[manipulation_days['mouse_id'] == mouse]
        earliest_manipulation_day = all_manipulation_days.min()

        index_to_remove = experiments[np.logical_and((experiments['mouse_id'] == mouse),
                                                     (pd.to_datetime(experiments['date']) >= earliest_manipulation_day['date']))].index
        if index_to_remove.shape[0] > 0:
            dates_being_removed = experiments['date'][index_to_remove].unique()
            experiments = experiments.drop(index=index_to_remove)
            print('removing {}: {}'.format(mouse, dates_being_removed))
    return experiments


def remove_manipulation_days(experiments):
    exemption_list = ['psychometric', 'state change medium cloud', 'value blocks', 'state change white noise', 'omissions and large rewards']
    exemptions = '|'.join(exemption_list)
    index_to_remove = experiments[np.logical_xor(experiments['include'] == 'no', experiments['experiment_notes'].str.contains(exemptions, na=False))].index
    cleaned_experiments = experiments.drop(index=index_to_remove)
    return cleaned_experiments


def remove_bad_recordings(experiments):
    index_to_remove = experiments[experiments['include'] == 'no'].index
    cleaned_experiments = experiments.drop(index=index_to_remove)
    print(index_to_remove)
    return cleaned_experiments

def open_experiment(experiment_to_add):
    for index, experiment in experiment_to_add.iterrows():
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + experiment['mouse_id'] + '\\'
        #saving_folder = 'C:\\Users\\francescag\\Documents\\PhD_Project\\SNL_photo_photometry\\processed_data' + experiment['mouse_id'] + '\\'
        restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)
        session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
    return session_traces, trial_data


def open_one_experiment(experiment):
    ## takes a row of a dataframe as input
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + experiment['mouse_id'] + '\\'
    restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
    dff_trace_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'smoothed_signal.npy'
    dff = np.load(saving_folder + dff_trace_filename)
    session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
    return trial_data, session_traces
