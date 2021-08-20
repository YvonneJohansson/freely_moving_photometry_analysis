import sys
sys.path.append('C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')
import numpy as np
import pandas as pd
import matplotlib
from utils.behavioural_utils import hernando_custom_functions as cuf
from utils.post_processing_utils import get_all_experimental_records, remove_manipulation_days
import matplotlib.pyplot as plt
import seaborn as sns
import bpod_open_ephys_analysis.utils.load_nested_structs as load_ns
from utils.post_processing_utils import find_manipulation_days


def remove_manipulation_days(df_to_plot, manipulation_days):
    for _,session_to_remove in manipulation_days.iterrows():
        mouse = session_to_remove['mouse_id']
        date = session_to_remove['date'].strftime('%b%d')
        mouse_date = '{} {}'.format(mouse, date)
        cleaned_df_to_plot = df_to_plot[~df_to_plot.SessionID.str.contains(mouse_date)]
    return cleaned_df_to_plot


def prep_data_for_learning_curve(DS_name, ans_to_remove, protocols_selected=['Auditory'], num_trials_to_remove_at_start=15):
    all_records = get_all_experimental_records()
    data_directory = cuf.get_data_folder() + 'photometry_2AC\\bpod_data\\' + DS_name + '_Analysis\\'

    df_name = DS_name + '_dataframe.pkl'
    df_to_plot = pd.read_pickle(data_directory + df_name)
    df_to_plot = df_to_plot[~df_to_plot.AnimalID.isin(ans_to_remove)]
    mice = df_to_plot.AnimalID.unique()
    manipulation_days = find_manipulation_days(all_records, mice)

    df_to_plot = df_to_plot[df_to_plot.Protocol.isin(protocols_selected)]


    df_to_plot = df_to_plot[df_to_plot['TrialIndex'] > num_trials_to_remove_at_start]
    first_poke_correct_frame = df_to_plot['FirstPokeCorrect'].to_frame()
    df_to_plot.FirstPokeCorrect = first_poke_correct_frame.fillna(value=0)
    df_to_plot = df_to_plot[df_to_plot['Contingency'] == 1]
    df_to_plot["TrialIndexBinned200"] = (df_to_plot.CumulativeTrialNumberByProtocol // 200) * 200 + 100
    if 'Auditory' in protocols_selected:
        for mouse in manipulation_days['mouse_id'].unique():
            all_manipulation_days = manipulation_days[manipulation_days['mouse_id'] == mouse]
            earliest_manipulation_day = all_manipulation_days.min()
            index_to_remove = df_to_plot[np.logical_and((df_to_plot['AnimalID'] == mouse),
                                                        (df_to_plot['FullSessionTime'] >= earliest_manipulation_day['date']))].index
            if index_to_remove.shape[0] > 0:
                dates_being_removed = df_to_plot['FullSessionTime'][index_to_remove].unique()
                df_to_plot = df_to_plot.drop(index=index_to_remove)
                print('removing {}, {}'.format(mouse, dates_being_removed))
    return df_to_plot


def discrimination_final_session(df_for_plot):
    mice = df_for_plot.AnimalID.unique()
    percents = []
    side = []
    mice_for_df = []
    for mouse in mice:
        mouse_df = df_for_plot[df_for_plot['AnimalID'] == mouse]
        last_session = mouse_df.FullSessionTime.max()
        last_session_df = mouse_df[mouse_df['FullSessionTime'] == last_session]
        percent_left = find_percentage_choices_to_side(last_session_df, 1)
        percent_right = find_percentage_choices_to_side(last_session_df, 2)
        percents.append(percent_left)
        side.append('high')
        mice_for_df.append(mouse)
        percents.append(percent_right)
        side.append('low')
        mice_for_df.append(mouse)

    final_session = {'mouse': mice_for_df, 'tone cloud': side, '% right choices': percents}
    final_session_df = pd.DataFrame(final_session)
    return final_session_df


def get_psychometric_sessions(data_set):
    df_for_plot = prep_data_for_learning_curve(data_set, [], protocols_selected=['Aud_Psycho'])
    mice = df_for_plot.AnimalID.unique()
    proportions_all_mice = []
    for mouse in mice:
        mouse_df = df_for_plot[df_for_plot['AnimalID'] == mouse]
        proportions = mouse_df.groupby('TrialHighPerc')['FirstPoke'].value_counts(normalize=True) * 100
        prop_2 = proportions.xs(2.0, level=1)  # indexes 2.0 into the second level of index
        reindexed = prop_2.reset_index(drop=False)
        prop_df = pd.DataFrame(reindexed)
        prop_df['mouse'] = mouse
        proportions_all_mice.append(prop_df)
    proportions_df = pd.concat(proportions_all_mice, ignore_index=True)
    proportions_df['TrialHighPerc'] = 100 - proportions_df['TrialHighPerc']
    proportions_df = proportions_df.rename(
        columns={'TrialHighPerc': 'Percentage low tones', 'FirstPoke': 'Percentage rightwards choices'})
    return proportions_df



def find_percentage_choices_to_side(df_for_plot, side):
    num_side_correct = df_for_plot[np.logical_and(df_for_plot['FirstPoke'] == 2, df_for_plot['TrialSide'] == side)].shape[0]
    num_side = df_for_plot[df_for_plot['TrialSide'] == side].shape[0]
    percentage_side = num_side_correct / num_side * 100
    return percentage_side


def last_session_discrimination_plot(axs, df_for_plot):
    df_for_line_plot = df_for_plot.pivot(index='tone cloud', columns='mouse', values='% right choices').sort_values(
        'tone cloud')
    sns.stripplot(x='tone cloud', y='% right choices', data=df_for_plot, ax=axs, alpha=0.4, linewidth=0.1, jitter=0, zorder=1)
    df_for_line_plot.plot(ax=axs, color='gray', alpha=0.4, legend=False, zorder=0)
    sns.pointplot(x='tone cloud', y='% right choices', data=df_for_plot, ax=axs, alpha=1, linewidth=0.1, color='k',
                  scale=0.5, ci=None, zorder=2)


def get_stimulus_examples(file):
    """open file, find a high cloud trial and low cloud trial and extract the stimulus"""
    loaded_bpod_file = load_ns.loadmat(file)
    # as RawEvents.Trial is a cell array of structs in MATLAB, we have to loop through the array and convert the structs to dicts
    trial_stimulus = loaded_bpod_file['SessionData']['Stimulus']
    trial_side = loaded_bpod_file['SessionData']['TrialSide']
    first_low = np.where(trial_side == 2)[0][0]
    first_high = np.where(trial_side == 1)[0][0]
    low_cloud = trial_stimulus[first_low]
    high_cloud = trial_stimulus[first_high]
    return low_cloud, high_cloud


def plot_spectrograms(low_cloud, low_ax, high_cloud, high_ax):
    my_cmap = sns.light_palette('gray', as_cmap=True)
    #1000, 800, 2000, 192000
    high_ax.specgram(x=high_cloud, Fs=192000, NFFT=1000, noverlap=800, cmap=my_cmap)
    low_ax.specgram(x=low_cloud, Fs=192000, NFFT=1000, noverlap=800, cmap=my_cmap)
    low_ax.set_ylabel('Frequency (kHz)')
    low_ax.set_xlabel('Time (s)')
    high_ax.set_xlabel('Time (s)')
    high_ax.get_shared_x_axes().join(high_ax, low_ax)
    high_ax.get_shared_y_axes().join(high_ax, low_ax)
    ticks = matplotlib.ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y * 0.001))
    low_ax.yaxis.set_major_formatter(ticks)
    high_ax.set_yticklabels([])
    high_ax.set_ylim([0, 40000])
    high_ax.set_xlim([0, 0.3])





