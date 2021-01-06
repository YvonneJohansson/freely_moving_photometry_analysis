from data_preprocessing.session_traces_and_mean import get_all_experimental_records
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm
import numpy as np
from utils.plotting import calculate_error_bars
import matplotlib
from scipy.signal import decimate


def plot_mean_traces(experiments_to_add, axis, side='contra', align_to = 'choice', error_bar_method=None):
    num_types = experiments_to_add.shape[0]
    experiments_to_add = experiments_to_add.reset_index()
    colours = cm.viridis(np.linspace(0, 0.8, num_types))
    for index, experiment in experiments_to_add.iterrows():
        mouse_id = experiment['mouse_id']
        date = experiment['date']
        print(mouse_id, date)
        saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse_id + '\\'
        aligned_filename = mouse_id + '_' + date + '_' + 'aligned_traces.p'
        save_filename = saving_folder + aligned_filename
        session_data = pickle.load(open(save_filename, "rb"))
        choice_data = session_data.choice_data
        reward_data = session_data.reward_data
        cue_data = session_data.cue_data

        if align_to == 'reward':
            if side == 'contra':
                data = reward_data.contra_data
                mean_trace = decimate(data.mean_trace, 10)
                sorted_traces = decimate(data, 10)
            elif side == 'ipsi':
                data = reward_data.ipsi_data
                mean_trace = decimate(data.mean_trace, 10)
                sorted_traces = decimate(data, 10)
            else:
                data = np.concatenate((reward_data.contra_data.sorted_traces, reward_data.ipsi_data.sorted_traces))
                sorted_traces = decimate(data, 10)
                mean_trace = np.mean(sorted_traces, 0)

        elif align_to == 'choice':
            if side == 'contra':
                data = choice_data.contra_data
                mean_trace = decimate(data.mean_trace, 10)
                sorted_traces = decimate(data.sorted_traces, 10)
            elif side == 'ipsi':
                data = choice_data.ipsi_data
                mean_trace = decimate(data.mean_trace, 10)
                sorted_traces = decimate(data.sorted_traces, 10)
            else:
                data = np.concatenate((choice_data.contra_data.sorted_traces, choice_data.ipsi_data.sorted_traces))
                sorted_traces = decimate(data, 10)
                mean_trace = np.mean(sorted_traces, 0)
        elif align_to == 'cue':
            if side == 'contra':
                data = cue_data.contra_data
                mean_trace = decimate(data.mean_trace, 10)
                sorted_traces = decimate(data.sorted_traces, 10)
            elif side == 'ipsi':
                data = cue_data.ipsi_data
                mean_trace = decimate(data.mean_trace, 10)
                sorted_traces = decimate(data.sorted_traces, 10)
            else:
                data = np.concatenate((cue_data.contra_data.sorted_traces, cue_data.ipsi_data.sorted_traces))
                sorted_traces = decimate(data, 10)
                mean_trace = np.mean(sorted_traces, 0)


        time_points = decimate(choice_data.ipsi_data.time_points, 10)
        axis.plot(time_points, mean_trace, color=colours[index])
        axis.set_xlim([-1, 2])
        axis.set_xlabel('Time (s)')
        axis.axvline(0, color='k')
        axis.set_ylabel('zscore')
        #axis.set_ylim([-1, 2])

        if error_bar_method is not None:
            error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                    sorted_traces,
                                                                    error_bar_method=error_bar_method)
            axis.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                             facecolor=colours[index], linewidth=0)


def remove_experiments(experiments, ones_to_remove):
    for mouse in ones_to_remove.keys():
        for date in ones_to_remove[mouse]:
            index_to_remove = experiments[(experiments['mouse_id'] == mouse) & (experiments['date'] == date)].index[0]
            experiments = experiments.drop(index=index_to_remove)
    return experiments



if __name__ == '__main__':
    mouse_ids = ['SNL_photo31']
    date = 'all'
    experiments_to_remove = {'SNL_photo21': ['20200805'], 'SNL_photo25': ['20200812'], 'SNL_photo31':  ['20201211', '20201214','20201216', '20201218', '20201219', '20201221', '20201222']}
    all_experiments = get_all_experimental_records()
    all_experiments = remove_experiments(all_experiments, experiments_to_remove)
    align_to = 'cue'
    recording_site = 'Nacc'
    side = ''

    for mouse_id in mouse_ids:
        if (mouse_id == 'all') & (date == 'all'):
            experiments_to_process = all_experiments
        elif (mouse_id == 'all') & (date != 'all'):
            experiments_to_process = all_experiments[all_experiments['date'] == date]
        elif (mouse_id != 'all') & (date == 'all'):
            experiments_to_process = all_experiments[all_experiments['mouse_id'] == mouse_id]
        elif (mouse_id != 'all') & (date != 'all'):
            experiments_to_process = all_experiments[
                (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]

        recording_sites = experiments_to_process['recording_site'].unique()
        num_sites = recording_sites.shape[0]

        font = {'size': 14}
        matplotlib.rc('font', **font)

        if recording_site == 'all':
            fig, axs = plt.subplots(nrows=2, ncols=num_sites, sharey=True)
            for site_num, site in enumerate(recording_sites):
                recording_site_experiments = experiments_to_process[experiments_to_process['recording_site'] == site]
                if num_sites > 1:
                    plot_mean_traces(recording_site_experiments, axs[1, site_num], axs[0, site_num], align_to=align_to)

                else:
                    plot_mean_traces(recording_site_experiments, axs[1], axs[0], align_to=align_to)
        elif side != 'both':
            fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
            recording_site_experiments = experiments_to_process[experiments_to_process['recording_site'] == recording_site]
            plot_mean_traces(recording_site_experiments, axs[0], side='contra', align_to=align_to, error_bar_method='sem')
            plot_mean_traces(recording_site_experiments, axs[1], side='ipsi', align_to=align_to, error_bar_method='sem')
        else:
            fig, axs = plt.subplots(nrows=1, ncols=1, sharey=True)
            recording_site_experiments = experiments_to_process[experiments_to_process['recording_site'] == recording_site]
            plot_mean_traces(recording_site_experiments, axs, side='both', align_to=align_to, error_bar_method='sem')


        #fig.suptitle(mouse_id)
    plt.show()