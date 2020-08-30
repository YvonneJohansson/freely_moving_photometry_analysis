from session_traces_and_mean import get_all_experimental_records
import matplotlib.pyplot as plt
import pickle
from matplotlib import colors, cm
import numpy as np


def plot_mean_traces(experiments_to_add, ipsi_axis, contra_axis, align_to = 'choice'):
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
            contra = reward_data.contra_data.mean_trace
            ipsi = reward_data.ipsi_data.mean_trace
        elif align_to == 'choice':
            contra = choice_data.contra_data.mean_trace
            ipsi = choice_data.ipsi_data.mean_trace
        elif align_to == 'cue':
            contra = cue_data.contra_data.mean_trace
            ipsi = cue_data.ipsi_data.mean_trace

        ipsi_axis.plot(choice_data.ipsi_data.time_points, ipsi, color=colours[index])
        ipsi_axis.set_xlim([-2, 2])
        ipsi_axis.set_xlabel('Time (s)')
        ipsi_axis.axvline(0, color='k')
        ipsi_axis.set_ylabel('zscore')
        contra_axis.plot(choice_data.contra_data.time_points, contra, color=colours[index])
        contra_axis.set_xlim([-2, 2])
        contra_axis.axvline(0, color='k')
        contra_axis.set_ylabel('zscore')

def remove_experiments(experiments, ones_to_remove):
    for mouse in ones_to_remove.keys():
        for date in ones_to_remove[mouse]:
            index_to_remove = experiments[(experiments['mouse_id'] == mouse) & (experiments['date'] == date)].index[0]
            experiments = experiments.drop(index=index_to_remove)
    return experiments



if __name__ == '__main__':
    mouse_ids = ['SNL_photo26']
    date = 'all'
    experiments_to_remove = {'SNL_photo21': ['20200805'], 'SNL_photo25': ['20200812']}
    all_experiments = get_all_experimental_records()
    all_experiments = remove_experiments(all_experiments, experiments_to_remove)
    align_to = 'choice'
    recording_site = 'tail'

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

        if recording_site == 'all':
            fig, axs = plt.subplots(nrows=2, ncols=num_sites, sharey=True)
            for site_num, site in enumerate(recording_sites):
                recording_site_experiments = experiments_to_process[experiments_to_process['recording_site'] == site]
                if num_sites > 1:
                    plot_mean_traces(recording_site_experiments, axs[1, site_num], axs[0, site_num], align_to=align_to)

                else:
                    plot_mean_traces(recording_site_experiments, axs[1], axs[0], align_to=align_to)
        else:
            fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True)
            recording_site_experiments = experiments_to_process[experiments_to_process['recording_site'] == recording_site]
            plot_mean_traces(recording_site_experiments, axs[1], axs[0], align_to=align_to)

        fig.suptitle(mouse_id)
    plt.show()