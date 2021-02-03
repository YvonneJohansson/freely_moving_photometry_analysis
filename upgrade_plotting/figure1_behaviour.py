import sys
sys.path.append('C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')
from matplotlib import cm
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utils.behavioural_utils.upgrade_fig1_utils import prep_data_for_learning_curve, discrimination_final_session, last_session_discrimination_plot, get_stimulus_examples, plot_spectrograms
from utils.plotting_visuals import makes_plots_pretty, get_continuous_cmap

# Makes figure 1 for upgrade
data_set = 'tail_and_acc_animals_new'  # This dataset has been pre-processed, but conditions have not been selected
ans_to_remove = []
max_trials = 10000
df_for_plot = prep_data_for_learning_curve(data_set, ans_to_remove)
final_session_data = discrimination_final_session(df_for_plot)
df_for_plot_limited = df_for_plot[df_for_plot['CumulativeTrialNumberByProtocol'] < max_trials]

final_session_data = discrimination_final_session(df_for_plot)

# plotting
font = {'size': 8}
matplotlib.rc('font', **font)
fig = plt.figure(constrained_layout=True, figsize=[5, 5.5])
gs = fig.add_gridspec(nrows=6, ncols=4)
task_diagram_ax = fig.add_subplot(gs[:4, 0:2])
task_diagram_ax.set_title('A', loc='left', fontweight='bold')
cloud_of_tones_low_ax = fig.add_subplot(gs[0:2, 2])
cloud_of_tones_low_ax.set_title('B', loc='left', fontweight='bold')
cloud_of_tones_high_ax = fig.add_subplot(gs[0:2, 3])
expert_mice_ax = fig.add_subplot(gs[2:4, 2:])
expert_mice_ax.set_title('C', loc='left', fontweight='bold')
learning_curves_ax = fig.add_subplot(gs[4:, :])
learning_curves_ax.set_title('D', loc='left', fontweight='bold')
#raw_data_ax = inset_axes(learning_curves_ax, width='40%', height='50%', loc='lower right',  borderpad=0.4)
makes_plots_pretty(fig.axes)

stim_file = 'W:\\photometry_2AC\\bpod_data\\SNL_photo22\\Two_Alternative_Choice\\Session Data\\SNL_photo22_Two_Alternative_Choice_20200818_135227'
ans_list = np.sort(df_for_plot_limited.AnimalID.unique())
num_ans = len(ans_list)
colours = cm.viridis(np.linspace(0, 0.8, num_ans))
low_cloud, high_cloud = get_stimulus_examples(stim_file)
last_session_discrimination_plot(expert_mice_ax, final_session_data)
plot_spectrograms(low_cloud, cloud_of_tones_low_ax, high_cloud, cloud_of_tones_high_ax)

sns.lineplot(data=df_for_plot_limited,
                x="CumulativeTrialNumberByProtocol",
                y='CurrentPastPerformance100',
                hue='AnimalID',
                legend=False,
                palette='PuBuGn_r',
                alpha=0.3,
                lw=0.8,
                ax=learning_curves_ax)

sns.lineplot(data=df_for_plot_limited,
                x="CumulativeTrialNumberByProtocol",
                y='CurrentPastPerformance100',
                ci=None,
                color='#2D606E',
                alpha=1,
                lw=1.5,
                ax=learning_curves_ax)

learning_curves_ax.axis('on')
learning_curves_ax.set_ylabel('Cumulative % correct')
learning_curves_ax.set_xlabel('Trial number')
learning_curves_ax.set_xlim([0, max_trials])
learning_curves_ax.set_yticks([0, 20, 40, 60, 80, 100])

# raw_data_ax.get_legend().remove()
# raw_data_ax.axis('on')
# raw_data_ax.set_xlim([0, max_trials])
# raw_data_ax.yaxis.set_label_text("")
# raw_data_ax.xaxis.set_label_text("")
# raw_data_ax.set_yticks([0, 20, 40, 60, 80, 100])
# raw_data_ax.set_xticklabels([])

#plt.savefig(data_directory + 'Performance_by_session_individual_animals.pdf', transparent=True, bbox_inches='tight')
plt.show()
