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
from utils.zscored_plots.zscored_plots_utils import plot_heat_map, get_data_for_recording_site, make_y_lims_same_heat_map, plot_all_heatmaps_same_scale
from utils.plotting_visuals import makes_plots_pretty

font = {'size': 8}
matplotlib.rc('font', **font)

fig = plt.figure(constrained_layout=True, figsize=[9, 9])
gs = fig.add_gridspec(nrows=9, ncols=4)


recording_diagram_ax = fig.add_subplot(gs[0:2, 0:2])
recording_diagram_ax.set_title('A', loc='left', fontweight='bold')
histology_ax = fig.add_subplot(gs[0:2, 2:])
histology_ax.set_title('B', loc='left', fontweight='bold')

ts_heatmap_contra_ax = fig.add_subplot(gs[2:4, 0])
ts_heatmap_contra_ax.set_title('C', loc='left', fontweight='bold')
ts_heatmap_ipsi_ax = fig.add_subplot(gs[2:4, 1])
ts_heatmap_rew_ax = fig.add_subplot(gs[2:4, 2])
ts_heatmap_unrew_ax = fig.add_subplot(gs[2:4, 3])

vs_heatmap_contra_ax = fig.add_subplot(gs[5:7, 0])
vs_heatmap_contra_ax.set_title('E', loc='left', fontweight='bold')
vs_heatmap_ipsi_ax = fig.add_subplot(gs[5:7, 1])
vs_heatmap_rew_ax = fig.add_subplot(gs[5:7, 2])
vs_heatmap_unrew_ax = fig.add_subplot(gs[5:7, 3])

ts_average_contra_ax = fig.add_subplot(gs[4:5, 0])
ts_average_contra_ax.set_title('D', loc='left', fontweight='bold')
ts_average_ipsi_ax = fig.add_subplot(gs[4:5, 1], sharey= ts_average_contra_ax)
ts_average_rew_ax = fig.add_subplot(gs[4:5, 2], sharey= ts_average_contra_ax)
ts_average_unrew_ax = fig.add_subplot(gs[4:5, 3], sharey= ts_average_contra_ax)

vs_average_contra_ax = fig.add_subplot(gs[7:8, 0], sharey= ts_average_contra_ax)
vs_average_contra_ax.set_title('F', loc='left', fontweight='bold')
vs_average_ipsi_ax = fig.add_subplot(gs[7:8, 1], sharey= ts_average_contra_ax)
vs_average_rew_ax = fig.add_subplot(gs[7:8, 2], sharey= ts_average_contra_ax)
vs_average_unrew_ax = fig.add_subplot(gs[7:8, 3], sharey= ts_average_contra_ax)

ts_heat_map_axs = {'contra': [ts_heatmap_contra_ax, ts_average_contra_ax],
                   'ipsi': [ts_heatmap_ipsi_ax, ts_average_ipsi_ax],
                   'rewarded': [ts_heatmap_rew_ax, ts_average_rew_ax],
                   'unrewarded': [ts_heatmap_unrew_ax, ts_average_unrew_ax]}
vs_heat_map_axs = {'contra': [vs_heatmap_contra_ax, vs_average_contra_ax],
                   'ipsi': [vs_heatmap_ipsi_ax, vs_average_ipsi_ax],
                   'rewarded': [vs_heatmap_rew_ax, vs_average_rew_ax],
                   'unrewarded': [vs_heatmap_unrew_ax, vs_average_unrew_ax]}

t_axs, t_data, t_wd, t_y_mins, t_y_maxs = get_data_for_recording_site('TS', ts_heat_map_axs)
v_axs, v_data, v_wd, v_y_mins, v_y_maxs = get_data_for_recording_site('VS', vs_heat_map_axs)

all_ymins = t_y_mins + v_y_mins
all_ymaxs = t_y_maxs + v_y_maxs
ymin, ymax = make_y_lims_same_heat_map(all_ymins, all_ymaxs)
cb_range = (ymin, ymax)

heat_map_t = plot_all_heatmaps_same_scale(fig, t_axs, t_data, t_wd, cb_range)
heat_map_v = plot_all_heatmaps_same_scale(fig, v_axs, v_data, v_wd, cb_range)
makes_plots_pretty([ts_average_contra_ax, ts_average_ipsi_ax, ts_average_rew_ax, ts_average_unrew_ax,
                    vs_average_contra_ax, vs_average_ipsi_ax, vs_average_rew_ax, vs_average_unrew_ax])
plt.tight_layout()

data_directory = 'W:\\upgrade\\'
plt.savefig(data_directory + 'Fig2_no_diagram.pdf', transparent=True, bbox_inches='tight')
plt.show()