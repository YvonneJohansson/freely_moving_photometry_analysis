import sys
sys.path.append('C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')
import matplotlib
import matplotlib.pylab as plt
from utils.regression.regression_plotting_utils import *
from utils.plotting_visuals import makes_plots_pretty
import matplotlib.gridspec as gridspec

font = {'size': 8}
matplotlib.rc('font', **font)

fig = plt.figure(constrained_layout=True, figsize=[8, 6.5])
gs = fig.add_gridspec(nrows=3, ncols=5)

model_description_ax1 = fig.add_subplot(gs[0, 0:2])
model_description_ax1.set_title('A', loc='left', fontweight='bold')
model_description_ax2 = fig.add_subplot(gs[1, 0:2])
model_description_ax2.set_title('C', loc='left', fontweight='bold')


ts_cue_ax = fig.add_subplot(gs[0, 2])
ts_cue_ax.set_title('B', loc='left', fontweight='bold')
ts_move_ax = fig.add_subplot(gs[0, 3])
ts_rew_ax = fig.add_subplot(gs[0, 4])

vs_cue_ax = fig.add_subplot(gs[1, 2], sharey=ts_cue_ax)
vs_cue_ax.set_title('D', loc='left', fontweight='bold')
vs_move_ax = fig.add_subplot(gs[1, 3], sharey=ts_move_ax)
vs_rew_ax = fig.add_subplot(gs[1, 4], sharey=ts_rew_ax)

total_perc_exp = fig.add_subplot(gs[2, 0:2])
total_perc_exp.set_title('E', loc='left', fontweight='bold')
perc_exp_cue_ax = fig.add_subplot(gs[2, 2])
perc_exp_cue_ax.set_title('F', loc='left', fontweight='bold')
perc_exp_move_ax = fig.add_subplot(gs[2, 3], sharey=perc_exp_cue_ax)
perc_exp_rew_ax = fig.add_subplot(gs[2, 4], sharey=perc_exp_cue_ax)

tail_time_stamps, tail_reg_means, tail_reg_sems = get_regression_data_for_plot(recording_site='tail')
nacc_time_stamps, nacc_reg_means, nacc_reg_sems = get_regression_data_for_plot(recording_site='Nacc')

plot_kernels_for_site(ts_move_ax, ts_cue_ax, ts_rew_ax, tail_reg_means, tail_reg_sems, tail_time_stamps, legend=True)
plot_kernels_for_site(vs_move_ax, vs_cue_ax, vs_rew_ax, nacc_reg_means, nacc_reg_sems, nacc_time_stamps, legend=True)

nacc_exp_var = load_exp_var_data_for_site('Nacc')
tail_exp_var = load_exp_var_data_for_site('tail')


full_df = get_data_both_sites_for_predictor(nacc_exp_var, tail_exp_var, 'full')
cue_df = get_data_both_sites_for_predictor(nacc_exp_var, tail_exp_var, 'cue')
choice_df = get_data_both_sites_for_predictor(nacc_exp_var, tail_exp_var, 'choice')
outcome_df = get_data_both_sites_for_predictor(nacc_exp_var, tail_exp_var, 'outcome')

make_box_plot(full_df, total_perc_exp, set_ylims=True, label='Full model')
make_box_plot(cue_df, perc_exp_cue_ax,label='Cue')
make_box_plot(choice_df, perc_exp_move_ax, label='Choice')
make_box_plot(outcome_df, perc_exp_rew_ax, label='Outcome')

makes_plots_pretty([ts_move_ax, ts_cue_ax, ts_rew_ax,vs_move_ax, vs_cue_ax, vs_rew_ax,  total_perc_exp, perc_exp_cue_ax, perc_exp_move_ax, perc_exp_rew_ax])
make_example_figure(model_description_ax1, model_description_ax2)
plt.show()