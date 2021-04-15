import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')
from scipy.interpolate import interp1d
from utils.plotting import calculate_error_bars
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import os
from matplotlib.lines import Line2D
import matplotlib
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import  *
font = {'size': 8}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(2, 2, constrained_layout=True)
tail_mice = ['SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26']
make_change_over_time_plot(tail_mice, ax[1, 0])
ax[1, 0].set_title('C', loc='left', fontweight='bold')
nacc_mice = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
make_change_over_time_plot(nacc_mice, ax[1, 1])
ax[1, 1].set_title('D', loc='left', fontweight='bold')

make_example_traces_plot('SNL_photo17', ax[0, 0])
ax[0, 0].set_title('A', loc='left', fontweight='bold')
make_example_traces_plot('SNL_photo35', ax[0, 1])
ax[0, 1].set_title('B', loc='left', fontweight='bold')
makes_plots_pretty([ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]])
plt.show()