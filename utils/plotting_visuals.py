import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def makes_plots_pretty(axs):
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')



