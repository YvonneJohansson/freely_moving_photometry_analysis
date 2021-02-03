import os
import sys
sys.path.append("C:\\Users\\francescag\Documents\\SourceTree_repos\\Python_git")
from utils.behavioural_utils import sigmoid_fitting
from utils.behavioural_utils import plotting
from matplotlib import cm
import ntpath
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import math
import seaborn as sns
import pandas as pd
import warnings
#from itertools import chain
from scipy import stats
import scipy.optimize as opt
import random
from datetime import date
from sklearn.linear_model import LinearRegression


def sigmoid_func(x_val, perf_end, slope, bias):
    return (perf_end - 0.5) / (1 + np.exp(-slope * (x_val - bias))) + 0.5


def sigmoid_func_sc(x_val, perf_end, slope, bias):
    return (perf_end - 50) / (1 + np.exp(-slope * (x_val - bias))) + 50


def der_sig(x_val, perf_end, slope, bias):
    return (perf_end - 50) * slope * np.exp(- slope * (x_val - bias)) / (1 + np.exp(- slope * (x_val - bias)))**2


def fit_sigmoids_to_behavioural_data(df_to_plot, max_num_trials=18000):
    df_bin200tr = df_to_plot.groupby(['AnimalID','TrialIndexBinned200','Protocol']).median().reset_index()
    mouse_max_perf = df_bin200tr.groupby('AnimalID').max().reset_index()[['AnimalID', 'CurrentPastPerformance50']]

    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)

    fit_dir = {}
    xmeans_list = []
    xsd_list = []

    for counter, animal in enumerate(ans_list):
        df = df_to_plot[df_to_plot.AnimalID == animal][
            ['CumulativeTrialNumberByProtocol',
             'CurrentPastPerformance50',
             'SessionID']
        ].dropna()

        xdata = np.array(df.CumulativeTrialNumberByProtocol)
        ydata = np.array(df.CurrentPastPerformance50)

        xdatasc = (xdata - xdata.mean()) / xdata.std()
        ydatasc = ydata / 100

        mp = mouse_max_perf[mouse_max_perf.AnimalID == animal].CurrentPastPerformance50.iloc[0] / 100

        cost_func = lambda x: np.mean(np.abs(sigmoid_func(xdatasc, x[0], x[1], x[2]) - ydatasc))
        res = opt.minimize(cost_func, [1, 0, 0], bounds=((0.5, mp), (0., 10.), (None, None)))

        fit_dir[animal] = res
        xmeans_list.append(xdata.mean())
        xsd_list.append(xdata.std())

    for key, value in fit_dir.items():
        print(key, '  ', value.x)

    fit_df = pd.DataFrame({
        'AnimalID': list(fit_dir.keys()),
        'maximum_performance': [v.x[0] for k, v in fit_dir.items()],
        'slope': [v.x[1] for k, v in fit_dir.items()],
        'bias': [v.x[2] for k, v in fit_dir.items()]
    })
    fit_df.maximum_performance = fit_df.maximum_performance * 100
    fit_df.slope = fit_df.slope / np.array(xsd_list)
    fit_df.bias = fit_df.bias * np.array(xsd_list) + np.array(xmeans_list)


    der_max_dir = {}
    for animal in ans_list:
        m_point = opt.fmin(lambda x_val: -der_sig(x_val, *[fit_df[fit_df.AnimalID == animal].maximum_performance.iloc[0],
                                                           fit_df[fit_df.AnimalID == animal].slope.iloc[0],
                                                           fit_df[fit_df.AnimalID == animal].bias.iloc[0]]), 0,
                           full_output=True)

        der_max_dir[animal] = (m_point[0][0], -m_point[1])


    x_val = np.linspace(1, max_num_trials)

    for animal_number, animal in enumerate(ans_list):
        y = sigmoid_func_sc(x_val, *[fit_df[fit_df.AnimalID==animal].maximum_performance.iloc[0],
                                                fit_df[fit_df.AnimalID==animal].slope.iloc[0],
                                                fit_df[fit_df.AnimalID==animal].bias.iloc[0]])
        data = pd.DataFrame({'x': x_val, 'y': y, 'AnimalID':animal})
        if animal_number > 0:
            all_animal_sigmoids = pd.concat([all_animal_sigmoids, data])
        else:
            all_animal_sigmoids = data
    # add the maximum value of the derivative to the fit_df dataframe
    for key, value in der_max_dir.items():
        fit_df.loc[fit_df.index[fit_df['AnimalID'] == key].tolist()[0], 'max_of_der'] = value[1]
    return all_animal_sigmoids, fit_df
