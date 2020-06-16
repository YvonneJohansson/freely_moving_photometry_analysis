from utils.plotting import heat_map_and_mean
import pickle
import numpy as np
import pandas as pd
from utils.individual_trial_analysis_utils import SessionData
from utils.mean_trace_utils import mouseDates, plot_multiple_days, plot_average_mouse
import matplotlib.pyplot as plt

experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
example_mouse = 'SNL_photo18'
example_date = '20200223'

mice_dates = []
mouse1 = 'SNL_photo16'
dates1 = '20200210'
mice_dates.append(mouseDates(mouse1, dates1))
mouse2 = 'SNL_photo17'
dates2 = '20200204'
mice_dates.append(mouseDates(mouse2, dates2))
mouse3 = 'SNL_photo18'
dates3 = '20200223'
mice_dates.append(mouseDates(mouse3, dates3))

saving_folder = 'W:\\photometry_2AC\\processed_data\\' + example_mouse + '\\'
choice_aligned_filename = example_mouse + '_' + example_date + '_' + 'aligned_traces.p'
save_filename = saving_folder + choice_aligned_filename
example_session_data = pickle.load(open(save_filename, "rb"))
example_cue_data = example_session_data.cue_data

contra_mean_traces = []
ipsi_mean_traces = []

for mouse_dates in mice_dates:
    mouse = mouse_dates.mouse
    date = mouse_dates.dates
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    aligned_filename = saving_folder + mouse + '_' + date + '_' + 'aligned_traces.p'
    session_data = pickle.load(open(aligned_filename, "rb"))
    cue_data = session_data.cue_data
    contra_mean_traces.append(cue_data.contra_data.mean_trace)
    ipsi_mean_traces.append(cue_data.ipsi_data.mean_trace)
average_ipsi = np.mean(np.array(ipsi_mean_traces), axis=0)
average_contra = np.mean(np.array(contra_mean_traces), axis=0)

figure = heat_map_and_mean(example_cue_data, average_ipsi, average_contra, sort=True, error_bar_method='sem', mean_across_mice=True)
plt.savefig('W:\\upgrade\\figure3_plots.pdf', transparent=True, optimize=True)
plt.show()

