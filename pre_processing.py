
# Add modules to the path
import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import nptdms
import copy
import numpy as np
import scipy.signal
import peakutils
from itertools import chain
import pandas as pd
from utils.demodulation import lerner_deisseroth_preprocess, demodulate
import utils.bpod_data_processing as bpod
import pylab as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize


mouse = 'BLA_photo2'
date = '20191011'
data = nptdms.TdmsFile("W://photometry_2AC//freely_moving_photometry_data//BLA_photo2/20191011_12_53_04/\AI.tdms")

main_session_file = bpod.find_bpod_file(mouse, date)
loaded_bpod_file, trial_raw_events = bpod.load_bpod_file(main_session_file)


chan_0 = data.group_channels('acq_task')[0].data
led405 = data.group_channels('acq_task')[2].data
led465 = data.group_channels('acq_task')[1].data
clock = data.group_channels('acq_task')[3].data
stim_trigger = data.group_channels('acq_task')[4].data
stim_trigger_gaps = np.diff(stim_trigger)
trial_start_ttls_daq_samples = np.where(stim_trigger_gaps > 2.6)
trial_start_ttls_daq = trial_start_ttls_daq_samples[0]/10000
daq_num_trials = trial_start_ttls_daq.shape[0]
bpod_num_trials = trial_raw_events.shape[0]
if daq_num_trials != bpod_num_trials:
    print('numbers of trials do not match!')
    print('daq: ', daq_num_trials)
    print('bpod: ', bpod_num_trials)
else: print(daq_num_trials, 'trials in session')

signal, back = demodulate(chan_0, led465, led405)

sampling_rate = 10000
GCaMP_raw = signal[sampling_rate:]
TdTom_raw = back[sampling_rate:]

time_seconds = np.arange(GCaMP_raw.shape[0])/sampling_rate

# Median filtering to remove electrical artifact.
GCaMP_denoised = medfilt(GCaMP_raw, kernel_size=5)
TdTom_denoised = medfilt(TdTom_raw, kernel_size=5)

# Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
b, a = butter(2, 10, btype='low', fs=sampling_rate)
GCaMP_denoised = filtfilt(b, a, GCaMP_denoised)
TdTom_denoised = filtfilt(b, a, TdTom_denoised)

b,a = butter(2, 0.001, btype='high', fs=sampling_rate)
GCaMP_highpass = filtfilt(b,a, GCaMP_denoised, padtype='even')
TdTom_highpass = filtfilt(b,a, TdTom_denoised, padtype='even')

slope, intercept, r_value, p_value, std_err = linregress(x=TdTom_highpass, y=GCaMP_highpass)

GCaMP_est_motion = intercept + slope * TdTom_highpass
GCaMP_corrected = GCaMP_highpass - GCaMP_est_motion

b,a = butter(2, 0.001, btype='low', fs=sampling_rate)
baseline_fluorescence = filtfilt(b,a, GCaMP_denoised, padtype='even')

GCaMP_dF_F = GCaMP_corrected/baseline_fluorescence
b,a = butter(2, 2, btype='low', fs=sampling_rate)
smoothed_GCaMP =  filtfilt(b,a, GCaMP_dF_F, padtype='even')

restructured_data = bpod.restructure_bpod_timestamps(loaded_bpod_file, trial_start_ttls_daq)

