import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')

import nptdms
import numpy as np
from data_preprocessing.demodulation import demodulate
import data_preprocessing.bpod_data_processing as bpod


from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress


mouse = 'SNL_photo26'
date = '20200812'
daq_file = bpod.find_daq_file(mouse, date)
data = nptdms.TdmsFile(daq_file)

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

sampling_rate = 10000
signal, back = demodulate(chan_0[sampling_rate * 6:], led465[sampling_rate * 6:], led405[sampling_rate * 6:], 10000)


GCaMP_raw = np.pad(signal, (6 * sampling_rate, 0), mode='median')
back_raw = np.pad(back, (6 * sampling_rate, 0), mode='median')


time_seconds = np.arange(GCaMP_raw.shape[0])/sampling_rate

# Median filtering to remove electrical artifact.
GCaMP_denoised = medfilt(GCaMP_raw, kernel_size=5)
back_denoised = medfilt(back_raw, kernel_size=5)
 
# Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
b,a = butter(2, 10, btype='low', fs=sampling_rate)
GCaMP_denoised = filtfilt(b,a, GCaMP_denoised)
back_denoised = filtfilt(b,a, back_denoised)


b,a = butter(2, 0.001, btype='high', fs=sampling_rate)
GCaMP_highpass = filtfilt(b,a, GCaMP_denoised, padtype='even')
back_highpass = filtfilt(b,a, back_denoised, padtype='even')

slope, intercept, r_value, p_value, std_err = linregress(x=back_highpass, y=GCaMP_highpass)

print('Slope    : {:.3f}'.format(slope))
print('R-squared: {:.3f}'.format(r_value**2))

GCaMP_est_motion = intercept + slope * back_highpass
GCaMP_corrected = GCaMP_highpass - GCaMP_est_motion


b,a = butter(2, 0.001, btype='low', fs=sampling_rate)
baseline_fluorescence = filtfilt(b,a, GCaMP_denoised, padtype='even')


GCaMP_dF_F = GCaMP_corrected/baseline_fluorescence
b,a = butter(2, 3, btype='low', fs=sampling_rate)
smoothed_GCaMP =  filtfilt(b,a, GCaMP_dF_F, padtype='even')

clock_diff = np.diff(clock)
clock_pulses = np.where(clock_diff > 2.6)[0]/10000

original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps']
daq_trials_start_ttls = trial_start_ttls_daq

restructured_data = bpod.restructure_bpod_timestamps(loaded_bpod_file, trial_start_ttls_daq, clock_pulses)

saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
demod_trace_filename = mouse + '_' + date + '_' + 'demod_signal.npy'
smoothed_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
background_filename = mouse + '_' + date + '_' + 'background.npy'
notdF_filename = mouse + '_' + date + '_' + 'not_regress.npy'
restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
#np.save(saving_folder + demod_trace_filename, GCaMP_dF_F)
np.save(saving_folder + smoothed_trace_filename, smoothed_GCaMP)
#np.save(saving_folder + background_filename, back_highpass)
#np.save(saving_folder + notdF_filename, GCaMP_highpass)
restructured_data.to_pickle(saving_folder + restructured_data_filename) 

