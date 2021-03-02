import pickle
import numpy as np
import pandas as pd
from utils.regression.linear_regression_utils import *

mice_dates = pd.DataFrame ({'mouse': [ 'SNL_photo17', 'SNL_photo35',], 'date': [ '20200206', '20201121',], 'site': [ 'tail', 'Nacc'], 'inds':[np.arange(39100, 39300) ,np.arange(112450, 112650)]})
fig, axs = plt.subplots(2, 1)
for ind, mouse_date in mice_dates.iterrows():
    mouse = mouse_date['mouse']
    date = mouse_date['date']
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    save_filename = saving_folder + mouse + '_' + date + '_'
    kernel_filename = save_filename + 'linear_regression_kernels_different_shifts.p'
    inputs_X_filename = save_filename + 'linear_regression_different_shifts_X.p'
    inputs_y_filename = save_filename + 'linear_regression_different_shifts_y.p'
    params_filename = save_filename + 'linear_regression_parameters.p'
    X = pickle.load(open(inputs_X_filename, 'rb'))
    y = pickle.load(open(inputs_y_filename, 'rb'))
    params = pickle.load(open(params_filename, 'rb'))
    kernels = pickle.load(open(kernel_filename, 'rb'))
    kernel_list = []
    param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards']
    for param_name in param_names:
        kernel = kernels['kernels'][param_name]
        kernel_list.append(kernel)
    coefs = np.array([item for sublist in kernel_list for item in sublist])
    intercept = kernels['intercept']
    shifts = kernels['shifts']
    windows = kernels['shift_window_lengths']
    all_shifts = [shifts[param_name] for param_name in param_names]
    all_windows = [windows[param_name] for param_name in param_names]
    params_to_remove = ['high cues', 'low cues']
    cue_pred, _ = remove_param_and_calculate_r2(param_names, params_to_remove, coefs, X,
                                                              intercept, y, all_shifts, all_windows, remove=False)
    params_to_remove = ['ipsi choices', 'contra choices']
    choice_pred, _ = remove_param_and_calculate_r2(param_names, params_to_remove, coefs, X,
                                                              intercept, y, all_shifts, all_windows, remove=False)
    params_to_remove = ['rewards', 'no rewards']
    outcome_pred, _ = remove_param_and_calculate_r2(param_names, params_to_remove, coefs, X,
                                                   intercept, y, all_shifts, all_windows, remove=False)
    i = mouse_date['inds']
    full_model = np.dot(X, coefs) + intercept
    #axs[0].plot(y[inds])
    #axs[0].plot(full_model[inds])
    axs[ind].plot(y[i])
    axs[ind].plot(choice_pred[i], label='choice')
    axs[ind].plot(outcome_pred[i], label='outcome')
    axs[ind].plot(cue_pred[i], label='cue')
    axs[ind].legend()
    axs[ind].plot(params[4][i], color='r')
    axs[ind].plot(params[3][i], color='k')
    axs[ind].plot(params[0][i], color='y')
    axs[ind].plot(params[1][i], color='m')
    save_out_filename = saving_folder + mouse + '_' + date + '_example.npz'
    np.savez(save_out_filename, choice_pred=choice_pred[i], outcome_pred=outcome_pred[i], cue_pred=cue_pred[i], dff=y[i],
                                contra_choices=params[3][i], rewards=params[4][i], high_cues=params[0][i], low_cues=params[1][i])
plt.show()