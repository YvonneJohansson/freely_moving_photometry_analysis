import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
import os
from utils.regression.linear_regression_utils import *
import gc
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
mouse_ids = ['SNL_photo28', 'SNL_photo29', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
site = 'Nacc'

experiment_record = pd.read_csv('W:\\photometry_2AC\\experimental_record.csv')
experiment_record['date'] = experiment_record['date'].astype(str)
good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
clean_experiments = remove_bad_recordings(good_experiments)
all_experiments_to_process = clean_experiments[(clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(drop=True)
experiments_to_process = get_first_x_sessions(all_experiments_to_process)

file_name = site + '_explained_variances_refitted.p'
processed_data_dir = os.path.join('W:\\photometry_2AC\\processed_data\\linear_regression_data\\')
saving_filename = os.path.join('W:\\photometry_2AC\\processed_data\\linear_regression_data\\', file_name)
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

choice = []
cue = []
outcome = []
for index, experiment in experiments_to_process.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    print('proccessing' + mouse + date)
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    save_filename = saving_folder + mouse + '_' + date + '_'
    kernel_filename = save_filename + 'linear_regression_kernels_no_repeated_cues_both_cues.p'
    inputs_X_filename = save_filename + 'linear_regression_X.p'
    inputs_y_filename = save_filename + 'linear_regression_y.p'
    X = pickle.load(open(inputs_X_filename, 'rb'))
    y = pickle.load(open(inputs_y_filename, 'rb'))
    kernels = pickle.load(open(kernel_filename, 'rb'))
    kernel_list = []
    param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards']
    for param_name in param_names:
        kernel = kernels['kernels'][param_name]
        kernel_list.append(kernel)
    coefs = np.array([item for sublist in kernel_list for item in sublist])
    intercept = kernels['intercept']
    param_names = ['high_cues', 'low_cues', 'ipsi_choices', 'contra_choices', 'rewards', 'no_rewards']
    params_to_remove = ['high_cues', 'low_cues']
    cue_pred, prop_due_to_cue = remove_param_and_refit_r2(param_names, params_to_remove, coefs, X,
                                                                intercept, y)
    print('cue: ', prop_due_to_cue)
    params_to_remove = ['ipsi_choices', 'contra_choices']
    choice_pred, prop_due_to_choice = remove_param_and_refit_r2(param_names, params_to_remove, coefs, X,
                                                                intercept, y)
    print('choice: ', prop_due_to_choice)
    params_to_remove = ['rewards', 'no_rewards']
    reward_pred, prop_due_to_outcome = remove_param_and_refit_r2(param_names, params_to_remove, coefs, X,
                                                                intercept, y)
    print('outcome: ', prop_due_to_outcome)
    choice.append(prop_due_to_choice)
    cue.append(prop_due_to_cue)
    outcome.append(prop_due_to_outcome)
    gc.collect()
regression_stats = experiments_to_process[['mouse_id', 'date']].reset_index(drop=True)
regression_stats['cue explained variance'] = cue
regression_stats['choice explained variance'] = choice
regression_stats['outcome explained variance'] = outcome
with open(saving_filename, "wb") as f:
    pickle.dump(regression_stats, f)

