import os
import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
import numpy as np
import pandas as pd
from utils.value_change_utils import open_experiment, CustomAlignedDataRewardBlocks, get_all_experimental_records, get_block_change_info, add_traces_and_peaks, plot_mean_trace_for_condition, one_session_get_block_changes

exp_name = 'value_change'
processed_data_dir = os.path.join(os.getcwd(), 'value_change_data')
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

all_experiments = get_all_experimental_records()
block_types = pd.DataFrame({'block type': [1, 2, 3, 4, 5], 'left reward': [6, 4, 2, 2, 2], 'right reward': [2, 2, 2, 4, 6]})

mice =['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35'] #['SNL_photo21', 'SNL_photo22', 'SNL_photo26'] for tail
sessions = ['20201219'] #['20200917', '20200918', '20200921'] for tail

block_data_file = os.path.join(processed_data_dir, 'block_data_nacc_mice.csv')

if os.path.isfile(block_data_file):
    all_reward_block_data = pd.read_pickle(block_data_file)
else:
    for mouse_num, mouse_id in enumerate(mice):
        for session_idx, date in enumerate(sessions):
            experiment_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
            behavioural_data, session_data = open_experiment(experiment_to_process)
            for reward_block in range(1,6):
                one_reward_block_data = {}
                print(reward_block)
                try:
                    params = {'state_type_of_interest': 3, # 5 for tail
                        'outcome': 2,
                        'last_outcome': 0,  # NOT USED CURRENTLY
                        'no_repeats' : 1,
                        'last_response': 0,
                        'align_to' : 'Time start',
                        'instance':- 1,
                        'plot_range': [-6, 6],
                        'first_choice_correct': 0,
                         'cue': 'None'}
                    reward_block_data = CustomAlignedDataRewardBlocks(session_data, params, reward_block)
                    contra_side = reward_block_data.contra_fiber_side_numeric
                    trial_nums = reward_block_data.contra_data.trial_nums
                    blocks = np.ones([len(trial_nums)]) * reward_block
                    session_nums = np.ones([len(trial_nums)]) * session_idx
                    block_traces = reward_block_data.contra_data.sorted_traces
                    list_traces = [block_traces[i,:] for i in range(block_traces.shape[0])]

                    if contra_side == 2:
                        new_reward_amount = block_types[block_types['block type'] == reward_block]['right reward'].values[0]
                        new_other_amount = block_types[block_types['block type'] == reward_block]['left reward'].values[0]
                    else:
                        new_reward_amount = block_types[block_types['block type'] == reward_block]['left reward'].values[0]
                        new_other_amount = block_types[block_types['block type'] == reward_block]['right reward'].values[0]
                    current_reward_amounts = np.ones([len(trial_nums)]) * new_reward_amount
                    relative_values = np.ones([len(trial_nums)]) * (new_reward_amount - new_other_amount)

                    one_reward_block_data['block number'] = blocks
                    one_reward_block_data['trial number'] = trial_nums
                    one_reward_block_data['peak size'] = reward_block_data.contra_data.trial_peaks
                    one_reward_block_data['relative reward amount'] = relative_values

                    one_reward_block_dataf = pd.DataFrame(one_reward_block_data)
                    one_reward_block_dataf['session'] = pd.Series([date] *
                                                                     len(list_traces), index=one_reward_block_dataf.index)
                    one_reward_block_dataf['mouse'] = pd.Series([mouse_id] *
                                                                        len(list_traces), index=one_reward_block_dataf.index)
                    one_reward_block_dataf['contra reward amount'] = current_reward_amounts
                    one_reward_block_dataf['traces'] = pd.Series(list_traces, index=one_reward_block_dataf.index)
                    one_reward_block_dataf['time points'] = pd.Series([reward_block_data.contra_data.time_points] *
                                                                     len(list_traces), index=one_reward_block_dataf.index)
                    if (reward_block > 1) or (session_idx > 0) or (mouse_num > 0):
                        all_reward_block_data = pd.concat([all_reward_block_data, one_reward_block_dataf], ignore_index=True)
                    else:
                        all_reward_block_data = one_reward_block_dataf

                except:
                    pass
    all_reward_block_data.to_pickle(block_data_file)

timepoints = all_reward_block_data['time points'].iloc[0]

all_reward_block_data['Experiment'] = exp_name

#plot_mean_trace_for_condition(all_reward_block_data[all_reward_block_data['mouse'] == 'SNL_photo28'], timepoints,
#                              'contra reward amount', error_bar_method='ci', save_location=processed_data_dir)

#plot_mean_trace_for_condition(all_reward_block_data[all_reward_block_data['mouse'] == 'SNL_photo28'], timepoints,
#                              'relative reward amount', error_bar_method = 'ci', save_location=processed_data_dir)