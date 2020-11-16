from utils.post_processing_utils import *
import os
processed_data_dir = os.path.join('W:\\photometry_2AC\\processed_data\\state_change_data')
state_change_data_file = os.path.join(processed_data_dir, 'state_change_data_all_mice.csv')

mice = ['SNL_photo21', 'SNL_photo22', 'SNL_photo26']
for mouse_num, mouse_id in enumerate(mice):
    state_change_data = {}
    date = '20200915'
    all_experiments = get_all_experimental_records()
    experiment_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
    session_data = open_experiment(experiment_to_process)

    params = {'state_type_of_interest': 5,
        'outcome': 2,
        'last_outcome': 0,  # NOT USED CURRENTLY
        'no_repeats' : 0,
        'last_response': 0,
        'align_to' : 'Time start',
        'instance': -1,
        'plot_range': [-6, 6],
        'first_choice_correct': 0,
         'cue': 'None'}
    test = CustomAlignedData(session_data, params)

    state_change_data['trial number'] = test.contra_data.trial_nums
    state_change_data['peak size'] = test.contra_data.trial_peaks
    state_change_dataFrame = pd.DataFrame(state_change_data)
    list_traces = [test.contra_data.sorted_traces[i,:] for i in range(test.contra_data.sorted_traces.shape[0])]
    state_change_dataFrame['traces'] = pd.Series(list_traces, index=state_change_dataFrame.index)

    state_change_dataFrame['trial type'] = np.where(state_change_dataFrame['trial number'] < 150, 'pre', 'post')
    state_change_dataFrame['mouse'] = mouse_id
    if mouse_num > 0:
        all_state_change_data = pd.concat([all_state_change_data, state_change_dataFrame], ignore_index=True)
    else:
        all_state_change_data = state_change_dataFrame

all_state_change_data.to_pickle(state_change_data_file)


