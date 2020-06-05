from utils.plotting import heat_map_and_mean
import pickle
from utils.individual_trial_analysis_utils import SessionData

mouse = 'SNL_photo20'
dates = ['20200225']
fiber_side = 'left'
for date in dates:
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse + '\\'
    choice_aligned_filename = mouse + '_' + date + '_' + 'choice_aligned_traces.p'
    save_filename = saving_folder + choice_aligned_filename
    session_data = pickle.load(open(save_filename, "rb"))
    choice_data = session_data.choice_data
    heat_map_and_mean(choice_data)