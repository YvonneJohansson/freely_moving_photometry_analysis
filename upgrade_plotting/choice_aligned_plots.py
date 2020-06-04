import pickle
import numpy as np

file_name = 'W:\\photometry_2AC\\processed_data\\SNL_photo17\\SNL_photo17_20200208_mean_correct_data.p'

correct_data = pickle.load(open(file_name, "rb"))
mean_contra = correct_data.contra_mean_y_vals