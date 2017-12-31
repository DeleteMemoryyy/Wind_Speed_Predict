import os
import numpy as np
import pandas as pd
from sklearn import preprocessing as prep

nearest_dir = 'data/preprocessed_data/history_data/scattered_speed/'
average_dir = 'data/preprocessed_data/history_data/accurate_biased_average/'
generated_dir = 'data/preprocessed_data/history_data/our_prediction_to_labels/'

history_data_name = ['machine{}_accurate.csv'.format(i) for i in range(1, 7)]

TEST_SIZE = 1920

# nearest_result_data = np.zeros((TEST_SIZE*60,6))
# nearest_result_data[:,:] = 3.0

# for generator in range(6):
#     history_data = pd.read_csv(nearest_dir+history_data_name[generator])
#     day1_speed=history_data['speed'][(history_data['month']==7) & (history_data['day']==11)].values
#     nearest_result_data[:day1_speed.shape[0],generator] = day1_speed
#     day2_speed = history_data['speed'][(history_data['month'] == 7) &  (history_data['day'] == 12)].values
#     nearest_result_data[day1_speed.shape[0]:day1_speed.shape[0]+day2_speed.shape[0], generator] = day2_speed

# nearest_result = pd.DataFrame(nearest_result_data,columns=['G{}'.format(i) for i in range(1,7)])
# nearest_result.to_csv('result/result_nearest.csv',index=False)

# average_result_data = np.zeros((TEST_SIZE * 60, 6))
# average_result_data[:, :] = 3.0

# for generator in range(6):
#     history_data = pd.read_csv(average_dir + history_data_name[generator])
#     day1_speed = history_data['speed'][(history_data['month'] == 7) & (
#         history_data['day'] == 11)].values
#     average_result_data[:day1_speed.shape[0], generator] = day1_speed
#     day2_speed = history_data['speed'][(history_data['month'] == 7) & (
#         history_data['day'] == 12)].values
#     average_result_data[day1_speed.shape[0]:day1_speed.shape[0] +
#                         day2_speed.shape[0], generator] = day2_speed

generated_result_data = np.zeros((TEST_SIZE * 60, 6))
generated_result_data[:, :] = 3.0

for generator in range(6):
    history_data = pd.read_csv(generated_dir + history_data_name[generator])
    day1_speed = history_data['speed'][(history_data['month'] == 7) & (
        history_data['day'] == 11)].values
    generated_result_data[:day1_speed.shape[0], generator] = day1_speed
    day2_speed = history_data['speed'][(history_data['month'] == 7) & (
        history_data['day'] == 12)].values
    generated_result_data[day1_speed.shape[0]:day1_speed.shape[0] +
                        day2_speed.shape[0], generator] = day2_speed


generated_result = pd.DataFrame(generated_result_data, columns=[
                              'G{}'.format(i) for i in range(1, 7)])
generated_result.to_csv('result/result_generated.csv',index=False)
