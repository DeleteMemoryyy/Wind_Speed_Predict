import os
import numpy as np
import pandas as pd
from sklearn import preprocessing as prep

# history_dir = 'data/preprocessed_data/history_data/scattered_speed/'
history_dir = 'data/preprocessed_data/history_data/accurate_biased_average/'
predict_dir = 'data/preprocessed_data/predict_data/'

history_data_name = ['machine{}_accurate.csv'.format(i) for i in range(1, 7)]
predict_data_name = ['machine{}_predict.csv'.format(i) for i in range(1, 7)]

days_of_month = {3:31,4:30,5:31,6:30,7:31}

train_date = (4,4)
valid_date = (6,21)
test_data = (7,10)
end_date = (7,31)

weight = {1:[1.0],2:[0.4,0.6],3:[0.1,0.3,0.6],4:[0.1,0.2,0.3,0.4]}
ori_columns = ['month', 'day', 'second', 'NT_speed', 'P_out', 'speed_out', 'u_out',
               'v_out', 'w_out']
reshape_columns = ['NT_speed', 'P_out', 'speed_out', 'u_out',
 'v_out', 'w_out']

# for generator in range(6):
#     predict_data = pd.read_csv(predict_dir + predict_data_name[generator])

#     new_predict_data = pd.DataFrame([],columns=ori_columns)
#     for month in range(4,8):
#         day = 1
#         if month == 4:
#             day = 4
#         while day <= days_of_month[month]:
#             for sec in range(0, 24 * 4 * 900, 900):
#                 new_sec_data = pd.DataFrame(np.array([month, day, sec, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1,9),columns=ori_columns)
#                 np.array([[month, day, sec, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
#                 sec_data = predict_data[(predict_data['month']==month) & (predict_data['day'] == day) & (predict_data['second']==sec)]
#                 item_num = sec_data.shape[0]
#                 assert item_num >= 0 and item_num <=4
#                 sec_data.index = range(item_num)
#                 for col in reshape_columns:
#                     for i in range(item_num):
#                         new_sec_data.loc[0,col] += sec_data[col][i] * weight[item_num][i]
#                 new_predict_data = pd.concat((new_predict_data,new_sec_data))
#             print(generator + 1,month,day)
#             day += 1

#     new_predict_data.index = range(new_predict_data.shape[0])

#     new_predict_data['month'] = new_predict_data['month'].astype('int')
#     new_predict_data['day'] = new_predict_data['day'].astype('int')
#     new_predict_data['second'] = new_predict_data['second'].astype('int')

#     fillin_st = new_predict_data[(new_predict_data['month'] == 6) & (new_predict_data['day'] == 9) & (new_predict_data['second'] == ((16 * 4 - 1) * 900))].index[0]
#     fillin_ed = new_predict_data[(new_predict_data['month'] == 6) & (new_predict_data['day'] == 9) & (new_predict_data['second'] == ((18 * 4) * 900))].index[0]

#     for i in range(1, 9):
#         for col in reshape_columns:
#             new_predict_data.loc[fillin_st + i, col] = new_predict_data[col][fillin_st] * (
#                 9 - i) / 9 + new_predict_data[col][fillin_ed] * i / 9

#     print('generator {} procession finish.'.format(generator + 1))
#     new_predict_data.to_csv(predict_dir+'re_'+predict_data_name[generator],index=False,float_format='%.4f')
#     print('generator {} file saved.'.format(generator + 1))

# for generator in range(6):
#     new_predict_data = pd.read_csv(predict_dir+'re_'+predict_data_name[generator])

#     fillin_st = new_predict_data[(new_predict_data['month'] == 6) & (
#         new_predict_data['day'] == 9) & (new_predict_data['second'] == ((16 * 4 - 1) * 900))].index[0]
#     fillin_ed = new_predict_data[(new_predict_data['month'] == 6) & (new_predict_data['day'] == 9) & (new_predict_data['second'] == ((18 * 4) * 900))].index[0]

#     for i in range(1, 9):
#         for col in reshape_columns:
#             new_predict_data.loc[fillin_st + i, col]=new_predict_data[col][fillin_st] * (9 - i) / 9 + new_predict_data[col][fillin_ed] * i / 9

#     new_predict_data.to_csv(predict_dir + 're_' + predict_data_name[generator], index=False, float_format='%.4f')

# for generator in range(6):
#     new_predict_data = pd.read_csv(predict_dir + 're_' + predict_data_name[generator])
#     new_predict_data = new_predict_data[((new_predict_data['month'] > 4) & (new_predict_data['month'] < 7)) | (
#         (new_predict_data['month'] == 4) & (new_predict_data['day'] >= 4)) | ((new_predict_data['month'] == 7) & (new_predict_data['day'] <= 30))]
#     ss_x = prep.StandardScaler()
#     new_predict_data[reshape_columns] = ss_x.fit_transform(new_predict_data[reshape_columns].values)
#     new_predict_data.to_csv(predict_dir + 'std_' + predict_data_name[generator], index=False, float_format='%.4f')

for generator in range(6):
    new_history_data = pd.read_csv(history_dir + history_data_name[generator])[['month','day','second','speed']]
    new_history_data = new_history_data[((new_history_data['month'] > 4) & (new_history_data['month'] < 7)) | (
        (new_history_data['month'] == 4) & (new_history_data['day'] >= 4)) | ((new_history_data['month'] == 7) & (new_history_data['day'] <= 30))]
    new_history_data.index = range(new_history_data.shape[0])
    ss_x = prep.StandardScaler()
    new_history_data['speed'] = ss_x.fit_transform(new_history_data['speed'].values.reshape(-1,1))
    new_history_data.to_csv(
        history_dir + 'std_' + history_data_name[generator], index=False, float_format='%.4f')
