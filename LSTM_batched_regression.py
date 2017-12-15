import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing as prep
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import Input, LSTM, TimeDistributed, Dense, Activation
from keras.optimizers import Adam

history_dir = 'data/preprocessed_data/history_data/accurate_biased_average/'
predict_dir ='data/preprocessed_data/predict_data/'
net_dir = 'net/'
net_name = 'batched_e30_b32_s32_net_model.h5'

history_data_name = ['machine{}_accurate.csv'.format(i) for i in range(1, 7)]
std_history_data_name = ['std_machine{}_accurate.csv'.format(i) for i in range(1, 7)]
std_predict_data_name = ['std_machine{}_predict.csv'.format(i) for i in range(1, 7)]

drop_columns = ['month','day','second','speed_out']

days_of_month = {3:31,4:30,5:31,6:30,7:31}
train_date = (4,6)
valid_date = (6,21)
test_data = (7,10)
end_date = (7,31)

TIME_STEPS = 32
TEST_SIZE = 1920

result_data = []
result_rmse = []

for generator in range(6):

    label_data = pd.read_csv(history_dir + history_data_name[generator])[['month','day','second','speed']]
    label_idx_st = label_data[(label_data['month'] == train_date[0]) & (label_data['day'] == train_date[1]) & (label_data['second'] == 0)].index[0]
    label_data = label_data[:][label_idx_st:]
    label_data.index = range(label_data.shape[0])
    real_speed = label_data['speed'].values.reshape(-1,60)
    history_data = pd.read_csv(history_dir + std_history_data_name[generator])
    predict_data = pd.read_csv(predict_dir + std_predict_data_name[generator])
    predict_data = predict_data[:][4*24*2:]
    predict_data.index = range(predict_data.shape[0])
    history_speed = history_data['speed'].values.reshape(-1,60)[:][:-4*24*2]
    train_data = pd.concat((predict_data,pd.DataFrame(history_speed,columns=['h_speed_{}'.format(i) for i in range(60)])),axis=1)

    x_data = train_data.drop(drop_columns,axis=1).values
    y_data = real_speed[TIME_STEPS:]

    sample_num = x_data.shape[0] - TIME_STEPS
    input_dim = x_data.shape[1]
    output_dim = y_data.shape[1]

    windows_data = []
    for i in range(sample_num):
        windows_data.append(x_data[i : i + TIME_STEPS])
    x_data = np.array(windows_data)
    x_test = x_data[-TEST_SIZE:]
    y_test = y_data[-TEST_SIZE:]

    generator_dir = '{0:s}/generator{1:d}/'.format(net_dir,generator+1)
    model = load_model(generator_dir+net_name)
    y_test_out = model.predict(x_test).reshape((-1,))
    result_data.append(y_test_out.tolist())
    result_rmse.append(np.sqrt(metrics.mean_squared_error(y_test.reshape((-1,)),y_test_out)))

result_data = np.transpose(np.array(result_data))
result = pd.DataFrame(result_data,columns=['G{}'.format(i) for i in range(1,7)])
print('LSTM_batched_test_rmse: {}'.format(np.array(result_rmse).mean()))
save_name = 'result/result_{}_LSTM_batched.csv'.format(time.strftime('%b_%d_%H-%M-%S',time.localtime()))
result.to_csv(save_name,header=True,index=False,encoding='utf-8')
print('save to {}\n'.format(save_name))

