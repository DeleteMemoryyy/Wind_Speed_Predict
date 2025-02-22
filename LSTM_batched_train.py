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
from keras import callbacks

history_dir = 'data/preprocessed_data/history_data/accurate_biased_average/'
predict_dir ='data/preprocessed_data/predict_data/'
net_dir = 'net/'
temp_save_name = 'temp_model.h5'

if (not os.path.exists(net_dir)):
    os.mkdir(net_dir)

history_data_name = ['machine{}_accurate.csv'.format(i) for i in range(1, 7)]
std_history_data_name = ['std_machine{}_accurate.csv'.format(i) for i in range(1, 7)]
std_predict_data_name = ['std_machine{}_predict.csv'.format(i) for i in range(1, 7)]

drop_columns = ['month','day','second','speed_out']

days_of_month = {3:31,4:30,5:31,6:30,7:31}
train_date = (4,6)
valid_date = (6,21)
test_data = (7,10)
end_date = (7,31)

EPOCH_SIZE = 80
BATCH_SIZE = 32
TIME_STEPS = 32
TEST_SIZE = 1920

EARLY_STOP_MIN_DELTA = -0.08
EARLY_STOP_PATIENCE = 3

INIT_LEARNING_RATE = 0.00006

LSTM_1_UNITS = 128
LSTM_2_UNITS = 256
LSTM_INPUT_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.2
DENSE_1_UNIT = 128

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
    x_train = x_data[: -TEST_SIZE]
    y_train = y_data[: -TEST_SIZE]
    x_test = x_data[-TEST_SIZE:]
    y_test = y_data[-TEST_SIZE:]

    model = Sequential()
    model.add(LSTM(units=LSTM_1_UNITS,input_shape=(TIME_STEPS,input_dim),return_sequences=True,dropout=LSTM_INPUT_DROPOUT,recurrent_dropout=LSTM_RECURRENT_DROPOUT))
    model.add(LSTM(units=LSTM_2_UNITS,input_shape=(TIME_STEPS,LSTM_1_UNITS),dropout=LSTM_INPUT_DROPOUT,recurrent_dropout=LSTM_RECURRENT_DROPOUT))
    model.add(Dense(output_dim,activation='relu'))
    adam = Adam(INIT_LEARNING_RATE,decay=INIT_LEARNING_RATE*0.8/EPOCH_SIZE)
    model.compile(loss='mean_squared_error',optimizer=adam)

    class LossHistory(callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.train_losses = []
            self.valid_losses = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(logs.get('loss'))
            self.valid_losses.append(logs.get('val_loss'))

    loss_history = LossHistory()
    early_stop = callbacks.EarlyStopping(monitor='val_loss',min_delta=EARLY_STOP_MIN_DELTA,patience=EARLY_STOP_PATIENCE,verbose=1,mode='auto')

    model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCH_SIZE,validation_data=(x_test,y_test),verbose=1,callbacks=[loss_history,early_stop])

    print('Training finish!')
    print('Finishing training loss: {0:f}, valid loss: {1:f}'.format(loss_history.train_losses[-1], loss_history.valid_losses[-1]))

    generator_dir = '{0:s}generator{1:d}/'.format(net_dir,generator+1)
    if (not os.path.exists(generator_dir)):
        os.mkdir(generator_dir)
    save_name = '{0:s}_e{1:d}_b{2:d}_s{3:d}_emd{4:.3f}_ep{5:d}_model.h5'.format(
        'batched', EPOCH_SIZE, BATCH_SIZE, TIME_STEPS,EARLY_STOP_MIN_DELTA,EARLY_STOP_PATIENCE)
    model.save('{0:s}{1:s}'.format(generator_dir, save_name))
    print('Save to: {0:s}{1:s}'.format(generator_dir, save_name))

    train_rmse = np.sqrt(np.array(loss_history.train_losses)).tolist()
    valid_rmse = np.sqrt(np.array(loss_history.valid_losses)).tolist()
    plot_data = pd.DataFrame({'train_rmse':train_rmse,'valid_rmse':valid_rmse})
    plot_data.plot()
    plt.savefig('img/generator{0:d}/{1:s}.png'.format(generator+1,save_name[:-3]))