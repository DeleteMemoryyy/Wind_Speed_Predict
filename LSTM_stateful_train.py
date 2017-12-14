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

if __name__ == '__main__':
    history_dir = 'data/preprocessed_data/history_data/scattered_speed/'
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

    EPOCH_SIZE = 1
    BATCH_SIZE = 1
    TIME_STEPS = 1
    VALID_SIZE = 960
    TEST_SIZE = 960

    PRINT_GAP = 1
    SAVE_GAP = 2

    early_stop = False
    early_stop_flag = False

    INIT_LEARNING_RATE = 0.0001

    LSTM_STATEFUL = True
    LSTM_UNITS = 1024
    LSTM_INPUT_DROPOUT = 0.2
    LSTM_RECURRENT_DROPOUT = 0.4

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
        y_data = real_speed

        sample_num, input_dim = x_data.shape
        output_dim = y_data.shape[1]

        train_num = sample_num - VALID_SIZE - TEST_SIZE
        model = Sequential()
        model.add(LSTM(units=LSTM_UNITS,batch_input_shape=(1,1,input_dim),activation='tanh',stateful=True))
        model.add(Dense(512))
        model.add(Dense(128))
        model.add(Dense(output_dim))
        adam = Adam(INIT_LEARNING_RATE)
        model.compile(loss='mean_squared_error',optimizer=adam)

        x_data = x_data.reshape(sample_num,1,input_dim)

        x_train = x_data[:train_num]
        y_train = y_data[:train_num]
        x_valid = x_data[train_num:train_num + VALID_SIZE]
        y_valid = y_data[train_num:train_num + VALID_SIZE]
        x_test = x_data[train_num + VALID_SIZE:]
        y_test = y_data[train_num + VALID_SIZE:]

        class LossHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.epoch_losses = []

            def on_epoch_begin(self, batch, logs={}):
                self.batch_losses = []

            def on_batch_end(self, batch, logs={}):
                self.last_loss = logs.get('loss')
                self.batch_losses.append(logs.get('loss'))

            def on_epoch_end(self, epoch, logs={}):
                batch_num = len(self.batch_losses)
                loss_of_eposh = np.sqrt(np.array(self.batch_losses).sum()/batch_num)
                self.epoch_losses.append(loss_of_eposh)

        loss_history = LossHistory()
        time_s = time.time()
        time_u = time_s
        recent_loss = 100
        train_err_list = []
        valid_err_list = []
        learning_time = []
        for epoch in range(EPOCH_SIZE):
            model.reset_states()
            model.fit(x_train,y_train,epochs=1,batch_size=1,verbose=1,shuffle=False,callbacks=[loss_history])

            time_c = time.time()
            # if ((epoch % PRINT_GAP) == 0):
            print('Process = {0:.3f}%'.format(
                (float(epoch + PRINT_GAP) / float(EPOCH_SIZE)) * 100.0))
            print('Training time = {0:.3f}s, this round = {1:.3f}s, remaining time = {2:.3f}s'.format(
                time_c - time_s, time_c - time_u, (time_c - time_u) * (EPOCH_SIZE - epoch - 1) / float(PRINT_GAP)))
            print('Training rmse = {0:.4f}'.format(loss_history.epoch_losses[-1]))
            time_u = time_c
                # if (not((epoch % SAVE_GAP) == 0)):
                    # print('')

            # if ((epoch % SAVE_GAP) == 0):
            #     model_valid = model
            #     y_valid_out=np.zeros(shape=(VALID_SIZE,output_dim))
            #     for sample_idx in range(VALID_SIZE):
            #         x_valid_temp = x_valid[sample_idx:sample_idx+1]
            #         y_valid_temp = y_valid[sample_idx:sample_idx+1]
            #         y_valid_out[sample_idx] = model_valid.predict(x_valid_temp, batch_size=1)
            #         model_valid.fit(x_valid_temp,y_valid_temp, epochs=1, batch_size=1, verbose=0, shuffle=False)

            #     this_loss = np.sqrt(metrics.mean_squared_error(y_valid.flatten(),y_valid_out.flatten()))
            #     train_err_list.append(loss_history.epoch_losses[-1])
            #     valid_err_list.append(this_loss)
            #     learning_time.append(epoch + 1)
            #     print('Validating rmse = {0:.4f}'.format(this_loss))
                # if(early_stop and epoch > EPOCH_SIZE * stop_process and recent_loss <= stop_mse
                # and this_loss > (recent_loss * (1 + tolerancing))):
                #     early_stop_flag = True
                #     epoch -= SAVE_GAP
                #     print('Early stop at epoch: {0:d}/{1:d}\n'.format(epoch,EPOCH_SIZE))
                #     break
                # recent_loss = this_loss
                # model_valid.save('{0:s}{1:s}'.format(net_dir, temp_save_name))
                # print('Temporary save to {0:s}{1:s} at {2:d}/{3:d}'.format(net_dir,temp_save_name,epoch + 1,EPOCH_SIZE))
                # print('')

        print('Training finish!')
        print('Finishing loss: {0:f}'.format(loss_history.epoch_losses[-1]))

        generator_dir = '{0:s}/generator{1:d}/'.format(net_dir,generator+1)
        existed_nets = os.listdir(generator_dir)
        counter = 0
        for e_net in existed_nets:
            if (os.path.isfile(generator_dir+e_net) and e_net.endswith('_net_model.h5')):
                counter += 1
        save_name = '{0:d}_{1:.4f}_{2:d}_stateful_net_model.h5'.format(
            counter, INIT_LEARNING_RATE, epoch + 1)
        if(early_stop_flag):
            os.rename('{0:s}{1:s}'.format(generator_dir, temp_save_name),
                    '{0:s}{1:s}'.format(generator_dir, save_name))
        else:
            model.fit(x_valid,y_valid,epochs=1,batch_size=1,verbose=1)
            model.save('{0:s}{1:s}'.format(generator_dir, save_name))
            print('Save to: {0:s}{1:s}'.format(generator_dir, save_name))

        # plot loss history
        # plt.plot(learning_time, train_err_list, 'r-')
        # plt.plot(learning_time, valid_err_list, 'b--')
        # plt.savefig('img/generator{0:d}{1:s}_loss.png'.format(generator+1,save_name[:-3]))
