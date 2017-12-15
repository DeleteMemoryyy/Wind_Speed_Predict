import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Concatenate, Input
from keras.optimizers import Adam
from keras import callbacks

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.valid_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.valid_losses.append(logs.get('val_loss'))

class LSTMConfig(object):
    def __init__(self):
        self.net_dir = 'net/'
        self.img_dir = 'img/'
        self.EPOCH_SIZE = 80
        self.BATCH_SIZE = 32
        self.EARLY_STOP_MIN_DELTA = -0.08
        self.EARLY_STOP_PATIENCE = 3
        self.INIT_LEARNING_RATE = 0.0006

class BatchedLSTM(object):
    def fit(self, generator, dl):
        assert dl.mode == 'lstm'

        pre_time_steps = dl.x_train_pre.shape[1]
        pre_input_dim = dl.x_train_pre.shape[2]
        his_time_steps = dl.x_train_his.shape[1]
        his_input_dim = dl.x_train_his.shape[2]
        output_dim = dl.y_train.shape[1]

        input_pre = Input(shape=(pre_time_steps,pre_input_dim))
        lstm_pre1 = LSTM(units=32 ,return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input_pre)
        lstm_pre2 = LSTM(units=64, input_shape=(pre_time_steps, 32), dropout=0.2, recurrent_dropout=0.2)(lstm_pre1)

        input_his = Input(shape=(his_time_steps,his_input_dim))
        lstm_his1 = LSTM(units=32 ,return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input_his)
        lstm_his2 = LSTM(units=64, input_shape=(his_time_steps, 32), dropout=0.2, recurrent_dropout=0.2)(lstm_his1)

        merged = Concatenate()([lstm_pre2, lstm_his2])
        out = Dense(output_dim, activation='relu')(merged)
        model = Model(inputs=[input_pre, input_his], outputs=out)

        conf = LSTMConfig()
        adam = Adam(conf.INIT_LEARNING_RATE,
                    decay=conf.INIT_LEARNING_RATE * 0.8 / conf.EPOCH_SIZE)
        loss_history = LossHistory()
        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=conf.EARLY_STOP_MIN_DELTA, patience=conf.EARLY_STOP_PATIENCE, verbose=1, mode='auto')

        model.compile(loss='mean_squared_error', optimizer=adam)
        model.fit([dl.x_train_pre,dl.x_train_his], dl.y_train, batch_size=conf.BATCH_SIZE, epochs=conf.EPOCH_SIZE, validation_data=(
            [dl.x_test_pre,dl.x_test_his], dl.y_test), verbose=1, callbacks=[loss_history, early_stop])

        print('Generator{} training finish!'.format(generator+1))
        print('Finishing training loss: {0:f}, valid loss: {1:f}'.format(
            loss_history.train_losses[-1], loss_history.valid_losses[-1]))

        save_name = '{0:s}_e{1:d}_b{2:d}_emd{3:.3f}_ep{4:d}_model{5:d}.h5'.format('batched', conf.EPOCH_SIZE, conf.BATCH_SIZE, conf.EARLY_STOP_MIN_DELTA, conf.EARLY_STOP_PATIENCE,generator+1)
        model.save('{0:s}{1:s}'.format(conf.net_dir, save_name))
        print('Save to: {0:s}{1:s}'.format(conf.net_dir, save_name))

        train_rmse = np.sqrt(np.array(loss_history.train_losses)).tolist()
        valid_rmse = np.sqrt(np.array(loss_history.valid_losses)).tolist()
        plot_data = pd.DataFrame({'train_rmse': train_rmse, 'valid_rmse': valid_rmse})
        plot_data.plot()
        plt.savefig('{0:s}{1:s}.png'.format(conf.img_dir, save_name[:-3]))

    def predict(self, generator, dl, net_name):
        assert dl.mode == 'lstm'

        conf = LSTMConfig()

        model = load_model('{0:s}{1:s}{2:d}.h5'.format(conf.net_dir,net_name[-4],generator+1))

        return model.predict([dl.x_test_pre,dl.x_test_his])
