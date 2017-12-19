import os
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import load_model, Model
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

class StackedEarlyStopping(callbacks.EarlyStopping):
    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.stacked_weights = 0
        temp_weights = os.listdir('temp/')
        for file in temp_weights:
            os.remove('temp/{}'.format(file))

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        self.total_epoch = epoch
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return

        if self.stacked_weights >= self.patience:
            os.remove('temp/weight0.h5')
            for i in range(1, self.stacked_weights):
                os.rename('temp/weight{}.h5'.format(i), 'temp/weight{}.h5'.format(i-1))
            self.stacked_weights -= 1
        self.model.save_weights('temp/weight{}.h5'.format(self.stacked_weights))
        self.stacked_weights += 1

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
            if self.stacked_weights > 0:
                self.model.load_weights('temp/weight0.h5')
                if self.verbose > 0:
                    print('Restore weights to epoch %05d' % (self.stopped_epoch + 1 - self.stacked_weights))
        elif self.wait > 0:
            self.model.load_weights('temp/weight{}.h5'.format(self.stacked_weights - self.wait))
            if self.verbose > 0:
                print('Restore weights to epoch %05d' % (self.total_epoch + 1 - self.wait))

        temp_weights = os.listdir('temp/')
        for file in temp_weights:
            os.remove('temp/{}'.format(file))


class LSTMConfig(object):
    def __init__(self):
        self.net_dir = 'net/'
        self.img_dir = 'img/'
        self.EPOCH_SIZE = 30
        self.BATCH_SIZE = 32
        self.EARLY_STOP_MIN_DELTA = 0
        self.EARLY_STOP_PATIENCE = 5
        self.INIT_LEARNING_RATE = 0.0001

class BatchedLSTM(object):
    def fit(self, generator, dl, date_name=''):
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
        early_stop = StackedEarlyStopping(monitor='val_loss', min_delta=conf.EARLY_STOP_MIN_DELTA, patience=conf.EARLY_STOP_PATIENCE, verbose=1, mode='auto')

        model.compile(loss='mean_squared_error', optimizer=adam)
        model.fit([dl.x_train_pre,dl.x_train_his], dl.y_train, batch_size=conf.BATCH_SIZE, epochs=conf.EPOCH_SIZE, validation_data=(
            [dl.x_test_pre,dl.x_test_his], dl.y_test), verbose=1, callbacks=[loss_history, early_stop])

        print('Generator{} training finish!'.format(generator+1))
        if early_stop.stopped_epoch > 0:
            print('Finishing training loss: {0:f}, valid loss: {1:f}'.format(
                loss_history.train_losses[early_stop.stopped_epoch - early_stop.patience], loss_history.valid_losses[early_stop.stopped_epoch - early_stop.patience]))
        else:
            print('Finishing training loss: {0:f}, valid loss: {1:f}'.format(
                loss_history.train_losses[- 1 - early_stop.wait], loss_history.valid_losses[- 1 - early_stop.wait]))

        save_name = date_name + '_{0:s}_ps{1:d}_hs{2:d}_e{3:d}_b{4:d}_lr{5:.5f}_emd{6:.3f}_ep{7:d}_model{8:d}.h5'.format('batched', pre_time_steps,his_time_steps, conf.EPOCH_SIZE, conf.BATCH_SIZE, conf.INIT_LEARNING_RATE, conf.EARLY_STOP_MIN_DELTA, conf.EARLY_STOP_PATIENCE, generator+1)
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

        model = load_model('{0:s}{1:s}{2:d}.h5'.format(conf.net_dir,net_name[:-4],generator+1))

        return model.predict([dl.x_test_pre,dl.x_test_his])
