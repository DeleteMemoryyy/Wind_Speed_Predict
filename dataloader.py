import numpy as np
import pandas as pd

class DataConfig(object):
    def __init__(self, mode, PRE_TIME_STEPS=None, HIS_TIME_STEPS=None, HIS_TIME_OFFSET=None):
        self.mode = mode
        self.history_dir = 'data/preprocessed_data/history_data/accurate_biased_average/'
        self.predict_dir = 'data/preprocessed_data/predict_data/'
        self.history_data_name = ['machine{}_accurate.csv'.format(i) for i in range(1, 7)]
        self.std_history_data_name = [
            'std_machine{}_accurate.csv'.format(i) for i in range(1, 7)]
        self.std_predict_data_name = [
            'std_machine{}_predict.csv'.format(i) for i in range(1, 7)]
        self.start_date = (4,4)
        self.TEST_SIZE = 1920

        self.pre_drop_columns = ['month', 'day', 'second', 'speed_out']

        if mode == 'norm':
            self.PRE_TIME_STEPS = PRE_TIME_STEPS or 16
            self.HIS_TIME_STEPS = HIS_TIME_STEPS or 2
            self.HIS_TIME_OFFSET = HIS_TIME_OFFSET or 4*36 # no less than 4*36

        elif mode == 'lstm':
            self.PRE_TIME_STEPS = PRE_TIME_STEPS or 32
            self.HIS_TIME_STEPS = HIS_TIME_STEPS or 4
            self.HIS_TIME_OFFSET = HIS_TIME_OFFSET or 4*36 # no less than 4*36

class DataLoader(object):
    def __init__(self, generator, conf):
        self.mode = conf.mode

        history_data = pd.read_csv(
            conf.history_dir + conf.std_history_data_name[generator])
        predict_data = pd.read_csv(
            conf.predict_dir + conf.std_predict_data_name[generator])

        label_data = pd.read_csv(
            conf.history_dir + conf.history_data_name[generator])[['month', 'day', 'second', 'speed']]
        label_idx_st = label_data[(label_data['month'] == conf.start_date[0]) & (
            label_data['day'] == conf.start_date[1]) & (label_data['second'] == 0)].index[0]

        x_data_his = history_data['speed'].values.reshape(
            -1, 60)[:][:-conf.HIS_TIME_OFFSET]
        x_data_pre = predict_data.drop(conf.pre_drop_columns, axis=1)[
            :][conf.HIS_TIME_OFFSET + conf.HIS_TIME_STEPS - conf.PRE_TIME_STEPS:].values
        y_data = label_data['speed'][label_idx_st:].values.reshape(-1, 60)[conf.HIS_TIME_OFFSET + conf.HIS_TIME_STEPS:]

        sample_num = y_data.shape[0]
        assert sample_num == x_data_his.shape[0] - conf.HIS_TIME_STEPS and sample_num == x_data_pre.shape[0] - conf.PRE_TIME_STEPS

        windows_data = []
        for i in range(sample_num):
            windows_data.append(x_data_his[i: i + conf.HIS_TIME_STEPS])
        x_window_his = np.array(windows_data)
        windows_data = []
        for i in range(sample_num):
            windows_data.append(x_data_pre[i: i + conf.PRE_TIME_STEPS])
        x_window_pre = np.array(windows_data)

        if self.mode == 'norm':
            x_data = np.concatenate(
                (x_data_his, x_data_pre), axis=1).reshape(sample_num, -1)

            self.x_train = x_data[: -conf.TEST_SIZE]
            self.x_test = x_data[-conf.TEST_SIZE:]

        elif self.mode == 'lstm':
            self.x_train_his = x_window_his[: -conf.TEST_SIZE]
            self.x_train_pre = x_window_pre[: -conf.TEST_SIZE]
            self.x_test_his = x_window_his[-conf.TEST_SIZE:]
            self.x_test_pre = x_window_pre[-conf.TEST_SIZE:]

        self.y_train = y_data[: -conf.TEST_SIZE]
        self.y_test = y_data[-conf.TEST_SIZE:]
