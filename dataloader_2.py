import numpy as np
import pandas as pd

WINDOW = 32

# some structs to help us
month_date = [0,30,61,91,122]

def get_col_num(bound):
    # change [month, date] into count(date - Apr.1st), from 0
    return (month_date[(bound[0] - 4)] + bound[1] - 1) * 96

def windowlize_his(data):
    # change [1,2,3, ... ,n] into
    #  [1,2,3,..., WINDOW  ]
    #  [2,3,4,..., WINDOW+1]
    #  [n-WINDOW,...,   n-1]
    Window = []
    for i in range(data.shape[0]-WINDOW):
        Window.append(data[i:i+WINDOW])
    return np.array(Window)

col_pre = ['predict1_speed','predict1_N']
def windowlize_pre(data):
    # change [1,2,3, ... ,n] into
    #  [2,3,..., WINDOW+1 ]
    #  [3,4,..., WINDOW+2 ]
    #  [n-WINDOW+1,..., n]
    Window = []
    for i in range(data.shape[0]-WINDOW):
        Window.append(data[i+1:i+WINDOW+1,:].reshape(-1))
    return np.array(Window)

class DataFrameMix(object):
    def __init__(self, path):
        self.mode = 'lstm'

        mixed_data = pd.read_csv(path)

        # training data [month, date]
        train_start_bound = [4,13] # from WINDOW:00 a.m.
        train_end_bound = [6,9] # to 23:45 p.m.
        # testing data [month, date]
        test_start_bound = [7,5] # from WINDOW:00 a.m.
        test_end_bound = [7,31] # to 23:45 p.m.

        train_start_bound = get_col_num(train_start_bound)
        train_end_bound = get_col_num(train_end_bound)+1
        test_start_bound = get_col_num(test_start_bound)
        test_end_bound = get_col_num(test_end_bound)+1

        x_train_his = mixed_data['accurate_speed'].values[train_start_bound:train_end_bound]
        x_train_pre = mixed_data[col_pre].values[train_start_bound:train_end_bound,:]
        x_test_his = mixed_data['accurate_speed'].values[test_start_bound:test_end_bound]
        x_test_pre = mixed_data[col_pre].values[test_start_bound:test_end_bound,:]

        self.y_train = mixed_data['accurate_speed'].values[train_start_bound+WINDOW:train_end_bound]
        self.x_train = np.concatenate((windowlize_pre(x_train_pre),windowlize_his(x_train_his)), axis = -1)

        self.y_test = mixed_data['accurate_speed'].values[test_start_bound+WINDOW:test_end_bound]
        self.x_test = np.concatenate((windowlize_pre(x_test_pre),windowlize_his(x_test_his)), axis = -1)
        