import time
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn import ensemble
from sklearn import linear_model as lm
from sklearn import kernel_ridge
from sklearn import svm as svm
from sklearn import tree as tree
from sklearn import metrics
import rgf.sklearn as rgf
import xgboost as xgb
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns

from dataloader import DataLoader, DataConfig
from stastical_regression import Stacking, CV
from lstm_batched import BatchedLSTM
import lstm_regression

enable_stacking = True
save_file = True
enable_lstm = True
lstm_file = 'result_Dec_16_11-06-09_LSTM_batched_1.89311.csv'


dl_norm = [DataLoader(g, DataConfig('norm')) for g in range(6)]

svr = svm.SVR(C=200, gamma=0.001)
regr = lm.Ridge(alpha=11.0)
lsr = lm.Lasso(alpha=0.0005547)
enr = lm.ElasticNet(alpha=0.0009649, l1_ratio=0.5)
krr = kernel_ridge.KernelRidge(kernel='polynomial')
gbr = ensemble.GradientBoostingRegressor(
    loss='huber', max_features='sqrt', n_estimators=400)
rfr = ensemble.RandomForestRegressor(n_estimators=90)
xgbr = xgb.XGBRegressor(booster='gbtree', gamma=0.001,
                        max_depth=3, min_child_weight=2, n_estimators=100)
xgblr = xgb.XGBRegressor(booster='gblinear', n_estimators=300, gamma=0.0001)
lgbr = lgb.LGBMRegressor(num_leaves=3, min_data_in_leaf=11,
                         max_bin=55, learning_rate=0.05, n_estimators=900)
rgfr = rgf.RGFRegressor(max_leaf=700, learning_rate=0.005,
                        min_samples_leaf=5, test_interval=50)

models = []
model_names = []
lstm_result = []
lstm_data = []
if enable_lstm:
    lstm_result = pd.read_csv('result/{}'.format(lstm_file))
    lstm_data = np.transpose(lstm_result.values).reshape((6, -1, 60))
#%%
if enable_stacking:
    base_models = [lsr]
    stacker = [lm.LinearRegression()]
    all_added_result = []
    weights = [0.5, 0.5]

    if enable_lstm:
        all_added_result.append(lstm_data.tolist())

    result_list = []
    result_rmse = []
    # for generator in range(6):
    generator = 0
    added_result = np.array(all_added_result)[:, generator, :]
    stacking = Stacking(5,base_models,stacker,added_result,weights)
    stacking_y_test_predict = stacking.fit_predict(dl_norm[generator].x_train,dl_norm[generator].y_train,dl_norm[generator].x_test)
    result_rmse.append(np.sqrt(stacking_y_test_predict,dl_norm[generator].y_test))

    # if save_file:
    #     result = pd.DataFrame(np.transpose(np.array(result_list)), columns=['G{}'.format(i) for i in range(1, 7)])
    #     test_rmse=np.array(result_rmse).mean()
    #     print('LSTM_batched_test_rmse: {}'.format(test_rmse))
    #     save_name='result/result_{0:s}_stacking_{1:.5f}.csv'.format(time.strftime('%b_%d_%H-%M-%S', time.localtime()), test_rmse)
    #     if enable_lstm:
    #         save_name='result/result_{0:s}_stacking_lstm_{1:.5f}.csv'.format(
    #             time.strftime('%b_%d_%H-%M-%S', time.localtime()), test_rmse)
    #     result.to_csv(save_name, header=True,
    #                 index=False, encoding='utf-8')
    #     print('save to {}\n'.format(save_name))




