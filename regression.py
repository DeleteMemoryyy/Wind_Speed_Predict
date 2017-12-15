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

from dataloader import DataLoader
from stastical_regression import Stacking, CV
from lstm_batched import BatchedLSTM

dl_norm = [DataLoader(g, 'norm') for g in range(6)]
dl_lstm = [DataLoader(g, 'lstm') for g in range(6)]

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