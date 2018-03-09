import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model as lm
from sklearn import kernel_ridge
from sklearn import svm as svm
from sklearn import metrics
import rgf.sklearn as rgf
import xgboost as xgb
import lightgbm as lgb

from dataloader import DataLoader, DataConfig
from dataloader_2 import DataFrameMix
from stastical_regression import grid_search, varify_on_test

data_file = ["data/new_mixed_machine{}.csv".format(i) for i in range(1,7)]
dl_norm = [DataFrameMix(data_file[i]) for i in range(6)]

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

# lgbr = lgb.LGBMRegressor(num_leaves=4, min_data_in_leaf=13,
#                          max_bin=55, learning_rate=0.05, n_estimators=1000)
# xgbr = xgb.XGBRegressor(booster='gbtree', gamma=0.001,
#                         max_depth=4, min_child_weight=2, n_estimators=150)
# xgblr = xgb.XGBRegressor(booster='gblinear', n_estimators=300, gamma=0.0001)

# models = [lsr, regr, enr]
# model_names = ['lsr','regr','enr']
# param_grid = [{'alpha':[0.01,0.1,1.0,10.0]},
# {'alpha':[0.01,0.1,1.0,10.0]},
# {'alpha':[0.01,0.1,1.0,10.0],'l1_ratio':[0.1,0.3,0.5,0.7,0.9]}]

models = [lgbr]
model_names = ['lgbr']
param_grid = [{'num_leaves':[3,4,5,6],'min_data_in_leaf':[3,6,9,12],'max_bin':[35,55,75],'n_estimators':[500,700,900,1100],'learning_rate':[0,005]}]

# if __name__ == '__main__':
    # dl_norm = [DataLoader(g, DataConfig('norm')) for g in range(6)]

#%%
for idx, model in enumerate(models):
    generator = 2
    best_model = grid_search(model, dl_norm[generator], param_grid[idx],cv=2,verbose=1,model_name=model_names[idx])
    train_rmse, test_rmse = varify_on_test(best_model, dl_norm[generator])

    print('{0:s}  train_rmse: {1:f}'.format(
        model_names[idx], train_rmse))
    print('{0:s}  test_rmse: {1:f}'.format(model_names[idx], test_rmse))

    # all_train_rmse = []
    # all_test_rmse = []
    # for generator in range(6):
    #     best_model = grid_search(
    #         model, dl_norm[generator], param_grid[idx], cv=2, verbose=0,model_name=model_names[idx])
    #     train_rmse, test_rmse = varify_on_test(best_model, dl_norm[generator],verbose=1)
    #     all_train_rmse.append(train_rmse)
    #     all_test_rmse.append(test_rmse)

    # result_train_rmse = np.sqrt(np.power(np.array(all_train_rmse), 2).mean())
    # result_test_rmse = np.sqrt(np.power(np.array(all_test_rmse), 2).mean())

    # print('{0:s}  train_rmse: {1:f}'.format(
    #     model_names[idx], result_train_rmse))
    # print('{0:s}  test_rmse: {1:f}'.format(model_names[idx], result_test_rmse))

# #%% LassoCV
# all_train_rmse = []
# all_test_rmse = []
# # lsr_cv = lm.MultiTaskLassoCV(cv=4, n_jobs=4)
# # lsr_cv = lm.MultiTaskLassoCV()
# for generator in range(6):
#     # lsr_cv.fit(dl_norm[generator].x_train, dl_norm[generator].y_train)
#     # best_alpha = lsr_cv.alpha_
#     # print('Best {0:s} parameters: alpha: {1}'.format(
#     #     'lsr', best_alpha))
#     # best_model = lm.MultiTaskLasso(alpha=best_alpha)
#     best_model = lm.Lasso()
#     train_rmse, test_rmse = varify_on_test(
#         best_model, dl_norm[generator])
#     all_train_rmse.append(train_rmse)
#     all_test_rmse.append(test_rmse)

# result_train_rmse = np.sqrt(
#     np.power(np.array(all_train_rmse), 2).mean())
# result_test_rmse = np.sqrt(np.power(np.array(all_test_rmse), 2).mean())

# print('{0:s}  train_rmse: {1:f}'.format(
#     'lsr', result_train_rmse))
# print('{0:s}  test_rmse: {1:f}'.format(
#     'lsr', result_test_rmse))





