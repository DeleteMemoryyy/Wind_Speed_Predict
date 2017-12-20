from scipy import io as scio
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model as lm
from dataloader import DataLoader,DataConfig
# data = scio.loadmat('data/HisRawData/201604/20160401.mat')

nearest_dir = 'data/preprocessed_data/history_data/scattered_speed/'
average_dir = 'data/preprocessed_data/history_data/accurate_biased_average/'
generated_dir = 'data/preprocessed_data/history_data/our_prediction_to_labels/'
predict_dir = 'data/preprocessed_data/predict_data/'

history_data_name = ['machine{}_accurate.csv'.format(i) for i in range(1, 7)]
predict_data_name = ['machine{}_predict.csv'.format(i) for i in range(1, 7)]

dlconf_ave = DataConfig('lstm')
dlconf_ave.history_dir = average_dir
dl_ave = DataLoader(0,dlconf_ave)

dlconf_gene = DataConfig('lstm')
dlconf_gene.history_dir = generated_dir
dl_gene = DataLoader(0, dlconf_gene)
from scipy import io as scio
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model as lm
from dataloader import DataLoader,DataConfig
# data = scio.loadmat('data/HisRawData/201604/20160401.mat')

nearest_dir = 'data/preprocessed_data/history_data/scattered_speed/'
average_dir = 'data/preprocessed_data/history_data/accurate_biased_average/'
generated_dir = 'data/preprocessed_data/history_data/our_prediction_to_labels/'
predict_dir = 'data/preprocessed_data/predict_data/'

history_data_name = ['machine{}_accurate.csv'.format(i) for i in range(1, 7)]
predict_data_name = ['machine{}_predict.csv'.format(i) for i in range(1, 7)]

dlconf_ave = DataConfig('lstm')
dlconf_ave.history_dir = average_dir
dl_ave = DataLoader(0,dlconf_ave)

dlconf_gene = DataConfig('lstm')
dlconf_gene.history_dir = generated_dir
dl_gene = DataLoader(0, dlconf_gene)

dlconf_ave.history_dir
dlconf_gene.history_dir
dl_ave.mode
dl_ave.y_test.shape

dl_ave.x_train_his.shape
dl_gene.x_train_his.shape

dl_ave.x_train_pre.shape
dl_gene.x_train_pre.shape
