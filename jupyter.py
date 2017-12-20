from scipy import io as scio
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model as lm
from dataloader import DataLoader,DataConfig
# data = scio.loadmat('data/HisRawData/201604/20160401.mat')

dl = DataLoader(0,DataConfig('lstm'))

# data['__header__']
# data['FJDATA'][0,1][0].shape

# df = pd.DataFrame(data)

# train = np.zeros((3,3))
# train[:,:] = 1.0
# test = np.zeros((3,3))
# test[:,:] = 3.0

# lr = lm.LinearRegression()
# lr. fit(train,test)
# output = lr.predict(train)
# output

data = np.arange(27).reshape((3,3,3))
data
data.mean(1)
data[:,:,0]

# data.tolist()