from scipy import io as scio
import pandas as pd
data = scio.loadmat('data/HisRawData/201604/20160401.mat')


data['__header__']
data['FJDATA'][0,1][0].shape

df = pd.DataFrame(data)
