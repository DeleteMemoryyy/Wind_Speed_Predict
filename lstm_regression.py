import time
import numpy as np
import pandas as pd
from sklearn import metrics
from dataloader import DataLoader, DataConfig
from lstm_batched import BatchedLSTM

net_name = 'Dec_16_13-19-31_batched_ps32_hs4_e30_b32_lr0.00010_emd0.000_ep5_model6.h5'

dl_lstm = [DataLoader(g, DataConfig('lstm')) for g in range(6)]

result_data = []
result_rmse = []
for generator in range(6):
    lstm = BatchedLSTM()
    y_test_out = lstm.predict(generator,dl_lstm[generator],net_name)
    result_data.append(y_test_out.reshape((-1,)).tolist())
    result_rmse.append(np.sqrt(metrics.mean_squared_error(dl_lstm[generator].y_test, y_test_out)))

result_data = np.transpose(np.array(result_data))
result = pd.DataFrame(result_data,columns=['G{}'.format(i) for i in range(1,7)])
test_rmse = np.array(result_rmse).mean()
print('LSTM_batched_test_rmse: {}'.format(test_rmse))
save_name = 'result/result_{0:s}_LSTM_batched_{1:.4f}.csv'.format(time.strftime('%b_%d_%H-%M-%S',time.localtime()), test_rmse)
result.to_csv(save_name,header=True,index=False,encoding='utf-8')
print('save to {}\n'.format(save_name))
