import time
from dataloader import DataLoader, DataConfig
from lstm_batched import BatchedLSTM

dl_lstm = [DataLoader(g, DataConfig('lstm')) for g in range(6)]
date_name = time.strftime('%b_%d_%H-%M-%S',time.localtime())

for generator in range(6):
    lstm = BatchedLSTM()
    lstm.fit(generator, dl_lstm[generator], date_name)