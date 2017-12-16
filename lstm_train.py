from dataloader import DataLoader, DataConfig
from lstm_batched import BatchedLSTM

dl_lstm = [DataLoader(g, DataConfig('lstm')) for g in range(6)]

for generator in range(6):
    lstm = BatchedLSTM()
    lstm.fit(generator, dl_lstm[generator])