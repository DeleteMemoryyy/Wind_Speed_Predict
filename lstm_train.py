from dataloader import DataLoader
from lstm_batched import BatchedLSTM

dl_lstm = [DataLoader(g, 'lstm') for g in range(6)]

for generator in range(6):
    lstm = BatchedLSTM()
    lstm.fit(generator, dl_lstm[generator])