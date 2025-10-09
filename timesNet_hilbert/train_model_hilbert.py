import torch
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from timesNet_hilbert import timesNet_model

"""
github: git add .
git commit -m ""
git push origin main
"""




# hyperparameters
n_channels = 10

seq_len = 64
batch_size = 8
d_embd = 64
dropout = 0.2

# timesNet
n_timeBlocks = 4
k_periods = 4

# filter
p_cutoff = 0.1
n_taps = 8  # seq_len has to be increased for this to be meaningful

# inception
n_blocks = 2
n_heads = 2
d_head = d_embd // n_heads
s_win = 8
levels = 3
s_region = [32, 16, 8]
s_pool = [1, 2, 8]
theta = 10000

# training
n_epochs = 4
eval_iter = 50


class weatherDataset(Dataset):
    def __init__(self, data, seq_len):
        self.X = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx : idx+self.seq_len, :]
        y = self.X[idx+seq_len//2+1: idx+self.seq_len+1, :]   # compare only half a seq_len to each other for loss
        return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iter, device='cuda')
        for it, (xb, yb) in enumerate(loader):
            if it == eval_iter:
                break

            xb, yb = xb.to('cuda'), yb.to('cuda')
            pred = model(xb)
            loss = F.mse_loss(pred[:, seq_len // 2:, :], yb)
            losses[it] = loss

        out[split] = losses.mean()
    model.train()
    return out

# read file
file = pd.read_csv('../weather_prediction_dataset.csv', usecols=range(104, 114)) #  munich weather data

# Train and test splits
data = torch.tensor(file.values, dtype=torch.float32)  # (T, C)
n = int(0.9*data.shape[0])
train_data = data[:n]
val_data = data[n:]

# Normalize data
mean = train_data.mean(dim=0)
std = train_data.std(dim=0)
train_data = (train_data - mean) / std
val_data = (val_data - mean) / std

# Dataset, Dataloader
train_dataset = weatherDataset(train_data, seq_len)
val_dataset = weatherDataset(val_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # samples as many as possible (B, T, C) chunks from train_dataset (with getitem) -> one epoch
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# create model
model = timesNet_model(n_channels, seq_len, d_embd, dropout, n_timeBlocks, k_periods, p_cutoff, n_taps, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta)
model = model.to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# training loop
model.train()
for epoch in range(n_epochs):
    for it, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to('cuda'), yb.to('cuda')
        pred = model(xb)
        loss = F.mse_loss(pred[:, seq_len//2:, :], yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % eval_iter == 0:
            losses = estimate_loss()
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# generate from model
model.eval()
context = data[-seq_len:, :]


