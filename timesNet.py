import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from inception import inception

class FeedForward(nn.Module):
     def __init__(self, d_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embd, 4 * d_embd),
            nn.GELU(),
            nn.Linear(4 * d_embd, d_embd),
            nn.Dropout(dropout)
         )

     def forward(self, x):
          return self.net(x)


class periods(nn.Module):
    def __init__(self, k_periods):
        super().__init__()
        self.k_periods = k_periods

    def forward(self, x):
        B,T,C = x.shape

        x_ft = torch.fft.rfft(x, dim=-2) # fft for frequencies 0,...,T//2
        x_ft = x_ft[:, 1:, :] # drop row 0 -> const term
        amps = torch.abs(x_ft)
        amps = torch.mean(amps, dim=-1) # avg across channel dim -> (B,F)
        amps_k, freq_k = torch.topk(amps, k=self.k_periods, dim=-1) # top k_periods amps, frequencies(indices): (B, k) each
        p_k = T // freq_k
        return amps_k, p_k

class times_block(nn.Module):
    def __init__(self, d_embd, dropout, k_periods, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta):
        super().__init__()
        self.periods = periods(k_periods)
        self.ffwd = FeedForward(d_embd, dropout)
        self.inception = inception(d_embd, dropout, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta)

        self.k_periods = k_periods

    def forward(self, x):
        B,T,C = x.shape
        amps, p = self.periods(x)  # amps, freq, periods: (B, k)

        # helper
        to_int = lambda x: int(x.item())

        fin_timeframes=[]

        for k in range(self.k_periods):
            nk_max = p[:, k].max()  # (1): max num columns of period k timeframes across batches
            mk_max = torch.ceil(T / p[:, k]).max()

            pad = (T - T % p[:, k]) % p[:, k]  # (B): pad for 1D time of period k for each batch
            n = p[:, k]                        # (B): num columns of 2D timeframe of period k for each batch
            m = (T + pad) // n

            batch_timeframes = []
            batch_mask = []
            for b in range(B):
                timeframe_b_k = F.pad(x[b], (0,0, 0,pad[b]),) # (T, C): pad time dim

                mask_b_k = x[b].new_ones(T, 1)
                mask_b_k = F.pad(mask_b_k,(0,0, 0,pad[b]))

                # 2D transform
                timeframe_b_k = rearrange(timeframe_b_k, '(m n) c -> m n c', n=to_int(n[b]))
                timeframe_b_k = F.pad(timeframe_b_k, (0,0, 0,nk_max - n[b], 0,mk_max - m[b]))  # pad to match dims across batches for fixed k

                mask_b_k = rearrange(mask_b_k,'(m n) c -> m n c', n=to_int(n[b]))
                mask_b_k = F.pad(mask_b_k,(0,0, 0,nk_max - n[b], 0,mk_max - m[b]))

                batch_timeframes.append(timeframe_b_k) # (M_max, N_max, C)
                batch_mask.append(mask_b_k)

            timeframes_k = torch.stack(batch_timeframes, dim=0) # (B, M_max, N_max C)
            mask = torch.stack(batch_mask, dim=0)

            # Inception
            timeframes_k = self.inception(timeframes_k, mask)  # (B,M_max,N_max,C)

            # flatten to 1D
            fin_timeframes_k = []
            for b in range(B):
                timeframe_b_k = timeframes_k[b, :m[b], :n[b], :]  # trunc away pads for M_max/N_max
                timeframe_b_k = rearrange(timeframe_b_k, 'm n c -> (m n) c')  # flatten to 1D
                timeframe_b_k = timeframe_b_k[:T, :] # trunc away padded time: (T, C)
                fin_timeframes_k.append(timeframe_b_k)

            fin_timeframes.append(torch.stack(fin_timeframes_k, dim=0)) # (B, T, C)

        timeframes_1d = torch.stack(fin_timeframes, dim=1) # (B, k, T, C)

        # MLP
        timeframes_1d = self.ffwd(timeframes_1d)

        # Adaptive Aggregation
        amps = F.softmax(amps, dim=-1)  # (B, k, 1, 1)
        timeframes_1d = timeframes_1d * amps[:, :, None, None]  # (B, k, T, C)
        deltaX = timeframes_1d.sum(dim=1)  # sum across k -> (B, T, C)

        # Residual Connection
        out = x + deltaX
        return out


class timesNet_model(nn.Module):
    def __init__(self, n_channels, seq_len, d_embd, dropout, n_timeBlocks, k_periods, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta):
        super().__init__()

        self.embd = nn.Linear(n_channels, d_embd)
        self.blocks = nn.Sequential(*[times_block(d_embd, dropout, k_periods, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta) for _ in range(n_timeBlocks)])
        self.embd_back = nn.Linear(d_embd, n_channels)

        self.seq_len = seq_len

        self.apply(self.init_weights)  # goes through every module and calls init_weights on it, passes in module as argument to init_weights

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input, targets=None):
        x = self.embd(input)
        x = self.blocks(x)
        pred = self.embd_back(x)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            loss = F.mse_loss(pred, targets)
        return pred, loss

    def generate(self, context, max_new_pred):   #  context: (B, T_context, C)
        for _ in range(max_new_pred):
            # crop idx to last seq_len tokens
            context_cond = context[:, -self.seq_len:, :]

            # get predictions
            pred, loss = self(context_cond)

            # only last time step
            pred_next = pred[:, [-1], :]  # (B, 1, C)

            # append pred_t to running sequence
            context = torch.concat([context, pred_next], dim=1)

        return context

