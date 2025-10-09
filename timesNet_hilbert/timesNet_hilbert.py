import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from scipy.signal import firwin, filtfilt, hilbert
import numpy as np

from inception_hilbert import inception

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

class film_mlp(nn.Module):
    def __init__(self, d_embd, d_head, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embd, 2*d_embd),
            nn.GELU(),
            nn.Linear(2*d_embd, 2*d_head),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class times_block(nn.Module):
    def __init__(self, d_embd, dropout, k_periods, p_cutoff, n_taps, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta):
        super().__init__()
        self.ffwd = FeedForward(d_embd, dropout)
        self.film_mlp = film_mlp(d_embd, d_head, dropout)
        self.inception = inception(d_embd, dropout, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta)

        self.k_periods = k_periods

        # filter
        self. p_cutoff = p_cutoff
        self.n_taps = n_taps

    def forward(self, x):
        B,T,C = x.shape
        _, p, freq_bin = self.get_periods(x)  # amps, periods, freq_bin: (B, k)

        # helper
        to_int = lambda x: int(x.item())

        fin_timeframes=[]
        fin_amps_t=[]
        for k in range(self.k_periods):
            nk_max = to_int(p[:, k].max())  # (1): max num columns of period k timeframes across batches
            mk_max = to_int(torch.ceil(T / p[:, k]).max())

            pad = (-T) % p[:, k]  # (B): pad for 1D time of period k for each batch
            n = p[:, k]           # (B): num columns of 2D timeframe of period k for each batch
            m = (T + pad) // n

            batch_timeframes = []
            batch_mask = []
            batch_amps_t = []
            batch_f_off = []
            for b in range(B):
                pad_b = pad[b]
                m_b = m[b]
                n_b = n[b]

                # filter and hilbert transform
                amp_t, f_offset = self.bandpass_hilbert(x[b], freq_bin[b, k] / T)  # (T, C) each
                f_off_embd = self.film_mlp(f_offset) # (T, 2*d_head)

                # shape/pad dims to create one tensor of 2D timeframes for each k
                timeframe_b_k = F.pad(x[b], (0, 0, 0,pad_b),) # (T+pad, C): pad time dim
                f_off_embd = F.pad(f_off_embd, (0, 0, 0, pad_b))

                mask_b_k = x[b].new_ones(T, 1)
                mask_b_k = F.pad(mask_b_k,(0,0, 0,pad_b))

                # 2D transform
                timeframe_b_k = rearrange(timeframe_b_k, '(m n) c -> m n c', n=n_b)
                timeframe_b_k = F.pad(timeframe_b_k, (0,0, 0,nk_max - n_b, 0,mk_max - m_b))  # pad to match dims across batches for fixed k
                f_off_embd = rearrange(f_off_embd, '(m n) c -> m n c', n=n_b)
                f_off_embd = F.pad(f_off_embd, (0,0, 0,nk_max - n_b, 0,mk_max - m_b))
                mask_b_k = rearrange(mask_b_k,'(m n) c -> m n c', n=n_b)
                mask_b_k = F.pad(mask_b_k,(0,0, 0,nk_max - n_b, 0,mk_max - m_b))

                batch_timeframes.append(timeframe_b_k) # (M_max, N_max, C)
                batch_f_off.append(f_off_embd)
                batch_amps_t.append(amp_t)
                batch_mask.append(mask_b_k)

            timeframes_k = torch.stack(batch_timeframes, dim=0) # (B, M_max, N_max C)
            f_off = torch.stack(batch_f_off, dim=0)  # (B, M_max, N_max, 2*d_head)
            mask = torch.stack(batch_mask, dim=0)

            amps_t = torch.stack(batch_amps_t, dim=0) # (B, T, C)
            fin_amps_t.append(amps_t)

            print(timeframes_k.shape)
            print(f_off.shape)

            # Inception
            timeframes_k = self.inception(timeframes_k, mask, f_off)  # returns (B,M_max,N_max,C)

            # flatten to 1D
            fin_timeframes_k = []
            for b in range(B):
                timeframe_b_k = timeframes_k[b, :to_int(m[b]), :to_int(n[b]), :]  # trunc away pads for M_max/N_max
                timeframe_b_k = rearrange(timeframe_b_k, 'm n c -> (m n) c')  # flatten to 1D
                timeframe_b_k = timeframe_b_k[:T, :] # trunc away padded time: (T, C)
                fin_timeframes_k.append(timeframe_b_k)

            fin_timeframes.append(torch.stack(fin_timeframes_k, dim=0)) # (B, T, C)

        timeframes_1d = torch.stack(fin_timeframes, dim=1) # (B, k, T, C)
        amps = torch.stack(fin_amps_t, dim=1)           # (B, k, T, C)

        # MLP
        timeframes_1d = self.ffwd(timeframes_1d)

        # Adaptive Aggregation
        amps = F.softmax(amps, dim=1)         # (B, k, T, C) -> softmax across k
        timeframes_1d = timeframes_1d * amps  # (B, k, T, C)
        deltaX = timeframes_1d.sum(dim=1)     # sum across k -> (B, T, C)

        # Residual Connection
        out = x + deltaX
        return out

    def bandpass_hilbert(self, x, f0):
        # move to cpu for scipy
        x = x.detach().cpu().numpy()
        f0 = float(f0.detach().cpu())

        # filter each channel c(t)
        fL, fH = f0 * (1 - self.p_cutoff), f0 * (1 + self.p_cutoff)
        taps = firwin(self.n_taps, [fL, fH], pass_zero=False)  # H(f) -> h(t)
        x_filtered = filtfilt(taps, [1.0], x, axis=-2)  # (T, C) -> filter across T, filtfilt -> no phase shift

        # hilbert transform the filtered c(t)
        z = hilbert(x_filtered, axis=-2)  # (T, C) -> stores analytic signals z(t) for each channel
        amp_t = np.abs(z)
        phase_t = np.unwrap(np.angle(z))  # angle returns [-pi, pi] -> unwrap
        freq_t = np.diff(phase_t / (2.0 * np.pi), axis=-2)  # cycles/timestep(t)
        freq_offset = f0-freq_t

        # move back to gpu
        amp_t = torch.from_numpy(amp_t).float().to('cuda')
        freq_offset = torch.from_numpy(freq_offset).float().to('cuda')

        freq_offset = F.pad(freq_offset, (0, 0, 1, 0)) # add time step that got lost at np.diff -> (T, C)

        return amp_t, freq_offset

    def get_periods(self, x):
        B, T, C = x.shape

        x_ft = torch.fft.rfft(x, dim=-2)  # fft for frequencies 0,...,T//2
        x_ft = x_ft[:, 1:, :]  # drop row 0 -> const term
        amps = torch.abs(x_ft)
        amps = torch.mean(amps,
                          dim=-1)  # avg across channel dim (identify most meaningful periods across channels) -> (B,F)
        amps_k, freq_k = torch.topk(amps, k=self.k_periods,dim=-1)  # top k_periods amps, frequencies(indices): (B, k) each
        p_k = T // (freq_k + 1)  # row 0 = freq 0 sliced out before -> new row 0 refers to freq 1 -> shift freq_k by one
        return amps_k, p_k, freq_k + 1


class timesNet_model(nn.Module):
    def __init__(self, n_channels, seq_len, d_embd, dropout, n_timeBlocks, k_periods, p_cutoff, n_taps, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta):
        super().__init__()

        self.embd = nn.Linear(n_channels, d_embd)
        self.blocks = nn.Sequential(*[times_block(d_embd, dropout, k_periods, p_cutoff, n_taps, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta) for _ in range(n_timeBlocks)])
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

