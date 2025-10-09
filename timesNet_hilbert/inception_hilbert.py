import torch
import torch.nn as nn
from einops import rearrange
from torch.cuda import device
from torch.nn import functional as F


class pool(nn.Module):
    def __init__(self, d_embd, levels, s_pool):
        super().__init__()
        self.pool_layers = nn.ModuleList(
            [nn.Linear(kernel * kernel * d_embd + 1, d_embd) for kernel in s_pool[1:]])  # no pooling for lvl 1
        self.levels = levels
        self.s_pool = s_pool

    def forward(self, feature_map, mask):
        feature_map = feature_map * mask  # ensure invalid entries are zero (for safety, attention masking should also ensure this)

        # Level 1
        pooled_maps = [feature_map]
        pooled_masks = [mask]

        # other levels
        B, M, N, C = feature_map.shape
        feature_map = rearrange(feature_map, 'b m n c -> b c m n')
        mask = rearrange(mask, 'b m n c -> b c m n')

        for l in range(1, self.levels):
            k = self.s_pool[l]
            feature_map_l = feature_map  # original feature_map/mask keep intact
            mask_l = mask

            # pad right and down to ensure Q-K/V alignment, masking is skipped since s_pool << M,N
            pad_y = (-M) % k
            pad_x = (-N) % k
            feature_map_l = F.pad(feature_map_l, (0, pad_x, 0, pad_y))
            mask_l = F.pad(mask_l, (0, pad_x, 0, pad_y))

            # prepare feature map for pooling
            Mk = (M + pad_y) // k
            Nk = (N + pad_x) // k
            feature_map_l = rearrange(feature_map_l, 'b c (Mk k1) (Nk k2) -> b Mk Nk (c k1 k2)',
                                      Mk=Mk, Nk=Nk, k1=k, k2=k)

            # pool mask
            percent_valid_tokens = F.avg_pool2d(mask_l, kernel_size=k)  # (B, 1, Mk, Nk) percentage of valid tokens in kernel
            percent_valid_tokens = rearrange(percent_valid_tokens, 'b c m n -> b m n c')
            mask_l = (percent_valid_tokens > 0).float()  # every kernel where at least one valid token -> valid
            pooled_masks.append(mask_l)  # (B Mk Nk 1)

            # concat percentage info to channel dim of feature map
            feature_map_l = torch.concat([feature_map_l, percent_valid_tokens], dim=-1)

            # pool feature map
            pooled_map = self.pool_layers[l-1](feature_map_l)  # (B, Mk, Nk, C), only level-1 pool layers
            pooled_maps.append(pooled_map)

        return pooled_maps, pooled_masks


class multi_head(nn.Module):
    def __init__(self, d_embd, d_head, n_heads, s_win, levels, s_region, s_pool, dropout, theta):
        super().__init__()
        self.query = nn.Linear(d_embd, n_heads * d_head, bias=False)
        self.key = nn.Linear(d_embd, n_heads * d_head, bias=False)
        self.val = nn.Linear(d_embd, n_heads * d_head, bias=False)
        self.proj_out = nn.Linear(n_heads * d_head, d_embd, bias=False)

        self.d_embd = d_embd
        self.d_head = d_head
        self.n_heads = n_heads
        self.s_win = s_win
        self.s_region = s_region
        self.levels = levels
        self.s_pool = s_pool
        self.dropout = dropout

        # Rope
        idx = torch.arange(0, d_head // 4, device='cuda')
        self.register_buffer('rad', theta ** (-2 * idx / d_head))

    def forward(self, pooled_maps, pooled_masks, feature_map, mask_q, cy, cx, freq_off):
        # query, key, value
        Q = self.query(feature_map)  # (B, M, N, H*C)
        K = [self.key(pooled_maps[i]) for i in range(self.levels)]  # l*(B, M, N, H*C)
        V = [self.val(pooled_maps[i]) for i in range(self.levels)]

        # queries
        Q = rearrange(Q, 'b m n (h c) -> b h m n c', h=self.n_heads, c=self.d_head)
        mask_q = rearrange(mask_q, 'b Mw Nw sw_sqr c -> b 1 Mw Nw sw_sqr c')  # add n_head dim

        # frequency and Rope embedding
        M_idx = torch.arange(Q.shape[-3], device='cuda')
        N_idx = torch.arange(Q.shape[-2], device='cuda')
        M_grid, N_grid = M_idx[:, None], N_idx[None, :]
        freq_off = rearrange(freq_off, 'b m n c -> b 1 m n c')  # add n_head dim
        queries = Q * freq_off[..., M_grid, N_grid, :self.d_head] + freq_off[..., M_grid, N_grid, self.d_head:]
        queries = self.Rope(queries, M_idx, N_idx)
        queries = rearrange(queries, 'b h (Mw sw1) (Nw sw2) c -> b h Mw Nw (sw1 sw2) c', sw1=self.s_win, sw2=self.s_win)

        K_levels_regions, mask_K_levels_regions = [], []
        V_levels_regions = []

        for l in range(self.levels):
            K_l = rearrange(K[l], 'b m n (h c) -> b h m n c', h=self.n_heads, c=self.d_head)
            V_l = rearrange(V[l], 'b m n (h c) -> b h m n c', h=self.n_heads, c=self.d_head)
            pooled_mask_l = rearrange(pooled_masks[l], 'b m n c -> b 1 m n c')  # add n_head dim

            # correct window centers by pooling
            cy = cy // self.s_pool[l]
            cx = cx // self.s_pool[l]

            # extract regions
            K_regions, mask_K, YY, XX = self.extract_regions(K_l, cy, cx, self.s_region[l], pooled_mask_l)  # (B, H, M, N, C)
            V_regions, _, _, _ = self.extract_regions(V_l, cy, cx, self.s_region[l], pooled_mask_l)  # YY, XX for K/V identical

            # prepare dims for frequency embedding and Rope
            M_idx = YY[:, 0, :, 0]  # (Mw, s_region)
            M_idx = rearrange(M_idx, 'Mw sr -> (Mw sr)')
            M_idx = M_idx * self.s_pool[l]  # correct indices by pool strength
            Mf = freq_off.shape[-3]
            M_idx = M_idx.clamp(0, Mf-1) # ensure valid idx after scaling with s_pool

            N_idx = XX[0, :, 0, :]  # (Nw, s_region)
            N_idx = rearrange(N_idx, 'Nw sr -> (Nw sr)')
            N_idx = N_idx * self.s_pool[l]
            Nf = freq_off.shape[-2]
            N_idx = N_idx.clamp(0,  Nf-1)

            # frequency embedding
            M_grid, N_grid = M_idx[:, None], N_idx[None, :]
            K_regions = K_regions * freq_off[..., M_grid, N_grid, :self.d_head] + freq_off[..., M_grid, N_grid, self.d_head:]
            V_regions = V_regions * freq_off[..., M_grid, N_grid, :self.d_head] + freq_off[..., M_grid, N_grid, self.d_head:]

            # Rope
            K_regions = self.Rope(K_regions, M_idx, N_idx)  # (B, H, M_idx, N_idx, C)
            V_regions = self.Rope(V_regions, M_idx, N_idx)

            # prepare dims for att, mask already prepared in extract_regions
            K_regions = rearrange(K_regions, 'b h (Mw sr1) (Nw sr2) c -> b h Mw Nw (sr1 sr2) c', sr1=self.s_region[l], sr2=self.s_region[l])
            V_regions = rearrange(V_regions, 'b h (Mw sr1) (Nw sr2) c -> b h Mw Nw (sr1 sr2) c', sr1=self.s_region[l], sr2=self.s_region[l])

            # append to lists
            K_levels_regions.append(K_regions), mask_K_levels_regions.append(mask_K)
            V_levels_regions.append(V_regions)

        # keys/vals for all windows and levels
        keys = torch.concat(K_levels_regions, dim=-2)  # (B, H, Mw, Nw, sr^2, C) -> (B, H, Mw Nw, levels * sr^2, C)
        vals = torch.concat(V_levels_regions, dim=-2)
        mask_keys = torch.concat(mask_K_levels_regions, dim=-2)

        mask_keys_t = rearrange(mask_keys, '... s c -> ... c s')
        attn_mask = mask_q.bool() & mask_keys_t.bool()  # (1, 1, Mw, Nw, sw^2, 1) & (1, 1, Mw, Nw, 1, s)

        # Attention
        att = F.scaled_dot_product_attention(queries, keys, vals, attn_mask=attn_mask, dropout_p=self.dropout)  # (B, H, Mw, Nw, s_win^2, C)

        # out
        att = rearrange(att, 'b h Mw Nw (sw1 sw2) c -> b (Mw sw1) (Nw sw2) (h c)', sw1=self.s_win, sw2=self.s_win)
        out = self.proj_out(att)
        return out  # (B, M, N, C)

    def extract_regions(self, X, cy, cx, k, pooled_mask):
        B, H, M, N, C = X.shape
        X = rearrange(X, '... m n c -> ... c m n')

        # pad
        pad = k // 2
        X = F.pad(X, (pad, pad, pad, pad))
        cy += pad
        cx += pad

        # mask
        mask = torch.ones((1, 1, 1, M, N), dtype=torch.bool, device='cuda')
        pooled_mask = rearrange(pooled_mask, '... m n c -> ... c m n')
        mask = mask * pooled_mask  # account for invalid tokens of feature_map
        mask = F.pad(mask, (pad, pad, pad, pad))

        # region offsets
        delta = torch.arange(k, device='cuda') - (k//2)  # (k) -> stores k different offsets ranging from -k/2 to k/2 (to create a region)

        # calc region coordinates for all windows
        YY, XX = torch.meshgrid(cy, cx, indexing='ij')  # (Mw, Nw) -> y and x center coords for each window (i,j)
        YY = YY[..., None, None] + delta.view(1, 1, -1, 1)  # (Mw, Nw, 1, 1) + (1, 1, k, 1) = (Mw, Nw, k, 1) -> for each window (i,j): stores absolute y coords of all vertical offsets k around center, YY[i,j,m,0] = cy[i] + dy[m]
        XX = XX[..., None, None] + delta.view(1, 1, 1, -1)  # (Mw, Nw, 1, 1) + (1, 1, 1, k) = (Mw, Nw, 1, k)

        YY = YY.clamp(0, M-1)  # ensure valid idx
        XX = XX.clamp(0, N-1)

        # gather values from X at all coords of XX, YY
        regions = X[:, :, :, YY, XX]  # broadcasting -> (B, H, C, Mw, Nw, k, k)
        mask_regions = mask[:, :, :, YY, XX]

        regions = rearrange(regions, 'b h c Mw Nw k1 k2 -> b h (Mw k1) (Nw k2) c')  # (B, H, M, N, C)
        mask_regions = rearrange(mask_regions, 'b h c Mw Nw k1 k2 -> b h Mw Nw (k1 k2) c')  # ready for att

        return regions, mask_regions, YY - pad, XX - pad  # subtract pad to get real positions in regard to query windows

    def Rope(self, q, M_idx, N_idx):
        cos_m = torch.cos(M_idx[:, None] * self.rad)  # (M, 1) * (d/4) = (M, d/4)
        sin_m = torch.sin(M_idx[:, None] * self.rad)
        cos_n = torch.cos(N_idx[:, None] * self.rad)  # (N, d/4)
        sin_n = torch.sin(N_idx[:, None] * self.rad)

        cos_m, sin_m = cos_m[:, None, :], sin_m[:, None, :]  # (M, 1, d/4)
        cos_n, sin_n = cos_n[None, :, :], sin_n[None, :, :]  # (1, N, d/4)

        d_half = self.d_head // 2

        # q is input tensor
        q_m = q[..., :d_half]  # (..., d/2)
        q_n = q[..., d_half:]

        q_m_pairs = q_m.unflatten(dim=-1, sizes=(d_half // 2, 2))  # (..., d/4, 2)
        q_n_pairs = q_n.unflatten(dim=-1, sizes=(d_half // 2, 2))

        # q_m -> rows embedding on first half of d_head
        q_m1 = cos_m * q_m_pairs[..., 0] - sin_m * q_m_pairs[..., 1]  # (M, 1, d/4) * (B, H, M, N, d/4) = (B, H, M, N, d/4)
        q_m2 = sin_m * q_m_pairs[..., 0] + cos_m * q_m_pairs[..., 1]

        q_m_rot = torch.stack([q_m1, q_m2], dim=-1)  # (..., M, N, d/4, 2)
        q_m_rot = q_m_rot.flatten(-2, -1)  # (..., M, N, d/2)

        # q_n -> columns embedding on second half of d_head
        q_n1 = cos_n * q_n_pairs[..., 0] - sin_n * q_n_pairs[..., 1]  # (1, N, d/4) * (B, H, M, N, d/4) = (B, H, M, N, d/4)
        q_n2 = sin_n * q_n_pairs[..., 0] + cos_n * q_n_pairs[..., 1]

        q_n_rot = torch.stack([q_n1, q_n2], dim=-1)  # (..., 1, N, d/4, 2)
        q_n_rot = q_n_rot.flatten(-2, -1)  # (..., M, N, d/2)

        q_rot = torch.concat([q_m_rot, q_n_rot], -1)  # (..., M, N, d)
        return q_rot


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

class block(nn.Module):
    def __init__(self, d_embd, d_head, n_heads, s_win, levels, s_region, s_pool, dropout, theta):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embd)
        self.pool = pool(d_embd, levels, s_pool)
        self.att = multi_head(d_embd, d_head, n_heads, s_win, levels, s_region, s_pool, dropout, theta)
        self.ffwd = FeedForward(d_embd, dropout)
        self.ln2 = nn.LayerNorm(d_embd)

        self.s_win = s_win

    def forward(self, feature_map_in, mask, freq_off):
        # Layer-Norm 1
        feature_map = self.ln1(feature_map_in)

        # pool feature map
        pooled_maps, pooled_masks = self.pool(feature_map, mask)

        # pad feature map for window alignment
        B, M, N, C = feature_map.shape
        pad_y = (self.s_win - M % self.s_win) % self.s_win
        pad_x = (self.s_win - N % self.s_win) % self.s_win
        feature_map = F.pad(feature_map, (0, 0, 0, pad_x, 0, pad_y))  # (B, C, M, N)
        freq_off = F.pad(freq_off, (0, 0, 0, pad_x, 0, pad_y))

        # window coords
        Mw = (M + pad_y) // self.s_win
        Nw = (N + pad_x) // self.s_win

        # queries mask
        mask_q = torch.ones((1, 1, M, N), dtype=torch.bool, device='cuda')
        mask = rearrange(mask, 'b m n c -> b c m n')
        mask_q = mask_q * mask  # account for invalid tokens of feature_map
        mask_q = F.pad(mask_q, (0, pad_x, 0, pad_y))
        mask_q = rearrange(mask_q, 'b c (Mw sw1) (Nw sw2) -> b Mw Nw (sw1 sw2) c',  # flattened for att
                           Mw=Mw, Nw=Nw, sw1=self.s_win, sw2=self.s_win)

        # window centers
        cy = torch.arange(Mw, device='cuda') * self.s_win + self.s_win // 2
        cx = torch.arange(Nw, device='cuda') * self.s_win + self.s_win // 2

        # MultiHead Attention
        out = self.att(pooled_maps, pooled_masks, feature_map, mask_q, cy, cx, freq_off)[:, :M, :N, :]
        out = out + feature_map_in

        # Layer Norm 2, MLP
        out_ln2 = self.ln2(out)
        out = self.ffwd(out_ln2) + out
        return out


class inception(nn.Module):
    def __init__(self, d_embd, dropout, n_blocks, n_heads, d_head, s_win, levels, s_region, s_pool, theta):
        super().__init__()
        self.blocks = nn.ModuleList(
            [block(d_embd, d_head, n_heads, s_win, levels, s_region, s_pool, dropout, theta) for _ in range(n_blocks)])

    def forward(self, feature_map, mask, freq_off):  # feature_map: (B, M, N, d_embd) -> already embedded in timesNet_model
        for blk in self.blocks:
            feature_map = blk(feature_map, mask, freq_off)
        return feature_map






