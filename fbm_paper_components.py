# Time–Frequency Modeling (TFM) module components: Trend, Seasonal, Interaction.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierBasisExpansion(nn.Module):
    """
    Pre‑computed Fourier basis expansion with learnable frequency weights.
    Implements the adaptive frequency selection mechanism described in the paper.
    """
    def __init__(self, seq_len: int, num_features: int, use_normalize: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.use_normalize = use_normalize

        # Pre‑compute Fourier basis functions
        ts = 1.0 / seq_len
        t = torch.arange(0, 1, ts, dtype=torch.float32)
        cos_list, sin_list = [], []
        for i in range(seq_len // 2 + 1):
            coef = 0.5 if i in (0, seq_len // 2) else 1.0
            cos_list.append(coef * torch.cos(2 * math.pi * i * t))
            sin_list.append(-coef * torch.sin(2 * math.pi * i * t))
        cos_basis = torch.stack(cos_list, dim=0)  # [P, T]
        sin_basis = torch.stack(sin_list, dim=0)  # [P, T]
        self.register_buffer("cos_basis", cos_basis)
        self.register_buffer("sin_basis", sin_basis)

        # Learnable frequency weights
        P = seq_len // 2 + 1
        self.freq_weights_raw = nn.Parameter(torch.zeros(P))

    def forward(self, x):
        """
        Input:  x [B, T, C]
        Output: time–frequency features [B, T, C]
        """
        B, T, C = x.shape
        xc = x.permute(0, 2, 1)                     # [B, C, T]
        X = torch.fft.rfft(xc, dim=-1) / T * 2    

        # Frequency weighting
        w = F.softplus(self.freq_weights_raw)        # non‑negative constraint
        if self.use_normalize:
            w = w / (w.sum() + 1e-8)
        w = w.view(1, 1, -1)
        X = X * w     

        basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.cos_basis)
        basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.sin_basis)
        z = basis_cos + basis_sin                    # [B, C, P, T]
        z_sum = z.sum(dim=2)                         # [B, C, T]
        return z_sum.permute(0, 2, 1)                # [B, T, C]


class TrendComponent(nn.Module):
    """
    Trend component – captures low‑frequency postural variations.
    Uses learnable frequency weights initialised to favour low‑frequency bands.
    """
    def __init__(self, seq_len: int, num_features: int, hidden_dim: int = 128, use_transformer: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer

        self.feature_projection = nn.Linear(num_features, hidden_dim)

        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
                dropout=0.1, batch_first=True
            )
            self.trend_net = nn.TransformerEncoder(enc_layer, num_layers=2)
        else:
            self.trend_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        self.output_projection = nn.Linear(hidden_dim, num_features)

        self.freq_weights = nn.Parameter(self.initialize_low_freq_weights(seq_len))

    def initialize_low_freq_weights(self, seq_len):
        """Initialise frequency weights with higher values for low‑frequency bins."""
        num_freqs = seq_len // 2 + 1
        freq_weights = torch.ones(num_freqs)
        low_freq_end = num_freqs // 2
        freq_weights[:low_freq_end] = 5.0          # boost low frequencies
        return freq_weights

    def forward(self, x):
        original_x = x
        x_proj = self.feature_projection(x)
        trend = self.trend_net(x_proj)
        trend = self.output_projection(trend)

        # Frequency‑domain weighting toward low‑frequency components
        B, T, C = trend.shape
        trend_fft = torch.fft.rfft(trend, dim=1)
        freq_len = trend_fft.size(1)

        # Align weight size
        w = F.pad(self.freq_weights, (0, max(0, freq_len - self.freq_weights.size(0))))[:freq_len]
        trend_fft = trend_fft * w.view(1, -1, 1)
        trend_filtered = torch.fft.irfft(trend_fft, n=T, dim=1)

        return trend_filtered + original_x           # residual connection


class SeasonalComponent(nn.Module):
    """
    Seasonal component – extracts periodic rhythmic patterns.
    Employs multi‑scale Fourier basis expansion and adaptive frequency weighting.
    """
    def __init__(self, seq_len: int, num_features: int, hidden_dim: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features

        self.fourier_expansion = FourierBasisExpansion(seq_len, num_features)

        self.seasonal_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_features)
        )

        # Multi‑scale processing (1, 1/2, 1/4 of original resolution)
        self.downsample_scales = [1, 2, 4]
        self.multiscale_weights = nn.Parameter(torch.ones(len(self.downsample_scales)))

    def forward(self, x):
        original_x = x
        time_freq = self.fourier_expansion(x)        # [B, T, C]

        # Multi‑scale down‑sampling
        multi_scale = [time_freq]
        cur = time_freq
        if cur.size(1) >= 4:
            down1 = F.avg_pool1d(cur.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            multi_scale.append(down1)
            if down1.size(1) >= 4:
                t2 = (down1.size(1) // 2) * 2
                d1_even = down1[:, :t2, :]
                down2 = d1_even.reshape(d1_even.size(0), t2 // 2, 2, d1_even.size(2)).sum(dim=2)
                multi_scale.append(down2)

        # Upsample & fuse with learnable weights
        target_size = multi_scale[0].size(1)
        upsampled = [
            F.interpolate(f.transpose(1, 2), size=target_size, mode='linear').transpose(1, 2)
            if f.size(1) != target_size else f
            for f in multi_scale
        ]
        weights = F.softmax(self.multiscale_weights[:len(upsampled)], dim=0)
        fused = sum(w * f for w, f in zip(weights, upsampled))

        seasonal_feature = self.seasonal_net(fused)
        return seasonal_feature + original_x          # residual connection


class InteractionComponent(nn.Module):
    """
    Interaction component – models non‑periodic transient dynamics
    using self‑attention and Transformer blocks.
    """
    def __init__(self, num_features: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.feature_projection = nn.Linear(num_features, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.output_projection = nn.Linear(hidden_dim, num_features)

    def forward(self, x):
        orig = x
        h = self.feature_projection(x)
        attn_out, _ = self.attn(h, h, h)
        h = self.encoder(attn_out)                   # Pre‑Norm Transformer
        out = self.output_projection(h)
        return out + orig                            # residual connection


def trend_component(seq_len, num_features, block_size=64, hidden_dim=128, use_transformer=False):
    return TrendComponent(seq_len, num_features, hidden_dim, use_transformer)

def seasonal_component(seq_len, num_features, block_size=64, hidden_dim=128, expansion_factor=2):
    return SeasonalComponent(seq_len, num_features, hidden_dim)

def interaction_component(seq_len, num_features, block_size=64, hidden_dim=128, num_heads=4):
    return InteractionComponent(num_features, hidden_dim, num_heads)
