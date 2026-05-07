# 效果好版本，预计算傅里叶基函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RevIN(nn.Module):
    def __init__(self, num_features: int, affine=True, eps=1e-5):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            mean = torch.mean(x, dim=1, keepdim=True)
            std = torch.std(x, dim=1, keepdim=True) + self.eps
            x_norm = (x - mean) / std
            if self.affine:
                x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            return x_norm
        elif mode == 'denorm':
            if self.affine:
                x_denorm = (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
            else:
                x_denorm = x
            return x_denorm
        else:
            raise ValueError(f"Unsupported mode: {mode}")

class FourierBasisExpansion(nn.Module):
    def __init__(self, seq_len: int, num_features: int, use_normalize: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.use_normalize = use_normalize

        # === 预计算傅里叶基函数 ===
        ts = 1.0 / seq_len
        t = torch.arange(0, 1, ts, dtype=torch.float32)
        cos_list, sin_list = [], []
        for i in range(seq_len // 2 + 1):
            coef = 0.5 if i in (0, seq_len // 2) else 1.0
            cos_list.append(coef * torch.cos(2 * math.pi * i * t))
            sin_list.append(-coef * torch.sin(2 * math.pi * i * t))
        cos_basis = torch.stack(cos_list, dim=0)  # [P,T]
        sin_basis = torch.stack(sin_list, dim=0)  # [P,T]
        self.register_buffer("cos_basis", cos_basis)
        self.register_buffer("sin_basis", sin_basis)

        # 可学习频率权重
        P = seq_len // 2 + 1
        self.freq_weights_raw = nn.Parameter(torch.zeros(P))

    def forward(self, x):
        """
        输入: x [B, T, C]
        输出: 时频特征 [B, T, C]
        """
        B, T, C = x.shape
        xc = x.permute(0, 2, 1)  # [B,C,T]
        X = torch.fft.rfft(xc, dim=-1) / T * 2  # [B,C,P]

        # 频率加权
        w = F.softplus(self.freq_weights_raw)
        if self.use_normalize:
            w = w / (w.sum() + 1e-8)
        w = w.view(1, 1, -1)
        X = X * w

        # 基函数展开
        basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.cos_basis)
        basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.sin_basis)
        z = basis_cos + basis_sin  # [B,C,P,T]
        z_sum = z.sum(dim=2)       # [B,C,T]
        return z_sum.permute(0, 2, 1)  # [B,T,C]


# Trend Component (低频趋势建模)
class TrendComponent(nn.Module):
    def __init__(self, seq_len: int, num_features: int, hidden_dim: int = 128, use_transformer: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer

        self.feature_projection = nn.Linear(num_features, hidden_dim)

        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
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

        # 低频权重
        self.freq_weights = nn.Parameter(torch.ones(seq_len // 2 + 1))

    def forward(self, x):
        original_x = x
        x_proj = self.feature_projection(x)
        trend = self.trend_net(x_proj)
        trend = self.output_projection(trend)

        # 频域加权低频分量
        B, T, C = trend.shape
        trend_fft = torch.fft.rfft(trend, dim=1)
        freq_len = trend_fft.size(1)
        w = F.pad(self.freq_weights, (0, max(0, freq_len - self.freq_weights.size(0))))[:freq_len]
        trend_fft = trend_fft * w.view(1, -1, 1)
        trend_filtered = torch.fft.irfft(trend_fft, n=T, dim=1)
        return trend_filtered + original_x

# Seasonal Component (周期建模)
class SeasonalComponent(nn.Module):
    def __init__(self, seq_len: int, num_features: int, hidden_dim: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features

        # 使用预计算的 Fourier 基函数
        self.fourier_expansion = FourierBasisExpansion(seq_len, num_features)

        self.seasonal_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_features)
        )

        # 多尺度融合
        self.downsample_scales = [1, 2, 4]
        self.multiscale_weights = nn.Parameter(torch.ones(len(self.downsample_scales)))

    def forward(self, x):
        original_x = x
        time_freq = self.fourier_expansion(x)  # [B,T,C]

        # 多尺度下采样
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

        # 上采样 & 融合
        target_size = multi_scale[0].size(1)
        upsampled = [F.interpolate(f.transpose(1, 2), size=target_size, mode='linear').transpose(1, 2)
                     if f.size(1) != target_size else f for f in multi_scale]
        weights = F.softmax(self.multiscale_weights[:len(upsampled)], dim=0)
        fused = sum(w * f for w, f in zip(weights, upsampled))

        seasonal_feature = self.seasonal_net(fused)
        return seasonal_feature + original_x

# Interaction Component (时域交互建模)

class InteractionComponent(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.feature_projection = nn.Linear(num_features, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                          dropout=0.1, batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                               dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.output_projection = nn.Linear(hidden_dim, num_features)

    def forward(self, x):
        orig = x
        h = self.feature_projection(x)
        attn_out, _ = self.attn(h, h, h)
        h = self.encoder(attn_out)
        out = self.output_projection(h)
        return out + orig


def trend_component(seq_len, num_features, block_size=64, hidden_dim=128, use_transformer=False):
    return TrendComponent(seq_len, num_features, hidden_dim, use_transformer)


def seasonal_component(seq_len, num_features, block_size=64, hidden_dim=128, expansion_factor=2):
    return SeasonalComponent(seq_len, num_features, hidden_dim)


def interaction_component(seq_len, num_features, block_size=64, hidden_dim=128, num_heads=8):
    return InteractionComponent(num_features, hidden_dim, num_heads)







# # 改进趋势组件，低频加权，预计算傅里叶基函数
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class FourierBasisExpansion(nn.Module):
#     def __init__(self, seq_len: int, num_features: int, use_normalize: bool = True):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.use_normalize = use_normalize

#         # === 预计算傅里叶基函数 ===
#         ts = 1.0 / seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos_list, sin_list = [], []
#         for i in range(seq_len // 2 + 1):
#             coef = 0.5 if i in (0, seq_len // 2) else 1.0
#             cos_list.append(coef * torch.cos(2 * math.pi * i * t))
#             sin_list.append(-coef * torch.sin(2 * math.pi * i * t))
#         cos_basis = torch.stack(cos_list, dim=0)  # [P,T]
#         sin_basis = torch.stack(sin_list, dim=0)  # [P,T]
#         self.register_buffer("cos_basis", cos_basis)
#         self.register_buffer("sin_basis", sin_basis)

#         # 可学习频率权重
#         P = seq_len // 2 + 1
#         self.freq_weights_raw = nn.Parameter(torch.zeros(P))

#     def forward(self, x):
#         """
#         输入: x [B, T, C]
#         输出: 时频特征 [B, T, C]
#         """
#         B, T, C = x.shape
#         xc = x.permute(0, 2, 1)  # [B,C,T]
#         X = torch.fft.rfft(xc, dim=-1) / T * 2  # [B,C,P]

#         # 频率加权
#         w = F.softplus(self.freq_weights_raw)
#         if self.use_normalize:
#             w = w / (w.sum() + 1e-8)
#         w = w.view(1, 1, -1)
#         X = X * w

#         # 基函数展开
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.cos_basis)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.sin_basis)
#         z = basis_cos + basis_sin  # [B,C,P,T]
#         z_sum = z.sum(dim=2)       # [B,C,T]
#         return z_sum.permute(0, 2, 1)  # [B,T,C]

# # Trend Component (低频趋势建模)
# class TrendComponent(nn.Module):
#     def __init__(self, seq_len: int, num_features: int, hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer

#         # 特征投影，将输入的 num_features 映射到 hidden_dim
#         self.feature_projection = nn.Linear(num_features, hidden_dim)

#         # 如果使用 Transformer 网络
#         if use_transformer:
#             enc_layer = nn.TransformerEncoderLayer(
#                 d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
#                 dropout=0.1, batch_first=True
#             )
#             self.trend_net = nn.TransformerEncoder(enc_layer, num_layers=2)
#         else:
#             self.trend_net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1)
#             )
        
#         # 输出投影，恢复到原始的 num_features 维度
#         self.output_projection = nn.Linear(hidden_dim, num_features)

#         # 低频权重初始化：强化低频区域
#         self.freq_weights = nn.Parameter(self.initialize_low_freq_weights(seq_len))

#     def initialize_low_freq_weights(self, seq_len):
#         # 初始化时给低频部分更高的权重
#         num_freqs = seq_len // 2 + 1
#         freq_weights = torch.ones(num_freqs)  # 初始化所有频率的权重为 1
        
#         # 给低频部分（例如 0 - 0.5 Hz）赋予较高的初始权重
#         low_freq_end = num_freqs // 2  # 假设低频的范围是前半部分
#         freq_weights[:low_freq_end] = 5.0  # 为低频部分赋予较高的权重（例如 5）
        
#         return freq_weights

#     def forward(self, x):
#         original_x = x
#         x_proj = self.feature_projection(x)
#         trend = self.trend_net(x_proj)
#         trend = self.output_projection(trend)

#         # 频域加权低频分量
#         B, T, C = trend.shape
#         trend_fft = torch.fft.rfft(trend, dim=1)  # 傅里叶变换
#         freq_len = trend_fft.size(1)

#         # 对齐权重大小
#         w = F.pad(self.freq_weights, (0, max(0, freq_len - self.freq_weights.size(0))))[:freq_len]
        
#         # 应用频率加权
#         trend_fft = trend_fft * w.view(1, -1, 1)

#         # 逆傅里叶变换回时域
#         trend_filtered = torch.fft.irfft(trend_fft, n=T, dim=1)

#         # 返回加权后的趋势加原始信号（残差连接）
#         return trend_filtered + original_x

# # Seasonal Component (周期建模)
# class SeasonalComponent(nn.Module):
#     def __init__(self, seq_len: int, num_features: int, hidden_dim: int = 128):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features

#         # 使用预计算的 Fourier 基函数
#         self.fourier_expansion = FourierBasisExpansion(seq_len, num_features)

#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )

#         # 多尺度融合
#         self.downsample_scales = [1, 2, 4]
#         self.multiscale_weights = nn.Parameter(torch.ones(len(self.downsample_scales)))

#     def forward(self, x):
#         original_x = x
#         time_freq = self.fourier_expansion(x)  # [B,T,C]

#         # 多尺度下采样
#         multi_scale = [time_freq]
#         cur = time_freq
#         if cur.size(1) >= 4:
#             down1 = F.avg_pool1d(cur.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
#             multi_scale.append(down1)
#             if down1.size(1) >= 4:
#                 t2 = (down1.size(1) // 2) * 2
#                 d1_even = down1[:, :t2, :]
#                 down2 = d1_even.reshape(d1_even.size(0), t2 // 2, 2, d1_even.size(2)).sum(dim=2)
#                 multi_scale.append(down2)

#         # 上采样 & 融合
#         target_size = multi_scale[0].size(1)
#         upsampled = [F.interpolate(f.transpose(1, 2), size=target_size, mode='linear').transpose(1, 2)
#                      if f.size(1) != target_size else f for f in multi_scale]
#         weights = F.softmax(self.multiscale_weights[:len(upsampled)], dim=0)
#         fused = sum(w * f for w, f in zip(weights, upsampled))

#         seasonal_feature = self.seasonal_net(fused)
#         return seasonal_feature + original_x

# # Interaction Component (时域交互建模)

# class InteractionComponent(nn.Module):
#     def __init__(self, num_features: int, hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
#         self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
#                                           dropout=0.1, batch_first=True)
#         enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
#                                                dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True)
#         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
#         self.output_projection = nn.Linear(hidden_dim, num_features)

#     def forward(self, x):
#         orig = x
#         h = self.feature_projection(x)
#         attn_out, _ = self.attn(h, h, h)
#         h = self.encoder(attn_out)
#         out = self.output_projection(h)
#         return out + orig

# def trend_component(seq_len, num_features, block_size=64, hidden_dim=128, use_transformer=False):
#     return TrendComponent(seq_len, num_features, hidden_dim, use_transformer)

# def seasonal_component(seq_len, num_features, block_size=64, hidden_dim=128, expansion_factor=2):
#     return SeasonalComponent(seq_len, num_features, hidden_dim)

# def interaction_component(seq_len, num_features, block_size=64, hidden_dim=128, num_heads=8):
#     return InteractionComponent(num_features, hidden_dim, num_heads)





# 完全采用时频联合特征，但是效果不好
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.fft import rfft, irfft
# from einops import rearrange
# import math

# class RevIN(nn.Module):
#     """
#     可逆实例归一化 (Reversible Instance Normalization)
#     基于FBM-S论文的实现，与fbm_paper_components.py完全一致
#     """
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             # 计算均值和标准差
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
            
#             # 归一化
#             x_norm = (x - mean) / std
            
#             # 仿射变换 - 使用gamma和beta参数
#             if self.affine:
#                 x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             return x_norm
        
#         elif mode == 'denorm':
#             # 反归一化
#             if self.affine:
#                 x_denorm = (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_denorm = x
#             return x_denorm
#         else:
#             raise ValueError(f"mode {mode} not supported.")

# class FourierBasisExpansion(nn.Module):
#     """
#     优化的傅里叶基函数扩展 - 流式处理，避免存储大的中间tensor
#     - 预计算基函数，避免重复计算
#     - 流式计算时频联合表示
#     - 内存优化版本
#     """
#     def __init__(self, seq_len: int, num_features: int, expansion_factor: int = 2,
#                  use_normalize: bool = True):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.expansion_factor = expansion_factor
#         self.expanded_len = seq_len * expansion_factor
#         self.use_normalize = use_normalize
        
#         # 预计算傅里叶基函数（向量化生成）
#         self.register_buffer('cos_basis', self._generate_cos_basis_vectorized())
#         self.register_buffer('sin_basis', self._generate_sin_basis_vectorized())
        
#         # 可学习频率权重
#         P = seq_len // 2 + 1
#         self.freq_weights_raw = nn.Parameter(torch.zeros(P))

#     def _generate_cos_basis_vectorized(self):
#         """向量化生成余弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         freqs = torch.arange(1, self.seq_len // 2 + 1, dtype=torch.float32)  # 跳过DC项
        
#         # 向量化计算所有频率的基函数
#         cos_basis = torch.cos(2 * math.pi * freqs.unsqueeze(1) * t.unsqueeze(0))  # [P-1, T]
        
#         # 处理边界情况
#         if self.seq_len % 2 == 0:  # 偶数长度
#             cos_basis = torch.cat([cos_basis, 0.5 * torch.cos(2 * math.pi * (self.seq_len // 2) * t).unsqueeze(0)], dim=0)
        
#         return cos_basis

#     def _generate_sin_basis_vectorized(self):
#         """向量化生成正弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         freqs = torch.arange(1, self.seq_len // 2 + 1, dtype=torch.float32)  # 跳过DC项
        
#         # 向量化计算所有频率的基函数
#         sin_basis = -torch.sin(2 * math.pi * freqs.unsqueeze(1) * t.unsqueeze(0))  # [P-1, T]
        
#         # 处理边界情况
#         if self.seq_len % 2 == 0:  # 偶数长度
#             sin_basis = torch.cat([sin_basis, -0.5 * torch.sin(2 * math.pi * (self.seq_len // 2) * t).unsqueeze(0)], dim=0)
        
#         return sin_basis

#     def forward(self, x):
#         """
#         流式计算时频联合表示 - 避免存储 [B,C,P,T]
#         输入: x [B, T, C]
#         输出: time_freq_features [B, T, C]
#         """
#         batch_size, seq_len, num_features = x.shape

#         # 1) FFT变换
#         xc = x.permute(0, 2, 1)  # [B,C,T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P] 复数

#         # 2) 应用频率权重
#         w = F.softplus(self.freq_weights_raw)  # 保证非负
#         if self.use_normalize:
#             w = w / (w.sum() + 1e-8)
#         w = w.view(1, 1, -1)  # [1,1,P]
#         X = X * w

#         # 3) 流式计算时域表示 - 避免存储 [B,C,P,T]
#         # 使用einsum进行批量矩阵乘法
#         cos_part = torch.einsum('bcp,pt->bct', X.real, self.cos_basis)  # [B,C,T]
#         sin_part = torch.einsum('bcp,pt->bct', X.imag, self.sin_basis)  # [B,C,T]
#         z_sum = cos_part + sin_part  # [B,C,T]

#         # 4) 转回 [B,T,C]
#         return z_sum.permute(0, 2, 1)


# class SeasonalComponent(nn.Module):
#     """
#     季节性组件：按照FBM-S论文实现
#     - 接收时频联合表示 [B,C,P,T] 和频域系数 [B,C,P]
#     - 重新进行基函数展开，使用可学习频域权重
#     - 输出时域特征 [B,T,C]
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64, 
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.expansion_factor = expansion_factor
        
#         # 预计算基函数（按照论文实现）
#         ts = 1.0 / seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
        
#         # 生成余弦和正弦基函数
#         cos_basis = []
#         sin_basis = []
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             else:
#                 if i == (seq_len // 2):
#                     cos = torch.vstack([cos, 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                     sin = torch.vstack([sin, -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#                 else:
#                     cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                     sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
        
#         # 注册为buffer
#         self.register_buffer('cos_basis', cos)
#         self.register_buffer('sin_basis', sin)
        
#         # 可学习的频域权重参数（按照论文实现）
#         # 确保parameter的维度与cos_basis/sin_basis去掉DC分量后一致
#         W_pos = torch.empty((seq_len // 2 + 1, seq_len))  # [P, T] 与cos_basis/sin_basis一致
#         nn.init.uniform_(W_pos, -0.001, 0.001)
#         self.parameter = nn.Parameter(W_pos, requires_grad=True)
        
#         # 季节性参数化网络
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )
    
#     def forward(self, time_freq_representation, freq_coefficients):
#         """
#         按照FBM-S论文实现的季节性组件
#         输入: 
#             time_freq_representation [B,C,P,T] - 时频联合表示
#             freq_coefficients [B,C,P] - 频域系数
#         输出: seasonal_features [B,T,C] - 时域特征
#         """
#         # 按照论文实现：去掉DC分量
#         x = time_freq_representation[:,:,1:,:]  # [B,C,P-1,T]
#         freq = freq_coefficients[:,:,1:]  # [B,C,P-1]
        
#         # 使用可学习参数进行频域加权
#         # 去掉DC分量，确保维度匹配
#         cos_basis_no_dc = self.cos_basis[1:,:]  # [P-1,T]
#         sin_basis_no_dc = self.sin_basis[1:,:]  # [P-1,T]
#         parameter_no_dc = self.parameter[1:,:]  # [P-1,T]
        
#         hidden_cos = torch.einsum('pt,pt->pt', cos_basis_no_dc, parameter_no_dc)  # [P-1,T]
#         hidden_sin = torch.einsum('pt,pt->pt', sin_basis_no_dc, parameter_no_dc)  # [P-1,T]
        
#         # 重新进行基函数展开
#         # freq.real/freq.imag: [B,C,P-1], hidden_cos/hidden_sin: [P-1,T]
#         # 计算 freq 与 hidden 的乘积
#         seasonal_features = torch.einsum('bcp,pt->bct', freq.real, hidden_cos) + \
#                           torch.einsum('bcp,pt->bct', freq.imag, hidden_sin)  # [B,C,T]
        
#         # 转置为 [B,T,C] 格式
#         seasonal_features = seasonal_features.transpose(1, 2)  # [B,T,C]
        
#         # 季节性参数化
#         seasonal_features = self.seasonal_net(seasonal_features)
        
#         return seasonal_features


# class InteractionComponent(nn.Module):
#     """
#     交互组件：基于时频联合表示工作（符合FBM论文设计）
#     - 接收时频联合表示 [B,C,P,T]
#     - 通过频域权重关注全频段
#     - 学习特征间的交互关系
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 4):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # 确保hidden_dim能被num_heads整除
#         if hidden_dim % num_heads != 0:
#             # 调整hidden_dim到最近的能被num_heads整除的值
#             adjusted_hidden_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads
#             print(f"Warning: hidden_dim {hidden_dim} not divisible by num_heads {num_heads}, adjusting to {adjusted_hidden_dim}")
#             hidden_dim = adjusted_hidden_dim
#             self.hidden_dim = hidden_dim  # 更新实例变量
        
#         # 时频联合表示处理网络 - 使用更轻量的方法
#         # 不直接处理展平的大维度，而是先降维
#         freq_len = seq_len // 2 + 1  # P
#         P_minus_1 = freq_len - 1     # P-1
        
#         # 内存优化：使用更简单的MLP，减少参数数量
#         # 直接映射：C -> hidden_dim，避免中间层
#         self.feature_projection = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),  # 直接映射
#             nn.ReLU(),
#             nn.Dropout(0.1)  # 添加dropout防止过拟合
#         )
        
#         # 恢复注意力机制，但使用更高效的方式
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # 使用单层Transformer，减少内存使用
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim,  # 减少feedforward维度
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 只用1层
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 论文中没有频域权重，移除不必要的参数
    
#     def forward(self, time_freq_representation, channel_mask=None):
#         """
#         按照FBM-S论文实现的交互组件 - 高效向量化版本
#         输入: 
#             time_freq_representation [B,C,P,T] - 时频联合表示
#             channel_mask - 通道掩码（可选）
#         输出: interaction_features [B,T,C] - 时域特征
#         """
#         # 按照论文实现：去掉DC分量
#         z = time_freq_representation[:,:,1:,:]  # [B,C,P-1,T]
#         B, C, P_minus_1, T = z.shape
        
#         # 高效向量化处理：避免循环，使用批量操作
#         # 1. 重塑为 [B*T, C, P-1] 以便进行批量处理
#         z_reshaped = z.permute(0, 3, 1, 2).contiguous()  # [B,T,C,P-1]
#         z_reshaped = z_reshaped.view(B * T, C, P_minus_1)  # [B*T, C, P-1]
        
#         # 2. 沿频率维度求和，得到每个时间步的通道特征
#         channel_features = z_reshaped.sum(dim=2)  # [B*T, C] - 沿频率维度求和
        
#         # 3. 通过 feature_projection 学习通道间交互
#         interaction_features = self.feature_projection(channel_features)  # [B*T, H]
        
#         # 4. 重塑为 [B*T, 1, H] 以便进行注意力计算
#         interaction_features = interaction_features.unsqueeze(1)  # [B*T, 1, H]
        
#         # 5. 使用多头注意力学习通道间交互
#         attn_output, _ = self.attention(
#             interaction_features, interaction_features, interaction_features
#         )  # [B*T, 1, H]
        
#         # 6. 重塑为 [B, T, H]
#         attn_output = attn_output.squeeze(1)  # [B*T, H]
#         attn_output = attn_output.view(B, T, -1)  # [B, T, H]
        
#         # 7. 使用Transformer进一步学习时间步间的交互
#         transformer_output = self.transformer(attn_output)  # [B, T, H]
        
#         # 8. 投影到目标通道数 C
#         if transformer_output.shape[-1] != C:
#             transformer_output = self.output_projection(transformer_output)  # [B, T, C]
        
#         return transformer_output


# class TrendComponent(nn.Module):
#     """
#     趋势组件：基于时频联合表示工作（符合FBM论文设计）
#     - 接收时频联合表示 [B,C,P,T]
#     - 通过频域权重关注低频分量
#     - 输出时域特征 [B,T,C]
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer
        
#         # 时频联合表示处理网络 - 使用轻量级MLP避免参数过多
#         # 计算频域长度
#         freq_len = seq_len // 2 + 1  # P
#         P_minus_1 = freq_len - 1     # P-1 (去掉DC分量)
        
#         # 按照论文：使用sum和mean进行多尺度下采样
#         self.multiscale = True  # 启用多尺度处理
        
#         if self.multiscale:
#             # 按照论文实现：多尺度下采样 + 线性层
#             # 原始尺度：P_minus_1
#             self.linear1 = nn.Linear(P_minus_1, hidden_dim)
            
#             # 1/2尺度：P_minus_1 // 2
#             self.linear2 = nn.Linear(P_minus_1 // 2, hidden_dim)
            
#             # 1/4尺度：P_minus_1 // 4  
#             self.linear3 = nn.Linear(P_minus_1 // 4, hidden_dim)
            
#             # 输出投影层
#             self.output_proj = nn.Linear(hidden_dim * 3, num_features)
#         else:
#             # 单尺度MLP
#             self.trend_net = nn.Sequential(
#                 nn.Linear(P_minus_1, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, num_features)
#             )
        
#         # 论文中没有频域权重，移除不必要的参数

#     def forward(self, time_freq_representation):
#         """
#         按照FBM-S论文实现的趋势组件 - 直接使用MLP处理时频联合表示
#         输入: time_freq_representation [B,C,P,T] - 时频联合表示
#         输出: trend_features [B,T,C] - 时域特征
#         """
#         # 按照论文实现：直接处理时频联合表示
#         # 去掉DC分量
#         x = time_freq_representation[:,:,1:,:]  # [B,C,P-1,T]
#         B, C, P_minus_1, T = x.shape
        
#         # 按照论文：使用sum和mean进行多尺度下采样
#         if self.multiscale:
#             # 按照论文实现：多尺度下采样
#             # 1. 重塑为 [B*C*T, P-1] 以便通过MLP
#             x_flat = x.permute(0, 1, 3, 2).contiguous()  # [B,C,T,P-1]
#             x_flat = x_flat.view(B * C * T, P_minus_1)  # [B*C*T, P-1]
            
#             # 2. 原始尺度处理
#             scale1 = self.linear1(x_flat)  # [B*C*T, hidden_dim]
            
#             # 3. 1/2尺度：按照论文使用sum和mean下采样
#             # 重塑为 [B*C*T, P_minus_1//2, 2] 然后进行下采样
#             x_half = x_flat.reshape(B * C * T, P_minus_1 // 2, 2)  # [B*C*T, P_minus_1//2, 2]
#             x_half_sum = x_half.sum(dim=-1)  # [B*C*T, P_minus_1//2] - sum下采样
#             x_half_mean = x_half.mean(dim=-1)  # [B*C*T, P_minus_1//2] - mean下采样
#             x_half_combined = x_half_sum + x_half_mean  # 组合sum和mean
#             scale2 = self.linear2(x_half_combined)  # [B*C*T, hidden_dim]
            
#             # 4. 1/4尺度：进一步下采样
#             # 从1/2尺度继续下采样到1/4尺度
#             x_quarter = x_half_combined.reshape(B * C * T, P_minus_1 // 4, 2)  # [B*C*T, P_minus_1//4, 2]
#             x_quarter_sum = x_quarter.sum(dim=-1)  # [B*C*T, P_minus_1//4] - sum下采样
#             x_quarter_mean = x_quarter.mean(dim=-1)  # [B*C*T, P_minus_1//4] - mean下采样
#             x_quarter_combined = x_quarter_sum + x_quarter_mean  # 组合sum和mean
#             scale3 = self.linear3(x_quarter_combined)  # [B*C*T, hidden_dim]
            
#             # 5. 融合多尺度特征
#             multiscale_features = torch.cat([scale1, scale2, scale3], dim=-1)  # [B*C*T, hidden_dim*3]
#             trend_output_flat = self.output_proj(multiscale_features)  # [B*C*T, num_features]
            
#         else:
#             # 单尺度处理
#             x_flat = x.permute(0, 1, 3, 2).contiguous()  # [B,C,T,P-1]
#             x_flat = x_flat.view(B * C * T, P_minus_1)  # [B*C*T, P-1]
#             trend_output_flat = self.trend_net(x_flat)  # [B*C*T, num_features]
        
#         # 4. 重新整形为 [B,C,T,num_features]
#         trend_output = trend_output_flat.view(B, C, T, -1)  # [B,C,T,num_features]
        
#         # 5. 转置为 [B,T,C] 格式
#         trend_output = trend_output.permute(0, 2, 1, 3)  # [B,T,C,num_features]
        
#         # 6. 确保输出是 [B,T,C] 格式
#         if trend_output.shape[-1] == 1:
#             trend_output = trend_output.squeeze(-1)  # [B,T,C]
#         else:
#             # 如果num_features > 1，取平均值
#             trend_output = trend_output.mean(dim=-1)  # [B,T,C]
        
#         return trend_output


# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 4) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)








# #  我的最终版本 去掉了分块处理和逐块中心化
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.fft import rfft, irfft
# from einops import rearrange
# import math

# class RevIN(nn.Module):
#     """
#     可逆实例归一化 (Reversible Instance Normalization)
#     基于FBM-S论文的实现，与fbm_paper_components.py完全一致
#     """
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             # 计算均值和标准差
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
            
#             # 归一化
#             x_norm = (x - mean) / std
            
#             # 仿射变换 - 使用gamma和beta参数
#             if self.affine:
#                 x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             return x_norm
        
#         elif mode == 'denorm':
#             # 反归一化
#             if self.affine:
#                 x_denorm = (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_denorm = x
#             return x_denorm
#         else:
#             raise ValueError(f"mode {mode} not supported.")


# class TimeBlocking(nn.Module):
#     """
#     时间分块策略：FBM-S论文的核心特性
#     将长序列分解为多个时间块，每个块独立处理
#     """
#     def __init__(self, seq_len: int, block_size: int = 64, overlap: int = 16):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
        
#         # 计算块数量
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
        
#         # 确保最后一个块能覆盖序列末尾
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         # x: [batch, seq_len, num_features]
#         batch_size, seq_len, num_features = x.shape
        
#         # 如果序列长度小于块大小，直接返回
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
        
#         blocks = []
#         block_positions = []
        
#         # 使用简单的滑动窗口策略
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             start_idx = i
#             end_idx = i + self.block_size
            
#             # 提取时间块
#             block = x[:, start_idx:end_idx, :]
#             blocks.append(block)
#             block_positions.append((start_idx, end_idx))
        
#         # 处理最后一个块，确保覆盖序列末尾
#         if block_positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             last_block = x[:, last_start:seq_len, :]
#             blocks.append(last_block)
#             block_positions.append((last_start, seq_len))
        
#         # 堆叠所有块
#         blocks = torch.stack(blocks, dim=1)  # [batch, num_blocks, block_size, num_features]
#         return blocks, block_positions
    
#     def reconstruct(self, blocks, block_positions):
#         # 重建原始序列
#         batch_size, num_blocks, block_size, num_features = blocks.shape
#         seq_len = max(pos[1] for pos in block_positions)
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
#         count = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
        
#         # 将每个块放回原位置
#         for i, (start, end) in enumerate(block_positions):
#             block = blocks[:, i, :end - start, :]
#             output[:, start:end, :] += block
#             count[:, start:end, :] += 1

        
#         # 平均重叠部分
#         count = torch.clamp(count, min=1)
#         output = output / count
        
#         return output


# class CenteringLayer(nn.Module):
#     """
#     中心化层：FBM-S论文的关键技术
#     每个块内部进行中心化处理，提高模型鲁棒性
#     """
#     def __init__(self, num_features: int, affine=True):
#         super().__init__()
#         self.num_features = num_features
#         self.affine = affine
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))
    
#     def forward(self, x, mode='center'):
#         if mode == 'center':
#             # 计算每个块的均值和标准差
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑为 [batch * num_blocks, block_size, num_features]
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 计算均值和标准差
#             mean = torch.mean(x_reshaped, dim=1, keepdim=True)
#             std = torch.std(x_reshaped, dim=1, keepdim=True) + 1e-5
            
#             # 中心化
#             x_centered = (x_reshaped - mean) / std
            
#             # 仿射变换
#             if self.affine:
#                 x_centered = x_centered * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             # 重塑回原形状
#             x_centered = x_centered.view(batch_size, num_blocks, block_size, num_features)
#             return x_centered, mean, std
        
#         elif mode == 'decenter':
#             x, mean, std = x
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 确保 mean 和 std 的维度匹配
#             if mean.size(0) != x_reshaped.size(0):
#                 # 如果维度不匹配，调整 mean 和 std 的维度
#                 target_batch = x_reshaped.size(0)
#                 if mean.size(0) > target_batch:
#                     mean_reshaped = mean[:target_batch].view(-1, 1, num_features)
#                     std_reshaped = std[:target_batch].view(-1, 1, num_features)
#                 else:
#                     # 重复 mean 和 std 到目标维度
#                     repeat_times = target_batch // mean.size(0)
#                     remainder = target_batch % mean.size(0)
#                     mean_repeated = mean.repeat(repeat_times, 1, 1)
#                     std_repeated = std.repeat(repeat_times, 1, 1)
#                     if remainder > 0:
#                         mean_repeated = torch.cat([mean_repeated, mean[:remainder]], dim=0)
#                         std_repeated = torch.cat([std_repeated, std[:remainder]], dim=0)
#                     mean_reshaped = mean_repeated.view(-1, 1, num_features)
#                     std_reshaped = std_repeated.view(-1, 1, num_features)
#             else:
#                 mean_reshaped = mean.view(-1, 1, num_features)
#                 std_reshaped = std.view(-1, 1, num_features)
            
#             # 反中心化
#             if self.affine:
#                 x_decentered = (x_reshaped - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_decentered = x_reshaped
            
#             x_decentered = x_decentered * std_reshaped + mean_reshaped
            
#             # 重塑回原形状
#             x_decentered = x_decentered.view(batch_size, num_blocks, block_size, num_features)
#             return x_decentered


# class FourierBasisExpansion(nn.Module):
#     """
#     傅里叶基函数扩展 (带可学习频率权重)
#     - 在论文基础上增加 freq_weights 参数，用于加权不同频段
#     - 适合分类任务，让模型学习关注哪些频段更有判别力
#     """
#     def __init__(self, seq_len: int, num_features: int, expansion_factor: int = 2,
#                  use_normalize: bool = True):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.expansion_factor = expansion_factor
#         self.expanded_len = seq_len * expansion_factor
#         self.use_normalize = use_normalize
        
#         # 生成傅里叶基函数
#         self.register_buffer('cos_basis', self._generate_cos_basis())
#         self.register_buffer('sin_basis', self._generate_sin_basis())
        
#         # === 新增：可学习频率权重 ===
#         P = seq_len // 2 + 1   # rfft 输出的频率数
#         self.freq_weights_raw = nn.Parameter(torch.zeros(P))  # raw 参数

#     def _generate_cos_basis(self):
#         """生成余弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#         else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#         cos = cos[1:]  # 去掉DC项
#         return cos

#     def _generate_sin_basis(self):
#         """生成正弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         sin = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         else:
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         sin = sin[1:]  # 去掉DC项
#         return sin

#     def forward(self, x):
#         """
#         构建时间-频域特征
#         输入: x [B, T, C]
#         输出: time_freq_features [B, T, C]
#         """
#         batch_size, seq_len, num_features = x.shape

#         # 1) rfft 变换
#         xc = x.permute(0, 2, 1)  # [B,C,T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P] 复数

#         # === 新增：应用可学习频率权重 ===
#         w = F.softplus(self.freq_weights_raw)  # 保证非负
#         if self.use_normalize:
#             w = w / (w.sum() + 1e-8)  # 归一化，避免权重无界增长
#         w = w.view(1, 1, -1)         # [1,1,P] 便于广播
#         X = X * w                    # 对实部/虚部同时加权

#         # 2) 使用构造函数中计算好的 cos_basis 和 sin_basis
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.cos_basis)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.sin_basis)
#         z_unified = basis_cos + basis_sin  # [B,C,P,T]
#         z_sum = z_unified.sum(dim=2)       # 按频率求和 → [B,C,T]

#         # 3) 转回 [B,T,C]
#         return z_sum.permute(0, 2, 1)


# class SeasonalComponent(nn.Module):

#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64, 
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.expansion_factor = expansion_factor
        
#         # 傅里叶基函数扩展（使用完整序列长度）
#         self.fourier_expansion = FourierBasisExpansion(seq_len, num_features, expansion_factor)
        
#         # 季节性网络
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )
        
#         # 多尺度下采样
#         self.downsample_scales = [1, 2, 4]  # 论文中的d₀, d₁, d₂
        
#         # 频域权重（季节性：中频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(seq_len // 2 + 1))
        
#         # 多尺度融合权重（可学习参数）
#         self.multiscale_weights = nn.Parameter(torch.ones(len(self.downsample_scales)))
    
#     def forward(self, x):
#         """
#         严格按照论文公式5实现（去掉分块处理）
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 获取傅里叶系数 [B, C, P] - 论文核心创新
#         xc = x.permute(0, 2, 1)  # [B, C, T]
#         X = torch.fft.rfft(xc, dim=-1) / xc.size(-1) * 2  # [B, C, P]

#         # 2. 论文核心创新：einsum基函数展开
#         # 动态生成基函数以匹配实际输入长度
#         actual_seq_len = x.size(1)
#         actual_freq_len = X.size(-1)
        
#         # 动态生成基函数
#         ts = 1.0 / actual_seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32, device=x.device)
        
#         cos_basis = None
#         sin_basis = None
#         for i in range(actual_freq_len):
#             if i == 0:
#                 cos_basis = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin_basis = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (actual_freq_len - 1):
#                 cos_basis = torch.vstack([cos_basis, 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin_basis = torch.vstack([sin_basis, -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#             else:
#                 cos_basis = torch.vstack([cos_basis, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin_basis = torch.vstack([sin_basis, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
        
#         # 确保基函数维度正确
#         cos_basis = cos_basis.to(x.device, dtype=x.dtype)  # [P, T]
#         sin_basis = sin_basis.to(x.device, dtype=x.dtype)  # [P, T]
        
#         # einsum基函数展开
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, cos_basis)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, sin_basis)
#         z_unified = basis_cos + basis_sin  # [B, C, P, T] - 时间-频域联合表示
        
#         # 3. 按频率求和得到时域表示 [B, C, T]
#         weighted_block = z_unified.sum(dim=2)  # [B, C, T]
#         weighted_block = weighted_block.transpose(1, 2)  # [B, T, C]
            
        
#         # 4. 多尺度下采样（针对 [B, T, C]，在时间维做离散 pairwise 聚合）
#         multi_scale_features = []
#         current_feature = weighted_block  # [B, T, C]
#         multi_scale_features.append(current_feature)

#         # 下采样尺度1 (T -> T/2)
#         if len(self.downsample_scales) > 1 and current_feature.size(1) >= 4:  # 确保有足够长度进行下采样
#             try:
#                 # 时间维度平均下采样
#                 down1 = F.avg_pool1d(current_feature.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
#                 # 边界检查：确保下采样后仍有有效长度
#                 if down1.size(1) > 0:
#                     multi_scale_features.append(down1)
#                     current_feature = down1  # 更新当前特征用于下一级下采样
#             except Exception as e:
#                 print(f"Warning: Downsampling scale 1 failed: {e}")
#                 # 如果下采样失败，跳过这一级
#                 pass

#         # 下采样尺度2 (T/2 -> T/4)
#         if len(self.downsample_scales) > 2 and current_feature.size(1) >= 4:  # 使用current_feature而不是检查变量存在性
#             try:
#                 # 确保长度为偶数
#                 t2 = (current_feature.size(1) // 2) * 2
#                 if t2 >= 2:  # 确保有足够长度
#                     d1_even = current_feature[:, :t2, :]
#                     down2 = d1_even.reshape(d1_even.size(0), t2 // 2, 2, d1_even.size(2)).sum(dim=2)  # [B,T/4,C]
#                     # 边界检查：确保下采样后仍有有效长度
#                     if down2.size(1) > 0:
#                         multi_scale_features.append(down2)
#             except Exception as e:
#                 print(f"Warning: Downsampling scale 2 failed: {e}")
#                 # 如果下采样失败，跳过这一级
#                 pass
        
#         # 5. 融合多尺度特征
#         if len(multi_scale_features) > 1:
#             # 上采样到相同大小
#             target_size = multi_scale_features[0].size(1)
#             upsampled_features = []
            
#             for feature in multi_scale_features:
#                 if feature.size(1) != target_size:
#                     feature = F.interpolate(
#                         feature.transpose(1, 2), 
#                         size=target_size, 
#                         mode='linear'
#                     ).transpose(1, 2)
#                 upsampled_features.append(feature)
            
#             # 使用可学习的多尺度融合权重
#             # 确保权重数量与特征数量匹配
#             num_features = len(upsampled_features)
#             if self.multiscale_weights.size(0) != num_features:
#                 # 调整权重维度
#                 if self.multiscale_weights.size(0) > num_features:
#                     weights = self.multiscale_weights[:num_features]
#                 else:
#                     # 用1.0填充不足的部分
#                     padding = torch.ones(num_features - self.multiscale_weights.size(0), 
#                                        device=self.multiscale_weights.device, 
#                                        dtype=self.multiscale_weights.dtype)
#                     weights = torch.cat([self.multiscale_weights, padding], dim=0)
#             else:
#                 weights = self.multiscale_weights
            
#             # 应用softmax确保权重和为1
#             weights = F.softmax(weights, dim=0)
            
#             # 加权融合
#             fused_feature = sum(w * f for w, f in zip(weights, upsampled_features))
#         else:
#             fused_feature = multi_scale_features[0]
        
#         # 6. 季节性参数化
#         seasonal_feature = self.seasonal_net(fused_feature)
        
#         # 7. 残差连接
#         if seasonal_feature.size() == original_x.size():
#             seasonal_feature = seasonal_feature + original_x
        
#         return seasonal_feature


# class InteractionComponent(nn.Module):
#     """
#     交互组件（去掉分块处理和逐块中心化）
#     - 保留投影、注意力、Transformer
#     - 删除固定掩码（C1, C2）
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         # 多头注意力
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
    
#     def forward(self, x):
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 特征投影
#         x_projected = self.feature_projection(x)  # [B, T, H]

#         # 2. 注意力 + Transformer
#         attn_output, _ = self.attention(x_projected, x_projected, x_projected)
#         transformer_output = self.transformer(attn_output)

#         # 3. 输出投影
#         interaction_feature = self.output_projection(transformer_output)
        
#         # 4. 残差连接
#         if interaction_feature.size() == original_x.size():
#             interaction_feature = interaction_feature + original_x
        
#         return interaction_feature


# class TrendComponent(nn.Module):
#     """
#     趋势组件：严格按照FBM-S论文实现（去掉分块处理）
#     MLP/Transformer + 频域权重
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         if use_transformer:
#             # Transformer编码器
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=8,
#                 dim_feedforward=hidden_dim * 2,
#                 dropout=0.1,
#                 batch_first=True
#             )
#             self.trend_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         else:
#             # MLP网络
#             self.trend_net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（趋势：低频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(seq_len // 2 + 1))

#     def forward(self, x):
#         """
#         严格按照论文实现趋势组件（去掉分块处理）
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 特征投影
#         x_projected = self.feature_projection(x)
        
#         # 2. 趋势建模
#         if self.use_transformer:
#             trend_output = self.trend_net(x_projected)
#         else:
#             trend_output = self.trend_net(x_projected)
        
#         # 3. 输出投影
#         trend_feature = self.output_projection(trend_output)

#         # 4. 在频域应用频率权重（趋势：低频分量权重大）
#         try:
#             # 对每个特征维度分别进行频域处理
#             trend_features_processed = []
#             for c in range(trend_feature.size(-1)):
#                 # 提取单个特征的时间序列
#                 feature_series = trend_feature[:, :, c]  # [batch, seq_len]
                
#                 # 傅里叶变换
#                 feature_fft = torch.fft.rfft(feature_series, dim=-1)  # [batch, freq_len]
                
#                 # 调整频域权重维度
#                 actual_freq_len = feature_fft.size(-1)
#                 if self.freq_weights.size(0) != actual_freq_len:
#                     if self.freq_weights.size(0) > actual_freq_len:
#                         freq_weights_resized = self.freq_weights[:actual_freq_len]
#                     else:
#                         # 用1.0填充不足的部分
#                         padding = torch.ones(actual_freq_len - self.freq_weights.size(0), 
#                                            device=self.freq_weights.device, 
#                                            dtype=self.freq_weights.dtype)
#                         freq_weights_resized = torch.cat([self.freq_weights, padding], dim=0)
#                 else:
#                     freq_weights_resized = self.freq_weights
                
#                 # 应用频域权重（趋势：低频分量权重大）
#                 freq_weights_resized = freq_weights_resized.to(feature_fft.device, dtype=feature_fft.dtype)
#                 feature_fft_weighted = feature_fft * freq_weights_resized.unsqueeze(0)
                
#                 # 逆傅里叶变换
#                 feature_processed = torch.fft.irfft(feature_fft_weighted, n=feature_series.size(-1), dim=-1)
#                 trend_features_processed.append(feature_processed)
            
#             # 重新组合特征
#             trend_feature = torch.stack(trend_features_processed, dim=-1)  # [batch, seq_len, num_features]
            
#         except Exception as e:
#             print(f"Warning: Frequency domain processing failed in trend component: {e}")
#             # 如果频域处理失败，使用原始特征
#             pass
        
#         # 5. 残差连接
#         if trend_feature.size() == original_x.size():
#             trend_feature = trend_feature + original_x
        
#         return trend_feature

# # 工厂函数
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)






# #  我的最终版本无滤波器  实现基函数
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.fft import rfft, irfft
# from einops import rearrange
# import math

# class RevIN(nn.Module):
#     """
#     可逆实例归一化 (Reversible Instance Normalization)
#     基于FBM-S论文的实现，与fbm_paper_components.py完全一致
#     """
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             # 计算均值和标准差
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
            
#             # 归一化
#             x_norm = (x - mean) / std
            
#             # 仿射变换 - 使用gamma和beta参数
#             if self.affine:
#                 x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             return x_norm
        
#         elif mode == 'denorm':
#             # 反归一化
#             if self.affine:
#                 x_denorm = (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_denorm = x
#             return x_denorm
#         else:
#             raise ValueError(f"mode {mode} not supported.")


# class TimeBlocking(nn.Module):
#     """
#     时间分块策略：FBM-S论文的核心特性
#     将长序列分解为多个时间块，每个块独立处理
#     """
#     def __init__(self, seq_len: int, block_size: int = 64, overlap: int = 16):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
        
#         # 计算块数量
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
        
#         # 确保最后一个块能覆盖序列末尾
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         # x: [batch, seq_len, num_features]
#         batch_size, seq_len, num_features = x.shape
        
#         # 如果序列长度小于块大小，直接返回
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
        
#         blocks = []
#         block_positions = []
        
#         # 使用简单的滑动窗口策略
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             start_idx = i
#             end_idx = i + self.block_size
            
#             # 提取时间块
#             block = x[:, start_idx:end_idx, :]
#             blocks.append(block)
#             block_positions.append((start_idx, end_idx))
        
#         # 处理最后一个块，确保覆盖序列末尾
#         if block_positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             last_block = x[:, last_start:seq_len, :]
#             blocks.append(last_block)
#             block_positions.append((last_start, seq_len))
        
#         # 堆叠所有块
#         blocks = torch.stack(blocks, dim=1)  # [batch, num_blocks, block_size, num_features]
#         return blocks, block_positions
    
#     def reconstruct(self, blocks, block_positions):
#         # 重建原始序列
#         batch_size, num_blocks, block_size, num_features = blocks.shape
#         seq_len = max(pos[1] for pos in block_positions)
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
#         count = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
        
#         # 将每个块放回原位置
#         for i, (start, end) in enumerate(block_positions):
#             block = blocks[:, i, :end - start, :]
#             output[:, start:end, :] += block
#             count[:, start:end, :] += 1

        
#         # 平均重叠部分
#         count = torch.clamp(count, min=1)
#         output = output / count
        
#         return output


# class CenteringLayer(nn.Module):
#     """
#     中心化层：FBM-S论文的关键技术
#     每个块内部进行中心化处理，提高模型鲁棒性
#     """
#     def __init__(self, num_features: int, affine=True):
#         super().__init__()
#         self.num_features = num_features
#         self.affine = affine
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))
    
#     def forward(self, x, mode='center'):
#         if mode == 'center':
#             # 计算每个块的均值和标准差
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑为 [batch * num_blocks, block_size, num_features]
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 计算均值和标准差
#             mean = torch.mean(x_reshaped, dim=1, keepdim=True)
#             std = torch.std(x_reshaped, dim=1, keepdim=True) + 1e-5
            
#             # 中心化
#             x_centered = (x_reshaped - mean) / std
            
#             # 仿射变换
#             if self.affine:
#                 x_centered = x_centered * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             # 重塑回原形状
#             x_centered = x_centered.view(batch_size, num_blocks, block_size, num_features)
#             return x_centered, mean, std
        
#         elif mode == 'decenter':
#             x, mean, std = x
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 确保 mean 和 std 的维度匹配
#             if mean.size(0) != x_reshaped.size(0):
#                 # 如果维度不匹配，调整 mean 和 std 的维度
#                 target_batch = x_reshaped.size(0)
#                 if mean.size(0) > target_batch:
#                     mean_reshaped = mean[:target_batch].view(-1, 1, num_features)
#                     std_reshaped = std[:target_batch].view(-1, 1, num_features)
#                 else:
#                     # 重复 mean 和 std 到目标维度
#                     repeat_times = target_batch // mean.size(0)
#                     remainder = target_batch % mean.size(0)
#                     mean_repeated = mean.repeat(repeat_times, 1, 1)
#                     std_repeated = std.repeat(repeat_times, 1, 1)
#                     if remainder > 0:
#                         mean_repeated = torch.cat([mean_repeated, mean[:remainder]], dim=0)
#                         std_repeated = torch.cat([std_repeated, std[:remainder]], dim=0)
#                     mean_reshaped = mean_repeated.view(-1, 1, num_features)
#                     std_reshaped = std_repeated.view(-1, 1, num_features)
#             else:
#                 mean_reshaped = mean.view(-1, 1, num_features)
#                 std_reshaped = std.view(-1, 1, num_features)
            
#             # 反中心化
#             if self.affine:
#                 x_decentered = (x_reshaped - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_decentered = x_reshaped
            
#             x_decentered = x_decentered * std_reshaped + mean_reshaped
            
#             # 重塑回原形状
#             x_decentered = x_decentered.view(batch_size, num_blocks, block_size, num_features)
#             return x_decentered


# class FourierBasisExpansion(nn.Module):
#     """
#     傅里叶基函数扩展：FBM-S论文的核心创新
#     实现时间-频域特征构建
#     """
#     def __init__(self, seq_len: int, num_features: int, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.expansion_factor = expansion_factor
#         self.expanded_len = seq_len * expansion_factor
        
#         # 生成傅里叶基函数
#         self.register_buffer('cos_basis', self._generate_cos_basis())
#         self.register_buffer('sin_basis', self._generate_sin_basis())
    
#     def _generate_cos_basis(self):
#         """生成余弦基函数 - 严格按照fbm_paper_components.py公式"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
        
#         # 关键修复：确保频率分量数量与rfft输出完全匹配
#         # rfft 输出频率分量数量是 seq_len // 2 + 1
#         num_freqs = self.seq_len // 2 + 1
        
#         for i in range(num_freqs):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 # Nyquist 频率（仅当偶数长度时存在半频点），系数取 0.5
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
        
#         # 重要：不要去掉DC项，因为rfft输出包含DC项
#         # cos = cos[1:]  # 注释掉这行
#         return cos
    
#     def _generate_sin_basis(self):
#         """生成正弦基函数 - 严格按照fbm_paper_components.py公式"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         sin = None
        
#         # 关键修复：确保频率分量数量与rfft输出完全匹配
#         # rfft 输出频率分量数量是 seq_len // 2 + 1
#         num_freqs = self.seq_len // 2 + 1
        
#         for i in range(num_freqs):
#             if i == 0:
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 # Nyquist 频率（仅当偶数长度时存在半频点），系数取 -0.5
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#             else:
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
        
#         # 重要：不要去掉DC项，因为rfft输出包含DC项
#         # sin = sin[1:]  # 注释掉这行
#         return sin
    
#     def forward(self, x):
#         """
#         构建时间-频域特征
#         输入: x [batch, seq_len, num_features]
#         输出: time_freq_features [batch, seq_len, num_features]
#         """
#         batch_size, seq_len, num_features = x.shape

#         # 1) 按论文规范进行RFFT归一化（/T*2），并在通道维上计算
#         xc = x.permute(0, 2, 1)  # [B,C,T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P]

#         # 2) 使用预计算的基函数，确保维度匹配
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.cos_basis)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.sin_basis)
#         z_unified = basis_cos + basis_sin  # [B,C,P,T]
#         z_sum = z_unified.sum(dim=2)       # 按频率求和 → [B,C,T]

#         # 3) 转回 [B,T,C]
#         return z_sum.permute(0, 2, 1)

# class SeasonalComponent(nn.Module):

#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64, 
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.expansion_factor = expansion_factor
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 傅里叶基函数扩展
#         self.fourier_expansion = FourierBasisExpansion(block_size, num_features, expansion_factor)
        
#         # 滚动窗口权重（论文公式5中的W）
#         # 确保维度与傅里叶基函数匹配
#         self.rolling_window = nn.Parameter(torch.randn(block_size, block_size // 2 + 1))
        
#         # 季节性网络
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )
        
#         # 多尺度下采样
#         self.downsample_scales = [1, 2, 4]  # 论文中的d₀, d₁, d₂
        
#         # 频域权重（季节性：中频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))
        
#         # 多尺度融合权重（可学习参数）
#         self.multiscale_weights = nn.Parameter(torch.ones(len(self.downsample_scales)))
    
#     def forward(self, x):
#         """
#         严格按照论文公式5实现
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用季节性处理
#         seasonal_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 1. 获取傅里叶系数 [B, C, P] - 论文核心创新
#             xc = block.permute(0, 2, 1)  # [B, C, T]
#             X = torch.fft.rfft(xc, dim=-1) / xc.size(-1) * 2  # [B, C, P]

#             # 2. 论文核心创新：einsum基函数展开
#             # 使用FourierBasisExpansion的基函数进行时间-频域联合表示
#             basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.fourier_expansion.cos_basis)
#             basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.fourier_expansion.sin_basis)
#             z_unified = basis_cos + basis_sin  # [B, C, P, T] - 时间-频域联合表示
            
#             # 3. 按频率求和得到时域表示 [B, C, T]
#             weighted_block = z_unified.sum(dim=2)  # [B, C, T]
#             weighted_block = weighted_block.transpose(1, 2)  # [B, T, C]
            
            
#             # 6. 多尺度下采样（针对 [B, T, C]，在时间维做离散 pairwise 聚合）
#             multi_scale_features = []
#             current_feature = weighted_block  # [B, T, C]
#             multi_scale_features.append(current_feature)

#             # 下采样尺度1 (T -> T/2)
#             if len(self.downsample_scales) > 1 and current_feature.size(1) >= 4:  # 确保有足够长度进行下采样
#                 try:
#                     # 时间维度平均下采样
#                     down1 = F.avg_pool1d(current_feature.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
#                     # 边界检查：确保下采样后仍有有效长度
#                     if down1.size(1) > 0:
#                         multi_scale_features.append(down1)
#                         current_feature = down1  # 更新当前特征用于下一级下采样
#                 except Exception as e:
#                     print(f"Warning: Downsampling scale 1 failed: {e}")
#                     # 如果下采样失败，跳过这一级
#                     pass

#             # 下采样尺度2 (T/2 -> T/4)
#             if len(self.downsample_scales) > 2 and current_feature.size(1) >= 4:  # 使用current_feature而不是检查变量存在性
#                 try:
#                     # 确保长度为偶数
#                     t2 = (current_feature.size(1) // 2) * 2
#                     if t2 >= 2:  # 确保有足够长度
#                         d1_even = current_feature[:, :t2, :]
#                         down2 = d1_even.reshape(d1_even.size(0), t2 // 2, 2, d1_even.size(2)).sum(dim=2)  # [B,T/4,C]
#                         # 边界检查：确保下采样后仍有有效长度
#                         if down2.size(1) > 0:
#                             multi_scale_features.append(down2)
#                 except Exception as e:
#                     print(f"Warning: Downsampling scale 2 failed: {e}")
#                     # 如果下采样失败，跳过这一级
#                     pass
            
#             # 7. 融合多尺度特征
#             if len(multi_scale_features) > 1:
#                 # 上采样到相同大小
#                 target_size = multi_scale_features[0].size(1)
#                 upsampled_features = []
                
#                 for feature in multi_scale_features:
#                     if feature.size(1) != target_size:
#                         feature = F.interpolate(
#                             feature.transpose(1, 2), 
#                             size=target_size, 
#                             mode='linear'
#                         ).transpose(1, 2)
#                     upsampled_features.append(feature)
                
#                 # 使用可学习的多尺度融合权重
#                 # 确保权重数量与特征数量匹配
#                 num_features = len(upsampled_features)
#                 if self.multiscale_weights.size(0) != num_features:
#                     # 调整权重维度
#                     if self.multiscale_weights.size(0) > num_features:
#                         weights = self.multiscale_weights[:num_features]
#                     else:
#                         # 用1.0填充不足的部分
#                         padding = torch.ones(num_features - self.multiscale_weights.size(0), 
#                                            device=self.multiscale_weights.device, 
#                                            dtype=self.multiscale_weights.dtype)
#                         weights = torch.cat([self.multiscale_weights, padding], dim=0)
#                 else:
#                     weights = self.multiscale_weights
                
#                 # 应用softmax确保权重和为1
#                 weights = F.softmax(weights, dim=0)
                
#                 # 加权融合
#                 fused_feature = sum(w * f for w, f in zip(weights, upsampled_features))
#             else:
#                 fused_feature = multi_scale_features[0]
            
#             # 8. 季节性参数化
#             seasonal_feature = self.seasonal_net(fused_feature)
#             seasonal_features.append(seasonal_feature)

#         # 9. 重建序列
#         seasonal_features = torch.stack(seasonal_features, dim=1)
#         # 10. 反中心化
#         seasonal_features = self.centering((seasonal_features, mean, std), mode='decenter')
        
#         # 11. 重建原始序列
#         seasonal_features = self.time_blocking.reconstruct(seasonal_features, block_positions)
        
#         # 12. 残差连接
#         if seasonal_features.size() == original_x.size():
#             seasonal_features = seasonal_features + original_x
        
#         return seasonal_features


# class InteractionComponent(nn.Module):
#     """
#     交互组件（去掉掩码版本）
#     - 保留时间分块、中心化、投影、注意力、Transformer
#     - 删除固定掩码（C1, C2）
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         # 多头注意力
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
    
#     def forward(self, x):
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用交互处理
#         interaction_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]

#             # 4. 特征投影
#             block_projected = self.feature_projection(block)  # [B, T, H]

#             # 5. 注意力 + Transformer
#             attn_output, _ = self.attention(block_projected, block_projected, block_projected)
#             transformer_output = self.transformer(attn_output)

#             # 6. 输出投影
#             interaction_feature = self.output_projection(transformer_output)
#             interaction_features.append(interaction_feature)
        
#         # 7. 重建序列
#         interaction_features = torch.stack(interaction_features, dim=1)
        
#         # 8. 反中心化
#         interaction_features = self.centering((interaction_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         interaction_features = self.time_blocking.reconstruct(interaction_features, block_positions)
        
#         # 10. 残差连接
#         if interaction_features.size() == original_x.size():
#             interaction_features = interaction_features + original_x
        
#         return interaction_features


# class TrendComponent(nn.Module):
#     """
#     趋势组件：严格按照FBM-S论文实现
#     时间分块 + MLP/Transformer + 频域权重
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         if use_transformer:
#             # Transformer编码器
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=8,
#                 dim_feedforward=hidden_dim * 2,
#                 dropout=0.1,
#                 batch_first=True
#             )
#             self.trend_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         else:
#             # MLP网络
#             self.trend_net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（趋势：低频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))

#     def forward(self, x):
#         """
#         严格按照论文实现趋势组件
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x

#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用趋势处理
#         trend_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 特征投影
#             block_projected = self.feature_projection(block)
            
#             # 5. 趋势建模
#             if self.use_transformer:
#                 trend_output = self.trend_net(block_projected)
#             else:
#                 trend_output = self.trend_net(block_projected)
            
#             # 6. 输出投影
#             trend_feature = self.output_projection(trend_output)
#             trend_features.append(trend_feature)
        
#         # 7. 重建序列
#         trend_features = torch.stack(trend_features, dim=1)
        
#         # 8. 反中心化
#         trend_features = self.centering((trend_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         trend_features = self.time_blocking.reconstruct(trend_features, block_positions)

#         # 在频域应用频率权重（趋势：低频分量权重大）
#         try:
#             # 对每个特征维度分别进行频域处理
#             trend_features_processed = []
#             for c in range(trend_features.size(-1)):
#                 # 提取单个特征的时间序列
#                 feature_series = trend_features[:, :, c]  # [batch, seq_len]
                
#                 # 傅里叶变换
#                 feature_fft = torch.fft.rfft(feature_series, dim=-1)  # [batch, freq_len]
                
#                 # 调整频域权重维度
#                 actual_freq_len = feature_fft.size(-1)
#                 if self.freq_weights.size(0) != actual_freq_len:
#                     if self.freq_weights.size(0) > actual_freq_len:
#                         freq_weights_resized = self.freq_weights[:actual_freq_len]
#                     else:
#                         # 用1.0填充不足的部分
#                         padding = torch.ones(actual_freq_len - self.freq_weights.size(0), 
#                                            device=self.freq_weights.device, 
#                                            dtype=self.freq_weights.dtype)
#                         freq_weights_resized = torch.cat([self.freq_weights, padding], dim=0)
#                 else:
#                     freq_weights_resized = self.freq_weights
                
#                 # 应用频域权重（趋势：低频分量权重大）
#                 freq_weights_resized = freq_weights_resized.to(feature_fft.device, dtype=feature_fft.dtype)
#                 feature_fft_weighted = feature_fft * freq_weights_resized.unsqueeze(0)
                
#                 # 逆傅里叶变换
#                 feature_processed = torch.fft.irfft(feature_fft_weighted, n=feature_series.size(-1), dim=-1)
#                 trend_features_processed.append(feature_processed)
            
#             # 重新组合特征
#             trend_features = torch.stack(trend_features_processed, dim=-1)  # [batch, seq_len, num_features]
            
#         except Exception as e:
#             print(f"Warning: Frequency domain processing failed in trend component: {e}")
#             # 如果频域处理失败，使用原始特征
#             pass
        
#         # 10. 残差连接
#         if trend_features.size() == original_x.size():
#             trend_features = trend_features + original_x
        
#         return trend_features

# # 工厂函数
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)






# #  我的最终版本 去除滤波器
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.fft import rfft, irfft
# from einops import rearrange
# import math

# class RevIN(nn.Module):
#     """
#     可逆实例归一化 (Reversible Instance Normalization)
#     基于FBM-S论文的实现，与fbm_paper_components.py完全一致
#     """
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             # 计算均值和标准差
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
            
#             # 归一化
#             x_norm = (x - mean) / std
            
#             # 仿射变换 - 使用gamma和beta参数
#             if self.affine:
#                 x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             return x_norm
        
#         elif mode == 'denorm':
#             # 反归一化
#             if self.affine:
#                 x_denorm = (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_denorm = x
#             return x_denorm
#         else:
#             raise ValueError(f"mode {mode} not supported.")


# class TimeBlocking(nn.Module):
#     """
#     时间分块策略：FBM-S论文的核心特性
#     将长序列分解为多个时间块，每个块独立处理
#     """
#     def __init__(self, seq_len: int, block_size: int = 64, overlap: int = 16):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
        
#         # 计算块数量
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
        
#         # 确保最后一个块能覆盖序列末尾
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         # x: [batch, seq_len, num_features]
#         batch_size, seq_len, num_features = x.shape
        
#         # 如果序列长度小于块大小，直接返回
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
        
#         blocks = []
#         block_positions = []
        
#         # 使用简单的滑动窗口策略
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             start_idx = i
#             end_idx = i + self.block_size
            
#             # 提取时间块
#             block = x[:, start_idx:end_idx, :]
#             blocks.append(block)
#             block_positions.append((start_idx, end_idx))
        
#         # 处理最后一个块，确保覆盖序列末尾
#         if block_positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             last_block = x[:, last_start:seq_len, :]
#             blocks.append(last_block)
#             block_positions.append((last_start, seq_len))
        
#         # 堆叠所有块
#         blocks = torch.stack(blocks, dim=1)  # [batch, num_blocks, block_size, num_features]
#         return blocks, block_positions
    
#     def reconstruct(self, blocks, block_positions):
#         # 重建原始序列
#         batch_size, num_blocks, block_size, num_features = blocks.shape
#         seq_len = max(pos[1] for pos in block_positions)
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
#         count = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
        
#         # 将每个块放回原位置
#         for i, (start, end) in enumerate(block_positions):
#             block = blocks[:, i, :end - start, :]
#             output[:, start:end, :] += block
#             count[:, start:end, :] += 1

        
#         # 平均重叠部分
#         count = torch.clamp(count, min=1)
#         output = output / count
        
#         return output


# class CenteringLayer(nn.Module):
#     """
#     中心化层：FBM-S论文的关键技术
#     每个块内部进行中心化处理，提高模型鲁棒性
#     """
#     def __init__(self, num_features: int, affine=True):
#         super().__init__()
#         self.num_features = num_features
#         self.affine = affine
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))
    
#     def forward(self, x, mode='center'):
#         if mode == 'center':
#             # 计算每个块的均值和标准差
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑为 [batch * num_blocks, block_size, num_features]
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 计算均值和标准差
#             mean = torch.mean(x_reshaped, dim=1, keepdim=True)
#             std = torch.std(x_reshaped, dim=1, keepdim=True) + 1e-5
            
#             # 中心化
#             x_centered = (x_reshaped - mean) / std
            
#             # 仿射变换
#             if self.affine:
#                 x_centered = x_centered * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             # 重塑回原形状
#             x_centered = x_centered.view(batch_size, num_blocks, block_size, num_features)
#             return x_centered, mean, std
        
#         elif mode == 'decenter':
#             x, mean, std = x
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 确保 mean 和 std 的维度匹配
#             if mean.size(0) != x_reshaped.size(0):
#                 # 如果维度不匹配，调整 mean 和 std 的维度
#                 target_batch = x_reshaped.size(0)
#                 if mean.size(0) > target_batch:
#                     mean_reshaped = mean[:target_batch].view(-1, 1, num_features)
#                     std_reshaped = std[:target_batch].view(-1, 1, num_features)
#                 else:
#                     # 重复 mean 和 std 到目标维度
#                     repeat_times = target_batch // mean.size(0)
#                     remainder = target_batch % mean.size(0)
#                     mean_repeated = mean.repeat(repeat_times, 1, 1)
#                     std_repeated = std.repeat(repeat_times, 1, 1)
#                     if remainder > 0:
#                         mean_repeated = torch.cat([mean_repeated, mean[:remainder]], dim=0)
#                         std_repeated = torch.cat([std_repeated, std[:remainder]], dim=0)
#                     mean_reshaped = mean_repeated.view(-1, 1, num_features)
#                     std_reshaped = std_repeated.view(-1, 1, num_features)
#             else:
#                 mean_reshaped = mean.view(-1, 1, num_features)
#                 std_reshaped = std.view(-1, 1, num_features)
            
#             # 反中心化
#             if self.affine:
#                 x_decentered = (x_reshaped - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_decentered = x_reshaped
            
#             x_decentered = x_decentered * std_reshaped + mean_reshaped
            
#             # 重塑回原形状
#             x_decentered = x_decentered.view(batch_size, num_blocks, block_size, num_features)
#             return x_decentered


# class FourierBasisExpansion(nn.Module):
#     """
#     傅里叶基函数扩展 (带可学习频率权重)
#     - 在论文基础上增加 freq_weights 参数，用于加权不同频段
#     - 适合分类任务，让模型学习关注哪些频段更有判别力
#     """
#     def __init__(self, seq_len: int, num_features: int, expansion_factor: int = 2,
#                  use_normalize: bool = True):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.expansion_factor = expansion_factor
#         self.expanded_len = seq_len * expansion_factor
#         self.use_normalize = use_normalize
        
#         # 生成傅里叶基函数
#         self.register_buffer('cos_basis', self._generate_cos_basis())
#         self.register_buffer('sin_basis', self._generate_sin_basis())
        
#         # === 新增：可学习频率权重 ===
#         P = seq_len // 2 + 1   # rfft 输出的频率数
#         self.freq_weights_raw = nn.Parameter(torch.zeros(P))  # raw 参数

#     def _generate_cos_basis(self):
#         """生成余弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#         else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#         cos = cos[1:]  # 去掉DC项
#         return cos

#     def _generate_sin_basis(self):
#         """生成正弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         sin = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         else:
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         sin = sin[1:]  # 去掉DC项
#         return sin

#     def forward(self, x):
#         """
#         构建时间-频域特征
#         输入: x [B, T, C]
#         输出: time_freq_features [B, T, C]
#         """
#         batch_size, seq_len, num_features = x.shape

#         # 1) rfft 变换
#         xc = x.permute(0, 2, 1)  # [B,C,T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P] 复数

#         # === 新增：应用可学习频率权重 ===
#         w = F.softplus(self.freq_weights_raw)  # 保证非负
#         if self.use_normalize:
#             w = w / (w.sum() + 1e-8)  # 归一化，避免权重无界增长
#         w = w.view(1, 1, -1)         # [1,1,P] 便于广播
#         X = X * w                    # 对实部/虚部同时加权

#         # 2) 使用构造函数中计算好的 cos_basis 和 sin_basis
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.cos_basis)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.sin_basis)
#         z_unified = basis_cos + basis_sin  # [B,C,P,T]
#         z_sum = z_unified.sum(dim=2)       # 按频率求和 → [B,C,T]

#         # 3) 转回 [B,T,C]
#         return z_sum.permute(0, 2, 1)


# class SeasonalComponent(nn.Module):

#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64, 
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.expansion_factor = expansion_factor
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 傅里叶基函数扩展
#         self.fourier_expansion = FourierBasisExpansion(block_size, num_features, expansion_factor)
        
#         # 滚动窗口权重（论文公式5中的W）
#         # 确保维度与傅里叶基函数匹配
#         self.rolling_window = nn.Parameter(torch.randn(block_size, block_size // 2 + 1))
        
#         # 季节性网络
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )
        
#         # 多尺度下采样
#         self.downsample_scales = [1, 2, 4]  # 论文中的d₀, d₁, d₂
        
#         # 频域权重（季节性：中频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))
        
#         # 多尺度融合权重（可学习参数）
#         self.multiscale_weights = nn.Parameter(torch.ones(len(self.downsample_scales)))
    
#     def forward(self, x):
#         """
#         严格按照论文公式5实现
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用季节性处理
#         seasonal_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 1. 获取傅里叶系数 [B, C, P]
#             xc = block.permute(0, 2, 1)  # [B, C, T]
#             X = torch.fft.rfft(xc, dim=-1) / xc.size(-1) * 2  # [B, C, P]

#             # 2. 应用频域权重进行有意义的频域处理
#             # 确保频域权重维度匹配
#             freq_len = X.size(-1)
#             if self.freq_weights.size(0) != freq_len:
#                 # 调整频域权重维度
#                 if self.freq_weights.size(0) > freq_len:
#                     freq_weights = self.freq_weights[:freq_len]
#                 else:
#                     # 用1.0填充不足的部分
#                     padding = torch.ones(freq_len - self.freq_weights.size(0), 
#                                        device=self.freq_weights.device, 
#                                        dtype=self.freq_weights.dtype)
#                     freq_weights = torch.cat([self.freq_weights, padding], dim=0)
#             else:
#                 freq_weights = self.freq_weights
            
#             # 应用频域权重（季节性：中频分量权重大）
#             freq_weights = freq_weights.view(1, 1, -1)  # [1, 1, P]
#             X_weighted = X * freq_weights.to(X.device, dtype=X.dtype)
            
#             # 3. 论文方式：直接逆傅里叶变换（无滤波器）
#             # 使用逆傅里叶变换回到时域，类似论文但保持频域权重
#             weighted_block = torch.fft.irfft(X_weighted, n=block.size(1), dim=-1)  # [B, C, T]
#             weighted_block = weighted_block.transpose(1, 2)  # [B, T, C]
            
            
#             # 6. 多尺度下采样（针对 [B, T, C]，在时间维做离散 pairwise 聚合）
#             multi_scale_features = []
#             current_feature = weighted_block  # [B, T, C]
#             multi_scale_features.append(current_feature)

#             # 下采样尺度1 (T -> T/2)
#             if len(self.downsample_scales) > 1 and current_feature.size(1) >= 4:  # 确保有足够长度进行下采样
#                 try:
#                     # 时间维度平均下采样
#                     down1 = F.avg_pool1d(current_feature.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
#                     # 边界检查：确保下采样后仍有有效长度
#                     if down1.size(1) > 0:
#                         multi_scale_features.append(down1)
#                         current_feature = down1  # 更新当前特征用于下一级下采样
#                 except Exception as e:
#                     print(f"Warning: Downsampling scale 1 failed: {e}")
#                     # 如果下采样失败，跳过这一级
#                     pass

#             # 下采样尺度2 (T/2 -> T/4)
#             if len(self.downsample_scales) > 2 and current_feature.size(1) >= 4:  # 使用current_feature而不是检查变量存在性
#                 try:
#                     # 确保长度为偶数
#                     t2 = (current_feature.size(1) // 2) * 2
#                     if t2 >= 2:  # 确保有足够长度
#                         d1_even = current_feature[:, :t2, :]
#                         down2 = d1_even.reshape(d1_even.size(0), t2 // 2, 2, d1_even.size(2)).sum(dim=2)  # [B,T/4,C]
#                         # 边界检查：确保下采样后仍有有效长度
#                         if down2.size(1) > 0:
#                             multi_scale_features.append(down2)
#                 except Exception as e:
#                     print(f"Warning: Downsampling scale 2 failed: {e}")
#                     # 如果下采样失败，跳过这一级
#                     pass
            
#             # 7. 融合多尺度特征
#             if len(multi_scale_features) > 1:
#                 # 上采样到相同大小
#                 target_size = multi_scale_features[0].size(1)
#                 upsampled_features = []
                
#                 for feature in multi_scale_features:
#                     if feature.size(1) != target_size:
#                         feature = F.interpolate(
#                             feature.transpose(1, 2), 
#                             size=target_size, 
#                             mode='linear'
#                         ).transpose(1, 2)
#                     upsampled_features.append(feature)
                
#                 # 使用可学习的多尺度融合权重
#                 # 确保权重数量与特征数量匹配
#                 num_features = len(upsampled_features)
#                 if self.multiscale_weights.size(0) != num_features:
#                     # 调整权重维度
#                     if self.multiscale_weights.size(0) > num_features:
#                         weights = self.multiscale_weights[:num_features]
#                     else:
#                         # 用1.0填充不足的部分
#                         padding = torch.ones(num_features - self.multiscale_weights.size(0), 
#                                            device=self.multiscale_weights.device, 
#                                            dtype=self.multiscale_weights.dtype)
#                         weights = torch.cat([self.multiscale_weights, padding], dim=0)
#                 else:
#                     weights = self.multiscale_weights
                
#                 # 应用softmax确保权重和为1
#                 weights = F.softmax(weights, dim=0)
                
#                 # 加权融合
#                 fused_feature = sum(w * f for w, f in zip(weights, upsampled_features))
#             else:
#                 fused_feature = multi_scale_features[0]
            
#             # 8. 季节性参数化
#             seasonal_feature = self.seasonal_net(fused_feature)
#             seasonal_features.append(seasonal_feature)

#         # 9. 重建序列
#         seasonal_features = torch.stack(seasonal_features, dim=1)
#         # 10. 反中心化
#         seasonal_features = self.centering((seasonal_features, mean, std), mode='decenter')
        
#         # 11. 重建原始序列
#         seasonal_features = self.time_blocking.reconstruct(seasonal_features, block_positions)
        
#         # 12. 残差连接
#         if seasonal_features.size() == original_x.size():
#             seasonal_features = seasonal_features + original_x
        
#         return seasonal_features


# class InteractionComponent(nn.Module):
#     """
#     交互组件（去掉掩码版本）
#     - 保留时间分块、中心化、投影、注意力、Transformer
#     - 删除固定掩码（C1, C2）
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         # 多头注意力
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
    
#     def forward(self, x):
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用交互处理
#         interaction_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]

#             # 4. 特征投影
#             block_projected = self.feature_projection(block)  # [B, T, H]

#             # 5. 注意力 + Transformer
#             attn_output, _ = self.attention(block_projected, block_projected, block_projected)
#             transformer_output = self.transformer(attn_output)

#             # 6. 输出投影
#             interaction_feature = self.output_projection(transformer_output)
#             interaction_features.append(interaction_feature)
        
#         # 7. 重建序列
#         interaction_features = torch.stack(interaction_features, dim=1)
        
#         # 8. 反中心化
#         interaction_features = self.centering((interaction_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         interaction_features = self.time_blocking.reconstruct(interaction_features, block_positions)
        
#         # 10. 残差连接
#         if interaction_features.size() == original_x.size():
#             interaction_features = interaction_features + original_x
        
#         return interaction_features


# class TrendComponent(nn.Module):
#     """
#     趋势组件：严格按照FBM-S论文实现
#     时间分块 + MLP/Transformer + 频域权重
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         if use_transformer:
#             # Transformer编码器
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=8,
#                 dim_feedforward=hidden_dim * 2,
#                 dropout=0.1,
#                 batch_first=True
#             )
#             self.trend_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         else:
#             # MLP网络
#             self.trend_net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（趋势：低频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))

#     def forward(self, x):
#         """
#         严格按照论文实现趋势组件
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x

#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用趋势处理
#         trend_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 特征投影
#             block_projected = self.feature_projection(block)
            
#             # 5. 趋势建模
#             if self.use_transformer:
#                 trend_output = self.trend_net(block_projected)
#             else:
#                 trend_output = self.trend_net(block_projected)
            
#             # 6. 输出投影
#             trend_feature = self.output_projection(trend_output)
#             trend_features.append(trend_feature)
        
#         # 7. 重建序列
#         trend_features = torch.stack(trend_features, dim=1)
        
#         # 8. 反中心化
#         trend_features = self.centering((trend_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         trend_features = self.time_blocking.reconstruct(trend_features, block_positions)

#         # 在频域应用频率权重（趋势：低频分量权重大）
#         try:
#             # 对每个特征维度分别进行频域处理
#             trend_features_processed = []
#             for c in range(trend_features.size(-1)):
#                 # 提取单个特征的时间序列
#                 feature_series = trend_features[:, :, c]  # [batch, seq_len]
                
#                 # 傅里叶变换
#                 feature_fft = torch.fft.rfft(feature_series, dim=-1)  # [batch, freq_len]
                
#                 # 调整频域权重维度
#                 actual_freq_len = feature_fft.size(-1)
#                 if self.freq_weights.size(0) != actual_freq_len:
#                     if self.freq_weights.size(0) > actual_freq_len:
#                         freq_weights_resized = self.freq_weights[:actual_freq_len]
#                     else:
#                         # 用1.0填充不足的部分
#                         padding = torch.ones(actual_freq_len - self.freq_weights.size(0), 
#                                            device=self.freq_weights.device, 
#                                            dtype=self.freq_weights.dtype)
#                         freq_weights_resized = torch.cat([self.freq_weights, padding], dim=0)
#                 else:
#                     freq_weights_resized = self.freq_weights
                
#                 # 应用频域权重（趋势：低频分量权重大）
#                 freq_weights_resized = freq_weights_resized.to(feature_fft.device, dtype=feature_fft.dtype)
#                 feature_fft_weighted = feature_fft * freq_weights_resized.unsqueeze(0)
                
#                 # 逆傅里叶变换
#                 feature_processed = torch.fft.irfft(feature_fft_weighted, n=feature_series.size(-1), dim=-1)
#                 trend_features_processed.append(feature_processed)
            
#             # 重新组合特征
#             trend_features = torch.stack(trend_features_processed, dim=-1)  # [batch, seq_len, num_features]
            
#         except Exception as e:
#             print(f"Warning: Frequency domain processing failed in trend component: {e}")
#             # 如果频域处理失败，使用原始特征
#             pass
        
#         # 10. 残差连接
#         if trend_features.size() == original_x.size():
#             trend_features = trend_features + original_x
        
#         return trend_features

# # 工厂函数
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)








# #  我的最终版本 采用滤波器
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.fft import rfft, irfft
# from einops import rearrange
# import math


# class RevIN(nn.Module):
#     """
#     可逆实例归一化 (Reversible Instance Normalization)
#     基于FBM-S论文的实现，与fbm_paper_components.py完全一致
#     """
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             # 计算均值和标准差
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
            
#             # 归一化
#             x_norm = (x - mean) / std
            
#             # 仿射变换 - 使用gamma和beta参数
#             if self.affine:
#                 x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             return x_norm
        
#         elif mode == 'denorm':
#             # 反归一化
#             if self.affine:
#                 x_denorm = (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_denorm = x
#             return x_denorm
#         else:
#             raise ValueError(f"mode {mode} not supported.")


# class TimeBlocking(nn.Module):
#     """
#     时间分块策略：FBM-S论文的核心特性
#     将长序列分解为多个时间块，每个块独立处理
#     """
#     def __init__(self, seq_len: int, block_size: int = 64, overlap: int = 16):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
        
#         # 计算块数量
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
        
#         # 确保最后一个块能覆盖序列末尾
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         # x: [batch, seq_len, num_features]
#         batch_size, seq_len, num_features = x.shape
        
#         # 如果序列长度小于块大小，直接返回
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
        
#         blocks = []
#         block_positions = []
        
#         # 使用简单的滑动窗口策略
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             start_idx = i
#             end_idx = i + self.block_size
            
#             # 提取时间块
#             block = x[:, start_idx:end_idx, :]
#             blocks.append(block)
#             block_positions.append((start_idx, end_idx))
        
#         # 处理最后一个块，确保覆盖序列末尾
#         if block_positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             last_block = x[:, last_start:seq_len, :]
#             blocks.append(last_block)
#             block_positions.append((last_start, seq_len))
        
#         # 堆叠所有块
#         blocks = torch.stack(blocks, dim=1)  # [batch, num_blocks, block_size, num_features]
#         return blocks, block_positions
    
#     def reconstruct(self, blocks, block_positions):
#         # 重建原始序列
#         batch_size, num_blocks, block_size, num_features = blocks.shape
#         seq_len = max(pos[1] for pos in block_positions)
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
#         count = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
        
#         # 将每个块放回原位置
#         for i, (start, end) in enumerate(block_positions):
#             block = blocks[:, i, :end - start, :]
#             output[:, start:end, :] += block
#             count[:, start:end, :] += 1

        
#         # 平均重叠部分
#         count = torch.clamp(count, min=1)
#         output = output / count
        
#         return output


# class CenteringLayer(nn.Module):
#     """
#     中心化层：FBM-S论文的关键技术
#     每个块内部进行中心化处理，提高模型鲁棒性
#     """
#     def __init__(self, num_features: int, affine=True):
#         super().__init__()
#         self.num_features = num_features
#         self.affine = affine
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))
    
#     def forward(self, x, mode='center'):
#         if mode == 'center':
#             # 计算每个块的均值和标准差
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑为 [batch * num_blocks, block_size, num_features]
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 计算均值和标准差
#             mean = torch.mean(x_reshaped, dim=1, keepdim=True)
#             std = torch.std(x_reshaped, dim=1, keepdim=True) + 1e-5
            
#             # 中心化
#             x_centered = (x_reshaped - mean) / std
            
#             # 仿射变换
#             if self.affine:
#                 x_centered = x_centered * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             # 重塑回原形状
#             x_centered = x_centered.view(batch_size, num_blocks, block_size, num_features)
#             return x_centered, mean, std
        
#         elif mode == 'decenter':
#             x, mean, std = x
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 确保 mean 和 std 的维度匹配
#             if mean.size(0) != x_reshaped.size(0):
#                 # 如果维度不匹配，调整 mean 和 std 的维度
#                 target_batch = x_reshaped.size(0)
#                 if mean.size(0) > target_batch:
#                     mean_reshaped = mean[:target_batch].view(-1, 1, num_features)
#                     std_reshaped = std[:target_batch].view(-1, 1, num_features)
#                 else:
#                     # 重复 mean 和 std 到目标维度
#                     repeat_times = target_batch // mean.size(0)
#                     remainder = target_batch % mean.size(0)
#                     mean_repeated = mean.repeat(repeat_times, 1, 1)
#                     std_repeated = std.repeat(repeat_times, 1, 1)
#                     if remainder > 0:
#                         mean_repeated = torch.cat([mean_repeated, mean[:remainder]], dim=0)
#                         std_repeated = torch.cat([std_repeated, std[:remainder]], dim=0)
#                     mean_reshaped = mean_repeated.view(-1, 1, num_features)
#                     std_reshaped = std_repeated.view(-1, 1, num_features)
#             else:
#                 mean_reshaped = mean.view(-1, 1, num_features)
#                 std_reshaped = std.view(-1, 1, num_features)
            
#             # 反中心化
#             if self.affine:
#                 x_decentered = (x_reshaped - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_decentered = x_reshaped
            
#             x_decentered = x_decentered * std_reshaped + mean_reshaped
            
#             # 重塑回原形状
#             x_decentered = x_decentered.view(batch_size, num_blocks, block_size, num_features)
#             return x_decentered


# class FourierBasisExpansion(nn.Module):
#     """
#     傅里叶基函数扩展 (带可学习频率权重)
#     - 在论文基础上增加 freq_weights 参数，用于加权不同频段
#     - 适合分类任务，让模型学习关注哪些频段更有判别力
#     """
#     def __init__(self, seq_len: int, num_features: int, expansion_factor: int = 2,
#                  use_normalize: bool = True):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.expansion_factor = expansion_factor
#         self.expanded_len = seq_len * expansion_factor
#         self.use_normalize = use_normalize
        
#         # 生成傅里叶基函数
#         self.register_buffer('cos_basis', self._generate_cos_basis())
#         self.register_buffer('sin_basis', self._generate_sin_basis())
        
#         # === 新增：可学习频率权重 ===
#         P = seq_len // 2 + 1   # rfft 输出的频率数
#         self.freq_weights_raw = nn.Parameter(torch.zeros(P))  # raw 参数

#     def _generate_cos_basis(self):
#         """生成余弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#         else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#         cos = cos[1:]  # 去掉DC项
#         return cos

#     def _generate_sin_basis(self):
#         """生成正弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         sin = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         else:
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         sin = sin[1:]  # 去掉DC项
#         return sin

#     def forward(self, x):
#         """
#         构建时间-频域特征
#         输入: x [B, T, C]
#         输出: time_freq_features [B, T, C]
#         """
#         batch_size, seq_len, num_features = x.shape

#         # 1) rfft 变换
#         xc = x.permute(0, 2, 1)  # [B,C,T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P] 复数

#         # === 新增：应用可学习频率权重 ===
#         w = F.softplus(self.freq_weights_raw)  # 保证非负
#         if self.use_normalize:
#             w = w / (w.sum() + 1e-8)  # 归一化，避免权重无界增长
#         w = w.view(1, 1, -1)         # [1,1,P] 便于广播
#         X = X * w                    # 对实部/虚部同时加权

#         # 2) 使用构造函数中计算好的 cos_basis 和 sin_basis
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.cos_basis)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.sin_basis)
#         z_unified = basis_cos + basis_sin  # [B,C,P,T]
#         z_sum = z_unified.sum(dim=2)       # 按频率求和 → [B,C,T]

#         # 3) 转回 [B,T,C]
#         return z_sum.permute(0, 2, 1)


# class SeasonalComponent(nn.Module):

#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64, 
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.expansion_factor = expansion_factor
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 傅里叶基函数扩展
#         self.fourier_expansion = FourierBasisExpansion(block_size, num_features, expansion_factor)
        
#         # 滚动窗口权重（论文公式5中的W）
#         # 确保维度与傅里叶基函数匹配
#         self.rolling_window = nn.Parameter(torch.randn(block_size, block_size // 2 + 1))
        
#         # 季节性网络
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )
        
#         # 多尺度下采样
#         self.downsample_scales = [1, 2, 4]  # 论文中的d₀, d₁, d₂
        
#         # 频域权重（季节性：中频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))
        
#         # 多尺度融合权重（可学习参数）
#         self.multiscale_weights = nn.Parameter(torch.ones(len(self.downsample_scales)))
    
#     def forward(self, x):
#         """
#         严格按照论文公式5实现
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用季节性处理
#         seasonal_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 1. 获取傅里叶系数 [B, C, P]
#             xc = block.permute(0, 2, 1)  # [B, C, T]
#             X = torch.fft.rfft(xc, dim=-1) / xc.size(-1) * 2  # [B, C, P]

#             # 2. 应用频域权重进行有意义的频域处理
#             # 确保频域权重维度匹配
#             freq_len = X.size(-1)
#             if self.freq_weights.size(0) != freq_len:
#                 # 调整频域权重维度
#                 if self.freq_weights.size(0) > freq_len:
#                     freq_weights = self.freq_weights[:freq_len]
#                 else:
#                     # 用1.0填充不足的部分
#                     padding = torch.ones(freq_len - self.freq_weights.size(0), 
#                                        device=self.freq_weights.device, 
#                                        dtype=self.freq_weights.dtype)
#                     freq_weights = torch.cat([self.freq_weights, padding], dim=0)
#             else:
#                 freq_weights = self.freq_weights
            
#             # 应用频域权重（季节性：中频分量权重大）
#             freq_weights = freq_weights.view(1, 1, -1)  # [1, 1, P]
#             X_weighted = X * freq_weights.to(X.device, dtype=X.dtype)
            
#             # 3. 频域滤波：保留中频分量，抑制高频噪声
#             # 创建中频滤波器（季节性模式通常在中等频率）
#             freq_indices = torch.arange(freq_len, device=X.device, dtype=torch.float32)  # 使用float32而不是复数类型
#             # 中频范围：20%-80%的频率分量
#             low_cutoff = 0.2 * freq_len
#             high_cutoff = 0.8 * freq_len
            
#             # 创建平滑的滤波器
#             filter_mask = torch.ones_like(freq_indices)
#             # 低频衰减
#             low_mask = torch.sigmoid((freq_indices - low_cutoff) * 10)
#             # 高频衰减  
#             high_mask = torch.sigmoid((high_cutoff - freq_indices) * 10)
#             filter_mask = filter_mask * low_mask * high_mask
#             filter_mask = filter_mask.view(1, 1, -1)
            
#             # 应用滤波器
#             X_filtered = X_weighted * filter_mask.to(X.device, dtype=X.dtype)
            
#             # 4. 逆傅里叶变换
#             weighted_block = torch.fft.irfft(X_filtered, n=block.size(1), dim=-1)  # [B, C, T]
#             weighted_block = weighted_block.transpose(1, 2)  # [B, T, C]
            
            
#             # 6. 多尺度下采样（针对 [B, T, C]，在时间维做离散 pairwise 聚合）
#             multi_scale_features = []
#             current_feature = weighted_block  # [B, T, C]
#             multi_scale_features.append(current_feature)

#             # 下采样尺度1 (T -> T/2)
#             if len(self.downsample_scales) > 1 and current_feature.size(1) >= 4:  # 确保有足够长度进行下采样
#                 try:
#                     # 时间维度平均下采样
#                     down1 = F.avg_pool1d(current_feature.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
#                     # 边界检查：确保下采样后仍有有效长度
#                     if down1.size(1) > 0:
#                         multi_scale_features.append(down1)
#                         current_feature = down1  # 更新当前特征用于下一级下采样
#                 except Exception as e:
#                     print(f"Warning: Downsampling scale 1 failed: {e}")
#                     # 如果下采样失败，跳过这一级
#                     pass

#             # 下采样尺度2 (T/2 -> T/4)
#             if len(self.downsample_scales) > 2 and current_feature.size(1) >= 4:  # 使用current_feature而不是检查变量存在性
#                 try:
#                     # 确保长度为偶数
#                     t2 = (current_feature.size(1) // 2) * 2
#                     if t2 >= 2:  # 确保有足够长度
#                         d1_even = current_feature[:, :t2, :]
#                         down2 = d1_even.reshape(d1_even.size(0), t2 // 2, 2, d1_even.size(2)).sum(dim=2)  # [B,T/4,C]
#                         # 边界检查：确保下采样后仍有有效长度
#                         if down2.size(1) > 0:
#                             multi_scale_features.append(down2)
#                 except Exception as e:
#                     print(f"Warning: Downsampling scale 2 failed: {e}")
#                     # 如果下采样失败，跳过这一级
#                     pass
            
#             # 7. 融合多尺度特征
#             if len(multi_scale_features) > 1:
#                 # 上采样到相同大小
#                 target_size = multi_scale_features[0].size(1)
#                 upsampled_features = []
                
#                 for feature in multi_scale_features:
#                     if feature.size(1) != target_size:
#                         feature = F.interpolate(
#                             feature.transpose(1, 2), 
#                             size=target_size, 
#                             mode='linear'
#                         ).transpose(1, 2)
#                     upsampled_features.append(feature)
                
#                 # 使用可学习的多尺度融合权重
#                 # 确保权重数量与特征数量匹配
#                 num_features = len(upsampled_features)
#                 if self.multiscale_weights.size(0) != num_features:
#                     # 调整权重维度
#                     if self.multiscale_weights.size(0) > num_features:
#                         weights = self.multiscale_weights[:num_features]
#                     else:
#                         # 用1.0填充不足的部分
#                         padding = torch.ones(num_features - self.multiscale_weights.size(0), 
#                                            device=self.multiscale_weights.device, 
#                                            dtype=self.multiscale_weights.dtype)
#                         weights = torch.cat([self.multiscale_weights, padding], dim=0)
#                 else:
#                     weights = self.multiscale_weights
                
#                 # 应用softmax确保权重和为1
#                 weights = F.softmax(weights, dim=0)
                
#                 # 加权融合
#                 fused_feature = sum(w * f for w, f in zip(weights, upsampled_features))
#             else:
#                 fused_feature = multi_scale_features[0]
            
#             # 8. 季节性参数化
#             seasonal_feature = self.seasonal_net(fused_feature)
#             seasonal_features.append(seasonal_feature)

#         # 9. 重建序列
#         seasonal_features = torch.stack(seasonal_features, dim=1)
#         # 10. 反中心化
#         seasonal_features = self.centering((seasonal_features, mean, std), mode='decenter')
        
#         # 11. 重建原始序列
#         seasonal_features = self.time_blocking.reconstruct(seasonal_features, block_positions)
        
#         # 12. 残差连接
#         if seasonal_features.size() == original_x.size():
#             seasonal_features = seasonal_features + original_x
        
#         return seasonal_features


# class InteractionComponent(nn.Module):
#     """
#     交互组件（去掉掩码版本）
#     - 保留时间分块、中心化、投影、注意力、Transformer
#     - 删除固定掩码（C1, C2）
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         # 多头注意力
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
    
#     def forward(self, x):
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用交互处理
#         interaction_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]

#             # 4. 特征投影
#             block_projected = self.feature_projection(block)  # [B, T, H]

#             # 5. 注意力 + Transformer
#             attn_output, _ = self.attention(block_projected, block_projected, block_projected)
#             transformer_output = self.transformer(attn_output)

#             # 6. 输出投影
#             interaction_feature = self.output_projection(transformer_output)
#             interaction_features.append(interaction_feature)
        
#         # 7. 重建序列
#         interaction_features = torch.stack(interaction_features, dim=1)
        
#         # 8. 反中心化
#         interaction_features = self.centering((interaction_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         interaction_features = self.time_blocking.reconstruct(interaction_features, block_positions)
        
#         # 10. 残差连接
#         if interaction_features.size() == original_x.size():
#             interaction_features = interaction_features + original_x
        
#         return interaction_features


# class TrendComponent(nn.Module):
#     """
#     趋势组件：严格按照FBM-S论文实现
#     时间分块 + MLP/Transformer + 频域权重
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         if use_transformer:
#             # Transformer编码器
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=8,
#                 dim_feedforward=hidden_dim * 2,
#                 dropout=0.1,
#                 batch_first=True
#             )
#             self.trend_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         else:
#             # MLP网络
#             self.trend_net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（趋势：低频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))

#     def forward(self, x):
#         """
#         严格按照论文实现趋势组件
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x

#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用趋势处理
#         trend_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 特征投影
#             block_projected = self.feature_projection(block)
            
#             # 5. 趋势建模
#             if self.use_transformer:
#                 trend_output = self.trend_net(block_projected)
#             else:
#                 trend_output = self.trend_net(block_projected)
            
#             # 6. 输出投影
#             trend_feature = self.output_projection(trend_output)
#             trend_features.append(trend_feature)
        
#         # 7. 重建序列
#         trend_features = torch.stack(trend_features, dim=1)
        
#         # 8. 反中心化
#         trend_features = self.centering((trend_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         trend_features = self.time_blocking.reconstruct(trend_features, block_positions)

#         # 在频域应用频率权重（趋势：低频分量权重大）
#         try:
#             # 对每个特征维度分别进行频域处理
#             trend_features_processed = []
#             for c in range(trend_features.size(-1)):
#                 # 提取单个特征的时间序列
#                 feature_series = trend_features[:, :, c]  # [batch, seq_len]
                
#                 # 傅里叶变换
#                 feature_fft = torch.fft.rfft(feature_series, dim=-1)  # [batch, freq_len]
                
#                 # 调整频域权重维度
#                 actual_freq_len = feature_fft.size(-1)
#                 if self.freq_weights.size(0) != actual_freq_len:
#                     if self.freq_weights.size(0) > actual_freq_len:
#                         freq_weights_resized = self.freq_weights[:actual_freq_len]
#                     else:
#                         # 用1.0填充不足的部分
#                         padding = torch.ones(actual_freq_len - self.freq_weights.size(0), 
#                                            device=self.freq_weights.device, 
#                                            dtype=self.freq_weights.dtype)
#                         freq_weights_resized = torch.cat([self.freq_weights, padding], dim=0)
#                 else:
#                     freq_weights_resized = self.freq_weights
                
#                 # 应用频域权重（趋势：低频分量权重大）
#                 freq_weights_resized = freq_weights_resized.to(feature_fft.device, dtype=feature_fft.dtype)
#                 feature_fft_weighted = feature_fft * freq_weights_resized.unsqueeze(0)
                
#                 # 逆傅里叶变换
#                 feature_processed = torch.fft.irfft(feature_fft_weighted, n=feature_series.size(-1), dim=-1)
#                 trend_features_processed.append(feature_processed)
            
#             # 重新组合特征
#             trend_features = torch.stack(trend_features_processed, dim=-1)  # [batch, seq_len, num_features]
            
#         except Exception as e:
#             print(f"Warning: Frequency domain processing failed in trend component: {e}")
#             # 如果频域处理失败，使用原始特征
#             pass
        
#         # 10. 残差连接
#         if trend_features.size() == original_x.size():
#             trend_features = trend_features + original_x
        
#         return trend_features

# # 工厂函数
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)








#  修复之后的最终版本，但是还是有问题
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.fft import rfft, irfft
# from einops import rearrange
# import math


# class RevIN(nn.Module):
#     """
#     可逆实例归一化 (Reversible Instance Normalization)
#     基于FBM-S论文的实现，与fbm_paper_components.py完全一致
#     """
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             # 计算均值和标准差
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
            
#             # 归一化
#             x_norm = (x - mean) / std
            
#             # 仿射变换 - 使用gamma和beta参数
#             if self.affine:
#                 x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             return x_norm
        
#         elif mode == 'denorm':
#             # 反归一化
#             if self.affine:
#                 x_denorm = (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_denorm = x
#             return x_denorm
#         else:
#             raise ValueError(f"mode {mode} not supported.")


# class TimeBlocking(nn.Module):
#     """
#     时间分块策略：FBM-S论文的核心特性
#     将长序列分解为多个时间块，每个块独立处理
#     """
#     def __init__(self, seq_len: int, block_size: int = 64, overlap: int = 16):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
        
#         # 计算块数量
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
        
#         # 确保最后一个块能覆盖序列末尾
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         # x: [batch, seq_len, num_features]
#         batch_size, seq_len, num_features = x.shape
        
#         # 如果序列长度小于块大小，直接返回
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
        
#         blocks = []
#         block_positions = []
        
#         # 使用简单的滑动窗口策略
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             start_idx = i
#             end_idx = i + self.block_size
            
#             # 提取时间块
#             block = x[:, start_idx:end_idx, :]
#             blocks.append(block)
#             block_positions.append((start_idx, end_idx))
        
#         # 处理最后一个块，确保覆盖序列末尾
#         if block_positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             last_block = x[:, last_start:seq_len, :]
#             blocks.append(last_block)
#             block_positions.append((last_start, seq_len))
        
#         # 堆叠所有块
#         blocks = torch.stack(blocks, dim=1)  # [batch, num_blocks, block_size, num_features]
#         return blocks, block_positions
    
#     def reconstruct(self, blocks, block_positions):
#         # 重建原始序列
#         batch_size, num_blocks, block_size, num_features = blocks.shape
#         seq_len = max(pos[1] for pos in block_positions)
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
#         count = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
        
#         # 将每个块放回原位置
#         for i, (start, end) in enumerate(block_positions):
#             block = blocks[:, i, :end - start, :]
#             output[:, start:end, :] += block
#             count[:, start:end, :] += 1

        
#         # 平均重叠部分
#         count = torch.clamp(count, min=1)
#         output = output / count
        
#         return output


# class CenteringLayer(nn.Module):
#     """
#     中心化层：FBM-S论文的关键技术
#     每个块内部进行中心化处理，提高模型鲁棒性
#     """
#     def __init__(self, num_features: int, affine=True):
#         super().__init__()
#         self.num_features = num_features
#         self.affine = affine
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))
    
#     def forward(self, x, mode='center'):
#         if mode == 'center':
#             # 计算每个块的均值和标准差
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑为 [batch * num_blocks, block_size, num_features]
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 计算均值和标准差
#             mean = torch.mean(x_reshaped, dim=1, keepdim=True)
#             std = torch.std(x_reshaped, dim=1, keepdim=True) + 1e-5
            
#             # 中心化
#             x_centered = (x_reshaped - mean) / std
            
#             # 仿射变换
#             if self.affine:
#                 x_centered = x_centered * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             # 重塑回原形状
#             x_centered = x_centered.view(batch_size, num_blocks, block_size, num_features)
#             return x_centered, mean, std
        
#         elif mode == 'decenter':
#             x, mean, std = x
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 确保 mean 和 std 的维度匹配
#             if mean.size(0) != x_reshaped.size(0):
#                 # 如果维度不匹配，调整 mean 和 std 的维度
#                 target_batch = x_reshaped.size(0)
#                 if mean.size(0) > target_batch:
#                     mean_reshaped = mean[:target_batch].view(-1, 1, num_features)
#                     std_reshaped = std[:target_batch].view(-1, 1, num_features)
#                 else:
#                     # 重复 mean 和 std 到目标维度
#                     repeat_times = target_batch // mean.size(0)
#                     remainder = target_batch % mean.size(0)
#                     mean_repeated = mean.repeat(repeat_times, 1, 1)
#                     std_repeated = std.repeat(repeat_times, 1, 1)
#                     if remainder > 0:
#                         mean_repeated = torch.cat([mean_repeated, mean[:remainder]], dim=0)
#                         std_repeated = torch.cat([std_repeated, std[:remainder]], dim=0)
#                     mean_reshaped = mean_repeated.view(-1, 1, num_features)
#                     std_reshaped = std_repeated.view(-1, 1, num_features)
#             else:
#                 mean_reshaped = mean.view(-1, 1, num_features)
#                 std_reshaped = std.view(-1, 1, num_features)
            
#             # 反中心化
#             if self.affine:
#                 x_decentered = (x_reshaped - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_decentered = x_reshaped
            
#             x_decentered = x_decentered * std_reshaped + mean_reshaped
            
#             # 重塑回原形状
#             x_decentered = x_decentered.view(batch_size, num_blocks, block_size, num_features)
#             return x_decentered


# class FourierBasisExpansion(nn.Module):
#     """
#     傅里叶基函数扩展 (带可学习频率权重)
#     - 在论文基础上增加 freq_weights 参数，用于加权不同频段
#     - 适合分类任务，让模型学习关注哪些频段更有判别力
#     """
#     def __init__(self, seq_len: int, num_features: int, expansion_factor: int = 2,
#                  use_normalize: bool = True):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.expansion_factor = expansion_factor
#         self.expanded_len = seq_len * expansion_factor
#         self.use_normalize = use_normalize
        
#         # 生成傅里叶基函数
#         self.register_buffer('cos_basis', self._generate_cos_basis())
#         self.register_buffer('sin_basis', self._generate_sin_basis())
        
#         # === 新增：可学习频率权重 ===
#         P = seq_len // 2 + 1   # rfft 输出的频率数
#         self.freq_weights_raw = nn.Parameter(torch.zeros(P))  # raw 参数

#     def _generate_cos_basis(self):
#         """生成余弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#         else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#         cos = cos[1:]  # 去掉DC项
#         return cos

#     def _generate_sin_basis(self):
#         """生成正弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         sin = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         else:
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         sin = sin[1:]  # 去掉DC项
#         return sin

#     def forward(self, x):
#         """
#         构建时间-频域特征
#         输入: x [B, T, C]
#         输出: time_freq_features [B, T, C]
#         """
#         batch_size, seq_len, num_features = x.shape

#         # 1) rfft 变换
#         xc = x.permute(0, 2, 1)  # [B,C,T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P] 复数

#         # === 新增：应用可学习频率权重 ===
#         w = F.softplus(self.freq_weights_raw)  # 保证非负
#         if self.use_normalize:
#             w = w / (w.sum() + 1e-8)  # 归一化，避免权重无界增长
#         w = w.view(1, 1, -1)         # [1,1,P] 便于广播
#         X = X * w                    # 对实部/虚部同时加权

#         # 2) 使用构造函数中计算好的 cos_basis 和 sin_basis
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.cos_basis)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.sin_basis)
#         z_unified = basis_cos + basis_sin  # [B,C,P,T]
#         z_sum = z_unified.sum(dim=2)       # 按频率求和 → [B,C,T]

#         # 3) 转回 [B,T,C]
#         return z_sum.permute(0, 2, 1)


# class SeasonalComponent(nn.Module):

#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64, 
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.expansion_factor = expansion_factor
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 傅里叶基函数扩展
#         self.fourier_expansion = FourierBasisExpansion(block_size, num_features, expansion_factor)
        
#         # 滚动窗口权重（论文公式5中的W）
#         # 确保维度与傅里叶基函数匹配
#         self.rolling_window = nn.Parameter(torch.randn(block_size, block_size // 2 + 1))
        
#         # 季节性网络
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )
        
#         # 多尺度下采样
#         self.downsample_scales = [1, 2, 4]  # 论文中的d₀, d₁, d₂
        
#         # 频域权重（季节性：中频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))
    
#     def forward(self, x):
#         """
#         严格按照论文公式5实现
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用季节性处理
#         seasonal_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 1. 获取傅里叶系数 [B, C, P]
#             xc = block.permute(0, 2, 1)  # [B, C, T]
#             X = torch.fft.rfft(xc, dim=-1) / xc.size(-1) * 2  # [B, C, P]

#             # 2. 确保基函数在正确设备上
#             cos_basis = self.fourier_expansion.cos_basis.to(block.device)  # [P, T]
#             sin_basis = self.fourier_expansion.sin_basis.to(block.device)  # [P, T]

#             # 3. 简化处理：直接使用傅里叶逆变换，避免复杂的维度匹配
#             # 直接对傅里叶系数进行逆变换
#             weighted_block = torch.fft.irfft(X, n=block.size(1), dim=-1)  # [B, C, T]
#             weighted_block = weighted_block.transpose(1, 2)  # [B, T, C]
            
            
#             # 6. 多尺度下采样（针对 [B, T, C]，在时间维做离散 pairwise 聚合）
#             multi_scale_features = []

#             # 原始尺度 (scale=1)
#             x1 = weighted_block  # [B, T, C]
#             multi_scale_features.append(x1)

#             # 下采样尺度1 (T -> T/2)
#             if len(self.downsample_scales) > 1 and x1.size(1) >= 2:
#                 # 时间维度平均下采样
#                 down1 = F.avg_pool1d(x1.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
#                 multi_scale_features.append(down1)

#             # 下采样尺度2 (T/2 -> T/4)
#             if len(self.downsample_scales) > 2 and 'down1' in locals() and down1.size(1) >= 2:
#                 t2 = (down1.size(1) // 2) * 2
#                 d1_even = down1[:, :t2, :]
#                 down2 = d1_even.reshape(d1_even.size(0), t2 // 2, 2, d1_even.size(2)).sum(dim=2)  # [B,T/4,C]
#                 multi_scale_features.append(down2)
            
#             # 7. 融合多尺度特征
#             if len(multi_scale_features) > 1:
#                 # 上采样到相同大小
#                 target_size = multi_scale_features[0].size(1)
#                 upsampled_features = []
                
#                 for feature in multi_scale_features:
#                     if feature.size(1) != target_size:
#                         feature = F.interpolate(
#                             feature.transpose(1, 2), 
#                             size=target_size, 
#                             mode='linear'
#                         ).transpose(1, 2)
#                     upsampled_features.append(feature)
                
#                 # 加权融合
#                 weights = F.softmax(torch.randn(len(upsampled_features)), dim=0)
#                 fused_feature = sum(w * f for w, f in zip(weights, upsampled_features))
#             else:
#                 fused_feature = multi_scale_features[0]
            
#             # 8. 季节性参数化
#             seasonal_feature = self.seasonal_net(fused_feature)
#             seasonal_features.append(seasonal_feature)

#         # 9. 重建序列
#         seasonal_features = torch.stack(seasonal_features, dim=1)
#         # 10. 反中心化
#         seasonal_features = self.centering((seasonal_features, mean, std), mode='decenter')
        
#         # 11. 重建原始序列
#         seasonal_features = self.time_blocking.reconstruct(seasonal_features, block_positions)
        
#         # 12. 残差连接
#         if seasonal_features.size() == original_x.size():
#             seasonal_features = seasonal_features + original_x
        
#         return seasonal_features


# class InteractionComponent(nn.Module):
#     """
#     交互组件（去掉掩码版本）
#     - 保留时间分块、中心化、投影、注意力、Transformer
#     - 删除固定掩码（C1, C2）
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         # 多头注意力
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
    
#     def forward(self, x):
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用交互处理
#         interaction_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]

#             # 4. 特征投影
#             block_projected = self.feature_projection(block)  # [B, T, H]

#             # 5. 注意力 + Transformer
#             attn_output, _ = self.attention(block_projected, block_projected, block_projected)
#             transformer_output = self.transformer(attn_output)

#             # 6. 输出投影
#             interaction_feature = self.output_projection(transformer_output)
#             interaction_features.append(interaction_feature)
        
#         # 7. 重建序列
#         interaction_features = torch.stack(interaction_features, dim=1)
        
#         # 8. 反中心化
#         interaction_features = self.centering((interaction_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         interaction_features = self.time_blocking.reconstruct(interaction_features, block_positions)
        
#         # 10. 残差连接
#         if interaction_features.size() == original_x.size():
#             interaction_features = interaction_features + original_x
        
#         return interaction_features


# class TrendComponent(nn.Module):
#     """
#     趋势组件：严格按照FBM-S论文实现
#     时间分块 + MLP/Transformer + 频域权重
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         if use_transformer:
#             # Transformer编码器
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=8,
#                 dim_feedforward=hidden_dim * 2,
#                 dropout=0.1,
#                 batch_first=True
#             )
#             self.trend_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         else:
#             # MLP网络
#             self.trend_net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（趋势：低频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))

#     def forward(self, x):
#         """
#         严格按照论文实现趋势组件
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x

#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用趋势处理
#         trend_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 特征投影
#             block_projected = self.feature_projection(block)
            
#             # 5. 趋势建模
#             if self.use_transformer:
#                 trend_output = self.trend_net(block_projected)
#             else:
#                 trend_output = self.trend_net(block_projected)
            
#             # 6. 输出投影
#             trend_feature = self.output_projection(trend_output)
#             trend_features.append(trend_feature)
        
#         # 7. 重建序列
#         trend_features = torch.stack(trend_features, dim=1)
        
#         # 8. 反中心化
#         trend_features = self.centering((trend_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         trend_features = self.time_blocking.reconstruct(trend_features, block_positions)

#         # 暂时注释掉频域权重应用，避免维度不匹配问题
#         # 在频域应用频率权重
#         # trend_features_fft = torch.fft.rfft(trend_features, dim=-2)
#         # # 根据实际序列长度调整频域权重
#         # actual_seq_len = trend_features.size(-2)
#         # target_freq_len = actual_seq_len // 2 + 1
        
#         # # 使用更简单的方法：直接创建匹配的权重
#         # if self.freq_weights.size(0) != target_freq_len:
#         #     # 创建与目标频域长度匹配的权重
#         #     freq_weights_resized = torch.ones(target_freq_len, device=trend_features_fft.device, dtype=trend_features_fft.dtype)
#         #     # 如果原始权重大于目标长度，取前target_freq_len个
#         #     if self.freq_weights.size(0) > target_freq_len:
#         #         freq_weights_resized = self.freq_weights[:target_freq_len].to(trend_features_fft.device)
#         #     # 如果原始权重小于目标长度，用1.0填充
#         #     else:
#         #         freq_weights_resized[:self.freq_weights.size(0)] = self.freq_weights.to(trend_features_fft.device)
#         # else:
#         #     freq_weights_resized = self.freq_weights.to(trend_features_fft.device)
        
#         # # 应用频域权重
#         # trend_features_fft = trend_features_fft * freq_weights_resized.unsqueeze(0).unsqueeze(0)
#         # trend_features = torch.fft.irfft(trend_features_fft, n=actual_seq_len, dim=-2)
        
#         # 10. 残差连接
#         if trend_features.size() == original_x.size():
#             trend_features = trend_features + original_x
        
#         return trend_features

# # 工厂函数
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)










# #  我的最终版本
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.fft import rfft, irfft
# from einops import rearrange
# import math


# class RevIN(nn.Module):
#     """
#     可逆实例归一化 (Reversible Instance Normalization)
#     基于FBM-S论文的实现，与fbm_paper_components.py完全一致
#     """
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             # 计算均值和标准差
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
            
#             # 归一化
#             x_norm = (x - mean) / std
            
#             # 仿射变换 - 使用gamma和beta参数
#             if self.affine:
#                 x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             return x_norm
        
#         elif mode == 'denorm':
#             # 反归一化
#             if self.affine:
#                 x_denorm = (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_denorm = x
#             return x_denorm
#         else:
#             raise ValueError(f"mode {mode} not supported.")


# class TimeBlocking(nn.Module):
#     """
#     时间分块策略：FBM-S论文的核心特性
#     将长序列分解为多个时间块，每个块独立处理
#     """
#     def __init__(self, seq_len: int, block_size: int = 64, overlap: int = 16):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
        
#         # 计算块数量
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
        
#         # 确保最后一个块能覆盖序列末尾
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         # x: [batch, seq_len, num_features]
#         batch_size, seq_len, num_features = x.shape
        
#         # 如果序列长度小于块大小，直接返回
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
        
#         blocks = []
#         block_positions = []
        
#         # 使用简单的滑动窗口策略
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             start_idx = i
#             end_idx = i + self.block_size
            
#             # 提取时间块
#             block = x[:, start_idx:end_idx, :]
#             blocks.append(block)
#             block_positions.append((start_idx, end_idx))
        
#         # 处理最后一个块，确保覆盖序列末尾
#         if block_positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             last_block = x[:, last_start:seq_len, :]
#             blocks.append(last_block)
#             block_positions.append((last_start, seq_len))
        
#         # 堆叠所有块
#         blocks = torch.stack(blocks, dim=1)  # [batch, num_blocks, block_size, num_features]
#         return blocks, block_positions
    
#     def reconstruct(self, blocks, block_positions):
#         # 重建原始序列
#         batch_size, num_blocks, block_size, num_features = blocks.shape
#         seq_len = max(pos[1] for pos in block_positions)
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
#         count = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
        
#         # 将每个块放回原位置
#         for i, (start, end) in enumerate(block_positions):
#             block = blocks[:, i, :end - start, :]
#             output[:, start:end, :] += block
#             count[:, start:end, :] += 1

        
#         # 平均重叠部分
#         count = torch.clamp(count, min=1)
#         output = output / count
        
#         return output


# class CenteringLayer(nn.Module):
#     """
#     中心化层：FBM-S论文的关键技术
#     每个块内部进行中心化处理，提高模型鲁棒性
#     """
#     def __init__(self, num_features: int, affine=True):
#         super().__init__()
#         self.num_features = num_features
#         self.affine = affine
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))
    
#     def forward(self, x, mode='center'):
#         if mode == 'center':
#             # 计算每个块的均值和标准差
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑为 [batch * num_blocks, block_size, num_features]
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 计算均值和标准差
#             mean = torch.mean(x_reshaped, dim=1, keepdim=True)
#             std = torch.std(x_reshaped, dim=1, keepdim=True) + 1e-5
            
#             # 中心化
#             x_centered = (x_reshaped - mean) / std
            
#             # 仿射变换
#             if self.affine:
#                 x_centered = x_centered * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             # 重塑回原形状
#             x_centered = x_centered.view(batch_size, num_blocks, block_size, num_features)
#             return x_centered, mean, std
        
#         elif mode == 'decenter':
#             x, mean, std = x
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑
#             x_reshaped = x.view(-1, block_size, num_features)
#             mean_reshaped = mean.view(-1, 1, num_features)
#             std_reshaped = std.view(-1, 1, num_features)
            
#             # 反中心化
#             if self.affine:
#                 x_decentered = (x_reshaped - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_decentered = x_reshaped
            
#             x_decentered = x_decentered * std_reshaped + mean_reshaped
            
#             # 重塑回原形状
#             x_decentered = x_decentered.view(batch_size, num_blocks, block_size, num_features)
#             return x_decentered


# class FourierBasisExpansion(nn.Module):
#     """
#     傅里叶基函数扩展 (带可学习频率权重)
#     - 在论文基础上增加 freq_weights 参数，用于加权不同频段
#     - 适合分类任务，让模型学习关注哪些频段更有判别力
#     """
#     def __init__(self, seq_len: int, num_features: int, expansion_factor: int = 2,
#                  use_normalize: bool = True):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.expansion_factor = expansion_factor
#         self.expanded_len = seq_len * expansion_factor
#         self.use_normalize = use_normalize
        
#         # 生成傅里叶基函数
#         self.register_buffer('cos_basis', self._generate_cos_basis())
#         self.register_buffer('sin_basis', self._generate_sin_basis())
        
#         # === 新增：可学习频率权重 ===
#         P = seq_len // 2 + 1   # rfft 输出的频率数
#         self.freq_weights_raw = nn.Parameter(torch.zeros(P))  # raw 参数

#     def _generate_cos_basis(self):
#         """生成余弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#         cos = cos[1:]  # 去掉DC项
#         return cos

#     def _generate_sin_basis(self):
#         """生成正弦基函数"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         sin = None
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#             else:
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         sin = sin[1:]  # 去掉DC项
#         return sin

#     def forward(self, x):
#         """
#         构建时间-频域特征
#         输入: x [B, T, C]
#         输出: time_freq_features [B, T, C]
#         """
#         batch_size, seq_len, num_features = x.shape

#         # 1) rfft 变换
#         xc = x.permute(0, 2, 1)  # [B,C,T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P] 复数

#         # === 新增：应用可学习频率权重 ===
#         w = F.softplus(self.freq_weights_raw)  # 保证非负
#         if self.use_normalize:
#             w = w / (w.sum() + 1e-8)  # 归一化，避免权重无界增长
#         w = w.view(1, 1, -1)         # [1,1,P] 便于广播
#         X = X * w                    # 对实部/虚部同时加权

#         # 2) 使用构造函数中计算好的 cos_basis 和 sin_basis
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, self.cos_basis)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, self.sin_basis)
#         z_unified = basis_cos + basis_sin  # [B,C,P,T]
#         z_sum = z_unified.sum(dim=2)       # 按频率求和 → [B,C,T]

#         # 3) 转回 [B,T,C]
#         return z_sum.permute(0, 2, 1)


# class SeasonalComponent(nn.Module):

#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64, 
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.expansion_factor = expansion_factor
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 傅里叶基函数扩展
#         self.fourier_expansion = FourierBasisExpansion(block_size, num_features, expansion_factor)
        
#         # 滚动窗口权重（论文公式5中的W）
#         # self.rolling_window = nn.Parameter(torch.randn(block_size, block_size // 2))
#         self.rolling_window = nn.Parameter(torch.randn(block_size, block_size // 2 + 1))
        
#         # 季节性网络
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )
        
#         # 多尺度下采样
#         self.downsample_scales = [1, 2, 4]  # 论文中的d₀, d₁, d₂
        
#         # 频域权重（季节性：中频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))
    
#     def forward(self, x):
#         """
#         严格按照论文公式5实现
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用季节性处理
#         seasonal_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 1. 获取傅里叶系数 [B, C, P]
#             xc = block.permute(0, 2, 1)  # [B, C, T]
#             X = torch.fft.rfft(xc, dim=-1) / xc.size(-1) * 2  # [B, C, P]

#             # 2. 确保基函数在正确设备上
#             cos_basis = self.fourier_expansion.cos_basis.to(block.device)  # [P, T]
#             sin_basis = self.fourier_expansion.sin_basis.to(block.device)  # [P, T]

#             # 3. 按照论文公式5：W 先与 basis 乘，再与 H_R/H_I 乘
#             # W × basis 先乘（滚动窗口权重与基函数卷积）
#             W_cos = torch.einsum('kp,pt->kt', self.rolling_window, cos_basis)  # [K, T]
#             W_sin = torch.einsum('kp,pt->kt', self.rolling_window, sin_basis)  # [K, T]

#             # 再与 H_R/H_I 乘（傅里叶系数与加权后的基函数乘）
#             seasonal_real = torch.einsum('bcp,kt->bckt', X.real, W_cos)  # [B, C, K, T]
#             seasonal_imag = torch.einsum('bcp,kt->bckt', X.imag, W_sin)  # [B, C, K, T]

#             # 合并实部和虚部，并求和得到最终输出 [B, T, C]
#             weighted_block = (seasonal_real + seasonal_imag).sum(dim=2).transpose(1, 2)
            
            
#             # 6. 多尺度下采样（针对 [B, T, C]，在时间维做离散 pairwise 聚合）
#             multi_scale_features = []

#             # 原始尺度 (scale=1)
#             x1 = weighted_block  # [B, T, C]
#             multi_scale_features.append(x1)

#             # 下采样尺度1 (T -> T/2)
#             if len(self.downsample_scales) > 1 and x1.size(1) >= 2:
#                 # 时间维度平均下采样
#                 down1 = F.avg_pool1d(x1.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
#                 multi_scale_features.append(down1)

#             # 下采样尺度2 (T/2 -> T/4)
#             if len(self.downsample_scales) > 2 and 'down1' in locals() and down1.size(1) >= 2:
#                 t2 = (down1.size(1) // 2) * 2
#                 d1_even = down1[:, :t2, :]
#                 down2 = d1_even.reshape(d1_even.size(0), t2 // 2, 2, d1_even.size(2)).sum(dim=2)  # [B,T/4,C]
#                 multi_scale_features.append(down2)
            
#             # 7. 融合多尺度特征
#             if len(multi_scale_features) > 1:
#                 # 上采样到相同大小
#                 target_size = multi_scale_features[0].size(1)
#                 upsampled_features = []
                
#                 for feature in multi_scale_features:
#                     if feature.size(1) != target_size:
#                         feature = F.interpolate(
#                             feature.transpose(1, 2), 
#                             size=target_size, 
#                             mode='linear'
#                         ).transpose(1, 2)
#                     upsampled_features.append(feature)
                
#                 # 加权融合
#                 weights = F.softmax(torch.randn(len(upsampled_features)), dim=0)
#                 fused_feature = sum(w * f for w, f in zip(weights, upsampled_features))
#             else:
#                 fused_feature = multi_scale_features[0]
            
#             # 8. 季节性参数化
#             seasonal_feature = self.seasonal_net(fused_feature)
#             seasonal_features.append(seasonal_feature)
        
#         # 9. 重建序列
#         seasonal_features = torch.stack(seasonal_features, dim=1)
        
#         # 10. 反中心化
#         seasonal_features = self.centering((seasonal_features, mean, std), mode='decenter')
        
#         # 11. 重建原始序列
#         seasonal_features = self.time_blocking.reconstruct(seasonal_features, block_positions)
        
#         # 12. 残差连接
#         if seasonal_features.size() == original_x.size():
#             seasonal_features = seasonal_features + original_x
        
#         return seasonal_features


# class InteractionComponent(nn.Module):
#     """
#     交互组件（去掉掩码版本）
#     - 保留时间分块、中心化、投影、注意力、Transformer
#     - 删除固定掩码（C1, C2）
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         # 多头注意力
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
    
#     def forward(self, x):
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用交互处理
#         interaction_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]

#             # 4. 特征投影
#             block_projected = self.feature_projection(block)  # [B, T, H]

#             # 5. 注意力 + Transformer
#             attn_output, _ = self.attention(block_projected, block_projected, block_projected)
#             transformer_output = self.transformer(attn_output)

#             # 6. 输出投影
#             interaction_feature = self.output_projection(transformer_output)
#             interaction_features.append(interaction_feature)
        
#         # 7. 重建序列
#         interaction_features = torch.stack(interaction_features, dim=1)
        
#         # 8. 反中心化
#         interaction_features = self.centering((interaction_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         interaction_features = self.time_blocking.reconstruct(interaction_features, block_positions)
        
#         # 10. 残差连接
#         if interaction_features.size() == original_x.size():
#             interaction_features = interaction_features + original_x
        
#         return interaction_features


# class TrendComponent(nn.Module):
#     """
#     趋势组件：严格按照FBM-S论文实现
#     时间分块 + MLP/Transformer + 频域权重
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         if use_transformer:
#             # Transformer编码器
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=8,
#                 dim_feedforward=hidden_dim * 2,
#                 dropout=0.1,
#                 batch_first=True
#             )
#             self.trend_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         else:
#             # MLP网络
#             self.trend_net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（趋势：低频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))

#     def forward(self, x):
#         """
#         严格按照论文实现趋势组件
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x

#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用趋势处理
#         trend_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 特征投影
#             block_projected = self.feature_projection(block)
            
#             # 5. 趋势建模
#             if self.use_transformer:
#                 trend_output = self.trend_net(block_projected)
#             else:
#                 trend_output = self.trend_net(block_projected)
            
#             # 6. 输出投影
#             trend_feature = self.output_projection(trend_output)
#             trend_features.append(trend_feature)
        
#         # 7. 重建序列
#         trend_features = torch.stack(trend_features, dim=1)
        
#         # 8. 反中心化
#         trend_features = self.centering((trend_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         trend_features = self.time_blocking.reconstruct(trend_features, block_positions)

#         # 在频域应用频率权重
#         trend_features_fft = torch.fft.rfft(trend_features, dim=-2)
#         trend_features_fft = trend_features_fft * self.freq_weights.unsqueeze(0).unsqueeze(0)  # 应用频域权重
#         trend_features = torch.fft.irfft(trend_features_fft, n=self.block_size, dim=-2)
        
#         # 10. 残差连接
#         if trend_features.size() == original_x.size():
#             trend_features = trend_features + original_x
        
#         return trend_features


# # 工厂函数
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)













# # # 调试对比 最终版本
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# # ============ RevIN ============
# class RevIN(nn.Module):
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super().__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
#             x_norm = (x - mean) / std
#             if self.affine:
#                 x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
#             return x_norm
#         elif mode == 'denorm':
#             if self.affine:
#                 return (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 return x
#         else:
#             raise ValueError(f"mode {mode} not supported.")


# # ============ 时间分块 ============
# class TimeBlocking(nn.Module):
#     def __init__(self, seq_len: int, block_size: int = 64, overlap: int = 16):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         batch, seq_len, _ = x.shape
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
#         blocks, positions = [], []
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             blocks.append(x[:, i:i+self.block_size, :])
#             positions.append((i, i+self.block_size))
#         if positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             blocks.append(x[:, last_start:seq_len, :])
#             positions.append((last_start, seq_len))
#         return torch.stack(blocks, dim=1), positions
    
#     def reconstruct(self, blocks, positions):
#         """
#         重建：将 blocks (可能经过下采样/变形导致 time 长度 != 原始 block length)
#         按 positions (s,e) 恢复到原始序列长度 seq_len。

#         blocks: Tensor [B, num_blocks, t_block, C]  — t_block 可能 != (e-s)
#         positions: list of (s,e) 对应每个 block 在原序列的起止索引

#         处理策略：
#          - 若 block 时间长度 > 需要长度 (e-s) -> 截断
#          - 若 block 时间长度 < 需要长度 (e-s) -> 在时间维度尾部用 0 补齐
#         """
#         b, nb, t_block, c = blocks.shape
#         seq_len = max(p[1] for p in positions)
#         out = torch.zeros(b, seq_len, c, device=blocks.device, dtype=blocks.dtype)
#         count = torch.zeros_like(out)

#         for i, (s, e) in enumerate(positions):
#             needed = e - s  # 目标长度
#             cur = blocks[:, i, :, :]  # [B, t_block, C]

#             if cur.size(1) == needed:
#                 aligned = cur
#             elif cur.size(1) > needed:
#                 # 截断多余部分
#                 aligned = cur[:, :needed, :]
#             else:
#                 # cur.size(1) < needed: 在时间维度尾部补零
#                 pad_len = needed - cur.size(1)
#                 pad = torch.zeros(b, pad_len, c, device=blocks.device, dtype=blocks.dtype)
#                 aligned = torch.cat([cur, pad], dim=1)

#             # 写回并计数（用于 later 平均）
#             out[:, s:e, :] += aligned
#             count[:, s:e, :] += 1.0

#         # 防止除以 0
#         return out / torch.clamp(count, min=1.0)



# # ============ 中心化 ============
# class CenteringLayer(nn.Module):
#     def __init__(self, num_features: int, affine=True):
#         super().__init__()
#         self.affine = affine
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))
    
#     def forward(self, x, mode='center'):
#         if mode == 'center':
#             b, nb, t, c = x.shape
#             x_flat = x.view(-1, t, c)
#             mean = x_flat.mean(dim=1, keepdim=True)
#             std = x_flat.std(dim=1, keepdim=True) + 1e-5
#             x_centered = (x_flat - mean) / std
#             if self.affine:
#                 x_centered = x_centered * self.gamma + self.beta
#             return x_centered.view(b, nb, t, c), mean, std
#         elif mode == 'decenter':
#             x, mean, std = x
#             b, nb, t, c = x.shape
#             x_flat = x.view(-1, t, c)
#             mean, std = mean.view(-1, 1, c), std.view(-1, 1, c)
#             if self.affine:
#                 x_flat = (x_flat - self.beta) / self.gamma
#             return (x_flat * std + mean).view(b, nb, t, c)


# # ============ Fourier ============
# class FourierBasisExpansion(nn.Module):
#     def __init__(self, seq_len, num_features, use_freq_weights=True, use_normalize=True):
#         super().__init__()
#         self.seq_len = seq_len
#         self.use_freq_weights = use_freq_weights
#         self.use_normalize = use_normalize
#         P = seq_len // 2 + 1
#         if self.use_freq_weights:
#             self.freq_weights_raw = nn.Parameter(torch.zeros(P))

#     def forward(self, x):
#         xc = x.permute(0, 2, 1)
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2
#         if self.use_freq_weights:
#             w = F.softplus(self.freq_weights_raw)
#             if self.use_normalize:
#                 w = w / (w.sum() + 1e-8)
#             X = X * w.view(1, 1, -1)
#         z_sum = torch.fft.irfft(X, n=norm, dim=-1)
#         return z_sum.permute(0, 2, 1)


# # ============ Mahalanobis 掩码工具 ============
# def mahalanobis_mask(x, thresh=0.5):
#     """
#     严格 per-channel 马氏距离掩码
#     x: [B, T, C] 特征
#     return: [C] 掩码 (1=保留, 0=屏蔽)
#     """
#     B, T, C = x.shape
#     x_flat = x.reshape(-1, C)   # [N, C], N = B*T

#     # 计算均值和协方差
#     mean = x_flat.mean(dim=0, keepdim=True)  # [1, C]
#     cov = torch.cov(x_flat.T)                # [C, C]
#     inv_cov = torch.inverse(cov + 1e-5 * torch.eye(C, device=x.device))  # [C, C]

#     # 计算每个样本的马氏距离
#     diffs = x_flat - mean                    # [N, C]
#     left = diffs @ inv_cov                   # [N, C]
#     dists = torch.sqrt(torch.sum(left * diffs, dim=1))  # [N]

#     # per-channel 平均距离
#     per_channel_dist = torch.zeros(C, device=x.device)
#     for c in range(C):
#         per_channel_dist[c] = torch.mean(torch.abs(diffs[:, c]) * dists)

#     # 阈值化
#     mask = (per_channel_dist < thresh).float()  # [C]

#     return mask

# # ============ Seasonal ============
# class SeasonalComponent(nn.Module):
#     def __init__(self, seq_len, num_features, block_size=64, hidden_dim=128,
#                  downsample_mode="flexible"):
#         super().__init__()
#         self.block_size = block_size
#         self.downsample_mode = downsample_mode
#         self.time_blocking = TimeBlocking(seq_len, block_size)
#         self.centering = CenteringLayer(num_features)
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_features)
#         )

#     def forward(self, x):
#         blocks, pos = self.time_blocking(x)
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
#         outs = []
#         for i in range(blocks_centered.size(1)):
#             b = blocks_centered[:, i, :, :]
#             if self.downsample_mode == "fixed":
#                 down = F.avg_pool1d(b.transpose(1, 2), kernel_size=2).transpose(1, 2)
#             else:
#                 down = F.avg_pool1d(b.transpose(1, 2), kernel_size=4, stride=2).transpose(1, 2)
#             outs.append(self.seasonal_net(down))
#         outs = torch.stack(outs, dim=1)
#         outs = self.centering((outs, mean, std), mode='decenter')
#         return self.time_blocking.reconstruct(outs, pos)


# # ============ Interaction ============
# class InteractionComponent(nn.Module):
#     def __init__(self, seq_len, num_features, block_size=64, hidden_dim=128,
#                  num_heads=8, use_mask=False, use_mahalanobis_mask=False, mask_thresh=0.5):
#         super().__init__()
#         self.use_mask = use_mask
#         self.use_mahalanobis_mask = use_mahalanobis_mask
#         self.mask_thresh = mask_thresh
#         self.time_blocking = TimeBlocking(seq_len, block_size)
#         self.centering = CenteringLayer(num_features)
#         self.proj = nn.Linear(num_features, hidden_dim)
#         self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
#         self.out = nn.Linear(hidden_dim, num_features)

#     def forward(self, x):
#         blocks, pos = self.time_blocking(x)
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
#         outs = []
#         for i in range(blocks_centered.size(1)):
#             b = blocks_centered[:, i, :, :]
#             b_proj = self.proj(b)
#             attn_mask = None
#             if self.use_mask:
#                 L = b_proj.size(1)
#                 attn_mask = torch.triu(torch.ones(L, L, device=b.device), 1).bool()
#             if self.use_mahalanobis_mask:
#                 channel_mask = mahalanobis_mask(b, self.mask_thresh)  # [C]
#                 b = b * channel_mask.view(1, 1, -1)
#                 b_proj = self.proj(b)
#             attn_out, _ = self.attn(b_proj, b_proj, b_proj, attn_mask=attn_mask)
#             outs.append(self.out(attn_out))
#         outs = torch.stack(outs, dim=1)
#         outs = self.centering((outs, mean, std), mode='decenter')
#         return self.time_blocking.reconstruct(outs, pos)


# # ============ Trend ============

# class TrendComponent(nn.Module):
#     def __init__(self, seq_len, num_features, block_size=64, hidden_dim=128,
#                  use_transformer=False, use_mahalanobis_mask=False, mask_thresh=0.5):
#         super().__init__()
#         self.use_transformer = use_transformer
#         self.use_mahalanobis_mask = use_mahalanobis_mask
#         self.mask_thresh = mask_thresh
#         self.time_blocking = TimeBlocking(seq_len, block_size)
#         self.centering = CenteringLayer(num_features)
#         self.proj = nn.Linear(num_features, hidden_dim)
#         if use_transformer:
#             enc_layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*2,
#                                                    dropout=0.1, batch_first=True)
#             self.net = nn.TransformerEncoder(enc_layer, 2)
#         else:
#             self.net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
#         self.out = nn.Linear(hidden_dim, num_features)

#     def forward(self, x):
#         blocks, pos = self.time_blocking(x)
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
#         outs = []
#         for i in range(blocks_centered.size(1)):
#             b = blocks_centered[:, i, :, :]
#             if self.use_mahalanobis_mask:
#                 channel_mask = mahalanobis_mask(b, self.mask_thresh)
#                 b = b * channel_mask.view(1, 1, -1)
#             b_proj = self.proj(b)
#             h = self.net(b_proj)
#             outs.append(self.out(h))
#         outs = torch.stack(outs, dim=1)
#         outs = self.centering((outs, mean, std), mode='decenter')
#         return self.time_blocking.reconstruct(outs, pos)


# # 工厂函数
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False,
#                    use_mahalanobis_mask: bool = False, mask_thresh: float = 0.5) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim,
#                           use_transformer, use_mahalanobis_mask, mask_thresh)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2,
#                       downsample_mode: str = "flexible") -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size,
#                              hidden_dim, downsample_mode)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8,
#                          use_mask: bool = False,
#                          use_mahalanobis_mask: bool = False,
#                          mask_thresh: float = 0.5) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size,
#                                 hidden_dim, num_heads,
#                                 use_mask, use_mahalanobis_mask, mask_thresh)










# # # 原来的组件修正公式

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.fft import rfft, irfft
# from einops import rearrange
# import math


# class RevIN(nn.Module):
#     """
#     可逆实例归一化 (Reversible Instance Normalization)
#     基于FBM-S论文的实现，与fbm_paper_components.py完全一致
#     """
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             # 计算均值和标准差
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
            
#             # 归一化
#             x_norm = (x - mean) / std
            
#             # 仿射变换 - 使用gamma和beta参数
#             if self.affine:
#                 x_norm = x_norm * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             return x_norm
        
#         elif mode == 'denorm':
#             # 反归一化
#             if self.affine:
#                 x_denorm = (x - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_denorm = x
#                 return x_denorm
#         else:
#             raise ValueError(f"mode {mode} not supported.")


# class TimeBlocking(nn.Module):
#     """
#     时间分块策略：FBM-S论文的核心特性
#     将长序列分解为多个时间块，每个块独立处理
#     """
#     def __init__(self, seq_len: int, block_size: int = 64, overlap: int = 16):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
        
#         # 计算块数量
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
        
#         # 确保最后一个块能覆盖序列末尾
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         # x: [batch, seq_len, num_features]
#         batch_size, seq_len, num_features = x.shape
        
#         # 如果序列长度小于块大小，直接返回
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
        
#         blocks = []
#         block_positions = []
        
#         # 使用简单的滑动窗口策略
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             start_idx = i
#             end_idx = i + self.block_size
            
#             # 提取时间块
#             block = x[:, start_idx:end_idx, :]
#             blocks.append(block)
#             block_positions.append((start_idx, end_idx))
        
#         # 处理最后一个块，确保覆盖序列末尾
#         if block_positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             last_block = x[:, last_start:seq_len, :]
#             blocks.append(last_block)
#             block_positions.append((last_start, seq_len))
        
#         # 堆叠所有块
#         blocks = torch.stack(blocks, dim=1)  # [batch, num_blocks, block_size, num_features]
#         return blocks, block_positions
    
#     def reconstruct(self, blocks, block_positions):
#         # 重建原始序列
#         batch_size, num_blocks, block_size, num_features = blocks.shape
#         seq_len = max(pos[1] for pos in block_positions)
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
#         count = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
        
#         # 将每个块放回原位置
#         for i, (start, end) in enumerate(block_positions):
#             output[:, start:end, :] += blocks[:, i, :, :]
#             count[:, start:end, :] += 1
        
#         # 平均重叠部分
#         count = torch.clamp(count, min=1)
#         output = output / count
        
#         return output


# class CenteringLayer(nn.Module):
#     """
#     中心化层：FBM-S论文的关键技术
#     每个块内部进行中心化处理，提高模型鲁棒性
#     """
#     def __init__(self, num_features: int, affine=True):
#         super().__init__()
#         self.num_features = num_features
#         self.affine = affine
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))
    
#     def forward(self, x, mode='center'):
#         if mode == 'center':
#             # 计算每个块的均值和标准差
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑为 [batch * num_blocks, block_size, num_features]
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 计算均值和标准差
#             mean = torch.mean(x_reshaped, dim=1, keepdim=True)
#             std = torch.std(x_reshaped, dim=1, keepdim=True) + 1e-5
            
#             # 中心化
#             x_centered = (x_reshaped - mean) / std
            
#             # 仿射变换
#             if self.affine:
#                 x_centered = x_centered * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             # 重塑回原形状
#             x_centered = x_centered.view(batch_size, num_blocks, block_size, num_features)
#             return x_centered, mean, std
        
#         elif mode == 'decenter':
#             x, mean, std = x
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑
#             x_reshaped = x.view(-1, block_size, num_features)
#             mean_reshaped = mean.view(-1, 1, num_features)
#             std_reshaped = std.view(-1, 1, num_features)
            
#             # 反中心化
#             if self.affine:
#                 x_decentered = (x_reshaped - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_decentered = x_reshaped
            
#             x_decentered = x_decentered * std_reshaped + mean_reshaped
            
#             # 重塑回原形状
#             x_decentered = x_decentered.view(batch_size, num_blocks, block_size, num_features)
#             return x_decentered


# class FourierBasisExpansion(nn.Module):
#     """
#     傅里叶基函数扩展：FBM-S论文的核心创新
#     实现时间-频域特征构建
#     """
#     def __init__(self, seq_len: int, num_features: int, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.expansion_factor = expansion_factor
#         self.expanded_len = seq_len * expansion_factor
        
#         # 生成傅里叶基函数
#         self.register_buffer('cos_basis', self._generate_cos_basis())
#         self.register_buffer('sin_basis', self._generate_sin_basis())
    
#     def _generate_cos_basis(self):
#         """生成余弦基函数 - 严格按照fbm_paper_components.py公式"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
        
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 # Nyquist 频率（仅当偶数长度时存在半频点），系数取 0.5
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
        
#         cos = cos[1:]  # 去掉DC项
#         return cos
    
#     def _generate_sin_basis(self):
#         """生成正弦基函数 - 严格按照fbm_paper_components.py公式"""
#         ts = 1.0 / self.seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         sin = None
        
#         for i in range(self.seq_len // 2 + 1):
#             if i == 0:
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             elif i == (self.seq_len // 2):
#                 # Nyquist 频率（仅当偶数长度时存在半频点），系数取 -0.5
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#             else:
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
        
#         sin = sin[1:]  # 去掉DC项
#         return sin
    
#     def forward(self, x):
#         """
#         构建时间-频域特征
#         输入: x [batch, seq_len, num_features]
#         输出: time_freq_features [batch, seq_len, num_features]
#         """
#         batch_size, seq_len, num_features = x.shape

#         # 1) 按论文规范进行RFFT归一化（/T*2），并在通道维上计算
#         xc = x.permute(0, 2, 1)  # [B,C,T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P]

#         # 2) 构建余弦/正弦基（DC项0.5，sin取负号），长度严格按当前序列长度
#         ts = 1.0 / norm
#         t = torch.arange(0, 1, ts, device=x.device, dtype=x.dtype)
#         cos = None
#         sin = None
#         for i in range(norm // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])

#         # 3) 用论文的einsum公式生成统一前端特征，并在频率维求和还原到 [B,C,T]
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, cos)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, sin)
#         z_unified = basis_cos + basis_sin  # [B,C,P,T]
#         z_sum = z_unified.sum(dim=2)       # 按频率求和 → [B,C,T]

#         # 4) 转回 [B,T,C]
#         return z_sum.permute(0, 2, 1)


# class SeasonalComponent(nn.Module):
#     """
#     季节性组件：严格按照FBM-S论文实现
#     使用滚动窗口在扩展的傅里叶基函数上操作
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64, 
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.expansion_factor = expansion_factor
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 傅里叶基函数扩展
#         self.fourier_expansion = FourierBasisExpansion(block_size, num_features, expansion_factor)
        
#         # 滚动窗口权重（论文公式5中的W）
#         self.rolling_window = nn.Parameter(torch.randn(block_size, num_features))
        
#         # 季节性网络
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )
        
#         # 多尺度下采样
#         self.downsample_scales = [1, 2, 4]  # 论文中的d₀, d₁, d₂
        
#         # 频域权重（季节性：中频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))
    
#     def forward(self, x):
#         """
#         严格按照论文公式5实现
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用季节性处理
#         seasonal_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 傅里叶基函数扩展
#             time_freq_block = self.fourier_expansion(block)
            
#             # 5. 应用滚动窗口权重（论文公式5）
#             # W × (扩展的傅里叶基函数)
#             weighted_block = time_freq_block * self.rolling_window.unsqueeze(0)
            
#             # 6. 多尺度下采样（针对 [B, T, C]，在时间维做离散 pairwise 聚合）
#             multi_scale_features = []

#             # 原始尺度 (scale=1)
#             x1 = weighted_block  # [B, T, C]
#             multi_scale_features.append(x1)

#             # 下采样尺度1 (T -> T/2)
#             if len(self.downsample_scales) > 1 and x1.size(1) >= 2:
#                 t1 = (x1.size(1) // 2) * 2  # 截断到偶数长度
#                 x1_even = x1[:, :t1, :]
#                 down1 = x1_even.reshape(x1_even.size(0), t1 // 2, 2, x1_even.size(2)).sum(dim=2)  # [B,T/2,C]
#                 multi_scale_features.append(down1)

#             # 下采样尺度2 (T/2 -> T/4)
#             if len(self.downsample_scales) > 2 and 'down1' in locals() and down1.size(1) >= 2:
#                 t2 = (down1.size(1) // 2) * 2
#                 d1_even = down1[:, :t2, :]
#                 down2 = d1_even.reshape(d1_even.size(0), t2 // 2, 2, d1_even.size(2)).sum(dim=2)  # [B,T/4,C]
#                 multi_scale_features.append(down2)
            
#             # 7. 融合多尺度特征
#             if len(multi_scale_features) > 1:
#                 # 上采样到相同大小
#                 target_size = multi_scale_features[0].size(1)
#                 upsampled_features = []
                
#                 for feature in multi_scale_features:
#                     if feature.size(1) != target_size:
#                         feature = F.interpolate(
#                             feature.transpose(1, 2), 
#                             size=target_size, 
#                             mode='linear'
#                         ).transpose(1, 2)
#                     upsampled_features.append(feature)
                
#                 # 加权融合
#                 weights = F.softmax(torch.randn(len(upsampled_features)), dim=0)
#                 fused_feature = sum(w * f for w, f in zip(weights, upsampled_features))
#             else:
#                 fused_feature = multi_scale_features[0]
            
#             # 8. 季节性参数化
#             seasonal_feature = self.seasonal_net(fused_feature)
#             seasonal_features.append(seasonal_feature)
        
#         # 9. 重建序列
#         seasonal_features = torch.stack(seasonal_features, dim=1)
        
#         # 10. 反中心化
#         seasonal_features = self.centering((seasonal_features, mean, std), mode='decenter')
        
#         # 11. 重建原始序列
#         seasonal_features = self.time_blocking.reconstruct(seasonal_features, block_positions)
        
#         # 12. 残差连接
#         if seasonal_features.size() == original_x.size():
#             seasonal_features = seasonal_features + original_x
        
#         return seasonal_features


# class InteractionComponent(nn.Module):
#     """
#     交互组件：严格按照FBM-S论文实现
#     固定掩码 + 中心化 + Transformer
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         # 固定掩码（论文中的C₁和C₂）
#         # C₁=24: 输入掩码，针对短期交互
#         # C₂=48: 输出掩码，针对短期交互
#         self.input_mask = self._create_fixed_mask(24, seq_len)
#         self.output_mask = self._create_fixed_mask(48, seq_len)
        
#         # 掩码多头注意力
#         self.masked_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（交互：高频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))
    
#     def _create_fixed_mask(self, mask_size: int, seq_len: int):
#         """创建固定掩码，基于实际物理意义的时间范围"""
#         # 创建1D掩码，用于序列长度限制
#         mask = torch.ones(seq_len)
        
#         # 应用固定掩码 - 只保留前mask_size个时间步
#         if mask_size < seq_len:
#             mask[mask_size:] = 0
        
#         return mask
    
#     def to(self, device):
#         """确保掩码在正确的设备上"""
#         super().to(device)
#         self.input_mask = self.input_mask.to(device)
#         self.output_mask = self.output_mask.to(device)
#         return self
    
#     def forward(self, x):
#         """
#         严格按照论文实现交互组件
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用交互处理
#         interaction_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 特征投影
#             block_projected = self.feature_projection(block)  # [batch, block_size, hidden_dim]
            
#             # 5. 应用固定掩码 - 确保掩码在正确的设备上
#             device = block_projected.device
#             input_mask = self.input_mask[:block.size(1)].to(device)  # [block_size]
#             output_mask = self.output_mask[:block.size(1)].to(device)  # [block_size]
            
#             # 输入掩码 - 用于序列长度限制
#             # 将掩码扩展到与特征维度匹配，确保广播正确
#             # block_projected: [batch, block_size, hidden_dim]
#             # input_mask: [block_size]
#             # 需要扩展到: [1, block_size, 1] 以便与 [batch, block_size, hidden_dim] 广播
#             input_mask_expanded = input_mask.unsqueeze(0).unsqueeze(-1)  # [1, block_size, 1]
            
#             # 应用掩码到序列维度 - 只保留有效的时间步
#             input_masked = block_projected * input_mask_expanded  # [batch, block_size, hidden_dim]
            
#             # 确保张量维度正确
#             assert input_masked.shape == block_projected.shape, f"Mask application failed: {block_projected.shape} vs {input_masked.shape}"
            
#             # 使用掩码后的张量进行注意力计算
#             input_for_attention = input_masked
            
#             # 6. 掩码多头注意力 - 确保输入是3D张量
#             # 检查并确保张量维度正确
#             if input_for_attention.dim() == 4:
#                 # 如果是4D，压缩最后两个维度
#                 batch_size, seq_len = input_for_attention.shape[:2]
#                 input_for_attention = input_for_attention.view(batch_size, seq_len, -1)
#             elif input_for_attention.dim() != 3:
#                 raise ValueError(f"Expected 3D tensor, got {input_for_attention.dim()}D tensor")
            
#             attn_output, _ = self.masked_attention(
#                 input_for_attention, input_for_attention, input_for_attention
#             )
            
#             # 7. Transformer编码
#             transformer_output = self.transformer(attn_output)
            
#             # 8. 应用输出掩码
#             # output_mask: [block_size]
#             # transformer_output: [batch, block_size, hidden_dim]
#             # 需要扩展到: [1, block_size, 1] 以便广播
#             output_mask_expanded = output_mask.unsqueeze(0).unsqueeze(-1)  # [1, block_size, 1]
#             output_masked = transformer_output * output_mask_expanded
            
#             # 9. 输出投影
#             interaction_feature = self.output_projection(output_masked)
#             interaction_features.append(interaction_feature)
        
#         # 10. 重建序列
#         interaction_features = torch.stack(interaction_features, dim=1)
        
#         # 11. 反中心化
#         interaction_features = self.centering((interaction_features, mean, std), mode='decenter')
        
#         # 12. 重建原始序列
#         interaction_features = self.time_blocking.reconstruct(interaction_features, block_positions)
        
#         # 13. 残差连接
#         if interaction_features.size() == original_x.size():
#             interaction_features = interaction_features + original_x
        
#         return interaction_features


# class TrendComponent(nn.Module):
#     """
#     趋势组件：严格按照FBM-S论文实现
#     时间分块 + MLP/Transformer + 频域权重
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         if use_transformer:
#             # Transformer编码器
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=8,
#                 dim_feedforward=hidden_dim * 2,
#                 dropout=0.1,
#                 batch_first=True
#             )
#             self.trend_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         else:
#             # MLP网络
#             self.trend_net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（趋势：低频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))

#     def forward(self, x):
#         """
#         严格按照论文实现趋势组件
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x

#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用趋势处理
#         trend_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 特征投影
#             block_projected = self.feature_projection(block)
            
#             # 5. 趋势建模
#             if self.use_transformer:
#                 trend_output = self.trend_net(block_projected)
#             else:
#                 trend_output = self.trend_net(block_projected)
            
#             # 6. 输出投影
#             trend_feature = self.output_projection(trend_output)
#             trend_features.append(trend_feature)
        
#         # 7. 重建序列
#         trend_features = torch.stack(trend_features, dim=1)
        
#         # 8. 反中心化
#         trend_features = self.centering((trend_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         trend_features = self.time_blocking.reconstruct(trend_features, block_positions)
        
#         # 10. 残差连接
#         if trend_features.size() == original_x.size():
#             trend_features = trend_features + original_x
        
#         return trend_features


# # 工厂函数
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)








# # ===== 分块论文,原始实现：季节 Base_seasonal、趋势 MLP_backbone、交互 Interaction_backbone =====
# import sys
# import os
# from RevIN import RevIN
# from torch import nn, Tensor
# import numpy as np
# import torch
# import torch.nn.functional as F
# import math


# # ========= 时间分块架构 =========
# class TimeBlocking(nn.Module):
#     """
#     时间分块策略：FBM-S论文的核心特性
#     将长序列分解为多个时间块，每个块独立处理
#     """
#     def __init__(self, seq_len: int, block_size: int = 128, overlap: int = 32):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
        
#         # 计算块数量
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
        
#         # 确保最后一个块能覆盖序列末尾
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         # x: [batch, seq_len, num_features]
#         batch_size, seq_len, num_features = x.shape
        
#         # 如果序列长度小于块大小，直接返回
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
        
#         blocks = []
#         block_positions = []
        
#         # 使用简单的滑动窗口策略
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             start_idx = i
#             end_idx = i + self.block_size
            
#             # 提取时间块
#             block = x[:, start_idx:end_idx, :]
#             blocks.append(block)
#             block_positions.append((start_idx, end_idx))
        
#         # 处理最后一个块，确保覆盖序列末尾
#         if block_positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             last_block = x[:, last_start:seq_len, :]
#             blocks.append(last_block)
#             block_positions.append((last_start, seq_len))
        
#         # 堆叠所有块
#         blocks = torch.stack(blocks, dim=1)  # [batch, num_blocks, block_size, num_features]
#         return blocks, block_positions
    
#     def reconstruct(self, blocks, block_positions):
#         # 重建原始序列
#         batch_size, num_blocks, block_size, num_features = blocks.shape
#         seq_len = max(pos[1] for pos in block_positions)
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
#         count = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
        
#         # 将每个块放回原位置
#         for i, (start, end) in enumerate(block_positions):
#             output[:, start:end, :] += blocks[:, i, :, :]
#             count[:, start:end, :] += 1
        
#         # 平均重叠部分
#         count = torch.clamp(count, min=1)
#         output = output / count
        
#         return output


# # ========= 统一前端 =========
# def _fourier_expand(x:torch.Tensor):
#     """输入 x:[B,T,C] → (z_unified:[B,C,P,T], X_oneside:[B,C,P])"""
#     xc = x.permute(0,2,1)  # [B,C,T]
#     norm = xc.size(-1)
#     X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P]
#     # 直接按当前序列长度生成基（无缓存，简单可靠）
#     ts = 1.0 / norm
#     t = torch.arange(0, 1, ts, device=x.device, dtype=x.dtype)
#     cos = None; sin = None
#     for i in range(norm // 2 + 1):
#         if i == 0:
#             cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#         else:
#             cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#             sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#     basis_cos = torch.einsum('bcp,pt->bcpt', X.real, cos)
#     basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, sin)
#     z_unified = basis_cos + basis_sin
#     return z_unified, X

# class Base_seasonal(nn.Module):
#     def __init__(self,context_window, target_window,multiscale):
#         super().__init__()
#         self.context_window=context_window
#         sr=self.context_window
#         self.multiscale=multiscale
#         ts = 1.0/sr
#         t = np.arange(0,1,ts)
#         t=torch.tensor(t, dtype=torch.float32)
#         for i in range(self.context_window//2+1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2*math.pi*i*t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2*math.pi*i*t).unsqueeze(0)
#             elif i == (self.context_window//2):
#                 # Nyquist 频率（仅当偶数长度时存在半频点），系数取 0.5
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2*math.pi*i*t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2*math.pi*i*t).unsqueeze(0)])
#             else:
#                 cos = torch.vstack([cos, torch.cos(2*math.pi*i*t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2*math.pi*i*t).unsqueeze(0)])

#         cos=cos[1:]
#         sin=sin[1:]

#         rolled_tensor_cos = torch.stack([cos.roll(shifts=-i,dims=-1) for i in range(target_window)], dim=0)
#         rolled_tensor_sin = torch.stack([sin.roll(shifts=-i,dims=-1) for i in range(target_window)], dim=0)

#         self.cos = nn.Parameter(rolled_tensor_cos, requires_grad=False)
#         self.sin = nn.Parameter(rolled_tensor_sin, requires_grad=False)

#         W_pos = torch.empty((self.context_window//2,self.context_window), dtype=torch.float32)
#         nn.init.uniform_(W_pos, -0.001, 0.001)
#         self.parameter=nn.Parameter(W_pos, requires_grad=True)

#     def forward(self, x,freq):   
#                                       # x: [bs x nvars x d_model x patch_num]
#         x=x[:,:,1:,:]
#         freq=freq[:,:,1:]
#         hidden_cos=torch.einsum('pkt,kt->pk', self.cos,  self.parameter)
#         hidden_sin=torch.einsum('pkt,kt->pk', self.sin,  self.parameter)
#         x=torch.einsum('bkt,pt->bkp', freq.real, hidden_cos)+torch.einsum('bkt,pt->bkp', freq.imag, hidden_sin)
#         return x

# class MLP_backbone(nn.Module):
#     def __init__(self,context_window, target_window,dropout,hidden1,hidden2,linear,multiscale,drop_initial):
#         super().__init__()

#         self.context_window=context_window
#         self.target_window=target_window

#         self.drop_initial=drop_initial

#         self.linear=linear
#         self.multiscale=multiscale

#         self.flatten = nn.Flatten(start_dim=-2)

#         if self.linear==1:
#             if self.multiscale==2:
#                 self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
#                 self.linear2 = nn.Linear((self.context_window//2)*(self.context_window//4),target_window)
#                 self.linear3 = nn.Linear((self.context_window//4)*(self.context_window//8),target_window)
#             elif self.multiscale==1:
#                 self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
#                 self.linear2 = nn.Linear((self.context_window//2)*(self.context_window//4),target_window)
#             else:
#                 self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
#         else:
#             if self.multiscale==2:
#                 self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )

#                 self.linear2 = nn.Sequential(   nn.Linear((self.context_window//2)*(self.context_window//4),hidden1),  
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )
#                 self.linear3 = nn.Sequential(   nn.Linear((self.context_window//4)*(self.context_window//8),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )
#             elif self.multiscale==1:
#                 self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )

#                 self.linear2 = nn.Sequential(   nn.Linear((self.context_window//2)*(self.context_window//4),hidden1),  
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )
#             else:
#                 self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )

#     def forward(self, x):                             
#         x=x[:,:,1:,:]
#         if self.multiscale==1:
#             down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
#             down1= down1.sum(dim=-2) 
#             down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
#             down1= down1.mean(dim=-1)

#             if self.drop_initial:
#                 add2=self.linear2(self.flatten(down1))
#             else:
#                 add1=self.linear1(self.flatten(x))
#                 add2=self.linear2(self.flatten(down1))

#             if self.drop_initial:
#                  x=add2
#             else:
#                  x=add1+add2

#         elif self.multiscale==2:
#             down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
#             down1= down1.sum(dim=-2) 
#             down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
#             down1= down1.mean(dim=-1) 

#             down2= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2]//2,2, down1.shape[3])
#             down2= down2.sum(dim=-2) 
#             down2= down2.reshape(down2.shape[0], down2.shape[1],down2.shape[2], down2.shape[3]//2,2 )
#             down2= down2.mean(dim=-1)

#             if self.drop_initial:
#                 add2=self.linear2(self.flatten(down1))     
#                 add3=self.linear3(self.flatten(down2))
#             else:
#                 add1=self.linear1(self.flatten(x))
#                 add2=self.linear2(self.flatten(down1))     
#                 add3=self.linear3(self.flatten(down2))
#             if self.drop_initial:
#                  x=add2+add3
#             else:
#                  x=add1+add2+add3
#         else:
#             x=self.linear1(self.flatten(x))
#         return x

# class Interaction_backbone(nn.Module):
#     def __init__(self,configs, context_window, target_window,cut1 ,cut2, d_model2,dropout2,n_heads,n_layers):
#         super().__init__()

#         # 夹紧短窗，避免因回退到全长导致参数暴增
#         self.cut1 = int(max(1, min(cut1 if cut1 is not None and cut1 > 0 else context_window // 4, context_window)))
#         self.cut2 = int(max(1, min(cut2 if cut2 is not None and cut2 > 0 else max(1, context_window // 8), target_window)))
#         self.context_window=context_window
#         self.target_window=target_window

#         # 通道掩码功能已移除

#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, 1, attention_dropout=dropout2,
#                                       output_attention=False), d_model2, n_heads),
#                     d_model2,
#                     d_model2,
#                     dropout=dropout2,
#                     activation='gelu'
#                 ) for l in range(n_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(d_model2)
#         )

#         self.flatten = nn.Flatten(start_dim=-2)
#         self.linear=nn.Linear(self.cut1*(context_window//2),d_model2)

#         if self.cut2 <self.target_window:
#             self.proj=nn.Linear(d_model2,cut2 )
#         else:
#             self.proj=nn.Linear(d_model2,self.target_window )


        
#     def forward(self, z):

#         z=z[:,:,1:,:]         
#         z=self.flatten(z[:,:,:,-self.cut1:])

#         z=self.linear(z)
        
#         z,attention=self.encoder(z)

#         z=self.proj(z)

#         if self.cut2 <self.target_window:
#             a,b,d=z.size()
#             zeros=torch.zeros((a,b,self.target_window), device=z.device, dtype=z.dtype)
#             zeros[:,:,:self.cut2]=z
#             return zeros
#         else:
#             return z

# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x = x + self.dropout(new_x)

#         y = x = self.norm1(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))

#         return self.norm2(x + y), attn

# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / math.sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)

#         if self.mask_flag:
#             if attn_mask is None:
#                 raise RuntimeError("attn_mask is required when mask_flag is True")

#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         if self.output_attention:
#             return (V.contiguous(), A)
#         else:
#             return (V.contiguous(), None)

# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
#         out = out.view(B, L, -1)

#         return self.out_projection(out), attn

# class Encoder(nn.Module):
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         if self.conv_layers is not None:
#             for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
#                 delta = delta if i == 0 else None
#                 x, _ = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
#                 x = conv_layer(x)
#             x, _ = self.attn_layers[-1](x, tau=tau, delta=None)
#         else:
#             for attn_layer in self.attn_layers:
#                 x, _ = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)

#         if self.norm is not None:
#             x = self.norm(x)

#         return x, None

# # ===== 适配您现有模型的工厂函数 =====
# class TrendComponent(nn.Module):
#     """趋势组件：基于论文MLP_backbone，集成时间分块处理"""
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
        
#         # 时间分块相关组件（默认启用）
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 为每个块创建趋势网络
#         self.trend_net = MLP_backbone(
#             context_window=block_size,
#             target_window=block_size,  # 输出与块大小同长度
#             dropout=0.1,
#             hidden1=hidden_dim,
#             hidden2=hidden_dim,
#             linear=0,  # 使用非线性网络
#             multiscale=1,  # 使用多尺度
#             drop_initial=False
#         )
        
#         # 预生成与论文主干一致的 Fourier 基（用于统一前端展开）
#         sr = seq_len
#         ts = 1.0 / sr
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         sin = None
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         self.register_buffer('trend_cos', cos, persistent=False)
#         self.register_buffer('trend_sin', sin, persistent=False)
    
#     def forward(self, x):
#         # 时间分块处理（默认启用）
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 对每个块应用趋势处理
#         trend_features = []
        
#         for i in range(blocks.size(1)):
#             block = blocks[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 使用傅里叶展开
#             z, _ = _fourier_expand(block)
#             trend_output = self.trend_net(z)  # [B, C, block_size]
            
#             # 转换回原格式
#             trend_output = trend_output.permute(0, 2, 1)  # [batch, block_size, num_features]
#             trend_features.append(trend_output)
        
#         # 3. 重建序列
#         trend_features = torch.stack(trend_features, dim=1)
#         trend_features = self.time_blocking.reconstruct(trend_features, block_positions)
        
#         # 4. 残差连接
#         if trend_features.size() == original_x.size():
#             trend_features = trend_features + original_x
        
#         return trend_features

# class SeasonalComponent(nn.Module):
#     """季节性组件：基于论文Base_seasonal，集成时间分块处理"""
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
        
#         # 时间分块相关组件（默认启用）
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 为每个块创建季节性网络
#         self.seasonal_net = Base_seasonal(
#             context_window=block_size,
#             target_window=block_size,  # 输出与块大小同长度
#             multiscale=1
#         )
    
#     def forward(self, x=None, *, z_unified:torch.Tensor=None, X_oneside:torch.Tensor=None):
#         # 时间分块处理（默认启用）
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 对每个块应用季节性处理
#         seasonal_features = []
        
#         for i in range(blocks.size(1)):
#             block = blocks[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 使用傅里叶展开
#             z_unified, X_oneside = _fourier_expand(block)
#             seasonal_output = self.seasonal_net(z_unified, X_oneside)  # [B, C, block_size]
            
#             # 转换回原格式
#             seasonal_output = seasonal_output.permute(0, 2, 1)  # [batch, block_size, num_features]
#             seasonal_features.append(seasonal_output)
        
#         # 3. 重建序列
#         seasonal_features = torch.stack(seasonal_features, dim=1)
#         seasonal_features = self.time_blocking.reconstruct(seasonal_features, block_positions)
        
#         # 4. 残差连接
#         if seasonal_features.size() == original_x.size():
#             seasonal_features = seasonal_features + original_x
        
#         return seasonal_features

# class InteractionComponent(nn.Module):
#     """交互组件：基于论文Interaction_backbone，集成时间分块处理"""
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
        
#         # 时间分块相关组件（默认启用）
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 创建简化的configs对象
#         class SimpleConfig:
#             def __init__(self):
#                 self.patch_num = 1
#                 self.enc_in = num_features
        
#         configs = SimpleConfig()
        
#         # 为每个块创建交互网络
#         self.interaction_net = Interaction_backbone(
#             configs=configs,
#             context_window=block_size,
#             target_window=block_size,  # 输出与块大小同长度
#             cut1=block_size,  # 块大小
#             cut2=block_size,
#             d_model2=hidden_dim,
#             dropout2=0.1,
#             n_heads=num_heads,
#             n_layers=2
#         )
        
#         # 预生成与论文主干一致的 Fourier 基（用于统一前端展开）
#         sr = seq_len
#         ts = 1.0 / sr
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         sin = None
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         self.register_buffer('inter_cos', cos, persistent=False)
#         self.register_buffer('inter_sin', sin, persistent=False)

#         # 可调优的短窗参数（从全局 config 读取时可覆盖）
#         self.cut1 = seq_len
#         self.cut2 = seq_len
#         # 通道掩码开关与阈值移除

#     def set_short_window_and_mask(self, cut1:int=None, cut2:int=None, enable_mask:bool=None, mask_thresh:float=None):
#         if cut1 is not None and cut1 > 0:
#             self.cut1 = cut1
#         if cut2 is not None and cut2 > 0:
#             self.cut2 = cut2
#         if enable_mask is not None:
#             self.enable_channel_mask = enable_mask
#         if mask_thresh is not None:
#             self.channel_mask_thresh = mask_thresh
#         # 同步给 backbone
#         self.interaction_net.cut1 = self.cut1
#         self.interaction_net.cut2 = self.cut2
#         self.interaction_net.channel_mask = self.enable_channel_mask

#     def forward(self, x=None, *, z_unified:torch.Tensor=None):
#         # 时间分块处理（默认启用）
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 对每个块应用交互处理
#         interaction_features = []
        
#         for i in range(blocks.size(1)):
#             block = blocks[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 使用傅里叶展开
#             z_unified, _ = _fourier_expand(block)
#             interaction_output = self.interaction_net(z_unified)  # [B, C, block_size]
            
#             # 转换回原格式
#             interaction_output = interaction_output.permute(0, 2, 1)  # [batch, block_size, num_features]
#             interaction_features.append(interaction_output)
        
#         # 3. 重建序列
#         interaction_features = torch.stack(interaction_features, dim=1)
#         interaction_features = self.time_blocking.reconstruct(interaction_features, block_positions)
        
#         # 4. 残差连接
#         if interaction_features.size() == original_x.size():
#             interaction_features = interaction_features + original_x
        
#         return interaction_features

# # =========== 统一前端 + 三分支一次性调用（论文式） ===========
# class FourierFrontEnd(nn.Module):
#     def __init__(self, seq_len:int):
#         super().__init__()
#         ts = 1.0 / seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None; sin = None
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         self.register_buffer('cos', cos, persistent=False)
#         self.register_buffer('sin', sin, persistent=False)

#     def expand(self, x:torch.Tensor):
#         # x: [B, T, C]
#         xc = x.permute(0, 2, 1)  # [B, C, T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B, C, P]
#         cos = self.cos.to(x.device, dtype=x.dtype)
#         sin = self.sin.to(x.device, dtype=x.dtype)
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, cos)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, sin)
#         z_unified = basis_cos + basis_sin  # [B, C, P, T]
#         return z_unified, X

# class FBMBlocks(nn.Module):
#     """论文式一次性前端 + 三分支（避免重复 FFT/基展开）"""
#     def __init__(self, seq_len:int, num_features:int,
#                  trend_hidden:int=128, inter_hidden:int=128, inter_heads:int=8,
#                  multiscale:int=1, linear:int=0, drop_initial:bool=False,
#                  cut1:int=None, cut2:int=None, channel_mask:bool=False):
#         super().__init__()
#         self.front = FourierFrontEnd(seq_len)
#         # Trend
#         self.trend = MLP_backbone(context_window=seq_len, target_window=seq_len,
#                                   dropout=0.1, hidden1=trend_hidden, hidden2=trend_hidden,
#                                   linear=linear, multiscale=multiscale, drop_initial=drop_initial)
#         # Seasonal
#         self.seasonal = Base_seasonal(context_window=seq_len, target_window=seq_len, multiscale=1)
#         # Interaction
#         class Cfg:
#             pass
#         cfg = Cfg()
#         cfg.enc_in = num_features
#         # 默认与论文对齐：若未传参，则取全长窗口
#         self.cut1 = int(max(1, min(cut1 if cut1 is not None and cut1 > 0 else seq_len, seq_len)))
#         self.cut2 = int(max(1, min(cut2 if cut2 is not None and cut2 > 0 else seq_len, seq_len)))
#         self.inter = Interaction_backbone(cfg, context_window=seq_len, target_window=seq_len,
#                                           cut1=self.cut1, cut2=self.cut2, d_model2=inter_hidden,
#                                           dropout2=0.1, n_heads=inter_heads, n_layers=2)

#     def forward(self, x:torch.Tensor):
#         # x: [B, T, C]
#         z_unified, X_oneside = self.front.expand(x)
#         # Trend
#         trend = self.trend(z_unified)  # [B, C, T]
#         # Seasonal（按论文接口需要 X_oneside）
#         seasonal = self.seasonal(z_unified, X_oneside)  # [B, C, T]
#         # Interaction（不再构造通道掩码）
#         inter = self.inter(z_unified)  # [B, C, T]
#         # 回到 [B, T, C]
#         return trend.permute(0,2,1), seasonal.permute(0,2,1), inter.permute(0,2,1)

# # ===== 您现有模型使用的工厂函数 =====
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """趋势组件工厂函数（默认启用时间分块）"""
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)

# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """季节性组件工厂函数（默认启用时间分块）"""
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)

# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """交互组件工厂函数（默认启用时间分块）"""
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)





# # ===== 论文原始实现：季节 Base_seasonal、趋势 MLP_backbone、交互 Interaction_backbone =====
# import sys
# import os
# from RevIN import RevIN
# from torch import nn, Tensor
# import numpy as np
# import torch
# import torch.nn.functional as F
# import math


# # ========= 统一前端 =========
# def _fourier_expand(x:torch.Tensor):
#     """输入 x:[B,T,C] → (z_unified:[B,C,P,T], X_oneside:[B,C,P])"""
#     xc = x.permute(0,2,1)  # [B,C,T]
#     norm = xc.size(-1)
#     X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P]
#     # 直接按当前序列长度生成基（无缓存，简单可靠）
#     ts = 1.0 / norm
#     t = torch.arange(0, 1, ts, device=x.device, dtype=x.dtype)
#     cos = None; sin = None
#     for i in range(norm // 2 + 1):
#         if i == 0:
#             cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#         else:
#             cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#             sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#     basis_cos = torch.einsum('bcp,pt->bcpt', X.real, cos)
#     basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, sin)
#     z_unified = basis_cos + basis_sin
#     return z_unified, X

# class Base_seasonal(nn.Module):
#     def __init__(self,context_window, target_window,multiscale):
#         super().__init__()
#         self.context_window=context_window
#         sr=self.context_window
#         self.multiscale=multiscale
#         ts = 1.0/sr
#         t = np.arange(0,1,ts)
#         t=torch.tensor(t, dtype=torch.float32)
#         for i in range(self.context_window//2+1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2*math.pi*i*t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2*math.pi*i*t).unsqueeze(0)
#             elif i == (self.context_window//2):
#                 # Nyquist 频率（仅当偶数长度时存在半频点），系数取 0.5
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2*math.pi*i*t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2*math.pi*i*t).unsqueeze(0)])
#             else:
#                 cos = torch.vstack([cos, torch.cos(2*math.pi*i*t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2*math.pi*i*t).unsqueeze(0)])

#         cos=cos[1:]
#         sin=sin[1:]

#         rolled_tensor_cos = torch.stack([cos.roll(shifts=-i,dims=-1) for i in range(target_window)], dim=0)
#         rolled_tensor_sin = torch.stack([sin.roll(shifts=-i,dims=-1) for i in range(target_window)], dim=0)

#         self.cos = nn.Parameter(rolled_tensor_cos, requires_grad=False)
#         self.sin = nn.Parameter(rolled_tensor_sin, requires_grad=False)

#         W_pos = torch.empty((self.context_window//2,self.context_window), dtype=torch.float32)
#         nn.init.uniform_(W_pos, -0.001, 0.001)
#         self.parameter=nn.Parameter(W_pos, requires_grad=True)

#     def forward(self, x,freq):   
#                                       # x: [bs x nvars x d_model x patch_num]
#         x=x[:,:,1:,:]
#         freq=freq[:,:,1:]
#         hidden_cos=torch.einsum('pkt,kt->pk', self.cos,  self.parameter)
#         hidden_sin=torch.einsum('pkt,kt->pk', self.sin,  self.parameter)
#         x=torch.einsum('bkt,pt->bkp', freq.real, hidden_cos)+torch.einsum('bkt,pt->bkp', freq.imag, hidden_sin)
#         return x

# class MLP_backbone(nn.Module):
#     def __init__(self,context_window, target_window,dropout,hidden1,hidden2,linear,multiscale,drop_initial):
#         super().__init__()

#         self.context_window=context_window
#         self.target_window=target_window

#         self.drop_initial=drop_initial

#         self.linear=linear
#         self.multiscale=multiscale

#         self.flatten = nn.Flatten(start_dim=-2)

#         if self.linear==1:
#             if self.multiscale==2:
#                 self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
#                 self.linear2 = nn.Linear((self.context_window//2)*(self.context_window//4),target_window)
#                 self.linear3 = nn.Linear((self.context_window//4)*(self.context_window//8),target_window)
#             elif self.multiscale==1:
#                 self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
#                 self.linear2 = nn.Linear((self.context_window//2)*(self.context_window//4),target_window)
#             else:
#                 self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
#         else:
#             if self.multiscale==2:
#                 self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )

#                 self.linear2 = nn.Sequential(   nn.Linear((self.context_window//2)*(self.context_window//4),hidden1),  
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )
#                 self.linear3 = nn.Sequential(   nn.Linear((self.context_window//4)*(self.context_window//8),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )
#             elif self.multiscale==1:
#                 self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )

#                 self.linear2 = nn.Sequential(   nn.Linear((self.context_window//2)*(self.context_window//4),hidden1),  
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )
#             else:
#                 self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )

#     def forward(self, x):                             
#         x=x[:,:,1:,:]
#         if self.multiscale==1:
#             down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
#             down1= down1.sum(dim=-2) 
#             down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
#             down1= down1.mean(dim=-1)

#             if self.drop_initial:
#                 add2=self.linear2(self.flatten(down1))
#             else:
#                 add1=self.linear1(self.flatten(x))
#                 add2=self.linear2(self.flatten(down1))

#             if self.drop_initial:
#                  x=add2
#             else:
#                  x=add1+add2

#         elif self.multiscale==2:
#             down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
#             down1= down1.sum(dim=-2) 
#             down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
#             down1= down1.mean(dim=-1) 

#             down2= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2]//2,2, down1.shape[3])
#             down2= down2.sum(dim=-2) 
#             down2= down2.reshape(down2.shape[0], down2.shape[1],down2.shape[2], down2.shape[3]//2,2 )
#             down2= down2.mean(dim=-1)

#             if self.drop_initial:
#                 add2=self.linear2(self.flatten(down1))     
#                 add3=self.linear3(self.flatten(down2))
#             else:
#                 add1=self.linear1(self.flatten(x))
#                 add2=self.linear2(self.flatten(down1))     
#                 add3=self.linear3(self.flatten(down2))
#             if self.drop_initial:
#                  x=add2+add3
#             else:
#                  x=add1+add2+add3
#         else:
#             x=self.linear1(self.flatten(x))
#         return x

# class Interaction_backbone(nn.Module):
#     def __init__(self,configs, context_window, target_window,cut1 ,cut2, d_model2,dropout2,n_heads,n_layers):
#         super().__init__()

#         # 夹紧短窗，避免因回退到全长导致参数暴增
#         self.cut1 = int(max(1, min(cut1 if cut1 is not None and cut1 > 0 else context_window // 4, context_window)))
#         self.cut2 = int(max(1, min(cut2 if cut2 is not None and cut2 > 0 else max(1, context_window // 8), target_window)))
#         self.context_window=context_window
#         self.target_window=target_window

#         # 通道掩码功能已移除

#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, 1, attention_dropout=dropout2,
#                                       output_attention=False), d_model2, n_heads),
#                     d_model2,
#                     d_model2,
#                     dropout=dropout2,
#                     activation='gelu'
#                 ) for l in range(n_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(d_model2)
#         )

#         self.flatten = nn.Flatten(start_dim=-2)
#         self.linear=nn.Linear(self.cut1*(context_window//2),d_model2)

#         if self.cut2 <self.target_window:
#             self.proj=nn.Linear(d_model2,cut2 )
#         else:
#             self.proj=nn.Linear(d_model2,self.target_window )


        
#     def forward(self, z):

#         z=z[:,:,1:,:]         
#         z=self.flatten(z[:,:,:,-self.cut1:])

#         z=self.linear(z)
        
#         z,attention=self.encoder(z)

#         z=self.proj(z)

#         if self.cut2 <self.target_window:
#             a,b,d=z.size()
#             zeros=torch.zeros((a,b,self.target_window), device=z.device, dtype=z.dtype)
#             zeros[:,:,:self.cut2]=z
#             return zeros
#         else:
#             return z

# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x = x + self.dropout(new_x)

#         y = x = self.norm1(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))

#         return self.norm2(x + y), attn

# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / math.sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)

#         if self.mask_flag:
#             if attn_mask is None:
#                 raise RuntimeError("attn_mask is required when mask_flag is True")

#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         if self.output_attention:
#             return (V.contiguous(), A)
#         else:
#             return (V.contiguous(), None)

# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
#         out = out.view(B, L, -1)

#         return self.out_projection(out), attn

# class Encoder(nn.Module):
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         if self.conv_layers is not None:
#             for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
#                 delta = delta if i == 0 else None
#                 x, _ = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
#                 x = conv_layer(x)
#             x, _ = self.attn_layers[-1](x, tau=tau, delta=None)
#         else:
#             for attn_layer in self.attn_layers:
#                 x, _ = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)

#         if self.norm is not None:
#             x = self.norm(x)

#         return x, None

# # ===== 适配您现有模型的工厂函数 =====
# class TrendComponent(nn.Module):
#     """趋势组件：基于论文MLP_backbone，适配为特征提取"""
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
        
#         # 使用论文的MLP_backbone作为趋势组件
#         self.trend_net = MLP_backbone(
#             context_window=seq_len,
#             target_window=seq_len,  # 输出与输入同长度
#             dropout=0.1,
#             hidden1=hidden_dim,
#             hidden2=hidden_dim,
#             linear=0,  # 使用非线性网络
#             multiscale=1,  # 使用多尺度
#             drop_initial=False
#         )
#         # 预生成与论文主干一致的 Fourier 基（用于统一前端展开）
#         sr = seq_len
#         ts = 1.0 / sr
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         sin = None
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         self.register_buffer('trend_cos', cos, persistent=False)
#         self.register_buffer('trend_sin', sin, persistent=False)
    
#     def forward(self, x):
#         # 输入: [B,T,C]；通过缓存前端实现“统一一次展开、三分支复用”
#         z, _ = _fourier_expand(x)
#         trend_out = self.trend_net(z)  # [B, C, T]
        
#         # 转换回原格式
#         trend_out = trend_out.permute(0, 2, 1)  # [batch, seq_len, num_features]
        
#         return trend_out

# class SeasonalComponent(nn.Module):
#     """季节性组件：基于论文Base_seasonal，适配为特征提取"""
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
        
#         # 使用论文的Base_seasonal作为季节性组件
#         self.seasonal_net = Base_seasonal(
#             context_window=seq_len,
#             target_window=seq_len,  # 输出与输入同长度
#             multiscale=1
#         )
    
#     def forward(self, x=None, *, z_unified:torch.Tensor=None, X_oneside:torch.Tensor=None):
#         # 两种调用方式：
#         # 1) forward(x=...)：内部统一基展开（不建议在高效路径使用）
#         # 2) forward(z_unified=..., X_oneside=...)：使用统一前端的输出（推荐）
#         if z_unified is None or X_oneside is None:
#             assert x is not None, "Provide x or (z_unified, X_oneside)."
#             z_unified, X_oneside = _fourier_expand(x)
#         # 通过季节性网络（与论文一致：用 X_oneside 与自身基做投影）
#         seasonal_out = self.seasonal_net(z_unified, X_oneside)  # [B, C, T]
        
#         # 转换回原格式
#         seasonal_out = seasonal_out.permute(0, 2, 1)  # [batch, seq_len, num_features]
        
#         return seasonal_out

# class InteractionComponent(nn.Module):
#     """交互组件：基于论文Interaction_backbone，适配为特征提取"""
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
        
#         # 创建简化的configs对象
#         class SimpleConfig:
#             def __init__(self):
#                 self.patch_num = 1
#                 self.enc_in = num_features
        
#         configs = SimpleConfig()
        
#         # 使用论文的Interaction_backbone作为交互组件
#         self.interaction_net = Interaction_backbone(
#             configs=configs,
#             context_window=seq_len,
#             target_window=seq_len,  # 输出与输入同长度
#             cut1=seq_len,  # 将由外部配置覆盖
#             cut2=seq_len,
#             d_model2=hidden_dim,
#             dropout2=0.1,
#             n_heads=num_heads,
#             n_layers=2
#         )
#         # 预生成与论文主干一致的 Fourier 基（用于统一前端展开）
#         sr = seq_len
#         ts = 1.0 / sr
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         sin = None
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#         else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         self.register_buffer('inter_cos', cos, persistent=False)
#         self.register_buffer('inter_sin', sin, persistent=False)

#         # 可调优的短窗参数（从全局 config 读取时可覆盖）
#         self.cut1 = seq_len
#         self.cut2 = seq_len
#         # 通道掩码开关与阈值移除

#     def set_short_window_and_mask(self, cut1:int=None, cut2:int=None, enable_mask:bool=None, mask_thresh:float=None):
#         if cut1 is not None and cut1 > 0:
#             self.cut1 = cut1
#         if cut2 is not None and cut2 > 0:
#             self.cut2 = cut2
#         if enable_mask is not None:
#             self.enable_channel_mask = enable_mask
#         if mask_thresh is not None:
#             self.channel_mask_thresh = mask_thresh
#         # 同步给 backbone
#         self.interaction_net.cut1 = self.cut1
#         self.interaction_net.cut2 = self.cut2
#         self.interaction_net.channel_mask = self.enable_channel_mask

#     def forward(self, x=None, *, z_unified:torch.Tensor=None):
#         # 两种调用：
#         # 1) forward(x=...)：内部自算统一基展开（不建议高效路径）
#         # 2) forward(z_unified=...)：使用统一前端（推荐）
#         if z_unified is None:
#             assert x is not None, "Provide x or z_unified."
#             z_unified, _ = _fourier_expand(x)
#         interaction_out = self.interaction_net(z_unified)  # [B, C, T]
        
#         # 转换回原格式
#         interaction_out = interaction_out.permute(0, 2, 1)  # [batch, seq_len, num_features]
        
#         return interaction_out

# # =========== 统一前端 + 三分支一次性调用（论文式） ===========
# class FourierFrontEnd(nn.Module):
#     def __init__(self, seq_len:int):
#         super().__init__()
#         ts = 1.0 / seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None; sin = None
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         self.register_buffer('cos', cos, persistent=False)
#         self.register_buffer('sin', sin, persistent=False)

#     def expand(self, x:torch.Tensor):
#         # x: [B, T, C]
#         xc = x.permute(0, 2, 1)  # [B, C, T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B, C, P]
#         cos = self.cos.to(x.device, dtype=x.dtype)
#         sin = self.sin.to(x.device, dtype=x.dtype)
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, cos)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, sin)
#         z_unified = basis_cos + basis_sin  # [B, C, P, T]
#         return z_unified, X

# class FBMBlocks(nn.Module):
#     """论文式一次性前端 + 三分支（避免重复 FFT/基展开）"""
#     def __init__(self, seq_len:int, num_features:int,
#                  trend_hidden:int=128, inter_hidden:int=128, inter_heads:int=8,
#                  multiscale:int=1, linear:int=0, drop_initial:bool=False,
#                  cut1:int=None, cut2:int=None, channel_mask:bool=False):
#         super().__init__()
#         self.front = FourierFrontEnd(seq_len)
#         # Trend
#         self.trend = MLP_backbone(context_window=seq_len, target_window=seq_len,
#                                   dropout=0.1, hidden1=trend_hidden, hidden2=trend_hidden,
#                                   linear=linear, multiscale=multiscale, drop_initial=drop_initial)
#         # Seasonal
#         self.seasonal = Base_seasonal(context_window=seq_len, target_window=seq_len, multiscale=1)
#         # Interaction
#         class Cfg:
#             pass
#         cfg = Cfg()
#         cfg.enc_in = num_features
#         # 默认与论文对齐：若未传参，则取全长窗口
#         self.cut1 = int(max(1, min(cut1 if cut1 is not None and cut1 > 0 else seq_len, seq_len)))
#         self.cut2 = int(max(1, min(cut2 if cut2 is not None and cut2 > 0 else seq_len, seq_len)))
#         self.inter = Interaction_backbone(cfg, context_window=seq_len, target_window=seq_len,
#                                           cut1=self.cut1, cut2=self.cut2, d_model2=inter_hidden,
#                                           dropout2=0.1, n_heads=inter_heads, n_layers=2)

#     def forward(self, x:torch.Tensor):
#         # x: [B, T, C]
#         z_unified, X_oneside = self.front.expand(x)
#         # Trend
#         trend = self.trend(z_unified)  # [B, C, T]
#         # Seasonal（按论文接口需要 X_oneside）
#         seasonal = self.seasonal(z_unified, X_oneside)  # [B, C, T]
#         # Interaction（不再构造通道掩码）
#         inter = self.inter(z_unified)  # [B, C, T]
#         # 回到 [B, T, C]
#         return trend.permute(0,2,1), seasonal.permute(0,2,1), inter.permute(0,2,1)

# # ===== 您现有模型使用的工厂函数 =====
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """趋势组件工厂函数"""
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)

# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """季节性组件工厂函数"""
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)

# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """交互组件工厂函数"""
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)



# # =====zsq结果 论文原始实现：季节 Base_seasonal、趋势 MLP_backbone、交互 Interaction_backbone =====
# import sys
# import os
# from RevIN import RevIN
# from torch import nn, Tensor
# import numpy as np
# import torch
# import torch.nn.functional as F
# import math


# # ========= 统一前端 =========
# def _fourier_expand(x:torch.Tensor):
#     """输入 x:[B,T,C] → (z_unified:[B,C,P,T], X_oneside:[B,C,P])"""
#     xc = x.permute(0,2,1)  # [B,C,T]
#     norm = xc.size(-1)
#     X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B,C,P]
#     # 直接按当前序列长度生成基（无缓存，简单可靠）
#     ts = 1.0 / norm
#     t = torch.arange(0, 1, ts, device=x.device, dtype=x.dtype)
#     cos = None; sin = None
#     for i in range(norm // 2 + 1):
#         if i == 0:
#             cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#             sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#         else:
#             cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#             sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#     basis_cos = torch.einsum('bcp,pt->bcpt', X.real, cos)
#     basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, sin)
#     z_unified = basis_cos + basis_sin
#     return z_unified, X

# class Base_seasonal(nn.Module):
#     def __init__(self,context_window, target_window,multiscale):
#         super().__init__()
#         self.context_window=context_window
#         sr=self.context_window
#         self.multiscale=multiscale
#         ts = 1.0/sr
#         t = np.arange(0,1,ts)
#         t=torch.tensor(t, dtype=torch.float32)
#         for i in range(self.context_window//2+1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2*math.pi*i*t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2*math.pi*i*t).unsqueeze(0)
#             elif i == (self.context_window//2):
#                 # Nyquist 频率（仅当偶数长度时存在半频点），系数取 0.5
#                 cos = torch.vstack([cos, 0.5 * torch.cos(2*math.pi*i*t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -0.5 * torch.sin(2*math.pi*i*t).unsqueeze(0)])
#             else:
#                 cos = torch.vstack([cos, torch.cos(2*math.pi*i*t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2*math.pi*i*t).unsqueeze(0)])

#         cos=cos[1:]
#         sin=sin[1:]

#         rolled_tensor_cos = torch.stack([cos.roll(shifts=-i,dims=-1) for i in range(target_window)], dim=0)
#         rolled_tensor_sin = torch.stack([sin.roll(shifts=-i,dims=-1) for i in range(target_window)], dim=0)

#         self.cos = nn.Parameter(rolled_tensor_cos, requires_grad=False)
#         self.sin = nn.Parameter(rolled_tensor_sin, requires_grad=False)

#         W_pos = torch.empty((self.context_window//2,self.context_window), dtype=torch.float32)
#         nn.init.uniform_(W_pos, -0.001, 0.001)
#         self.parameter=nn.Parameter(W_pos, requires_grad=True)

#     def forward(self, x,freq):   
#                                       # x: [bs x nvars x d_model x patch_num]
#         x=x[:,:,1:,:]
#         freq=freq[:,:,1:]
#         hidden_cos=torch.einsum('pkt,kt->pk', self.cos,  self.parameter)
#         hidden_sin=torch.einsum('pkt,kt->pk', self.sin,  self.parameter)
#         x=torch.einsum('bkt,pt->bkp', freq.real, hidden_cos)+torch.einsum('bkt,pt->bkp', freq.imag, hidden_sin)
#         return x

# class MLP_backbone(nn.Module):
#     def __init__(self,context_window, target_window,dropout,hidden1,hidden2,linear,multiscale,drop_initial):
#         super().__init__()

#         self.context_window=context_window
#         self.target_window=target_window

#         self.drop_initial=drop_initial

#         self.linear=linear
#         self.multiscale=multiscale

#         self.flatten = nn.Flatten(start_dim=-2)

#         if self.linear==1:
#             if self.multiscale==2:
#                 self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
#                 self.linear2 = nn.Linear((self.context_window//2)*(self.context_window//4),target_window)
#                 self.linear3 = nn.Linear((self.context_window//4)*(self.context_window//8),target_window)
#             elif self.multiscale==1:
#                 self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
#                 self.linear2 = nn.Linear((self.context_window//2)*(self.context_window//4),target_window)
#             else:
#                 self.linear1 = nn.Linear((self.context_window)*(self.context_window//2),target_window)
#         else:
#             if self.multiscale==2:
#                 self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )

#                 self.linear2 = nn.Sequential(   nn.Linear((self.context_window//2)*(self.context_window//4),hidden1),  
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )
#                 self.linear3 = nn.Sequential(   nn.Linear((self.context_window//4)*(self.context_window//8),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )
#             elif self.multiscale==1:
#                 self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )

#                 self.linear2 = nn.Sequential(   nn.Linear((self.context_window//2)*(self.context_window//4),hidden1),  
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )
#             else:
#                 self.linear1 = nn.Sequential(   nn.Linear(self.context_window*(self.context_window//2),hidden1), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden1,hidden2), 
#                                                 nn.Dropout(p=dropout),
#                                                 nn.ReLU(),
#                                                 nn.Linear(hidden2, target_window),
#                                             )

#     def forward(self, x):                             
#         x=x[:,:,1:,:]
#         if self.multiscale==1:
#             down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
#             down1= down1.sum(dim=-2) 
#             down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
#             down1= down1.mean(dim=-1)

#             if self.drop_initial:
#                 add2=self.linear2(self.flatten(down1))
#             else:
#                 add1=self.linear1(self.flatten(x))
#                 add2=self.linear2(self.flatten(down1))

#             if self.drop_initial:
#                  x=add2
#             else:
#                  x=add1+add2

#         elif self.multiscale==2:
#             down1= x.reshape(x.shape[0], x.shape[1],x.shape[2]//2,2, x.shape[3])
#             down1= down1.sum(dim=-2) 
#             down1= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2], down1.shape[3]//2,2 )
#             down1= down1.mean(dim=-1) 

#             down2= down1.reshape(down1.shape[0], down1.shape[1],down1.shape[2]//2,2, down1.shape[3])
#             down2= down2.sum(dim=-2) 
#             down2= down2.reshape(down2.shape[0], down2.shape[1],down2.shape[2], down2.shape[3]//2,2 )
#             down2= down2.mean(dim=-1)

#             if self.drop_initial:
#                 add2=self.linear2(self.flatten(down1))     
#                 add3=self.linear3(self.flatten(down2))
#             else:
#                 add1=self.linear1(self.flatten(x))
#                 add2=self.linear2(self.flatten(down1))     
#                 add3=self.linear3(self.flatten(down2))
#             if self.drop_initial:
#                  x=add2+add3
#             else:
#                  x=add1+add2+add3
#         else:
#             x=self.linear1(self.flatten(x))
#         return x

# class Interaction_backbone(nn.Module):
#     def __init__(self,configs, context_window, target_window,cut1 ,cut2, d_model2,dropout2,n_heads,n_layers):
#         super().__init__()

#         # 夹紧短窗，避免因回退到全长导致参数暴增
#         self.cut1 = int(max(1, min(cut1 if cut1 is not None and cut1 > 0 else context_window // 4, context_window)))
#         self.cut2 = int(max(1, min(cut2 if cut2 is not None and cut2 > 0 else max(1, context_window // 8), target_window)))
#         self.context_window=context_window
#         self.target_window=target_window

#         # 通道掩码功能已移除

#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, 1, attention_dropout=dropout2,
#                                       output_attention=False), d_model2, n_heads),
#                     d_model2,
#                     d_model2,
#                     dropout=dropout2,
#                     activation='gelu'
#                 ) for l in range(n_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(d_model2)
#         )

#         self.flatten = nn.Flatten(start_dim=-2)

#         self.patch_num=configs.patch_num
#         patch_len=int((context_window*(context_window//2))/self.patch_num)
#         self.linear=nn.Linear(self.cut1*(context_window//2),d_model2)

#         if self.cut2 <self.target_window:
#             self.proj=nn.Linear(d_model2,cut2 )
#         else:
#             self.proj=nn.Linear(d_model2,self.target_window )
#         self.revin = RevIN(configs.enc_in)

        
#     def forward(self, z):

#         z=z[:,:,1:,:]         
#         z=self.flatten(z[:,:,:,-self.cut1:])
#         if self.cut1<self.context_window:
#             z=z.permute(0,2,1)
#             z = self.revin(z, 'norm')
#             z=z.permute(0,2,1)

#         z=self.linear(z)
        
#         if self.cut1<self.context_window:

#             z=z.permute(0,2,1)
#             z = self.revin(z, 'denorm')
#             z=z.permute(0,2,1)

#         z,attention=self.encoder(z)

#         z=self.proj(z)

#         if self.cut2 <self.target_window:
#             a,b,d=z.size()
#             zeros=torch.zeros((a,b,self.target_window), device=z.device, dtype=z.dtype)
#             zeros[:,:,:self.cut2]=z
#             return zeros
#         else:
#             return z

# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x = x + self.dropout(new_x)

#         y = x = self.norm1(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))

#         return self.norm2(x + y), attn

# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / math.sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)

#         if self.mask_flag:
#             if attn_mask is None:
#                 raise RuntimeError("attn_mask is required when mask_flag is True")

#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         if self.output_attention:
#             return (V.contiguous(), A)
#         else:
#             return (V.contiguous(), None)

# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
#         out = out.view(B, L, -1)

#         return self.out_projection(out), attn

# class Encoder(nn.Module):
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         attns = []
#         if self.conv_layers is not None:
#             for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
#                 delta = delta if i == 0 else None
#                 x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
#                 x = conv_layer(x)
#                 attns.append(attn)
#             x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
#             attns.append(attn)
#         else:
#             for attn_layer in self.attn_layers:
#                 x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
#                 attns.append(attn)

#         if self.norm is not None:
#             x = self.norm(x)

#         return x, attns

# # ===== 适配您现有模型的工厂函数 =====
# class TrendComponent(nn.Module):
#     """趋势组件：基于论文MLP_backbone，适配为特征提取"""
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
        
#         # 使用论文的MLP_backbone作为趋势组件
#         self.trend_net = MLP_backbone(
#             context_window=seq_len,
#             target_window=seq_len,  # 输出与输入同长度
#             dropout=0.1,
#             hidden1=hidden_dim,
#             hidden2=hidden_dim,
#             linear=0,  # 使用非线性网络
#             multiscale=1,  # 使用多尺度
#             drop_initial=False
#         )
#         # 预生成与论文主干一致的 Fourier 基（用于统一前端展开）
#         sr = seq_len
#         ts = 1.0 / sr
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         sin = None
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         self.register_buffer('trend_cos', cos, persistent=False)
#         self.register_buffer('trend_sin', sin, persistent=False)
    
#     def forward(self, x):
#         # 输入: [B,T,C]；通过缓存前端实现“统一一次展开、三分支复用”
#         z, _ = _fourier_expand(x)
#         trend_out = self.trend_net(z)  # [B, C, T]
        
#         # 转换回原格式
#         trend_out = trend_out.permute(0, 2, 1)  # [batch, seq_len, num_features]
        
#         return trend_out

# class SeasonalComponent(nn.Module):
#     """季节性组件：基于论文Base_seasonal，适配为特征提取"""
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
        
#         # 使用论文的Base_seasonal作为季节性组件
#         self.seasonal_net = Base_seasonal(
#             context_window=seq_len,
#             target_window=seq_len,  # 输出与输入同长度
#             multiscale=1
#         )
    
#     def forward(self, x=None, *, z_unified:torch.Tensor=None, X_oneside:torch.Tensor=None):
#         # 两种调用方式：
#         # 1) forward(x=...)：内部统一基展开（不建议在高效路径使用）
#         # 2) forward(z_unified=..., X_oneside=...)：使用统一前端的输出（推荐）
#         if z_unified is None or X_oneside is None:
#             assert x is not None, "Provide x or (z_unified, X_oneside)."
#             z_unified, X_oneside = _fourier_expand(x)
#         # 通过季节性网络
#         seasonal_out = self.seasonal_net(z_unified, X_oneside)  # [B, C, T]
        
#         # 转换回原格式
#         seasonal_out = seasonal_out.permute(0, 2, 1)  # [batch, seq_len, num_features]
        
#         return seasonal_out

# class InteractionComponent(nn.Module):
#     """交互组件：基于论文Interaction_backbone，适配为特征提取"""
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
        
#         # 创建简化的configs对象
#         class SimpleConfig:
#             def __init__(self):
#                 self.patch_num = 1
#                 self.enc_in = num_features
        
#         configs = SimpleConfig()
        
#         # 使用论文的Interaction_backbone作为交互组件
#         self.interaction_net = Interaction_backbone(
#             configs=configs,
#             context_window=seq_len,
#             target_window=seq_len,  # 输出与输入同长度
#             cut1=seq_len,  # 将由外部配置覆盖
#             cut2=seq_len,
#             d_model2=hidden_dim,
#             dropout2=0.1,
#             n_heads=num_heads,
#             n_layers=2
#         )
#         # 预生成与论文主干一致的 Fourier 基（用于统一前端展开）
#         sr = seq_len
#         ts = 1.0 / sr
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None
#         sin = None
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#         else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         self.register_buffer('inter_cos', cos, persistent=False)
#         self.register_buffer('inter_sin', sin, persistent=False)

#         # 可调优的短窗参数（从全局 config 读取时可覆盖）
#         self.cut1 = seq_len
#         self.cut2 = seq_len

#     def set_short_window_and_mask(self, cut1:int=None, cut2:int=None, enable_mask:bool=None, mask_thresh:float=None):
#         if cut1 is not None and cut1 > 0:
#             self.cut1 = cut1
#         if cut2 is not None and cut2 > 0:
#             self.cut2 = cut2
#         if enable_mask is not None:
#             self.enable_channel_mask = enable_mask
#         if mask_thresh is not None:
#             self.channel_mask_thresh = mask_thresh
#         # 同步给 backbone
#         self.interaction_net.cut1 = self.cut1
#         self.interaction_net.cut2 = self.cut2
#         self.interaction_net.channel_mask = self.enable_channel_mask

#     def forward(self, x=None, *, z_unified:torch.Tensor=None):
#         # 两种调用：
#         # 1) forward(x=...)：内部自算统一基展开（不建议高效路径）
#         # 2) forward(z_unified=...)：使用统一前端（推荐）
#         if z_unified is None:
#             assert x is not None, "Provide x or z_unified."
#             z_unified, _ = _fourier_expand(x)
#         interaction_out = self.interaction_net(z_unified)  # [B, C, T]
        
#         # 转换回原格式
#         interaction_out = interaction_out.permute(0, 2, 1)  # [batch, seq_len, num_features]
        
#         return interaction_out

# # =========== 统一前端 + 三分支一次性调用 ===========
# class FourierFrontEnd(nn.Module):
#     def __init__(self, seq_len:int):
#         super().__init__()
#         ts = 1.0 / seq_len
#         t = torch.arange(0, 1, ts, dtype=torch.float32)
#         cos = None; sin = None
#         for i in range(seq_len // 2 + 1):
#             if i == 0:
#                 cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
#                 sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
#             else:
#                 cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
#                 sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])
#         self.register_buffer('cos', cos, persistent=False)
#         self.register_buffer('sin', sin, persistent=False)

#     def expand(self, x:torch.Tensor):
#         # x: [B, T, C]
#         xc = x.permute(0, 2, 1)  # [B, C, T]
#         norm = xc.size(-1)
#         X = torch.fft.rfft(xc, dim=-1) / norm * 2  # [B, C, P]
#         cos = self.cos.to(x.device, dtype=x.dtype)
#         sin = self.sin.to(x.device, dtype=x.dtype)
#         basis_cos = torch.einsum('bcp,pt->bcpt', X.real, cos)
#         basis_sin = torch.einsum('bcp,pt->bcpt', X.imag, sin)
#         z_unified = basis_cos + basis_sin  # [B, C, P, T]
#         return z_unified, X

# class FBMBlocks(nn.Module):
#     """论文式一次性前端 + 三分支（避免重复 FFT/基展开）"""
#     def __init__(self, seq_len:int, num_features:int,
#                  trend_hidden:int=128, inter_hidden:int=128, inter_heads:int=8,
#                  multiscale:int=1, linear:int=0, drop_initial:bool=False,
#                  cut1:int=None, cut2:int=None, channel_mask:bool=False):
#         super().__init__()
#         self.front = FourierFrontEnd(seq_len)
#         # Trend
#         self.trend = MLP_backbone(context_window=seq_len, target_window=seq_len,
#                                   dropout=0.1, hidden1=trend_hidden, hidden2=trend_hidden,
#                                   linear=linear, multiscale=multiscale, drop_initial=drop_initial)
#         # Seasonal
#         self.seasonal = Base_seasonal(context_window=seq_len, target_window=seq_len, multiscale=1)
#         # Interaction
#         class Cfg:
#             pass
#         cfg = Cfg()
#         cfg.patch_num = 1
#         cfg.enc_in = num_features
#         # 缺省短窗采用保守默认，避免全长导致参数暴增
#         self.cut1 = int(max(1, min(cut1 if cut1 is not None and cut1 > 0 else seq_len // 4, seq_len)))
#         self.cut2 = int(max(1, min(cut2 if cut2 is not None and cut2 > 0 else max(1, seq_len // 8), seq_len)))
#         self.inter = Interaction_backbone(cfg, context_window=seq_len, target_window=seq_len,
#                                           cut1=self.cut1, cut2=self.cut2, d_model2=inter_hidden,
#                                           dropout2=0.1, n_heads=inter_heads, n_layers=2)

#     def forward(self, x:torch.Tensor):
#         # x: [B, T, C]
#         z_unified, X_oneside = self.front.expand(x)
#         # Trend
#         trend = self.trend(z_unified)  # [B, C, T]
#         # Seasonal（按论文接口需要 X_oneside）
#         seasonal = self.seasonal(z_unified, X_oneside)  # [B, C, T]
#         # Interaction（不再构造通道掩码）
#         inter = self.inter(z_unified)  # [B, C, T]
#         # 回到 [B, T, C]
#         return trend.permute(0,2,1), seasonal.permute(0,2,1), inter.permute(0,2,1)

# # ===== 您现有模型使用的工厂函数 =====
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """趋势组件工厂函数"""
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)

# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """季节性组件工厂函数"""
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)

# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """交互组件工厂函数"""
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)



# """
# FBM-S论文：时间分块
# 严格按照论文方法实现时间-频域特征构建、季节性组件、掩码机制等
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.fft import rfft, irfft
# from einops import rearrange
# import math


# class RevIN(nn.Module):
#     """
#     可逆实例归一化 (Reversible Instance Normalization)
#     基于FBM-S论文的实现
#     """
#     def __init__(self, num_features: int, affine=True, eps=1e-5):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
        
#         if self.affine:
#             self.W = nn.Parameter(torch.ones(num_features))
#             self.b = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x, mode='norm'):
#         if mode == 'norm':
#             # 计算均值和标准差
#             mean = torch.mean(x, dim=1, keepdim=True)
#             std = torch.std(x, dim=1, keepdim=True) + self.eps
            
#             # 归一化
#             x_norm = (x - mean) / std
            
#             # 仿射变换
#             if self.affine:
#                 x_norm = x_norm * self.W.unsqueeze(0).unsqueeze(0) + self.b.unsqueeze(0).unsqueeze(0)
            
#             return x_norm
        
#         elif mode == 'denorm':
#             # 反归一化
#             if self.affine:
#                 x_denorm = (x - self.b.unsqueeze(0).unsqueeze(0)) / self.W.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_denorm = x
#                 return x_denorm
#         else:
#             raise ValueError(f"mode {mode} not supported.")


# class TimeBlocking(nn.Module):
#     """
#     时间分块策略：FBM-S论文的核心特性
#     将长序列分解为多个时间块，每个块独立处理
#     """
#     def __init__(self, seq_len: int, block_size: int = 64, overlap: int = 16):
#         super().__init__()
#         self.seq_len = seq_len
#         self.block_size = block_size
#         self.overlap = overlap
#         self.stride = block_size - overlap
        
#         # 计算块数量
#         self.num_blocks = max(1, (seq_len - overlap) // self.stride)
        
#         # 确保最后一个块能覆盖序列末尾
#         if (seq_len - overlap) % self.stride != 0:
#             self.num_blocks += 1
    
#     def forward(self, x):
#         # x: [batch, seq_len, num_features]
#         batch_size, seq_len, num_features = x.shape
        
#         # 如果序列长度小于块大小，直接返回
#         if seq_len <= self.block_size:
#             return x, [(0, seq_len)]
        
#         blocks = []
#         block_positions = []
        
#         # 使用简单的滑动窗口策略
#         for i in range(0, seq_len - self.block_size + 1, self.stride):
#             start_idx = i
#             end_idx = i + self.block_size
            
#             # 提取时间块
#             block = x[:, start_idx:end_idx, :]
#             blocks.append(block)
#             block_positions.append((start_idx, end_idx))
        
#         # 处理最后一个块，确保覆盖序列末尾
#         if block_positions[-1][1] < seq_len:
#             last_start = max(0, seq_len - self.block_size)
#             last_block = x[:, last_start:seq_len, :]
#             blocks.append(last_block)
#             block_positions.append((last_start, seq_len))
        
#         # 堆叠所有块
#         blocks = torch.stack(blocks, dim=1)  # [batch, num_blocks, block_size, num_features]
#         return blocks, block_positions
    
#     def reconstruct(self, blocks, block_positions):
#         # 重建原始序列
#         batch_size, num_blocks, block_size, num_features = blocks.shape
#         seq_len = max(pos[1] for pos in block_positions)
        
#         # 初始化输出张量
#         output = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
#         count = torch.zeros(batch_size, seq_len, num_features, device=blocks.device)
        
#         # 将每个块放回原位置
#         for i, (start, end) in enumerate(block_positions):
#             output[:, start:end, :] += blocks[:, i, :, :]
#             count[:, start:end, :] += 1
        
#         # 平均重叠部分
#         count = torch.clamp(count, min=1)
#         output = output / count
        
#         return output


# class CenteringLayer(nn.Module):
#     """
#     中心化层：FBM-S论文的关键技术
#     每个块内部进行中心化处理，提高模型鲁棒性
#     """
#     def __init__(self, num_features: int, affine=True):
#         super().__init__()
#         self.num_features = num_features
#         self.affine = affine
        
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(num_features))
#             self.beta = nn.Parameter(torch.zeros(num_features))
    
#     def forward(self, x, mode='center'):
#         if mode == 'center':
#             # 计算每个块的均值和标准差
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑为 [batch * num_blocks, block_size, num_features]
#             x_reshaped = x.view(-1, block_size, num_features)
            
#             # 计算均值和标准差
#             mean = torch.mean(x_reshaped, dim=1, keepdim=True)
#             std = torch.std(x_reshaped, dim=1, keepdim=True) + 1e-5
            
#             # 中心化
#             x_centered = (x_reshaped - mean) / std
            
#             # 仿射变换
#             if self.affine:
#                 x_centered = x_centered * self.gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
            
#             # 重塑回原形状
#             x_centered = x_centered.view(batch_size, num_blocks, block_size, num_features)
#             return x_centered, mean, std
        
#         elif mode == 'decenter':
#             x, mean, std = x
#             batch_size, num_blocks, block_size, num_features = x.shape
            
#             # 重塑
#             x_reshaped = x.view(-1, block_size, num_features)
#             mean_reshaped = mean.view(-1, 1, num_features)
#             std_reshaped = std.view(-1, 1, num_features)
            
#             # 反中心化
#             if self.affine:
#                 x_decentered = (x_reshaped - self.beta.unsqueeze(0).unsqueeze(0)) / self.gamma.unsqueeze(0).unsqueeze(0)
#             else:
#                 x_decentered = x_reshaped
            
#             x_decentered = x_decentered * std_reshaped + mean_reshaped
            
#             # 重塑回原形状
#             x_decentered = x_decentered.view(batch_size, num_blocks, block_size, num_features)
#             return x_decentered


# class FourierBasisExpansion(nn.Module):
#     """
#     傅里叶基函数扩展：FBM-S论文的核心创新
#     实现时间-频域特征构建
#     """
#     def __init__(self, seq_len: int, num_features: int, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.expansion_factor = expansion_factor
#         self.expanded_len = seq_len * expansion_factor
        
#         # 生成傅里叶基函数
#         self.register_buffer('cos_basis', self._generate_cos_basis())
#         self.register_buffer('sin_basis', self._generate_sin_basis())
    
#     def _generate_cos_basis(self):
#         """生成余弦基函数"""
#         freqs = torch.arange(self.expanded_len, dtype=torch.float32)
#         basis = torch.zeros(self.expanded_len, self.expanded_len)
        
#         for k in range(self.expanded_len):
#             basis[k, :] = torch.cos(2 * math.pi * k * freqs / self.expanded_len)
        
#         return basis
    
#     def _generate_sin_basis(self):
#         """生成正弦基函数"""
#         freqs = torch.arange(self.expanded_len, dtype=torch.float32)
#         basis = torch.zeros(self.expanded_len, self.expanded_len)
        
#         for k in range(self.expanded_len):
#             basis[k, :] = torch.sin(2 * math.pi * k * freqs / self.expanded_len)
        
#         return basis
    
#     def forward(self, x):
#         """
#         构建时间-频域特征
#         输入: x [batch, seq_len, num_features]
#         输出: time_freq_features [batch, seq_len, num_features]
#         """
#         batch_size, seq_len, num_features = x.shape
        
#         # 1. 傅里叶变换
#         x_fft = rfft(x, dim=1)  # [batch, seq_len//2 + 1, num_features]
        
#         # 2. 分离实部和虚部
#         H_R = x_fft.real  # 实部
#         H_I = x_fft.imag  # 虚部
        
#         # 3. 扩展傅里叶系数长度
#         H_R_expanded = F.pad(H_R, (0, 0, 0, self.expanded_len - H_R.size(1)), mode='replicate')
#         H_I_expanded = F.pad(H_I, (0, 0, 0, self.expanded_len - H_I.size(1)), mode='replicate')
        
#         # 4. 与基函数相乘（论文公式：时间频域特征 = H_R × cos_basis + H_I × sin_basis）
#         # 重塑为 [batch * num_features, freq_len] 以便进行矩阵乘法
#         H_R_reshaped = H_R_expanded.view(-1, H_R_expanded.size(1))  # [batch * num_features, freq_len]
#         H_I_reshaped = H_I_expanded.view(-1, H_I_expanded.size(1))  # [batch * num_features, freq_len]
        
#         # 选择合适大小的基函数
#         cos_basis = self.cos_basis[:H_R_expanded.size(1), :seq_len]  # [freq_len, seq_len]
#         sin_basis = self.sin_basis[:H_I_expanded.size(1), :seq_len]  # [freq_len, seq_len]
        
#         # 矩阵乘法
#         cos_features = torch.matmul(H_R_reshaped, cos_basis)  # [batch * num_features, seq_len]
#         sin_features = torch.matmul(H_I_reshaped, sin_basis)  # [batch * num_features, seq_len]
        
#         # 5. 组合时间-频域特征
#         time_freq_features = cos_features + sin_features
        
#         # 6. 重塑回 [batch, seq_len, num_features]
#         time_freq_features = time_freq_features.view(batch_size, num_features, seq_len).transpose(1, 2)
        
#         return time_freq_features


# class SeasonalComponent(nn.Module):
#     """
#     季节性组件：严格按照FBM-S论文实现
#     使用滚动窗口在扩展的傅里叶基函数上操作
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64, 
#                  hidden_dim: int = 128, expansion_factor: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.expansion_factor = expansion_factor
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 傅里叶基函数扩展
#         self.fourier_expansion = FourierBasisExpansion(block_size, num_features, expansion_factor)
        
#         # 滚动窗口权重（论文公式5中的W）
#         self.rolling_window = nn.Parameter(torch.randn(block_size, num_features))
        
#         # 季节性网络
#         self.seasonal_net = nn.Sequential(
#             nn.Linear(num_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_features)
#         )
        
#         # 多尺度下采样
#         self.downsample_scales = [1, 2, 4]  # 论文中的d₀, d₁, d₂
        
#         # 频域权重（季节性：中频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))
    
#     def forward(self, x):
#         """
#         严格按照论文公式5实现
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用季节性处理
#         seasonal_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 傅里叶基函数扩展
#             time_freq_block = self.fourier_expansion(block)
            
#             # 5. 应用滚动窗口权重（论文公式5）
#             # W × (扩展的傅里叶基函数)
#             weighted_block = time_freq_block * self.rolling_window.unsqueeze(0)
            
#             # 6. 多尺度下采样
#             multi_scale_features = []
#             for scale in self.downsample_scales:
#                 if scale == 1:
#                     scaled_feature = weighted_block
#                 else:
#                     # 平均下采样：沿时间维度平均
#                     scaled_feature = F.avg_pool1d(
#                         weighted_block.transpose(1, 2), 
#                         kernel_size=scale, 
#                         stride=scale
#                     ).transpose(1, 2)
                
#                 multi_scale_features.append(scaled_feature)
            
#             # 7. 融合多尺度特征
#             if len(multi_scale_features) > 1:
#                 # 上采样到相同大小
#                 target_size = multi_scale_features[0].size(1)
#                 upsampled_features = []
                
#                 for feature in multi_scale_features:
#                     if feature.size(1) != target_size:
#                         feature = F.interpolate(
#                             feature.transpose(1, 2), 
#                             size=target_size, 
#                             mode='linear'
#                         ).transpose(1, 2)
#                     upsampled_features.append(feature)
                
#                 # 加权融合
#                 weights = F.softmax(torch.randn(len(upsampled_features)), dim=0)
#                 fused_feature = sum(w * f for w, f in zip(weights, upsampled_features))
#             else:
#                 fused_feature = multi_scale_features[0]
            
#             # 8. 季节性参数化
#             seasonal_feature = self.seasonal_net(fused_feature)
#             seasonal_features.append(seasonal_feature)
        
#         # 9. 重建序列
#         seasonal_features = torch.stack(seasonal_features, dim=1)
        
#         # 10. 反中心化
#         seasonal_features = self.centering((seasonal_features, mean, std), mode='decenter')
        
#         # 11. 重建原始序列
#         seasonal_features = self.time_blocking.reconstruct(seasonal_features, block_positions)
        
#         # 12. 残差连接
#         if seasonal_features.size() == original_x.size():
#             seasonal_features = seasonal_features + original_x
        
#         return seasonal_features


# class InteractionComponent(nn.Module):
#     """
#     交互组件：严格按照FBM-S论文实现
#     固定掩码 + 中心化 + Transformer
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, num_heads: int = 8):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         # 固定掩码（论文中的C₁和C₂）
#         # C₁=24: 输入掩码，针对短期交互
#         # C₂=48: 输出掩码，针对短期交互
#         self.input_mask = self._create_fixed_mask(24, seq_len)
#         self.output_mask = self._create_fixed_mask(48, seq_len)
        
#         # 掩码多头注意力
#         self.masked_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（交互：高频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))
    
#     def _create_fixed_mask(self, mask_size: int, seq_len: int):
#         """创建固定掩码，基于实际物理意义的时间范围"""
#         # 创建1D掩码，用于序列长度限制
#         mask = torch.ones(seq_len)
        
#         # 应用固定掩码 - 只保留前mask_size个时间步
#         if mask_size < seq_len:
#             mask[mask_size:] = 0
        
#         return mask
    
#     def to(self, device):
#         """确保掩码在正确的设备上"""
#         super().to(device)
#         self.input_mask = self.input_mask.to(device)
#         self.output_mask = self.output_mask.to(device)
#         return self
    
#     def forward(self, x):
#         """
#         严格按照论文实现交互组件
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x
        
#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用交互处理
#         interaction_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 特征投影
#             block_projected = self.feature_projection(block)  # [batch, block_size, hidden_dim]
            
#             # 5. 应用固定掩码 - 确保掩码在正确的设备上
#             device = block_projected.device
#             input_mask = self.input_mask[:block.size(1)].to(device)  # [block_size]
#             output_mask = self.output_mask[:block.size(1)].to(device)  # [block_size]
            
#             # 输入掩码 - 用于序列长度限制
#             # 将掩码扩展到与特征维度匹配，确保广播正确
#             # block_projected: [batch, block_size, hidden_dim]
#             # input_mask: [block_size]
#             # 需要扩展到: [1, block_size, 1] 以便与 [batch, block_size, hidden_dim] 广播
#             input_mask_expanded = input_mask.unsqueeze(0).unsqueeze(-1)  # [1, block_size, 1]
            
#             # 应用掩码到序列维度 - 只保留有效的时间步
#             input_masked = block_projected * input_mask_expanded  # [batch, block_size, hidden_dim]
            
#             # 确保张量维度正确
#             assert input_masked.shape == block_projected.shape, f"Mask application failed: {block_projected.shape} vs {input_masked.shape}"
            
#             # 使用掩码后的张量进行注意力计算
#             input_for_attention = input_masked
            
#             # 6. 掩码多头注意力 - 确保输入是3D张量
#             # 检查并确保张量维度正确
#             if input_for_attention.dim() == 4:
#                 # 如果是4D，压缩最后两个维度
#                 batch_size, seq_len = input_for_attention.shape[:2]
#                 input_for_attention = input_for_attention.view(batch_size, seq_len, -1)
#             elif input_for_attention.dim() != 3:
#                 raise ValueError(f"Expected 3D tensor, got {input_for_attention.dim()}D tensor")
            
#             attn_output, _ = self.masked_attention(
#                 input_for_attention, input_for_attention, input_for_attention
#             )
            
#             # 7. Transformer编码
#             transformer_output = self.transformer(attn_output)
            
#             # 8. 应用输出掩码
#             # output_mask: [block_size]
#             # transformer_output: [batch, block_size, hidden_dim]
#             # 需要扩展到: [1, block_size, 1] 以便广播
#             output_mask_expanded = output_mask.unsqueeze(0).unsqueeze(-1)  # [1, block_size, 1]
#             output_masked = transformer_output * output_mask_expanded
            
#             # 9. 输出投影
#             interaction_feature = self.output_projection(output_masked)
#             interaction_features.append(interaction_feature)
        
#         # 10. 重建序列
#         interaction_features = torch.stack(interaction_features, dim=1)
        
#         # 11. 反中心化
#         interaction_features = self.centering((interaction_features, mean, std), mode='decenter')
        
#         # 12. 重建原始序列
#         interaction_features = self.time_blocking.reconstruct(interaction_features, block_positions)
        
#         # 13. 残差连接
#         if interaction_features.size() == original_x.size():
#             interaction_features = interaction_features + original_x
        
#         return interaction_features


# class TrendComponent(nn.Module):
#     """
#     趋势组件：严格按照FBM-S论文实现
#     时间分块 + MLP/Transformer + 频域权重
#     """
#     def __init__(self, seq_len: int, num_features: int, block_size: int = 64,
#                  hidden_dim: int = 128, use_transformer: bool = False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.block_size = block_size
#         self.hidden_dim = hidden_dim
#         self.use_transformer = use_transformer
        
#         # 时间分块
#         self.time_blocking = TimeBlocking(seq_len, block_size)
        
#         # 中心化层
#         self.centering = CenteringLayer(num_features)
        
#         # 特征投影层
#         self.feature_projection = nn.Linear(num_features, hidden_dim)
        
#         if use_transformer:
#             # Transformer编码器
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=8,
#                 dim_feedforward=hidden_dim * 2,
#                 dropout=0.1,
#                 batch_first=True
#             )
#             self.trend_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         else:
#             # MLP网络
#             self.trend_net = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
        
#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, num_features)
        
#         # 频域权重（趋势：低频分量权重大）
#         self.freq_weights = nn.Parameter(torch.ones(block_size // 2 + 1))

#     def forward(self, x):
#         """
#         严格按照论文实现趋势组件
#         """
#         batch_size, seq_len, num_features = x.shape
#         original_x = x

#         # 1. 时间分块
#         blocks, block_positions = self.time_blocking(x)
        
#         # 2. 中心化
#         blocks_centered, mean, std = self.centering(blocks, mode='center')
        
#         # 3. 对每个块应用趋势处理
#         trend_features = []
        
#         for i in range(blocks_centered.size(1)):
#             block = blocks_centered[:, i, :, :]  # [batch, block_size, num_features]
            
#             # 4. 特征投影
#             block_projected = self.feature_projection(block)
            
#             # 5. 趋势建模
#             if self.use_transformer:
#                 trend_output = self.trend_net(block_projected)
#             else:
#                 trend_output = self.trend_net(block_projected)
            
#             # 6. 输出投影
#             trend_feature = self.output_projection(trend_output)
#             trend_features.append(trend_feature)
        
#         # 7. 重建序列
#         trend_features = torch.stack(trend_features, dim=1)
        
#         # 8. 反中心化
#         trend_features = self.centering((trend_features, mean, std), mode='decenter')
        
#         # 9. 重建原始序列
#         trend_features = self.time_blocking.reconstruct(trend_features, block_positions)
        
#         # 10. 残差连接
#         if trend_features.size() == original_x.size():
#             trend_features = trend_features + original_x
        
#         return trend_features


# # 工厂函数
# def trend_component(seq_len: int, num_features: int, block_size: int = 64,
#                    hidden_dim: int = 128, use_transformer: bool = False) -> nn.Module:
#     """
#     趋势组件工厂函数
#     """
#     return TrendComponent(seq_len, num_features, block_size, hidden_dim, use_transformer)


# def seasonal_component(seq_len: int, num_features: int, block_size: int = 64,
#                       hidden_dim: int = 128, expansion_factor: int = 2) -> nn.Module:
#     """
#     季节性组件工厂函数
#     """
#     return SeasonalComponent(seq_len, num_features, block_size, hidden_dim, expansion_factor)


# def interaction_component(seq_len: int, num_features: int, block_size: int = 64,
#                          hidden_dim: int = 128, num_heads: int = 8) -> nn.Module:
#     """
#     交互组件工厂函数
#     """
#     return InteractionComponent(seq_len, num_features, block_size, hidden_dim, num_heads)
