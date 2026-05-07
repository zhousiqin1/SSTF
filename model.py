import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math
import numpy as np
from fbm_paper_components import trend_component, seasonal_component, interaction_component


# 完整FBM分支
class HybridFBM_LSTM(nn.Module):
    """
    仅使用FBM分支的模型（兼容train.py）：
    在HybridFBM_LSTM_CNN_2D的基础上去掉了CNN和文本分支，保留FBM分支、注意力机制和LSTM模块。
    """

    def __init__(self, raw_input_size, lstm_hidden, mlp_dim, output_size,
                 seq_len, dropout, fbm_block_size, fbm_hidden_dim, attention_heads):
        super(HybridFBM_LSTM, self).__init__()

        self.raw_input_size = raw_input_size
        self.seq_len = seq_len

        # ===== FBM 主干 =====
        from fbm_paper_components import trend_component, seasonal_component, interaction_component
        self.fbm_trend = trend_component(
            seq_len, raw_input_size, block_size=fbm_block_size,
            use_transformer=True, hidden_dim=fbm_hidden_dim
        )
        self.fbm_seasonal = seasonal_component(
            seq_len, raw_input_size, block_size=fbm_block_size, hidden_dim=fbm_hidden_dim
        )
        self.fbm_interaction = interaction_component(
            seq_len, raw_input_size, block_size=fbm_block_size, hidden_dim=fbm_hidden_dim * 2
        )

        # ===== 特征学习层（简单 MLP） =====
        self.deep_feature_learning = nn.Sequential(
            nn.Linear(raw_input_size * 3, raw_input_size * 6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(raw_input_size * 6, raw_input_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(raw_input_size * 4, raw_input_size * 4),
            nn.LayerNorm(raw_input_size * 4)
        )

        # ===== 时序注意力机制 =====
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=raw_input_size * 4,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # ===== LSTM 时序建模 =====
        self.lstm = nn.LSTM(raw_input_size * 4, lstm_hidden, batch_first=True)

        # ===== 分类头 =====
        self.fusion_norm = nn.LayerNorm(lstm_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, raw_sensor_data, text_feat=None, return_x=False):
        """
        Forward 函数：
        Args:
            raw_sensor_data: [B, T, C] 传感器时序数据
            text_feat: 占位参数，为兼容 train.py，可忽略
            return_x: 是否返回中间特征（bool）
        """

        # ===== FBM 主干特征提取 =====
        fbm_trend = self.fbm_trend(raw_sensor_data)
        fbm_seasonal = self.fbm_seasonal(raw_sensor_data)
        fbm_interaction = self.fbm_interaction(raw_sensor_data)
        multi_scale_features = torch.cat(
            [fbm_trend, fbm_seasonal, fbm_interaction], dim=-1
        )  # [B, T, C*3]

        # ===== 深度特征学习 =====
        deep_features = self.deep_feature_learning(multi_scale_features)
        deep_features = self.dropout(deep_features)

        # ===== 时序注意力机制 =====
        attended_features, _ = self.temporal_attention(
            deep_features, deep_features, deep_features
        )
        attended_features = self.dropout(attended_features)

        # ===== LSTM 时序建模 =====
        lstm_out, _ = self.lstm(attended_features)
        fbm_vec = lstm_out[:, -1, :]  # [B, lstm_hidden]

        # ===== 分类层 =====
        fused = self.fusion_norm(fbm_vec)
        fused = self.dropout(fused)
        output = self.classifier(fused)

        # ===== 输出 =====
        if isinstance(return_x, bool) and return_x:
            return fused, output
        else:
            return output


# # 复杂组件  效果好
class HybridFBM_LSTM_CNN_2D_Dynamic(nn.Module):
    """
    动态权重融合的双模态模型（无文本特征）：
    在 HybridFBM_LSTM_CNN_2D 基础上，将固定权重替换为基于内容的动态权重
    
    创新点：
    1. 动态权重计算：基于FBM和CNN特征内容自适应计算融合权重
    2. 内容感知融合：权重随输入特征动态变化
    3. 双模态权重：FBM和CNN之间的智能权重分配
    """
    def __init__(self, raw_input_size, lstm_hidden, mlp_dim, output_size,
                 seq_len, dropout, 
                 fbm_block_size, fbm_hidden_dim, attention_heads, 
                 cnn_hidden_dim, conv_time_kernel):
        super(HybridFBM_LSTM_CNN_2D_Dynamic, self).__init__()

        self.raw_input_size = raw_input_size
        self.seq_len = seq_len

        # ===== FBM 主干（与原始模型一致） =====
        from fbm_paper_components import trend_component, seasonal_component, interaction_component
        self.fbm_trend = trend_component(seq_len, raw_input_size, block_size=fbm_block_size, use_transformer=True, hidden_dim=fbm_hidden_dim)
        self.fbm_seasonal = seasonal_component(seq_len, raw_input_size, block_size=fbm_block_size, hidden_dim=fbm_hidden_dim)
        self.fbm_interaction = interaction_component(seq_len, raw_input_size, block_size=fbm_block_size, hidden_dim=fbm_hidden_dim * 2)

        self.deep_feature_learning = nn.Sequential(
            nn.Linear(raw_input_size * 3, raw_input_size * 6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(raw_input_size * 6, raw_input_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(raw_input_size * 4, raw_input_size * 4),
            nn.LayerNorm(raw_input_size * 4)
        )

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=raw_input_size * 4,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.lstm = nn.LSTM(raw_input_size * 4, lstm_hidden, batch_first=True)

        # ===== Conv2D 空间分支（与原始模型一致） =====
        mid_channels = 64
        k_t = conv_time_kernel
        self.spatial_conv1 = nn.Conv2d(1, mid_channels, kernel_size=(3, k_t), padding=(0, k_t // 2))
        self.spatial_bn1 = nn.BatchNorm2d(mid_channels)
        self.spatial_act1 = nn.ReLU(inplace=True)
        self.spatial_conv2 = nn.Conv2d(mid_channels, cnn_hidden_dim, kernel_size=(1, 1))
        self.spatial_bn2 = nn.BatchNorm2d(cnn_hidden_dim)
        self.spatial_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_vec_dropout = nn.Dropout(p=max(0.0, min(0.5, dropout * 0.5)))

        # ===== 动态权重融合网络（双模态版本） =====
        fusion_dim = lstm_hidden + cnn_hidden_dim
        
        # 权重计算网络：基于FBM和CNN特征内容计算动态权重
        self.weight_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 2),  # 输出2个权重：FBM, CNN
            nn.Softmax(dim=-1)
        )
        
        # 内容感知权重网络：基于每个模态的特征质量计算权重
        self.content_aware_weights = nn.ModuleDict({
            'fbm_quality': nn.Sequential(
                nn.Linear(lstm_hidden, lstm_hidden // 2),
                nn.ReLU(),
                nn.Linear(lstm_hidden // 2, 1),
                nn.Sigmoid()
            ),
            'cnn_quality': nn.Sequential(
                nn.Linear(cnn_hidden_dim, cnn_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(cnn_hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        })
        
        # 多尺度权重融合：结合全局和局部权重
        self.multi_scale_weight_fusion = nn.Sequential(
            nn.Linear(4, 8),  # 2个全局权重 + 2个局部权重
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(8, 2),
            nn.Softmax(dim=-1)
        )

        # ===== 双路融合与分类 =====
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_size)
        )
        self.dropout = nn.Dropout(dropout)

    def compute_dynamic_weights(self, fbm_vec, cnn_vec):
        """
        计算动态融合权重（双模态版本）
        
        Args:
            fbm_vec: FBM+LSTM特征 [B, lstm_hidden]
            cnn_vec: CNN特征 [B, cnn_hidden_dim]
        
        Returns:
            weights: 动态权重 [B, 2] (FBM, CNN)
        """
        # 1. 全局权重：基于所有特征的联合信息
        all_features = torch.cat([fbm_vec, cnn_vec], dim=-1)
        global_weights = self.weight_net(all_features)  # [B, 2]
        
        # 2. 局部权重：基于每个模态的特征质量
        fbm_quality = self.content_aware_weights['fbm_quality'](fbm_vec)  # [B, 1]
        cnn_quality = self.content_aware_weights['cnn_quality'](cnn_vec)  # [B, 1]
        
        local_weights = torch.cat([fbm_quality, cnn_quality], dim=-1)  # [B, 2]
        local_weights = torch.softmax(local_weights, dim=-1)
        
        # 3. 多尺度融合：结合全局和局部权重
        combined_weights = torch.cat([global_weights, local_weights], dim=-1)  # [B, 4]
        final_weights = self.multi_scale_weight_fusion(combined_weights)  # [B, 2]
        
        return final_weights

    def forward(self, raw_sensor_data, text_feat=None, return_x=False):
        # 忽略text_feat参数，保持接口兼容性（此模型不使用文本特征）
        
        # ===== FBM 主干 =====
        fbm_trend = self.fbm_trend(raw_sensor_data)
        fbm_seasonal = self.fbm_seasonal(raw_sensor_data)
        fbm_interaction = self.fbm_interaction(raw_sensor_data)
        multi_scale_features = torch.cat([fbm_trend, fbm_seasonal, fbm_interaction], dim=-1)
        deep_features = self.deep_feature_learning(multi_scale_features)
        deep_features = self.dropout(deep_features)
        attended_features, _ = self.temporal_attention(deep_features, deep_features, deep_features)
        attended_features = self.dropout(attended_features)
        lstm_out, _ = self.lstm(attended_features)
        fbm_vec = lstm_out[:, -1, :]  # [B, lstm_hidden]

        # ===== CNN 空间分支 =====
        x2d = raw_sensor_data.transpose(1, 2).unsqueeze(1)   # [B,1,3,T]
        h = self.spatial_act1(self.spatial_bn1(self.spatial_conv1(x2d)))        # [B,Cm,1,T]
        h = self.spatial_bn2(self.spatial_conv2(h))                             # [B,d,1,T]
        h = self.spatial_gap(h)                               # [B,d,1,1]
        cnn_vec = h.squeeze(-1).squeeze(-1)                   # [B,d]
        cnn_vec = self.spatial_vec_dropout(cnn_vec)

        # ===== 动态权重融合 =====
        # 计算动态权重
        weights = self.compute_dynamic_weights(fbm_vec, cnn_vec)  # [B, 2]
        fbm_weight = weights[:, 0:1]    # [B, 1]
        cnn_weight = weights[:, 1:2]    # [B, 1]
        
        # 应用动态权重
        fbm_vec_weighted = fbm_weight * fbm_vec
        cnn_vec_weighted = cnn_weight * cnn_vec
        
        # 双路拼接
        fused = torch.cat([fbm_vec_weighted, cnn_vec_weighted], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        output = self.classifier(fused)

        if return_x:
            return fused, output
        else:
            return output


# 复杂版本  
class HybridFBM_LSTM_CNN_2D_Text_Dynamic(nn.Module):
    """
    动态权重融合的多模态模型：
    在 HybridFBM_LSTM_CNN_2D_Text 基础上，将固定权重替换为基于内容的动态权重
    
    创新点：
    1. 动态权重计算：基于特征内容自适应计算融合权重
    2. 内容感知融合：权重随输入特征动态变化
    3. 多尺度权重：不同层次的特征使用不同的权重策略
    """
    def __init__(self, raw_input_size, lstm_hidden, mlp_dim, output_size,
                 text_feat_dim, seq_len, dropout, 
                 fbm_block_size, fbm_hidden_dim, attention_heads, 
                 cnn_hidden_dim, conv_time_kernel):
        super(HybridFBM_LSTM_CNN_2D_Text_Dynamic, self).__init__()

        self.raw_input_size = raw_input_size
        self.seq_len = seq_len
        self.text_feat_dim = text_feat_dim

        # ===== FBM 主干（与原始模型一致） =====
        from fbm_paper_components import trend_component, seasonal_component, interaction_component
        self.fbm_trend = trend_component(seq_len, raw_input_size, block_size=fbm_block_size, use_transformer=True, hidden_dim=fbm_hidden_dim)
        self.fbm_seasonal = seasonal_component(seq_len, raw_input_size, block_size=fbm_block_size, hidden_dim=fbm_hidden_dim)
        self.fbm_interaction = interaction_component(seq_len, raw_input_size, block_size=fbm_block_size, hidden_dim=fbm_hidden_dim * 2)

        self.deep_feature_learning = nn.Sequential(
            nn.Linear(raw_input_size * 3, raw_input_size * 6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(raw_input_size * 6, raw_input_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(raw_input_size * 4, raw_input_size * 4),
            nn.LayerNorm(raw_input_size * 4)
        )

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=raw_input_size * 4,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.lstm = nn.LSTM(raw_input_size * 4, lstm_hidden, batch_first=True)

        # ===== Conv2D 空间分支（与原始模型一致） =====
        mid_channels = 64
        k_t = conv_time_kernel
        self.spatial_conv1 = nn.Conv2d(1, mid_channels, kernel_size=(3, k_t), padding=(0, k_t // 2))
        self.spatial_bn1 = nn.BatchNorm2d(mid_channels)
        self.spatial_act1 = nn.ReLU(inplace=True)
        self.spatial_conv2 = nn.Conv2d(mid_channels, cnn_hidden_dim, kernel_size=(1, 1))
        self.spatial_bn2 = nn.BatchNorm2d(cnn_hidden_dim)
        self.spatial_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_vec_dropout = nn.Dropout(p=max(0.0, min(0.5, dropout * 0.5)))

        # ===== 文本特征处理（与原始模型一致） =====
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, text_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(text_feat_dim)
        )

        # ===== 动态权重融合网络 =====
        fusion_dim = lstm_hidden + cnn_hidden_dim + text_feat_dim
        
        # 权重计算网络：基于特征内容计算动态权重
        self.weight_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 3),  # 输出3个权重：FBM, CNN, Text
            nn.Softmax(dim=-1)
        )
        
        # 内容感知权重网络：基于每个模态的特征质量计算权重
        self.content_aware_weights = nn.ModuleDict({
            'fbm_quality': nn.Sequential(
                nn.Linear(lstm_hidden, lstm_hidden // 2),
                nn.ReLU(),
                nn.Linear(lstm_hidden // 2, 1),
                nn.Sigmoid()
            ),
            'cnn_quality': nn.Sequential(
                nn.Linear(cnn_hidden_dim, cnn_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(cnn_hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'text_quality': nn.Sequential(
                nn.Linear(text_feat_dim, text_feat_dim // 2),
                nn.ReLU(),
                nn.Linear(text_feat_dim // 2, 1),
                nn.Sigmoid()
            )
        })
        
        # 多尺度权重融合：结合全局和局部权重
        self.multi_scale_weight_fusion = nn.Sequential(
            nn.Linear(6, 12),  # 3个全局权重 + 3个局部权重
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(12, 3),
            nn.Softmax(dim=-1)
        )

        # ===== 三路融合与分类 =====
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_size)
        )
        self.dropout = nn.Dropout(dropout)

    def compute_dynamic_weights(self, fbm_vec, cnn_vec, text_vec):
        """
        计算动态融合权重
        
        Args:
            fbm_vec: FBM+LSTM特征 [B, lstm_hidden]
            cnn_vec: CNN特征 [B, cnn_hidden_dim]
            text_vec: 文本特征 [B, text_feat_dim]
        
        Returns:
            weights: 动态权重 [B, 3] (FBM, CNN, Text)
        """
        # 1. 全局权重：基于所有特征的联合信息
        all_features = torch.cat([fbm_vec, cnn_vec, text_vec], dim=-1)
        global_weights = self.weight_net(all_features)  # [B, 3]
        
        # 2. 局部权重：基于每个模态的特征质量
        fbm_quality = self.content_aware_weights['fbm_quality'](fbm_vec)  # [B, 1]
        cnn_quality = self.content_aware_weights['cnn_quality'](cnn_vec)  # [B, 1]
        text_quality = self.content_aware_weights['text_quality'](text_vec)  # [B, 1]
        
        local_weights = torch.cat([fbm_quality, cnn_quality, text_quality], dim=-1)  # [B, 3]
        local_weights = torch.softmax(local_weights, dim=-1)
        
        # 3. 多尺度融合：结合全局和局部权重
        combined_weights = torch.cat([global_weights, local_weights], dim=-1)  # [B, 6]
        final_weights = self.multi_scale_weight_fusion(combined_weights)  # [B, 3]
        
        return final_weights

    def forward(self, raw_sensor_data, text_feat=None, return_x=False):
        # ===== FBM 主干 =====
        fbm_trend = self.fbm_trend(raw_sensor_data)
        fbm_seasonal = self.fbm_seasonal(raw_sensor_data)
        fbm_interaction = self.fbm_interaction(raw_sensor_data)
        multi_scale_features = torch.cat([fbm_trend, fbm_seasonal, fbm_interaction], dim=-1)
        deep_features = self.deep_feature_learning(multi_scale_features)
        deep_features = self.dropout(deep_features)
        attended_features, _ = self.temporal_attention(deep_features, deep_features, deep_features)
        attended_features = self.dropout(attended_features)
        lstm_out, _ = self.lstm(attended_features)
        fbm_vec = lstm_out[:, -1, :]  # [B, lstm_hidden]

        # ===== CNN 空间分支 =====
        x2d = raw_sensor_data.transpose(1, 2).unsqueeze(1)   # [B,1,3,T]
        h = self.spatial_act1(self.spatial_bn1(self.spatial_conv1(x2d)))        # [B,Cm,1,T]
        h = self.spatial_bn2(self.spatial_conv2(h))                             # [B,d,1,T]
        h = self.spatial_gap(h)                               # [B,d,1,1]
        cnn_vec = h.squeeze(-1).squeeze(-1)                   # [B,d]
        cnn_vec = self.spatial_vec_dropout(cnn_vec)

        # ===== 文本特征处理 =====
        if text_feat is not None:
            text_vec = self.text_projection(text_feat)  # [B, text_feat_dim]
        else:
            text_vec = torch.zeros(raw_sensor_data.size(0), self.text_feat_dim, 
                                 device=raw_sensor_data.device, dtype=raw_sensor_data.dtype)

        # ===== 动态权重融合 =====
        # 计算动态权重
        weights = self.compute_dynamic_weights(fbm_vec, cnn_vec, text_vec)  # [B, 3]
        fbm_weight = weights[:, 0:1]    # [B, 1]
        cnn_weight = weights[:, 1:2]    # [B, 1]
        text_weight = weights[:, 2:3]   # [B, 1]
        
        # 应用动态权重
        fbm_vec_weighted = fbm_weight * fbm_vec
        cnn_vec_weighted = cnn_weight * cnn_vec
        text_vec_weighted = text_weight * text_vec
        
        # 三路拼接
        fused = torch.cat([fbm_vec_weighted, cnn_vec_weighted, text_vec_weighted], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        output = self.classifier(fused)

        if return_x:
            return fused, output
        else:
            return output



# 强
class HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive(nn.Module):
    """
    动态权重融合 + 对比学习（增强版）
    - 共享可学习的传感器联合投影头（替代简单平均）
    - 三重对齐目标：Sensor↔Text、FBM↔CNN、Supervised Contrastive
    - 保持与原有代码完全兼容的 forward 返回接口
    """

    def __init__(
        self,
        raw_input_size,
        lstm_hidden,
        mlp_dim,
        output_size,
        text_feat_dim,
        seq_len,
        dropout,
        fbm_block_size,
        fbm_hidden_dim,
        attention_heads,
        cnn_hidden_dim,
        conv_time_kernel,
        projection_dim=128,
        temperature=0.07,
    ):
        super().__init__()

        # ---- 基本属性 ----
        self.raw_input_size = raw_input_size
        self.seq_len = seq_len
        self.text_feat_dim = text_feat_dim
        self.lstm_hidden = lstm_hidden
        self.cnn_hidden_dim = cnn_hidden_dim
        self.projection_dim = projection_dim
        self.temperature = float(temperature)
        self.contrastive_weight = 0.3

        # ---- FBM 三组件 ----
        from fbm_paper_components import trend_component, seasonal_component, interaction_component
        self.fbm_trend = trend_component(seq_len, raw_input_size, block_size=fbm_block_size,
                                         use_transformer=True, hidden_dim=fbm_hidden_dim)
        self.fbm_seasonal = seasonal_component(seq_len, raw_input_size, block_size=fbm_block_size,
                                               hidden_dim=fbm_hidden_dim)
        self.fbm_interaction = interaction_component(seq_len, raw_input_size, block_size=fbm_block_size,
                                                     hidden_dim=fbm_hidden_dim * 2)

        # ---- FBM 后续深特征（时序）----
        self.deep_feature_learning = nn.Sequential(
            nn.Linear(raw_input_size * 3, raw_input_size * 6),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(raw_input_size * 6, raw_input_size * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(raw_input_size * 4, raw_input_size * 4),
            nn.LayerNorm(raw_input_size * 4),
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=raw_input_size * 4,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.lstm = nn.LSTM(raw_input_size * 4, lstm_hidden, batch_first=True)

        # ---- CNN 空间分支 ----
        mid_channels = 64
        k_t = conv_time_kernel
        self.spatial_conv1 = nn.Conv2d(1, mid_channels, kernel_size=(3, k_t), padding=(0, k_t // 2))
        self.spatial_bn1 = nn.BatchNorm2d(mid_channels)
        self.spatial_act1 = nn.ReLU(inplace=True)
        self.spatial_conv2 = nn.Conv2d(mid_channels, cnn_hidden_dim, kernel_size=(1, 1))
        self.spatial_bn2 = nn.BatchNorm2d(cnn_hidden_dim)
        self.spatial_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_vec_dropout = nn.Dropout(p=max(0.0, min(0.5, dropout * 0.5)))

        # ---- 文本特征处理 ----
        self.text_feature_projection = nn.Sequential(
            nn.Linear(text_feat_dim, text_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(text_feat_dim),
        )

        # ---- 单模态投影头（先各自到 proj 维度）----
        self.fbm_projection = nn.Sequential(
            nn.Linear(lstm_hidden, projection_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(projection_dim),
        )
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_hidden_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(projection_dim),
        )

        # ---- 传感器联合可学习投影头（替代简单平均）----
        self.sensor_joint_proj = nn.Sequential(
            nn.Linear(2 * projection_dim, 2 * projection_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(2 * projection_dim),
            nn.Linear(2 * projection_dim, projection_dim),
        )

        # ---- 动态权重融合 ----
        fusion_dim = lstm_hidden + cnn_hidden_dim + text_feat_dim
        self.weight_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 3),
            nn.Softmax(dim=-1),
        )
        self.content_aware_weights = nn.ModuleDict({
            "fbm_quality": nn.Sequential(
                nn.Linear(lstm_hidden, lstm_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(lstm_hidden // 2, 1),
                nn.Sigmoid(),
            ),
            "cnn_quality": nn.Sequential(
                nn.Linear(cnn_hidden_dim, cnn_hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(cnn_hidden_dim // 2, 1),
                nn.Sigmoid(),
            ),
            "text_quality": nn.Sequential(
                nn.Linear(text_feat_dim, text_feat_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(text_feat_dim // 2, 1),
                nn.Sigmoid(),
            ),
        })
        self.multi_scale_weight_fusion = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(12, 3),
            nn.Softmax(dim=-1),
        )

        # ---- 分类头 ----
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_size),
        )
        self.dropout = nn.Dropout(dropout)

    # ---------------- 动态权重 ---------------- #
    def compute_dynamic_weights(self, fbm_vec, cnn_vec, text_vec):
        all_features = torch.cat([fbm_vec, cnn_vec, text_vec], dim=-1)
        global_weights = self.weight_net(all_features)

        fbm_quality = self.content_aware_weights["fbm_quality"](fbm_vec)
        cnn_quality = self.content_aware_weights["cnn_quality"](cnn_vec)
        text_quality = self.content_aware_weights["text_quality"](text_vec)

        local_weights = torch.cat([fbm_quality, cnn_quality, text_quality], dim=-1)
        local_weights = torch.softmax(local_weights, dim=-1)

        combined = torch.cat([global_weights, local_weights], dim=-1)  # [B, 6]
        final_weights = self.multi_scale_weight_fusion(combined)       # [B, 3]
        return final_weights

    # ---------------- Contrastive: InfoNCE（双向） ---------------- #
    def _infonce(self, q, k, temperature=0.07):
        # q,k 已 L2 normalize
        logits = torch.matmul(q, k.T) / temperature  # [B,B]
        labels = torch.arange(q.size(0), device=q.device)
        li = F.cross_entropy(logits, labels)
        lt = F.cross_entropy(logits.T, labels)
        return 0.5 * (li + lt)

    # ---------------- Contrastive: Supervised（同类聚合） ---------------- #
    def _sup_contrastive(self, z, labels, temperature=0.07):
        # z 已 L2 normalize
        sim = torch.matmul(z, z.T) / temperature             # [B,B]
        with torch.no_grad():
            pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
            eye = torch.eye(z.size(0), device=z.device)
            pos_mask = pos_mask * (1.0 - eye)               # 自身不算正样本
            logits_mask = 1.0 - eye

        sim_exp = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss

    # ---------------- 前向传播 ---------------- #
    def forward(
        self,
        raw_sensor_data,
        text_feat=None,
        batch_y=None,
        return_x=False,
        contrastive=True,
    ):
        """
        return_x:
          - False: 仅返回分类输出（训练/推理）
          - True:  返回 (fbm_vec, cnn_vec, text_vec, fbm_proj, cnn_proj, text_proj, output, contrastive_loss_val)
          - "fused": 返回融合后的特征向量（不经过分类器）
        """

        # ===== FBM 主干 =====
        fbm_trend = self.fbm_trend(raw_sensor_data)
        fbm_seasonal = self.fbm_seasonal(raw_sensor_data)
        fbm_interaction = self.fbm_interaction(raw_sensor_data)
        multi_scale = torch.cat([fbm_trend, fbm_seasonal, fbm_interaction], dim=-1)

        deep_feat = self.dropout(self.deep_feature_learning(multi_scale))
        attn_out, _ = self.temporal_attention(deep_feat, deep_feat, deep_feat)
        lstm_out, _ = self.lstm(self.dropout(attn_out))
        fbm_vec = lstm_out[:, -1, :]  # [B, lstm_hidden]

        # ===== CNN 空间分支 =====
        x2d = raw_sensor_data.transpose(1, 2).unsqueeze(1)  # [B,1,C,T]
        h = self.spatial_act1(self.spatial_bn1(self.spatial_conv1(x2d)))
        h = self.spatial_bn2(self.spatial_conv2(h))
        cnn_vec = self.spatial_gap(h).squeeze(-1).squeeze(-1)  # [B, cnn_hidden_dim]
        cnn_vec = self.spatial_vec_dropout(cnn_vec)

        # ===== 文本分支 =====
        if text_feat is not None:
            text_vec = self.text_feature_projection(text_feat)
        else:
            text_vec = torch.zeros(
                raw_sensor_data.size(0),
                self.text_feat_dim,
                device=raw_sensor_data.device,
                dtype=raw_sensor_data.dtype,
            )

        # ===== 动态权重融合（原始特征级）=====
        weights = self.compute_dynamic_weights(fbm_vec, cnn_vec, text_vec)  # [B,3]
        fbm_vec_w = weights[:, 0:1] * fbm_vec
        cnn_vec_w = weights[:, 1:2] * cnn_vec
        text_vec_w = weights[:, 2:3] * text_vec

        fused = torch.cat([fbm_vec_w, cnn_vec_w, text_vec_w], dim=-1)  # [B, fusion_dim]
        fused = self.dropout(self.fusion_norm(fused))
        output = self.classifier(fused)

        # ===== 仅在对比学习时计算投影（contrastive=True） =====
        if contrastive:
            # 投影计算（L2 normalization）
            fbm_proj = self.fbm_projection(fbm_vec)     # [B, P]
            cnn_proj = self.cnn_projection(cnn_vec)     # [B, P]
            text_proj = self.text_projection(text_vec)  # [B, P]

            # 联合传感器投影
            sensor_cat = torch.cat([fbm_proj, cnn_proj], dim=-1)  # [B, 2P]
            sensor_proj = self.sensor_joint_proj(sensor_cat)      # [B, P]

            # L2 normalization
            fbm_proj = F.normalize(fbm_proj, dim=-1)
            cnn_proj = F.normalize(cnn_proj, dim=-1)
            sensor_proj = F.normalize(sensor_proj, dim=-1)
            text_proj = F.normalize(text_proj, dim=-1)

            contrastive_loss_val = None
            if batch_y is not None:
                # 计算对比学习损失
                L_st = self._infonce(sensor_proj, text_proj, temperature=self.temperature)  # 传感器↔文本
                L_sc = self._infonce(fbm_proj, cnn_proj, temperature=self.temperature)      # 传感器内部对齐
                L_sup = self._sup_contrastive(sensor_proj, batch_y, temperature=self.temperature)  # 监督对比
                contrastive_loss_val = L_st + 0.5 * L_sc + 0.5 * L_sup
        else:
            # contrastive=False，不计算投影和对比损失
            if batch_y is not None:  
                sensor_proj = None
                fbm_proj = cnn_proj = text_proj = None
                contrastive_loss_val = torch.tensor(0.0, device=fused.device)
            else:
                # 测试：不算 loss，但需要 projection（用于画图）
                fbm_proj = F.normalize(self.fbm_projection(fbm_vec), dim=-1)
                cnn_proj = F.normalize(self.cnn_projection(cnn_vec), dim=-1)
                text_proj = F.normalize(self.text_projection(text_vec), dim=-1)
                sensor_cat = torch.cat([fbm_proj, cnn_proj], dim=-1)
                sensor_proj = F.normalize(self.sensor_joint_proj(sensor_cat), dim=-1)
                contrastive_loss_val = torch.tensor(0.0, device=fused.device)

        # ===== 返回接口兼容 =====
        if return_x == "fused":
            return fused
        elif return_x:
            return (
                fbm_vec, cnn_vec, text_vec, fbm_proj, cnn_proj, text_proj, output, contrastive_loss_val)
        else:
            # 训练：contrastive=True → 返回 output + loss
            if contrastive:
                return output, contrastive_loss_val
            # 验证：contrastive=False → 只返回 output
            else:
                return output

