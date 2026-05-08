# model.py
# SSTF model variants: TFM‑only baseline, TFM+LST dynamic fusion, and the full
# TFM+LST+TXT contrastive model (SSTF).

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math
import numpy as np
from fbm_paper_components import trend_component, seasonal_component, interaction_component


class HybridFBM_LSTM(nn.Module):


    def __init__(self, raw_input_size, lstm_hidden, mlp_dim, output_size,
                 seq_len, dropout, fbm_block_size, fbm_hidden_dim, attention_heads):
        super(HybridFBM_LSTM, self).__init__()

        self.raw_input_size = raw_input_size
        self.seq_len = seq_len

        # ===== TFM backbone =====
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

        # ===== Deep feature learning (MLP) =====
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

        # ===== Multi‑head temporal attention =====
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=raw_input_size * 4,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # ===== LSTM temporal modelling =====
        self.lstm = nn.LSTM(raw_input_size * 4, lstm_hidden, batch_first=True)

        # ===== Classification head =====
        self.fusion_norm = nn.LayerNorm(lstm_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, raw_sensor_data, text_feat=None, return_x=False):

        # ===== TFM forward =====
        fbm_trend = self.fbm_trend(raw_sensor_data)
        fbm_seasonal = self.fbm_seasonal(raw_sensor_data)
        fbm_interaction = self.fbm_interaction(raw_sensor_data)
        multi_scale_features = torch.cat(
            [fbm_trend, fbm_seasonal, fbm_interaction], dim=-1
        )  # [B, T, C*3]

        # ===== Deep feature learning =====
        deep_features = self.deep_feature_learning(multi_scale_features)
        deep_features = self.dropout(deep_features)

        # ===== Temporal attention =====
        attended_features, _ = self.temporal_attention(
            deep_features, deep_features, deep_features
        )
        attended_features = self.dropout(attended_features)

        # ===== LSTM =====
        lstm_out, _ = self.lstm(attended_features)
        fbm_vec = lstm_out[:, -1, :]  # [B, lstm_hidden]

        # ===== Classification =====
        fused = self.fusion_norm(fbm_vec)
        fused = self.dropout(fused)
        output = self.classifier(fused)

        if isinstance(return_x, bool) and return_x:
            return fused, output
        else:
            return output



class HybridFBM_LSTM_CNN_2D_Dynamic(nn.Module):


    def __init__(self, raw_input_size, lstm_hidden, mlp_dim, output_size,
                 seq_len, dropout, 
                 fbm_block_size, fbm_hidden_dim, attention_heads, 
                 cnn_hidden_dim, conv_time_kernel):
        super(HybridFBM_LSTM_CNN_2D_Dynamic, self).__init__()

        self.raw_input_size = raw_input_size
        self.seq_len = seq_len

        # ===== TFM backbone =====
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

        # ===== LST stream (2‑D CNN) =====
        mid_channels = 64
        k_t = conv_time_kernel
        self.spatial_conv1 = nn.Conv2d(1, mid_channels, kernel_size=(3, k_t), padding=(0, k_t // 2))
        self.spatial_bn1 = nn.BatchNorm2d(mid_channels)
        self.spatial_act1 = nn.ReLU(inplace=True)
        self.spatial_conv2 = nn.Conv2d(mid_channels, cnn_hidden_dim, kernel_size=(1, 1))
        self.spatial_bn2 = nn.BatchNorm2d(cnn_hidden_dim)
        self.spatial_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_vec_dropout = nn.Dropout(p=max(0.0, min(0.5, dropout * 0.5)))

        # ===== CADF dynamic‑weight network (two‑modality) =====
        fusion_dim = lstm_hidden + cnn_hidden_dim
        
        self.weight_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 2),  # TFM, CNN
            nn.Softmax(dim=-1)
        )
        
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
        
        self.multi_scale_weight_fusion = nn.Sequential(
            nn.Linear(4, 8),  # 2 global + 2 local
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(8, 2),
            nn.Softmax(dim=-1)
        )

        # ===== Fusion & classification =====
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_size)
        )
        self.dropout = nn.Dropout(dropout)

    def compute_dynamic_weights(self, fbm_vec, cnn_vec):

        # (1) Global importance
        all_features = torch.cat([fbm_vec, cnn_vec], dim=-1)
        global_weights = self.weight_net(all_features)  # [B, 2]
        
        # (2) Local quality estimator
        fbm_quality = self.content_aware_weights['fbm_quality'](fbm_vec)  # [B, 1]
        cnn_quality = self.content_aware_weights['cnn_quality'](cnn_vec)  # [B, 1]
        
        local_weights = torch.cat([fbm_quality, cnn_quality], dim=-1)  # [B, 2]
        local_weights = torch.softmax(local_weights, dim=-1)
        
        # (3) Multi‑scale aggregation
        combined_weights = torch.cat([global_weights, local_weights], dim=-1)  # [B, 4]
        final_weights = self.multi_scale_weight_fusion(combined_weights)  # [B, 2]
        
        return final_weights

    def forward(self, raw_sensor_data, text_feat=None, return_x=False):

        
        # ===== TFM =====
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

        # ===== LST (2‑D CNN) =====
        x2d = raw_sensor_data.transpose(1, 2).unsqueeze(1)   # [B,1,3,T]
        h = self.spatial_act1(self.spatial_bn1(self.spatial_conv1(x2d)))        # [B,Cm,1,T]
        h = self.spatial_bn2(self.spatial_conv2(h))                             # [B,d,1,T]
        h = self.spatial_gap(h)                               # [B,d,1,1]
        cnn_vec = h.squeeze(-1).squeeze(-1)                   # [B,d]
        cnn_vec = self.spatial_vec_dropout(cnn_vec)

        # ===== CADF dynamic fusion =====
        weights = self.compute_dynamic_weights(fbm_vec, cnn_vec)  # [B, 2]
        fbm_weight = weights[:, 0:1]    # [B, 1]
        cnn_weight = weights[:, 1:2]    # [B, 1]
        
        fbm_vec_weighted = fbm_weight * fbm_vec
        cnn_vec_weighted = cnn_weight * cnn_vec
        
        fused = torch.cat([fbm_vec_weighted, cnn_vec_weighted], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        output = self.classifier(fused)

        if return_x:
            return fused, output
        else:
            return output



class HybridFBM_LSTM_CNN_2D_Text_Dynamic(nn.Module):


    def __init__(self, raw_input_size, lstm_hidden, mlp_dim, output_size,
                 text_feat_dim, seq_len, dropout, 
                 fbm_block_size, fbm_hidden_dim, attention_heads, 
                 cnn_hidden_dim, conv_time_kernel):
        super(HybridFBM_LSTM_CNN_2D_Text_Dynamic, self).__init__()

        self.raw_input_size = raw_input_size
        self.seq_len = seq_len
        self.text_feat_dim = text_feat_dim

        # ===== TFM backbone =====
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

        # ===== LST stream (2‑D CNN) =====
        mid_channels = 64
        k_t = conv_time_kernel
        self.spatial_conv1 = nn.Conv2d(1, mid_channels, kernel_size=(3, k_t), padding=(0, k_t // 2))
        self.spatial_bn1 = nn.BatchNorm2d(mid_channels)
        self.spatial_act1 = nn.ReLU(inplace=True)
        self.spatial_conv2 = nn.Conv2d(mid_channels, cnn_hidden_dim, kernel_size=(1, 1))
        self.spatial_bn2 = nn.BatchNorm2d(cnn_hidden_dim)
        self.spatial_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_vec_dropout = nn.Dropout(p=max(0.0, min(0.5, dropout * 0.5)))

        # ===== TXT stream (text processing) =====
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, text_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(text_feat_dim)
        )

        # ===== CADF dynamic‑weight network (three‑modality) =====
        fusion_dim = lstm_hidden + cnn_hidden_dim + text_feat_dim
        
        self.weight_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 3),  # TFM, CNN, Text
            nn.Softmax(dim=-1)
        )
        
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
        
        self.multi_scale_weight_fusion = nn.Sequential(
            nn.Linear(6, 12),  # 3 global + 3 local
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(12, 3),
            nn.Softmax(dim=-1)
        )

        # ===== Fusion & classification =====
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_size)
        )
        self.dropout = nn.Dropout(dropout)

    def compute_dynamic_weights(self, fbm_vec, cnn_vec, text_vec):

        # (1) Global importance
        all_features = torch.cat([fbm_vec, cnn_vec, text_vec], dim=-1)
        global_weights = self.weight_net(all_features)  # [B, 3]
        
        # (2) Local quality estimator
        fbm_quality = self.content_aware_weights['fbm_quality'](fbm_vec)  # [B, 1]
        cnn_quality = self.content_aware_weights['cnn_quality'](cnn_vec)  # [B, 1]
        text_quality = self.content_aware_weights['text_quality'](text_vec)  # [B, 1]
        
        local_weights = torch.cat([fbm_quality, cnn_quality, text_quality], dim=-1)  # [B, 3]
        local_weights = torch.softmax(local_weights, dim=-1)
        
        # (3) Multi‑scale aggregation
        combined_weights = torch.cat([global_weights, local_weights], dim=-1)  # [B, 6]
        final_weights = self.multi_scale_weight_fusion(combined_weights)  # [B, 3]
        
        return final_weights

    def forward(self, raw_sensor_data, text_feat=None, return_x=False):
        # ===== TFM =====
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

        # ===== LST (2‑D CNN) =====
        x2d = raw_sensor_data.transpose(1, 2).unsqueeze(1)   # [B,1,3,T]
        h = self.spatial_act1(self.spatial_bn1(self.spatial_conv1(x2d)))
        h = self.spatial_bn2(self.spatial_conv2(h))
        h = self.spatial_gap(h)
        cnn_vec = h.squeeze(-1).squeeze(-1)                   # [B,d]
        cnn_vec = self.spatial_vec_dropout(cnn_vec)

        # ===== TXT =====
        if text_feat is not None:
            text_vec = self.text_projection(text_feat)  # [B, text_feat_dim]
        else:
            text_vec = torch.zeros(raw_sensor_data.size(0), self.text_feat_dim, 
                                   device=raw_sensor_data.device, dtype=raw_sensor_data.dtype)

        # ===== CADF dynamic fusion =====
        weights = self.compute_dynamic_weights(fbm_vec, cnn_vec, text_vec)  # [B, 3]
        fbm_weight = weights[:, 0:1]    # [B, 1]
        cnn_weight = weights[:, 1:2]    # [B, 1]
        text_weight = weights[:, 2:3]   # [B, 1]
        
        fbm_vec_weighted = fbm_weight * fbm_vec
        cnn_vec_weighted = cnn_weight * cnn_vec
        text_vec_weighted = text_weight * text_vec
        
        fused = torch.cat([fbm_vec_weighted, cnn_vec_weighted, text_vec_weighted], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        output = self.classifier(fused)

        if return_x:
            return fused, output
        else:
            return output


class HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive(nn.Module):
    """
    SSTF — the full TFM + LST + TXT model with multi‑task contrastive learning
    and CADF dynamic fusion.
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

        # ---- basic attributes ----
        self.raw_input_size = raw_input_size
        self.seq_len = seq_len
        self.text_feat_dim = text_feat_dim
        self.lstm_hidden = lstm_hidden
        self.cnn_hidden_dim = cnn_hidden_dim
        self.projection_dim = projection_dim
        self.temperature = float(temperature)
        self.contrastive_weight = 0.3

        # ---- TFM components (trend, seasonal, interaction) ----
        from fbm_paper_components import trend_component, seasonal_component, interaction_component
        self.fbm_trend = trend_component(seq_len, raw_input_size, block_size=fbm_block_size,
                                         use_transformer=True, hidden_dim=fbm_hidden_dim)
        self.fbm_seasonal = seasonal_component(seq_len, raw_input_size, block_size=fbm_block_size,
                                               hidden_dim=fbm_hidden_dim)
        self.fbm_interaction = interaction_component(seq_len, raw_input_size, block_size=fbm_block_size,
                                                     hidden_dim=fbm_hidden_dim * 2)

        # ---- TFM post‑processing (deep features, attention, LSTM) ----
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

        # ---- LST stream (2‑D CNN) ----
        mid_channels = 64
        k_t = conv_time_kernel
        self.spatial_conv1 = nn.Conv2d(1, mid_channels, kernel_size=(3, k_t), padding=(0, k_t // 2))
        self.spatial_bn1 = nn.BatchNorm2d(mid_channels)
        self.spatial_act1 = nn.ReLU(inplace=True)
        self.spatial_conv2 = nn.Conv2d(mid_channels, cnn_hidden_dim, kernel_size=(1, 1))
        self.spatial_bn2 = nn.BatchNorm2d(cnn_hidden_dim)
        self.spatial_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_vec_dropout = nn.Dropout(p=max(0.0, min(0.5, dropout * 0.5)))

        # ---- TXT stream (text feature processing) ----
        self.text_feature_projection = nn.Sequential(
            nn.Linear(text_feat_dim, text_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(text_feat_dim),
        )

        # ---- Projection heads (for contrastive learning) ----
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

        # ---- Joint sensor projection (TFM+LST) ----
        self.sensor_joint_proj = nn.Sequential(
            nn.Linear(2 * projection_dim, 2 * projection_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(2 * projection_dim),
            nn.Linear(2 * projection_dim, projection_dim),
        )

        # ---- CADF dynamic fusion ----
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

        # ---- Classification head ----
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_size),
        )
        self.dropout = nn.Dropout(dropout)

    # ---------------- Dynamic weights ---------------- #
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

    # ---------------- Contrastive losses --------------- #
    def _infonce(self, q, k, temperature=0.07):

        logits = torch.matmul(q, k.T) / temperature  # [B,B]
        labels = torch.arange(q.size(0), device=q.device)
        li = F.cross_entropy(logits, labels)
        lt = F.cross_entropy(logits.T, labels)
        return 0.5 * (li + lt)

    def _sup_contrastive(self, z, labels, temperature=0.07):

        sim = torch.matmul(z, z.T) / temperature             # [B,B]
        with torch.no_grad():
            pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
            eye = torch.eye(z.size(0), device=z.device)
            pos_mask = pos_mask * (1.0 - eye)               # exclude self
            logits_mask = 1.0 - eye

        sim_exp = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss

    # ---------------- Forward pass ---------------- #
    def forward(
        self,
        raw_sensor_data,
        text_feat=None,
        batch_y=None,
        return_x=False,
        contrastive=True,
    ):
        

        # ===== TFM =====
        fbm_trend = self.fbm_trend(raw_sensor_data)
        fbm_seasonal = self.fbm_seasonal(raw_sensor_data)
        fbm_interaction = self.fbm_interaction(raw_sensor_data)
        multi_scale = torch.cat([fbm_trend, fbm_seasonal, fbm_interaction], dim=-1)

        deep_feat = self.dropout(self.deep_feature_learning(multi_scale))
        attn_out, _ = self.temporal_attention(deep_feat, deep_feat, deep_feat)
        lstm_out, _ = self.lstm(self.dropout(attn_out))
        fbm_vec = lstm_out[:, -1, :]  # [B, lstm_hidden]

        # ===== LST (2‑D CNN) =====
        x2d = raw_sensor_data.transpose(1, 2).unsqueeze(1)  # [B,1,C,T]
        h = self.spatial_act1(self.spatial_bn1(self.spatial_conv1(x2d)))
        h = self.spatial_bn2(self.spatial_conv2(h))
        cnn_vec = self.spatial_gap(h).squeeze(-1).squeeze(-1)  # [B, cnn_hidden_dim]
        cnn_vec = self.spatial_vec_dropout(cnn_vec)

        # ===== TXT =====
        if text_feat is not None:
            text_vec = self.text_feature_projection(text_feat)
        else:
            text_vec = torch.zeros(
                raw_sensor_data.size(0),
                self.text_feat_dim,
                device=raw_sensor_data.device,
                dtype=raw_sensor_data.dtype,
            )


        weights = self.compute_dynamic_weights(fbm_vec, cnn_vec, text_vec)  # [B,3]
        fbm_vec_w = weights[:, 0:1] * fbm_vec
        cnn_vec_w = weights[:, 1:2] * cnn_vec
        text_vec_w = weights[:, 2:3] * text_vec

        fused = torch.cat([fbm_vec_w, cnn_vec_w, text_vec_w], dim=-1)  # [B, fusion_dim]
        fused = self.dropout(self.fusion_norm(fused))
        output = self.classifier(fused)

        # ===== Contrastive projections =====
        if contrastive:
            fbm_proj = self.fbm_projection(fbm_vec)     # [B, P]
            cnn_proj = self.cnn_projection(cnn_vec)     # [B, P]
            text_proj = self.text_projection(text_vec)  # [B, P]

            sensor_cat = torch.cat([fbm_proj, cnn_proj], dim=-1)  # [B, 2P]
            sensor_proj = self.sensor_joint_proj(sensor_cat)      # [B, P]

            # L2 normalization
            fbm_proj = F.normalize(fbm_proj, dim=-1)
            cnn_proj = F.normalize(cnn_proj, dim=-1)
            sensor_proj = F.normalize(sensor_proj, dim=-1)
            text_proj = F.normalize(text_proj, dim=-1)

            contrastive_loss_val = None
            if batch_y is not None:
                L_st = self._infonce(sensor_proj, text_proj, temperature=self.temperature)
                L_sc = self._infonce(fbm_proj, cnn_proj, temperature=self.temperature)
                L_sup = self._sup_contrastive(sensor_proj, batch_y, temperature=self.temperature)
                contrastive_loss_val = L_st + 0.5 * L_sc + 0.5 * L_sup
        else:
            if batch_y is not None:  
                sensor_proj = None
                fbm_proj = cnn_proj = text_proj = None
                contrastive_loss_val = torch.tensor(0.0, device=fused.device)
            else:
                # test mode: compute projections for visualisation
                fbm_proj = F.normalize(self.fbm_projection(fbm_vec), dim=-1)
                cnn_proj = F.normalize(self.cnn_projection(cnn_vec), dim=-1)
                text_proj = F.normalize(self.text_projection(text_vec), dim=-1)
                sensor_cat = torch.cat([fbm_proj, cnn_proj], dim=-1)
                sensor_proj = F.normalize(self.sensor_joint_proj(sensor_cat), dim=-1)
                contrastive_loss_val = torch.tensor(0.0, device=fused.device)

        # ===== Return interface =====
        if return_x == "fused":
            return fused
        elif return_x:
            return (
                fbm_vec, cnn_vec, text_vec, fbm_proj, cnn_proj, text_proj,
                output, contrastive_loss_val
            )
        else:
            if contrastive:
                return output, contrastive_loss_val
            else:
                return output