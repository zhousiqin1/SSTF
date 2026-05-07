from tqdm import tqdm
import torch
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import os, numpy as np, torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import matplotlib as mpl
from sklearn.feature_selection import VarianceThreshold
# 设置全局样式
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff"
})
sns.set_style("whitegrid", {'axes.facecolor': '#ffffff'})

# 多模态特征空间对比分析   
def plot_tsne_modal_comparison_dynamic(fbm_vec, cnn_vec, text_vec, fused_vec, labels,
                                       save_dir="./fig", filename_prefix="tsne_modal_dynamic",
                                       k_per_group=10000, perplexity=25):
    """
    📊 绘制单模态与动态融合特征的 t-SNE 可视化（TFM / LST / TXT / Dynamic Fused）
    - 各模态统一标准化尺度；
    - 动态 perplexity、自适应 learning_rate；
    - 每类每模态等量抽样；
    - 输出四张独立图，风格与 plot_tsne 一致；
    """
    os.makedirs(save_dir, exist_ok=True)
    y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    print(f"\n🌀 开始绘制单模态与动态融合特征空间 t-SNE 图 (≤{k_per_group}/class×modality)...")

    # ===== 统一数据准备 =====
    modalities = {
        "TFM": fbm_vec.detach().cpu().numpy(),
        "LST": cnn_vec.detach().cpu().numpy(),
        "TXT": text_vec.detach().cpu().numpy(),
        "Dynamic-Fused": fused_vec.detach().cpu().numpy(),
    }
    behavior_names = {0: '0', 1: '1', 2: '2'}
    unique_classes = np.unique(y)

    # ===== 绘制函数（统一风格） =====
    def draw_tsne(X, modality_name, file_suffix):
        colors = ['#548FBC', '#39823A', '#A194C7']
        print(f"\n🔹 绘制 {modality_name} 模态特征空间 t-SNE ...")
        print(f"📏 Feature STD → {modality_name}={X.std():.4f}")

        # 抽样每类数据（避免样本数过多）
        rng = np.random.default_rng(42)
        idx_keep = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) > k_per_group:
                idx = rng.choice(idx, size=k_per_group, replace=False)
            idx_keep.append(idx)
        idx_keep = np.concatenate(idx_keep, axis=0)
        X = X[idx_keep]
        y_sub = y[idx_keep]
        print(f"✅ 抽样后样本数: {len(y_sub)}")

        X -= X.mean(axis=0, keepdims=True)
        X = StandardScaler().fit_transform(X)

        # # 标准化 + 动态参数
        # X = StandardScaler().fit_transform(X)
        # tsne = TSNE(
        #     n_components=2,
        #     perplexity=30,
        #     learning_rate=250,
        #     n_iter=3500, 
        #     init="pca",
        #     random_state=42
        # )
        from sklearn.feature_selection import VarianceThreshold
        X = np.clip(X, -3, 3)  # 防极端值
        X = VarianceThreshold(threshold=1e-6).fit_transform(X)  # 去除低方差特征

        if X.shape[1] > 50:
                X = PCA(n_components=50, random_state=42).fit_transform(X)

        tsne = TSNE(
            n_components=2,
            perplexity=32,          # 稍大
            learning_rate=240,
            n_iter=4000,
            init='random',          # 避免PCA带状
            metric='cosine',        # 方向相似性更适合频谱
            random_state=42,
            verbose=0
        )

        # # 针对 TFM 模态使用单独策略
        # if "TFM" in modality_name or "FBM" in modality_name:
        #     from sklearn.feature_selection import VarianceThreshold
        #     X = np.clip(X, -3, 3)  # 防极端值
        #     X = VarianceThreshold(threshold=1e-6).fit_transform(X)  # 去除低方差特征
        #     X = PCA(n_components=50, whiten=True, random_state=42).fit_transform(X)  # PCA降维

        #     tsne = TSNE(
        #         n_components=2,
        #         perplexity=32,          # 稍大
        #         learning_rate=240,
        #         n_iter=4000,
        #         init='random',          # 避免PCA带状
        #         metric='cosine',        # 方向相似性更适合频谱
        #         random_state=42,
        #         verbose=0
        #     )
        # else:
        #     if X.shape[1] > 50:
        #         X = PCA(n_components=50, random_state=42).fit_transform(X)
        #     tsne = TSNE(
        #         n_components=2,
        #         perplexity=25,
        #         learning_rate=200,
        #         n_iter=3500,
        #         init='pca',
        #         random_state=42,
        #         verbose=0
        #     )
        
        tsne_emb = tsne.fit_transform(X)

        # ===== 绘图部分 =====
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1],
                    c=[colors[int(lbl) % len(colors)] for lbl in y_sub],
                    s=45, alpha=0.75, edgecolors='none')
        plt.title(f"{modality_name} Feature Space (Standardized, {k_per_group}/class)", fontsize=12, pad=10)

        # 类别图例
        handles = [
            Line2D([0], [0], marker='o', color='w',
                   label=behavior_names.get(int(i), f'Class {i}'),
                   markerfacecolor=colors[int(i) % len(colors)], markersize=7)
            for i in unique_classes
        ]
        plt.legend(handles=handles, title="Behavior", loc="upper right", fontsize=8)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{filename_prefix}_{file_suffix}.png")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"✅ {modality_name} 模态 t-SNE 图已保存到: {save_path}")

    # ===== 绘制四张模态图 =====
    for name, X in modalities.items():
        draw_tsne(X, modality_name=name, file_suffix=name.replace('-', '_'))

# 对比学习
# def plot_tsne(fbm_vec, cnn_vec, text_vec, labels,
#               save_dir="./fig", filename="tsne_contrastive_embeddings.png",
#               k_per_group=10000, perplexity=20):
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, filename)

#     # ======== 1️⃣ 数据准备 ========
#     fbm_emb = fbm_vec.detach().cpu().numpy()
#     cnn_emb = cnn_vec.detach().cpu().numpy()
#     text_emb = text_vec.detach().cpu().numpy()
#     y = labels.detach().cpu().numpy()

#     # 检查维度一致性
#     print(f"FBM shape={fbm_emb.shape}, CNN shape={cnn_emb.shape}, Text shape={text_emb.shape}")

#     # 拼接三模态
#     all_emb = np.concatenate([fbm_emb, cnn_emb, text_emb], axis=0)
#     all_labels = np.tile(y, 3)
#     modality = np.array(['TFM'] * len(y) + ['LST'] * len(y) + ['TXT'] * len(y))

#     # ======== 2️⃣ 打印特征统计 (检测尺度差异) ========
#     print(f"📏 Feature STD → FBM={fbm_emb.std():.4f}, CNN={cnn_emb.std():.4f}, TEXT={text_emb.std():.4f}")

#     # ======== 3️⃣ 分层等量抽样 ========
#     rng = np.random.default_rng(42)
#     idx_keep = []
#     for c in np.unique(all_labels):
#         for m in ['TFM', 'LST', 'TXT']:
#             idx = np.where((all_labels == c) & (modality == m))[0]
#             if len(idx) > k_per_group:
#                 idx = rng.choice(idx, size=k_per_group, replace=False)
#             idx_keep.append(idx)
#     idx_keep = np.concatenate(idx_keep, axis=0)
#     all_emb = all_emb[idx_keep]
#     all_labels = all_labels[idx_keep]
#     modality = modality[idx_keep]
#     print(f"✅ 抽样后样本数: {len(all_labels)}")

#     # ======== 4️⃣ 标准化（关键步骤） ========
#     scaler = StandardScaler()
#     all_emb = scaler.fit_transform(all_emb)

#     n_samples = len(all_emb)

#     tsne = TSNE(
#         n_components=2,
#         perplexity=30,      # ✅ 原默认20 → 30
#         learning_rate=240,  # ✅ 原自动200 → 240
#         n_iter=3000,        # ✅ 原2000 → 3000
#         init="pca",         # ✅ 改为 'pca'，三模态共图更稳定
#         random_state=42
#     )

#     tsne_emb = tsne.fit_transform(all_emb)

#     # ======== 6️⃣ 绘图 ========
#     unique_classes = np.unique(all_labels)
#     colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
#     markers = {'TFM': 'o', 'LST': 's', 'TXT': '^'}
#     plt.figure(figsize=(9, 7))
#     for m in ['TFM', 'LST', 'TXT']:
#         idx = modality == m
#         plt.scatter(
#             tsne_emb[idx, 0], tsne_emb[idx, 1],
#             c=[colors[int(lbl) % len(colors)] for lbl in all_labels[idx]],
#             marker=markers[m], alpha=0.65, label=m, s=24, edgecolors='none'
#         )

#     plt.title(f"t-SNE Visualization of Multi-modal Contrastive Embeddings", fontsize=13, pad=12)

#     # 模态图例
#     legend1 = plt.legend(title="Modality", loc="upper right", fontsize=9)
#     plt.gca().add_artist(legend1)

#     # 类别图例
#     behavior_names = {0: 'Other', 1: 'Rumination', 2: 'Eating'}  # 根据你数据集类别定义修改
#     handles = [
#         Line2D([0], [0], marker='o', color='w',
#             label=behavior_names.get(int(i), f'Class {i}'),
#             markerfacecolor=colors[int(i) % len(colors)], markersize=7)
#         for i in unique_classes
#     ]
#     plt.legend(handles=handles, title="Behavior", loc="lower right", fontsize=8)

#     plt.xticks([]); plt.yticks([])
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=400, bbox_inches='tight')
#     plt.close()

#     print(f"✅ t-SNE 图已保存到: {save_path}")

def plot_tsne_no_contrastive(fbm_vec, cnn_vec, text_vec, labels,
              save_dir="./fig", filename="tsne_no_contrastive_embeddings.png",
              k_per_group=500, perplexity=25):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # ======== 1️⃣ 数据准备 ========
    fbm = F.normalize(fbm_vec, dim=-1).detach().cpu().numpy()
    cnn = F.normalize(cnn_vec, dim=-1).detach().cpu().numpy()
    text = F.normalize(text_vec, dim=-1).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    # 检查维度一致性
    print(f"[shape] FBM={fbm.shape}, CNN={cnn.shape}, TEXT={text.shape}")
    # ========= 对各模态进行PCA降维到相同维度（例如32）=========
    from sklearn.decomposition import PCA
    target_dim = 32  # 与文本模态对齐
    pca_fbm = PCA(n_components=target_dim, random_state=42)
    pca_cnn = PCA(n_components=target_dim, random_state=42)
    pca_text = PCA(n_components=target_dim, random_state=42)

    fbm = pca_fbm.fit_transform(fbm)
    cnn = pca_cnn.fit_transform(cnn)
    text = pca_text.fit_transform(text)

    print(f"[after PCA] FBM={fbm.shape}, CNN={cnn.shape}, TEXT={text.shape}")


    # ========= 2) 模态内去均值（消除模态之间的零点偏移）=========
    fbm -= fbm.mean(axis=0, keepdims=True)
    cnn -= cnn.mean(axis=0, keepdims=True)
    text -= text.mean(axis=0, keepdims=True)

    X = np.concatenate([fbm, cnn, text], axis=0)
    Y = np.tile(y, 3)
    M = np.array(['TFM'] * len(y) + ['LST'] * len(y) + ['TXT'] * len(y))


    # ========= 4) 分层等量抽样：每“类别×模态” ≤ 500 =========
    rng = np.random.default_rng(42)
    keep_idx = []
    classes = np.unique(Y)
    modalities = ['TFM', 'LST', 'TXT']

    print("=== 抽样前计数 ===")
    for c in classes:
        for m in modalities:
            cnt = np.sum((Y == c) & (M == m))
            print(f"class {int(c)}, modality {m}: {cnt}")

    for c in classes:
        for m in modalities:
            idx = np.where((Y == c) & (M == m))[0]
            if len(idx) > k_per_group:
                idx = rng.choice(idx, size=k_per_group, replace=False)
            keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)

    X, Y, M = X[keep_idx], Y[keep_idx], M[keep_idx]

    print("=== 抽样后计数 ===")
    for c in classes:
        for m in modalities:
            cnt = np.sum((Y == c) & (M == m))
            print(f"class {int(c)}, modality {m}: {cnt}")
    print(f"Total plotted: {len(Y)}")

    # ========= 5) 全局标准化（在拼接后统一做）=========
    X = StandardScaler().fit_transform(X)


    tsne = TSNE(
        n_components=2,
        perplexity=32,
        learning_rate=240,
        n_iter=4000,
        init='random',
        random_state=42
    )
    Z = tsne.fit_transform(X)

    # ========= 7) 绘图 =========
    # plt.figure(figsize=(9, 7))
    # unique_classes = np.unique(Y)
    # palette = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    # color_map = {int(c): palette[i % len(palette)] for i, c in enumerate(unique_classes)}
    # markers = {'TFM': 'o', 'LST': 's', 'TXT': '^'}

    # for m in modalities:
    #     mask = (M == m)
    #     plt.scatter(
    #         Z[mask, 0], Z[mask, 1],
    #         c=[color_map[int(lbl)] for lbl in Y[mask]],
    #         marker=markers[m], alpha=0.6, s=16, edgecolors='none', label=m
    #     )
    # ===== 颜色和点样式改成和 plot_tsne_modal_comparison_dynamic 一致 =====
    plt.figure(figsize=(8, 6))   # 8x6 和 draw_tsne 一样
    unique_classes = np.unique(Y)
    colors = ['#548FBC', '#39823A', '#A194C7']
    color_map = {int(c): colors[i % len(colors)] for i, c in enumerate(unique_classes)}

    markers = {'TFM': 'o', 'LST': 's', 'TXT': 'D'}

    for m in modalities:
        mask = (M == m)
        plt.scatter(
            Z[mask, 0], Z[mask, 1],
            c=[color_map[int(lbl)] for lbl in Y[mask]],
            marker=markers[m],
            alpha=1,       # 和 draw_tsne 接近
            s=40,             # 点稍微大一点，风格更像
            edgecolors='none',
            label=m
        )

    plt.title("Multi-modal no Contrastive Embeddings", fontsize=12, pad=12)

    # 模态图例
    legend1 = plt.legend(title="Modality", loc="upper right", fontsize=8)
    plt.gca().add_artist(legend1)

    # 类别图例
    behavior_names = {0: '0', 1: '1', 2: '2'}
    handles = [
        Line2D([0], [0], marker='o', color='w',
               label=behavior_names.get(int(c), f'Class {int(c)}'),
               markerfacecolor=color_map[int(c)], markersize=7)
        for c in unique_classes
    ]
    plt.legend(handles=handles, title="Behavior", loc="lower right", fontsize=8)

    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE 图已保存到: {save_path}")

def plot_tsne(fbm_vec, cnn_vec, text_vec, labels,
              save_dir="./fig", filename="tsne_contrastive_embeddings.png",
              k_per_group=500, perplexity=25):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # ======== 1️⃣ 数据准备 ========
    fbm = F.normalize(fbm_vec, dim=-1).detach().cpu().numpy()
    cnn = F.normalize(cnn_vec, dim=-1).detach().cpu().numpy()
    text = F.normalize(text_vec, dim=-1).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    # 检查维度一致性
    print(f"[shape] FBM={fbm.shape}, CNN={cnn.shape}, TEXT={text.shape}")

    # ========= 2) 模态内去均值（消除模态之间的零点偏移）=========
    fbm -= fbm.mean(axis=0, keepdims=True)
    cnn -= cnn.mean(axis=0, keepdims=True)
    text -= text.mean(axis=0, keepdims=True)

    X = np.concatenate([fbm, cnn, text], axis=0)
    Y = np.tile(y, 3)
    M = np.array(['TFM'] * len(y) + ['LST'] * len(y) + ['TXT'] * len(y))


    # ========= 4) 分层等量抽样：每“类别×模态” ≤ 500 =========
    rng = np.random.default_rng(42)
    keep_idx = []
    classes = np.unique(Y)
    modalities = ['TFM', 'LST', 'TXT']

    print("=== 抽样前计数 ===")
    for c in classes:
        for m in modalities:
            cnt = np.sum((Y == c) & (M == m))
            print(f"class {int(c)}, modality {m}: {cnt}")

    for c in classes:
        for m in modalities:
            idx = np.where((Y == c) & (M == m))[0]
            if len(idx) > k_per_group:
                idx = rng.choice(idx, size=k_per_group, replace=False)
            keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)

    X, Y, M = X[keep_idx], Y[keep_idx], M[keep_idx]

    print("=== 抽样后计数 ===")
    for c in classes:
        for m in modalities:
            cnt = np.sum((Y == c) & (M == m))
            print(f"class {int(c)}, modality {m}: {cnt}")
    print(f"Total plotted: {len(Y)}")

    # ========= 5) 全局标准化（在拼接后统一做）=========
    X -= X.mean(axis=0, keepdims=True)
    X = StandardScaler().fit_transform(X)
    X = np.clip(X, -3, 3)  # 防极端值
    X = VarianceThreshold(threshold=1e-6).fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=32,
        learning_rate=240,
        n_iter=3500,
        init='random',
        random_state=42
    )
    Z = tsne.fit_transform(X)

    # ========= 7) 绘图 =========
    # plt.figure(figsize=(9, 7))
    # unique_classes = np.unique(Y)
    # palette = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    # color_map = {int(c): palette[i % len(palette)] for i, c in enumerate(unique_classes)}
    # markers = {'TFM': 'o', 'LST': 's', 'TXT': '^'}

    # for m in modalities:
    #     mask = (M == m)
    #     plt.scatter(
    #         Z[mask, 0], Z[mask, 1],
    #         c=[color_map[int(lbl)] for lbl in Y[mask]],
    #         marker=markers[m], alpha=0.6, s=16, edgecolors='none', label=m
    #     )
    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(Y)

    colors = ['#548FBC', '#39823A', '#A194C7']
    color_map = {int(c): colors[i % len(colors)] for i, c in enumerate(unique_classes)}

    markers = {'TFM': 'o', 'LST': 's', 'TXT': 'D'}

    for m in modalities:
        mask = (M == m)
        plt.scatter(
            Z[mask, 0], Z[mask, 1],
            c=[color_map[int(lbl)] for lbl in Y[mask]],
            marker=markers[m],
            alpha=1,
            s=40,
            edgecolors='none',
            label=m
        )

    plt.title("Multi-modal Contrastive Embeddings", fontsize=12, pad=12)

    # 模态图例
    legend1 = plt.legend(title="Modality", loc="upper right", fontsize=8)
    plt.gca().add_artist(legend1)

    # 类别图例
    behavior_names = {0: '0', 1: '1', 2: '2'}
    handles = [
        Line2D([0], [0], marker='o', color='w',
               label=behavior_names.get(int(c), f'Class {int(c)}'),
               markerfacecolor=color_map[int(c)], markersize=7)
        for c in unique_classes
    ]
    plt.legend(handles=handles, title="Behavior", loc="lower right", fontsize=8)

    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE 图已保存到: {save_path}")

# 对比学习之后的相似  空白
def plot_modal_similarity(fbm_vec, cnn_vec, text_vec,
                          save_dir="./fig",
                          filename="modal_similarity_heatmap.png"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("开始绘制模态特征相似度热力图...")
    fbm_emb = fbm_vec.detach().cpu()
    cnn_emb = cnn_vec.detach().cpu()
    text_emb = text_vec.detach().cpu()

    # ===== 计算相似度 =====
    fbm_cnn_sim = F.cosine_similarity(fbm_emb, cnn_emb, dim=1).numpy()
    fbm_text_sim = F.cosine_similarity(fbm_emb, text_emb, dim=1).numpy()
    cnn_text_sim = F.cosine_similarity(cnn_emb, text_emb, dim=1).numpy()

    sim_matrix = np.vstack([fbm_cnn_sim, fbm_text_sim, cnn_text_sim])

    # ===== 绘图 =====
    plt.figure(figsize=(7.5, 3.5))
    sns.set_style("whitegrid")

    ax = sns.heatmap(
        sim_matrix,
        cmap="Blues",
        vmin=-1, vmax=1, center=0,   # 保持色彩对齐
        linewidths=0.05, linecolor='white',  # 分隔线
        cbar=True,
        cbar_kws={"label": "Cosine Similarity"},
        yticklabels=["TFM–LST", "TFM–SA", "LST–SA"],
        xticklabels=False
    )

    plt.title("Cross-Modal Feature Similarity Heatmap", fontsize=12, pad=8)
    plt.xlabel("Samples", fontsize=10)
    plt.ylabel("Modal Pairs", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"✅ 模态特征相似度热力图已保存到 {save_path}")

# 全局统计图
def plot_avg_modal_similarity(fbm_vec, cnn_vec, text_vec, save_dir="./fig", filename="modal_similarity_bar.png"):
    """
    绘制三模态对之间的平均余弦相似度条形图
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("开始绘制平均模态相似度条形图...")

    fbm = fbm_vec.detach().cpu()
    cnn = cnn_vec.detach().cpu()
    text = text_vec.detach().cpu()

    fbm_cnn_sim = F.cosine_similarity(fbm, cnn, dim=1).mean().item()
    fbm_text_sim = F.cosine_similarity(fbm, text, dim=1).mean().item()
    cnn_text_sim = F.cosine_similarity(cnn, text, dim=1).mean().item()

    # pairs = ["FBM–CNN", "FBM–Text", "CNN–Text"]
    pairs = ["TFM–LST", "TFM–SA", "LST–SA"]
    sims = [fbm_cnn_sim, fbm_text_sim, cnn_text_sim]

    plt.figure(figsize=(5, 4))
    sns.barplot(x=pairs, y=sims, palette="coolwarm")

    # palette=["#4C72B0", "#55A868", "#C44E52"]
    plt.ylim(0, 1)
    plt.title("Average Cross-Modal Cosine Similarity", fontsize=12)
    plt.ylabel("Mean Similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"✅ 平均模态相似度条形图已保存到 {save_path}")

def plot_embedding_ridge(fbm_vec, cnn_vec, text_vec, save_dir="./fig", filename="embedding_ridgeplot.png"):
    """
    绘制嵌入特征分布重叠的岭形图（KDE）
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("开始绘制嵌入分布岭形图...")

    fbm = fbm_vec.detach().cpu().numpy().flatten()
    cnn = cnn_vec.detach().cpu().numpy().flatten()
    text = text_vec.detach().cpu().numpy().flatten()

    df = pd.DataFrame({
        'value': np.concatenate([fbm, cnn, text]),
        'modality': ['FBM'] * len(fbm) + ['CNN'] * len(cnn) + ['Text'] * len(text)
    })

    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x="value", hue="modality", fill=True, alpha=0.5)
    plt.title("Embedding Distribution Overlap (Ridge Plot)", fontsize=12)
    plt.xlabel("Feature Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"✅ 嵌入分布岭形图已保存到 {save_path}")

# 动态融合
def plot_fusion_weight_distribution(model, fbm_vec, cnn_vec, text_vec,
                                    save_dir="./fig",
                                    filename="fusion_weights_distribution.png"):
    """
    🎨 绘制模型计算出的动态权重分布箱线图
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("开始绘制动态权重分布箱线图...")

    # ======= 提取权重 =======
    model.eval()
    with torch.no_grad():
        weights = model.compute_dynamic_weights(
            fbm_vec.to(next(model.parameters()).device),
            cnn_vec.to(next(model.parameters()).device),
            text_vec.to(next(model.parameters()).device)
        ).cpu().numpy()  # [N, 3]

    # ======= 美化绘图样式 =======
    plt.figure(figsize=(7, 5))
    sns.set_style("whitegrid")

    # 柔和论文级配色（Set2）
    palette = sns.color_palette("Set2", 3)
    box_colors = dict(boxes=palette, whiskers=palette,
                      medians=palette, caps=palette)

    # 箱线图
    bplot = plt.boxplot(
        [weights[:, 0], weights[:, 1], weights[:, 2]],
        # labels=["FBM", "CNN", "Text"],
        labels=["TFM", "LST", "TXT"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2)
    )

    # 设置填充色
    for patch, color in zip(bplot['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(1.1)

    # 均值点
    means = [np.mean(weights[:, 0]),
             np.mean(weights[:, 1]),
             np.mean(weights[:, 2])]
    plt.scatter(range(1, 4), means, color="#222222", s=45,
                marker="D", label="Mean Value", zorder=3)

    # 文字与样式
    plt.title("Distribution of Dynamic Fusion Weights", fontsize=13, pad=10)
    plt.ylabel("Weight Value", fontsize=11)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()

    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"✅ 动态权重分布箱线图已保存到 {save_path}")

def plot_classwise_modal_similarity(fbm_vec, cnn_vec, text_vec, labels,
                                    save_dir="./fig",
                                    filename="classwise_modal_similarity_heatmap.png"):
    """
    📊 绘制按行为类别聚合的模态相似度热力图：
    - 每行代表一个行为类别（如 Rumination / Feeding / Other）
    - 每列代表一种模态对（FBM–CNN / FBM–Text / CNN–Text）
    - 颜色表示该类别下该模态对的平均余弦相似度
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("开始绘制按类别聚合的模态相似度热力图...")

    # 转为CPU numpy
    fbm = fbm_vec.detach().cpu()
    cnn = cnn_vec.detach().cpu()
    text = text_vec.detach().cpu()
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    classes = np.unique(labels)
    sim_means = []

    # 计算每个类别下三种模态对的平均相似度
    for c in classes:
        mask = (labels == c)
        fbm_cnn_sim = F.cosine_similarity(fbm[mask], cnn[mask], dim=1).mean().item()
        fbm_text_sim = F.cosine_similarity(fbm[mask], text[mask], dim=1).mean().item()
        cnn_text_sim = F.cosine_similarity(cnn[mask], text[mask], dim=1).mean().item()
        sim_means.append([fbm_cnn_sim, fbm_text_sim, cnn_text_sim])

    # 构造 DataFrame 用于热力图
    sim_means = np.array(sim_means)
    behavior_names = {0: 'Other', 1: 'Rumination', 2: 'Eating'}
    class_labels = [behavior_names.get(int(c), f'Class {int(c)}') for c in classes]
    modality_pairs = ["TFM–LST", "TFM–SA", "LST–SA"]

    plt.figure(figsize=(6, 4))
    sns.heatmap(sim_means, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=modality_pairs, yticklabels=class_labels,
                cbar_kws={"label": "Mean Cosine Similarity"})
    plt.title("Class-wise Cross-Modal Similarity", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"✅ 按类别聚合模态相似度热力图已保存到 {save_path}")


def test_model(model, test_loader):
    """
    只做推理 + 收集特征，不画图
    （画图在 main.py 或外部函数里单独做）
    """
    model.eval()
    test_accs = []
    device = next(model.parameters()).device

    all_y = []
    all_raw = {'fbm': [], 'cnn': [], 'text': []}
    # all_proj = {'fbm': [], 'cnn': [], 'text': []}
    all_proj = {'fbm': [], 'cnn': [], 'text': []}
    predicted_labels, true_labels = [], []

    with torch.no_grad():
        for batch_sensor, batch_text, batch_y in tqdm(test_loader, desc="Testing"):
            batch_sensor = batch_sensor.to(device, non_blocking=True)
            batch_text = batch_text.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # 只 forward 一次，不计算对比损失
            fbm_vec, cnn_vec, text_vec, fbm_proj, cnn_proj, text_proj,  outputs, _ = model(
                batch_sensor, batch_text,
                batch_y=None,               # 不计算对比损失
                contrastive=False,
                return_x=True               # 取特征用于画图
            )

            _, predicted = torch.max(outputs, 1)
            predicted_labels.append(predicted.cpu())
            true_labels.append(batch_y.cpu())

            # 保存特征
            all_raw['fbm'].append(fbm_vec.cpu())
            all_raw['cnn'].append(cnn_vec.cpu())
            all_raw['text'].append(text_vec.cpu())

            all_proj['fbm'].append(fbm_proj.cpu())
            all_proj['cnn'].append(cnn_proj.cpu())
            all_proj['text'].append(text_proj.cpu())

            all_y.append(batch_y.cpu())

    # 拼接
    y_all = torch.cat(all_y, dim=0)
    raw_fbm = torch.cat(all_raw['fbm'], dim=0)
    raw_cnn = torch.cat(all_raw['cnn'], dim=0)
    raw_text = torch.cat(all_raw['text'], dim=0)

    proj_fbm = torch.cat(all_proj['fbm'], dim=0)
    proj_cnn = torch.cat(all_proj['cnn'], dim=0)
    proj_text = torch.cat(all_proj['text'], dim=0)


    predicted_labels = torch.cat(predicted_labels, dim=0).numpy()
    true_labels = torch.cat(true_labels, dim=0).numpy()

    # 指标
    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    print(f"\n✅ Test Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    # ✅ 计算指标
    recalls = recall_score(true_labels, predicted_labels, average=None)
    test_ba_accuracy = recalls.mean()
    test_accs.append(test_ba_accuracy)

    print("Test Balanced Accuracy: {:.4f}".format(test_ba_accuracy))
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))

    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    print("Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(precision, recall, f1))
    

    # 不画图，把特征全部返回出去
    return (
        acc, f1, predicted_labels,
        y_all, raw_fbm, raw_cnn, raw_text, 
        proj_fbm, proj_cnn, proj_text
    )


