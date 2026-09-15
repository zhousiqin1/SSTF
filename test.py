import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    accuracy_score
)
from matplotlib.lines import Line2D


# ---- Global plot style ----
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


# ============================================================================
#  t‑SNE visualisation helpers
# ============================================================================

def plot_tsne_modal_comparison_dynamic(fbm_vec, cnn_vec, text_vec, fused_vec, labels,
                                       save_dir="./fig", filename_prefix="tsne_modal_dynamic",
                                       k_per_group=10000, perplexity=25):
    """
    Plot t‑SNE embeddings separately for each modality (TFM, LST, TXT, fused).
    """
    os.makedirs(save_dir, exist_ok=True)
    y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    print(f"\nPlotting per‑modality t‑SNE (≤{k_per_group} samples / class / modality)...")

    modalities = {
        "TFM": fbm_vec.detach().cpu().numpy(),
        "LST": cnn_vec.detach().cpu().numpy(),
        "TXT": text_vec.detach().cpu().numpy(),
        "Dynamic-Fused": fused_vec.detach().cpu().numpy(),
    }
    behavior_names = {0: '0', 1: '1', 2: '2'}
    unique_classes = np.unique(y)

    def _draw(X, modality_name, file_suffix):
        colors = ['#548FBC', '#39823A', '#A194C7']
        print(f"  [{modality_name}] t‑SNE ...")
        print(f"  Feature STD before processing = {X.std():.4f}")

        # Stratified sampling
        rng = np.random.default_rng(42)
        idx_keep = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) > k_per_group:
                idx = rng.choice(idx, size=k_per_group, replace=False)
            idx_keep.append(idx)
        idx_keep = np.concatenate(idx_keep)
        X = X[idx_keep]
        y_sub = y[idx_keep]
        print(f"  Subsampled to {len(y_sub)} points.")

        # Pre‑processing
        X -= X.mean(axis=0, keepdims=True)
        X = StandardScaler().fit_transform(X)
        X = np.clip(X, -3, 3)
        X = VarianceThreshold(threshold=1e-6).fit_transform(X)
        if X.shape[1] > 50:
            X = PCA(n_components=50, random_state=42).fit_transform(X)

        tsne = TSNE(
            n_components=2, perplexity=32, learning_rate=240,
            n_iter=4000, init='random', metric='cosine', random_state=42
        )
        emb = tsne.fit_transform(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(emb[:, 0], emb[:, 1],
                    c=[colors[int(lbl) % len(colors)] for lbl in y_sub],
                    s=45, alpha=0.75, edgecolors='none')
        plt.title(f"{modality_name} Feature Space (≤{k_per_group}/class)", fontsize=12, pad=10)

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
        print(f"  Saved to {save_path}")

    for name, X in modalities.items():
        _draw(X, modality_name=name, file_suffix=name.replace('-', '_'))


def plot_tsne_no_contrastive(fbm_vec, cnn_vec, text_vec, labels,
                             save_dir="./fig", filename="tsne_no_contrastive_embeddings.png",
                             k_per_group=500, perplexity=25):
    """t‑SNE of multi‑modal embeddings **without** contrastive learning."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # ---- Prepare data ----
    fbm = F.normalize(fbm_vec, dim=-1).detach().cpu().numpy()
    cnn = F.normalize(cnn_vec, dim=-1).detach().cpu().numpy()
    text = F.normalize(text_vec, dim=-1).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    print(f"[shape] TFM={fbm.shape}, LST={cnn.shape}, TXT={text.shape}")

    # PCA to a common dimension (32)
    target_dim = 32
    fbm = PCA(n_components=target_dim, random_state=42).fit_transform(fbm)
    cnn = PCA(n_components=target_dim, random_state=42).fit_transform(cnn)
    text = PCA(n_components=target_dim, random_state=42).fit_transform(text)
    print(f"[after PCA] TFM={fbm.shape}, LST={cnn.shape}, TXT={text.shape}")

    # Remove mean per modality
    fbm -= fbm.mean(axis=0, keepdims=True)
    cnn -= cnn.mean(axis=0, keepdims=True)
    text -= text.mean(axis=0, keepdims=True)

    X = np.concatenate([fbm, cnn, text], axis=0)
    Y = np.tile(y, 3)
    M = np.array(['TFM'] * len(y) + ['LST'] * len(y) + ['TXT'] * len(y))

    # Stratified sampling (≤ k_per_group per class×modality)
    rng = np.random.default_rng(42)
    keep_idx = []
    classes = np.unique(Y)
    modalities = ['TFM', 'LST', 'TXT']

    print("=== Before sampling ===")
    for c in classes:
        for m in modalities:
            cnt = np.sum((Y == c) & (M == m))
            print(f"  class {int(c)}, modality {m}: {cnt}")

    for c in classes:
        for m in modalities:
            idx = np.where((Y == c) & (M == m))[0]
            if len(idx) > k_per_group:
                idx = rng.choice(idx, size=k_per_group, replace=False)
            keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    X, Y, M = X[keep_idx], Y[keep_idx], M[keep_idx]

    print("=== After sampling ===")
    for c in classes:
        for m in modalities:
            cnt = np.sum((Y == c) & (M == m))
            print(f"  class {int(c)}, modality {m}: {cnt}")
    print(f"  Total points plotted: {len(Y)}")

    # Global standardization
    X = StandardScaler().fit_transform(X)

    tsne = TSNE(
        n_components=2, perplexity=32, learning_rate=240,
        n_iter=4000, init='random', random_state=42
    )
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    colors = ['#548FBC', '#39823A', '#A194C7']
    color_map = {int(c): colors[i % len(colors)] for i, c in enumerate(classes)}
    markers = {'TFM': 'o', 'LST': 's', 'TXT': 'D'}

    for m in modalities:
        mask = (M == m)
        plt.scatter(Z[mask, 0], Z[mask, 1],
                    c=[color_map[int(lbl)] for lbl in Y[mask]],
                    marker=markers[m], alpha=1, s=40, edgecolors='none', label=m)

    plt.title("Multi-modal Embeddings (No Contrastive)", fontsize=12, pad=12)

    legend1 = plt.legend(title="Modality", loc="upper right", fontsize=8)
    plt.gca().add_artist(legend1)

    behavior_names = {0: '0', 1: '1', 2: '2'}
    handles = [
        Line2D([0], [0], marker='o', color='w',
               label=behavior_names.get(int(c), f'Class {int(c)}'),
               markerfacecolor=color_map[int(c)], markersize=7)
        for c in classes
    ]
    plt.legend(handles=handles, title="Behavior", loc="lower right", fontsize=8)

    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"t‑SNE (no contrastive) saved to {save_path}")


def plot_tsne(fbm_vec, cnn_vec, text_vec, labels,
              save_dir="./fig", filename="tsne_contrastive_embeddings.png",
              k_per_group=500, perplexity=25):
    """t‑SNE of multi‑modal embeddings **with** contrastive learning."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    fbm = F.normalize(fbm_vec, dim=-1).detach().cpu().numpy()
    cnn = F.normalize(cnn_vec, dim=-1).detach().cpu().numpy()
    text = F.normalize(text_vec, dim=-1).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    print(f"[shape] TFM={fbm.shape}, LST={cnn.shape}, TXT={text.shape}")

    # Remove mean per modality
    fbm -= fbm.mean(axis=0, keepdims=True)
    cnn -= cnn.mean(axis=0, keepdims=True)
    text -= text.mean(axis=0, keepdims=True)

    X = np.concatenate([fbm, cnn, text], axis=0)
    Y = np.tile(y, 3)
    M = np.array(['TFM'] * len(y) + ['LST'] * len(y) + ['TXT'] * len(y))

    # Stratified sampling
    rng = np.random.default_rng(42)
    keep_idx = []
    classes = np.unique(Y)
    modalities = ['TFM', 'LST', 'TXT']

    print("=== Before sampling ===")
    for c in classes:
        for m in modalities:
            cnt = np.sum((Y == c) & (M == m))
            print(f"  class {int(c)}, modality {m}: {cnt}")

    for c in classes:
        for m in modalities:
            idx = np.where((Y == c) & (M == m))[0]
            if len(idx) > k_per_group:
                idx = rng.choice(idx, size=k_per_group, replace=False)
            keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    X, Y, M = X[keep_idx], Y[keep_idx], M[keep_idx]

    print("=== After sampling ===")
    for c in classes:
        for m in modalities:
            cnt = np.sum((Y == c) & (M == m))
            print(f"  class {int(c)}, modality {m}: {cnt}")
    print(f"  Total points plotted: {len(Y)}")

    # Pre‑processing
    X -= X.mean(axis=0, keepdims=True)
    X = StandardScaler().fit_transform(X)
    X = np.clip(X, -3, 3)
    X = VarianceThreshold(threshold=1e-6).fit_transform(X)

    tsne = TSNE(
        n_components=2, perplexity=32, learning_rate=240,
        n_iter=3500, init='random', random_state=42
    )
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    colors = ['#548FBC', '#39823A', '#A194C7']
    color_map = {int(c): colors[i % len(colors)] for i, c in enumerate(classes)}
    markers = {'TFM': 'o', 'LST': 's', 'TXT': 'D'}

    for m in modalities:
        mask = (M == m)
        plt.scatter(Z[mask, 0], Z[mask, 1],
                    c=[color_map[int(lbl)] for lbl in Y[mask]],
                    marker=markers[m], alpha=1, s=40, edgecolors='none', label=m)

    plt.title("Multi-modal Contrastive Embeddings", fontsize=12, pad=12)

    legend1 = plt.legend(title="Modality", loc="upper right", fontsize=8)
    plt.gca().add_artist(legend1)

    behavior_names = {0: '0', 1: '1', 2: '2'}
    handles = [
        Line2D([0], [0], marker='o', color='w',
               label=behavior_names.get(int(c), f'Class {int(c)}'),
               markerfacecolor=color_map[int(c)], markersize=7)
        for c in classes
    ]
    plt.legend(handles=handles, title="Behavior", loc="lower right", fontsize=8)

    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"t‑SNE (contrastive) saved to {save_path}")


# ============================================================================
#  Cross‑modal similarity plots
# ============================================================================

def plot_modal_similarity(fbm_vec, cnn_vec, text_vec,
                          save_dir="./fig", filename="modal_similarity_heatmap.png"):
    """Heatmap of per‑sample cosine similarity between modality pairs."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("Plotting cross‑modal similarity heatmap...")

    fbm = fbm_vec.detach().cpu()
    cnn = cnn_vec.detach().cpu()
    text = text_vec.detach().cpu()

    sim_tfm_lst = F.cosine_similarity(fbm, cnn, dim=1).numpy()
    sim_tfm_txt = F.cosine_similarity(fbm, text, dim=1).numpy()
    sim_lst_txt = F.cosine_similarity(cnn, text, dim=1).numpy()

    sim_mat = np.vstack([sim_tfm_lst, sim_tfm_txt, sim_lst_txt])

    plt.figure(figsize=(7.5, 3.5))
    sns.heatmap(sim_mat, cmap="Blues", vmin=-1, vmax=1, center=0,
                linewidths=0.05, linecolor='white', cbar=True,
                cbar_kws={"label": "Cosine Similarity"},
                yticklabels=["TFM–LST", "TFM–TXT", "LST–TXT"],
                xticklabels=False)
    plt.title("Cross-Modal Feature Similarity Heatmap", fontsize=12, pad=8)
    plt.xlabel("Samples", fontsize=10)
    plt.ylabel("Modal Pairs", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Cross‑modal similarity heatmap saved to {save_path}")


def plot_avg_modal_similarity(fbm_vec, cnn_vec, text_vec,
                              save_dir="./fig", filename="modal_similarity_bar.png"):
    """Bar plot of average cosine similarity for each modality pair."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("Plotting average cross‑modal similarity ...")

    fbm = fbm_vec.detach().cpu()
    cnn = cnn_vec.detach().cpu()
    text = text_vec.detach().cpu()

    sims = [
        F.cosine_similarity(fbm, cnn, dim=1).mean().item(),
        F.cosine_similarity(fbm, text, dim=1).mean().item(),
        F.cosine_similarity(cnn, text, dim=1).mean().item()
    ]
    pairs = ["TFM–LST", "TFM–TXT", "LST–TXT"]

    plt.figure(figsize=(5, 4))
    sns.barplot(x=pairs, y=sims, palette="coolwarm")
    plt.ylim(0, 1)
    plt.title("Average Cross-Modal Cosine Similarity", fontsize=12)
    plt.ylabel("Mean Similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Average similarity bar plot saved to {save_path}")


def plot_embedding_ridge(fbm_vec, cnn_vec, text_vec,
                         save_dir="./fig", filename="embedding_ridgeplot.png"):
    """Ridge (KDE) plot of flattened feature distributions per modality."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("Plotting embedding distribution ridge plot ...")

    fbm = fbm_vec.detach().cpu().numpy().flatten()
    cnn = cnn_vec.detach().cpu().numpy().flatten()
    text = text_vec.detach().cpu().numpy().flatten()

    import pandas as pd
    df = pd.DataFrame({
        'value': np.concatenate([fbm, cnn, text]),
        'modality': ['TFM'] * len(fbm) + ['LST'] * len(cnn) + ['TXT'] * len(text)
    })

    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x="value", hue="modality", fill=True, alpha=0.5)
    plt.title("Embedding Distribution Overlap (Ridge Plot)", fontsize=12)
    plt.xlabel("Feature Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Ridge plot saved to {save_path}")


# ============================================================================
#  Dynamic fusion weight visualisation
# ============================================================================

def plot_fusion_weight_distribution(model, fbm_vec, cnn_vec, text_vec,
                                    save_dir="./fig",
                                    filename="fusion_weights_distribution.png"):
    """Box plot of the dynamic fusion weights (TFM, LST, TXT)."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("Plotting dynamic weight distribution ...")

    model.eval()
    with torch.no_grad():
        weights = model.compute_dynamic_weights(
            fbm_vec.to(next(model.parameters()).device),
            cnn_vec.to(next(model.parameters()).device),
            text_vec.to(next(model.parameters()).device)
        ).cpu().numpy()  # [N, 3]

    plt.figure(figsize=(7, 5))
    palette = sns.color_palette("Set2", 3)
    box = plt.boxplot(
        [weights[:, 0], weights[:, 1], weights[:, 2]],
        labels=["TFM", "LST", "TXT"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2)
    )
    for patch, color in zip(box['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(1.1)

    means = [np.mean(weights[:, i]) for i in range(3)]
    plt.scatter(range(1, 4), means, color="#222222", s=45,
                marker="D", label="Mean Value", zorder=3)

    plt.title("Distribution of Dynamic Fusion Weights", fontsize=13, pad=10)
    plt.ylabel("Weight Value", fontsize=11)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Dynamic weight box plot saved to {save_path}")


def plot_classwise_modal_similarity(fbm_vec, cnn_vec, text_vec, labels,
                                    save_dir="./fig",
                                    filename="classwise_modal_similarity_heatmap.png"):
    """Heatmap of average cross‑modal similarity for each behaviour class."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print("Plotting class‑wise cross‑modal similarity ...")

    fbm = fbm_vec.detach().cpu()
    cnn = cnn_vec.detach().cpu()
    text = text_vec.detach().cpu()
    y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    classes = np.unique(y)
    sim_means = []
    for c in classes:
        mask = (y == c)
        sim_means.append([
            F.cosine_similarity(fbm[mask], cnn[mask], dim=1).mean().item(),
            F.cosine_similarity(fbm[mask], text[mask], dim=1).mean().item(),
            F.cosine_similarity(cnn[mask], text[mask], dim=1).mean().item()
        ])

    sim_means = np.array(sim_means)
    behavior_names = {0: 'Other', 1: 'Rumination', 2: 'Eating'}
    class_labels = [behavior_names.get(int(c), f'Class {int(c)}') for c in classes]
    modality_pairs = ["TFM–LST", "TFM–TXT", "LST–TXT"]

    plt.figure(figsize=(6, 4))
    sns.heatmap(sim_means, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=modality_pairs, yticklabels=class_labels,
                cbar_kws={"label": "Mean Cosine Similarity"})
    plt.title("Class-wise Cross-Modal Similarity", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Class‑wise similarity heatmap saved to {save_path}")


# ============================================================================
#  Core evaluation function
# ============================================================================

def test_model(model, test_loader):
    """
    Run inference on the test set and collect raw & projected features
    for later visualisation. No plots are generated inside this function.
    """
    model.eval()
    device = next(model.parameters()).device

    all_y = []
    all_raw = {'fbm': [], 'cnn': [], 'text': []}
    all_proj = {'fbm': [], 'cnn': [], 'text': []}
    predicted_labels, true_labels = [], []

    with torch.no_grad():
        for batch_sensor, batch_text, batch_y in tqdm(test_loader, desc="Testing"):
            batch_sensor = batch_sensor.to(device, non_blocking=True)
            batch_text = batch_text.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            (
                fbm_vec, cnn_vec, text_vec,
                fbm_proj, cnn_proj, text_proj,
                outputs, _
            ) = model(batch_sensor, batch_text, batch_y=None,
                      contrastive=False, return_x=True)

            _, predicted = torch.max(outputs, 1)
            predicted_labels.append(predicted.cpu())
            true_labels.append(batch_y.cpu())

            all_raw['fbm'].append(fbm_vec.cpu())
            all_raw['cnn'].append(cnn_vec.cpu())
            all_raw['text'].append(text_vec.cpu())
            all_proj['fbm'].append(fbm_proj.cpu())
            all_proj['cnn'].append(cnn_proj.cpu())
            all_proj['text'].append(text_proj.cpu())
            all_y.append(batch_y.cpu())

    y_all = torch.cat(all_y, dim=0)
    raw_fbm = torch.cat(all_raw['fbm'], dim=0)
    raw_cnn = torch.cat(all_raw['cnn'], dim=0)
    raw_text = torch.cat(all_raw['text'], dim=0)
    proj_fbm = torch.cat(all_proj['fbm'], dim=0)
    proj_cnn = torch.cat(all_proj['cnn'], dim=0)
    proj_text = torch.cat(all_proj['text'], dim=0)

    predicted_labels = torch.cat(predicted_labels, dim=0).numpy()
    true_labels = torch.cat(true_labels, dim=0).numpy()

    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    print(f"\nTest Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    recalls = recall_score(true_labels, predicted_labels, average=None)
    test_ba_accuracy = recalls.mean()
    print(f"Test Balanced Accuracy: {test_ba_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))

    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    return (
        acc, f1, predicted_labels,
        y_all, raw_fbm, raw_cnn, raw_text,
        proj_fbm, proj_cnn, proj_text
    )
