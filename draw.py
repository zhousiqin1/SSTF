# draw.py
# Visualization utilities for the SSTF framework.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib as mpl

# === Global publication-ready style ===
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff"
})
sns.set_style("whitegrid", {'axes.facecolor': '#ffffff'})


def plot_confusion(labels, predicted, save_dir="./fig", experiment_name=""):
    """Plot normalized confusion matrix."""
    cm = confusion_matrix(labels, predicted, normalize='true')
    class_names = ['Other', 'Rumination', 'Eating']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cbar=True, cmap='Blues', fmt='.2f',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    if experiment_name:
        ax.set_title(f'Confusion Matrix - {experiment_name}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

    ax.set_xlim(0, len(class_names))
    ax.set_ylim(len(class_names), 0)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.85, wspace=0.15, hspace=0.15)
    plt.tight_layout()

    filename = f"confusion_{experiment_name}.png" if experiment_name else "confusion.png"
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_raw(labels, predicted, save_dir="./fig", experiment_name=""):
    """Plot confusion matrix with raw counts."""
    cm = confusion_matrix(labels, predicted)
    class_names = ['Other', 'Rumination', 'Eating']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cbar=True, cmap='Blues', fmt='d',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    title_text = f'Confusion Matrix (Raw Counts) - {experiment_name}' if experiment_name else 'Confusion Matrix (Raw Counts)'
    ax.set_title(title_text, fontsize=14, fontweight='bold')

    ax.set_xlim(0, len(class_names))
    ax.set_ylim(len(class_names), 0)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.85, wspace=0.15, hspace=0.15)
    plt.tight_layout()

    filename = f"confusion_raw_{experiment_name}.png" if experiment_name else "confusion_raw.png"
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()


def drawScatter(ds, names, save_dir="./fig", experiment_name=""):
    """Scatter plot of predicted vs true labels."""
    markers = ["x", "o"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(ds[0]))
    for d, name, marker in zip(ds, names, markers):
        ax.scatter(x, d, alpha=0.3, label=name, marker=marker)

    ax.legend(fontsize=16, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Class Label')
    if experiment_name:
        ax.set_title(f'Predictions vs True Labels - {experiment_name}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Predictions vs True Labels', fontsize=14, fontweight='bold')

    filename = f"predictions_{experiment_name}.png" if experiment_name else "pre.png"
    plt.savefig(f"{save_dir}/{filename}", dpi=600, bbox_inches='tight')
    plt.close()


def plot_train_val_loss(train_losses, val_losses, save_dir="./fig"):
    """Plot training and validation loss."""
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)

    plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    min_val_loss = min(val_losses)
    min_val_epoch = val_losses.index(min_val_loss) + 1
    plt.axvline(x=min_val_epoch, color='g', linestyle='--', alpha=0.7,
                label=f'Best Val Loss (Epoch {min_val_epoch})')
    plt.axhline(y=min_val_loss, color='g', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/train_val_loss.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_train_val_accuracy(train_accs, val_accs, save_dir="./fig"):
    """Plot training and validation accuracy."""
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_accs) + 1)

    plt.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    plt.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)

    plt.title('Training vs Validation Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    max_val_acc = max(val_accs)
    max_val_epoch = val_accs.index(max_val_acc) + 1
    plt.axvline(x=max_val_epoch, color='g', linestyle='--', alpha=0.7,
                label=f'Best Val Acc (Epoch {max_val_epoch})')
    plt.axhline(y=max_val_acc, color='g', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/train_val_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_dir="./fig", experiment_name=""):
    """Plot combined training curves (loss and accuracy)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # Accuracy
    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    if experiment_name:
        fig.suptitle(f'Training Curves - {experiment_name}', fontsize=16, fontweight='bold')

    plt.tight_layout()
    filename = f"training_curves_{experiment_name}.png" if experiment_name else "training_curves.png"
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_performance_summary(train_losses, train_accs, val_losses, val_accs,
                                   test_acc, predicted, labels, save_dir="./fig"):
    """Create a multi-panel model performance summary."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    epochs = range(1, len(train_losses) + 1)

    # 1. Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # 2. Training accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_accs, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)

    # 3. Validation loss
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, val_losses, 'r-', linewidth=2, marker='s', markersize=4)
    ax3.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)

    # 4. Validation accuracy
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, val_accs, 'orange', linewidth=2, marker='s', markersize=4)
    ax4.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True, alpha=0.3)

    # 5. Test accuracy
    ax5 = fig.add_subplot(gs[1, 1])
    if isinstance(test_acc, list):
        test_acc_value = test_acc[0] if test_acc else 0.0
    else:
        test_acc_value = test_acc
    ax5.bar(['Test Accuracy'], [test_acc_value], color='purple', alpha=0.7)
    ax5.set_title('Test Accuracy', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Accuracy')
    ax5.set_ylim(0, 1)
    ax5.text(0, test_acc_value + 0.01, f'{test_acc_value:.4f}', ha='center', va='bottom', fontweight='bold')

    # 6. Confusion matrix
    ax6 = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(labels, predicted, normalize='true')
    class_names = ['Other', 'Rumination', 'Eating']
    sns.heatmap(cm, annot=True, cbar=False, cmap='Blues', fmt='.2f',
                xticklabels=class_names, yticklabels=class_names, ax=ax6)
    ax6.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('True')

    plt.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(f"{save_dir}/model_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()