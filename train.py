# train.py
# Training and validation loops for the SSTF framework.
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    cohen_kappa_score, confusion_matrix
)

try:
    from thop import profile
except ImportError:
    profile = None


def train_model(model, loss_function, train_loader, optimizer, train_losses, train_accs, device):
    """Run one epoch of training."""
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    lambda_contrastive = getattr(model, "contrastive_weight", 0.3)

    for batch_sensor, batch_text, batch_y in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        batch_sensor = batch_sensor.to(device, non_blocking=True)
        batch_text = batch_text.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        # Training: compute classification + contrastive loss
        outputs, contrastive_loss_value = model(
            batch_sensor, batch_text, batch_y=batch_y,
            contrastive=True, return_x=False
        )

        cls_loss = loss_function(outputs, batch_y)
        total_loss = cls_loss + lambda_contrastive * contrastive_loss_value

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += total_loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == batch_y).sum().item()
        train_total += batch_y.size(0)

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total
    train_losses.append(train_loss)
    train_accs.append(train_accuracy)

    return train_losses, train_accs


def val_model(model, loss_function, val_loader, val_losses, val_accs, device):
    """Run validation and compute FLOPs once."""
    model.eval()
    val_loss = 0.0
    predicted_labels = []
    true_labels = []

    val_correct = 0
    val_total = 0

    total_inference_time = 0.0
    inference_times = []

    # ---- FLOPs calculation (run once) ----
    flops_g = 0.0
    params_m = 0.0

    if profile is not None and len(val_loader) > 0:
        try:
            dummy_sensor, dummy_text, dummy_y = next(iter(val_loader))
            dummy_sensor = dummy_sensor.to(device)
            dummy_text = dummy_text.to(device)
            dummy_y = dummy_y.to(device)

            # Strip DataParallel wrapper if present
            model_to_profile = model
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                model_to_profile = model.module

            class InferenceWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, sensor, text, y):
                    return self.model(
                        sensor, text, batch_y=y,
                        contrastive=False, return_x=False
                    )

            wrapper_model = InferenceWrapper(model_to_profile)
            macs, params = profile(wrapper_model, inputs=(dummy_sensor, dummy_text, dummy_y), verbose=False)

            batch_size_dummy = dummy_sensor.shape[0]
            flops_g = (macs * 2) / 1e9 / batch_size_dummy
            params_m = params / 1e6

            print(f"[Debug] Raw MACs: {macs}, Batch: {batch_size_dummy}, FLOPs: {flops_g:.4f} G")
        except Exception as e:
            print(f"[Warning] FLOPs calculation failed: {e}")

    # ---- Validation loop ----
    with torch.no_grad():
        for batch_sensor, batch_text, batch_y in tqdm(val_loader, desc="Validation"):
            torch.cuda.synchronize()
            start_time = time.time()

            batch_sensor = batch_sensor.to(device)
            batch_text = batch_text.to(device)
            batch_y = batch_y.to(device)

            # Validation: classification output only (no projections, no contrastive loss)
            outputs = model(
                batch_sensor, batch_text, batch_y=batch_y,
                contrastive=False, return_x=False
            )

            torch.cuda.synchronize()
            end_time = time.time()

            inference_times.append(end_time - start_time)
            total_inference_time += end_time - start_time

            val_loss += loss_function(outputs, batch_y).item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_y).sum().item()
            val_total += batch_y.size(0)

            predicted_labels.append(predicted.cpu())
            true_labels.append(batch_y.cpu())

    predicted_labels = torch.cat(predicted_labels, dim=0).numpy()
    true_labels = torch.cat(true_labels, dim=0).numpy()

    # ---- Metrics ----
    val_loss /= len(val_loader)
    accuracy = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    # ---- Additional metrics ----
    try:
        kappa = cohen_kappa_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        specificity_per_class = TN / (TN + FP + 1e-8)
        specificity = np.mean(specificity_per_class)
        recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        gmean = np.sqrt(recall_macro * specificity)
        print(f"[Extra Metrics] Kappa={kappa:.4f}, Specificity={specificity:.4f}, GMean={gmean:.4f}")
    except Exception as e:
        print(f"[Extra Metrics skipped] {e}")

    val_accs.append(accuracy)
    val_losses.append(val_loss)

    avg_inference_time = total_inference_time / len(val_loader)
    single_sample_time = avg_inference_time / val_loader.batch_size

    inference_stats = {
        'avg_batch_time': avg_inference_time,
        'single_sample_time': single_sample_time,
        'total_time': total_inference_time,
        'min_batch_time': min(inference_times),
        'max_batch_time': max(inference_times),
        'batch_size': val_loader.batch_size,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'flops_G': flops_g,
        'params_M': params_m
    }

    return val_losses, val_accs, inference_stats