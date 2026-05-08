# main.py
import torch
import torch.nn as nn
from config import *
from model import *
from train import *
from test import *
from early_stopping import EarlyStopping
from data import *
from draw import *
from clip_integration import load_clip_model
from set_random_seed import set_reproducible_training
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

if __name__ == '__main__':

    args = parse_args()

    set_reproducible_training()


    if args.use_gpu:
        device_list = args.devices.split(',')
        device_ids = [int(d) for d in device_list]
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device = torch.device('cpu')
        device_ids = []


    clip_model, preprocess, text_embedder = load_clip_model(
        device, text_input_size=512, output_dim=args.text_feat_dim
    )
    clip_model = clip_model.to(device)
    text_embedder = text_embedder.to(device)

    if args.model in ['HybridFBM_Text_LSTM', 'HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive']:
        sensor_seqs, text_feats, labels, cow_ids = load_data(
            args.data_path, args.window_size, args.stride,
            clip_model, device, text_embedder, args,
            raw_sensor_only=True
        )
    elif args.model in ['HybridFBM_LSTM']:
        sensor_seqs, text_feats, labels, cow_ids = load_data(
            args.data_path, args.window_size, args.stride,
            clip_model, device, text_embedder, args,
            raw_sensor_only=True, skip_text=True
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")


    train_data, val_data, test_data = preprocess_data(
        sensor_seqs, text_feats, labels, device,
        cow_ids=cow_ids, use_cow_split=args.use_cow_split
    )
    print(train_data[0].shape, train_data[2].shape)
    print(val_data[0].shape, val_data[2].shape)
    print(test_data[0].shape, test_data[2].shape)


    print("\n" + "=" * 60)
    print("=" * 60)

    label_names = {0: 'Other', 1: 'Rumination', 2: 'Eating'}
    train_labels = train_data[2].numpy()
    val_labels = val_data[2].numpy()
    test_labels = test_data[2].numpy()

    print(f"{'Behavior Category':<20} {'Training (60%)':<15} {'Validation (20%)':<15} {'Test (20%)':<15}")
    print("-" * 65)
    for label_id, name in label_names.items():
        train_cnt = (train_labels == label_id).sum()
        val_cnt = (val_labels == label_id).sum()
        test_cnt = (test_labels == label_id).sum()
        print(f"{name:<20} {train_cnt:<15} {val_cnt:<15} {test_cnt:<15}")

    train_total = len(train_labels)
    val_total = len(val_labels)
    test_total = len(test_labels)
    print("-" * 65)
    print(f"{'Total':<20} {train_total:<15} {val_total:<15} {test_total:<15}")
    print("=" * 60 + "\n")

    train_loader, val_loader, test_loader = create_dataloader(
        train_data, val_data, test_data, args.batch_size, device
    )

    try:
        print(f">>>>>  Initializing {args.model} model  <<<<<")
        if args.model == 'HybridFBM_LSTM':
            model = HybridFBM_LSTM(
                raw_input_size=args.raw_input_size,
                lstm_hidden=args.lstm_hidden,
                mlp_dim=args.mlp_dim,
                output_size=args.output_size,
                seq_len=args.fbm_seq_len,
                dropout=args.dropout,
                fbm_block_size=args.fbm_block_size,
                fbm_hidden_dim=args.fbm_hidden_dim,
                attention_heads=args.attention_heads
            )
        elif args.model == 'HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive':
            model = HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive(
                raw_input_size=args.raw_input_size,
                lstm_hidden=args.lstm_hidden,
                mlp_dim=args.mlp_dim,
                output_size=args.output_size,
                text_feat_dim=args.text_feat_dim,
                seq_len=args.fbm_seq_len,
                dropout=args.dropout,
                fbm_block_size=args.fbm_block_size,
                fbm_hidden_dim=args.fbm_hidden_dim,
                attention_heads=args.attention_heads,
                cnn_hidden_dim=args.cnn_hidden_dim,
                conv_time_kernel=getattr(args, 'conv_time_kernel', 31)
            )
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        if args.use_gpu and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
        else:
            model = model.to(device)
        print(model)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters of {args.model}: {total_params}")
        print(f">>>>>  {args.model} model initialized successfully.  <<<<<")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        exit(1)


    print(f">>>>>  Starting {args.model} training and validation  <<<<<")

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 2e-4)
    )

    scheduler_type = "warmup_cosine"
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5,
            threshold=1e-3, cooldown=1, verbose=True, min_lr=1e-6
        )
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif scheduler_type == "warmup_cosine":
        warmup_epochs = max(1, min(10, args.epochs // 10))
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=8,
            threshold=1e-3, cooldown=3, verbose=True, min_lr=1e-6
        )

    early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.005)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    val_inference_times = []
    val_inference_stats = []

    for epoch in range(args.epochs):
        train_losses, train_accs = train_model(
            model, loss_function, train_loader, optimizer,
            train_losses, train_accs, device
        )
        val_losses, val_accs, inference_stats = val_model(
            model, loss_function, val_loader, val_losses, val_accs, device
        )
        val_inference_times.append(inference_stats['avg_batch_time'])
        val_inference_stats.append(inference_stats)

        print(
            f'Epoch {epoch + 1}/{args.epochs}, '
            f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, '
            f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}'
        )
        print(
            f'Inference time - Batch: {inference_stats["avg_batch_time"]:.4f}s, '
            f'Per sample: {inference_stats["single_sample_time"]:.6f}s, '
            f'Total: {inference_stats["total_time"]:.4f}s'
        )

        if scheduler_type == "plateau":
            scheduler.step(val_losses[-1])
        else:
            scheduler.step()
        print("Learning rate:", optimizer.param_groups[0]['lr'])

        if early_stopping(val_losses[-1]):
            print("Early stopping")
            break


    experiment_dir = args.save_dir
    os.makedirs(f"{experiment_dir}/models", exist_ok=True)
    os.makedirs(f"{experiment_dir}/fig", exist_ok=True)

    print(f">>>>>  Testing {args.model} model  <<<<<")
    (
        test_acc, test_f1, predicted,
        y_all, fbm_raw, cnn_raw, text_raw,
        fbm_proj, cnn_proj, text_proj
    ) = test_model(model, test_loader)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")


    print("Extracting dynamic fusion features for visualization ...")
    fused_all = []
    model.eval()
    with torch.no_grad():
        for batch_sensor, batch_text, _ in tqdm(test_loader, desc="Extracting fusion features"):
            batch_sensor = batch_sensor.to(device, non_blocking=True)
            batch_text = batch_text.to(device, non_blocking=True)
            fused = model(batch_sensor, batch_text, return_x="fused")
            fused_all.append(fused.cpu())
    fused_all = torch.cat(fused_all, dim=0)

    labels = test_data[2].cpu().numpy()

    from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
    classification_rep = classification_report(labels, predicted, zero_division=0)
    precision = precision_score(labels, predicted, average='weighted', zero_division=0)
    recall = recall_score(labels, predicted, average='weighted', zero_division=0)
    f1 = f1_score(labels, predicted, average='weighted', zero_division=0)

    print(f"=== Classification Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if val_inference_stats:
        print(f"FLOPs (per sample): {val_inference_stats[-1].get('flops_G', 0):.4f} G")

    print(f"\n=== Detailed Classification Report ===")
    print(classification_rep)


    with open(f"{experiment_dir}/{args.experiment_name}_results.txt", 'w', encoding='utf-8') as f:
        f.write(f"Experiment Name: {args.experiment_name}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Input Dim: {args.raw_input_size}\n")
        f.write(f"LSTM Hidden: {args.lstm_hidden}\n")
        f.write(f"MLP Dim: {args.mlp_dim}\n")
        f.write(f"FBM Seq Len: {args.fbm_seq_len}\n")
        f.write(f"FBM Hidden Dim: {args.fbm_hidden_dim}\n")
        f.write(f"\n=== Training Results ===\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")

        last_stats = val_inference_stats[-1] if val_inference_stats else {}
        flops_val = last_stats.get('flops_G', 0.0)
        params_val = last_stats.get('params_M', 0.0)
        f.write(f"\n=== Model Complexity ===\n")
        f.write(f"Params: {params_val:.2f} M\n")
        f.write(f"FLOPs/sample: {flops_val:.4f} GFLOPs\n")

        f.write(f"Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Train Acc: {train_accs[-1]:.4f}\n")
        f.write(f"Final Val Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final Val Acc: {val_accs[-1]:.4f}\n")
        f.write(f"Best Val Acc: {max(val_accs):.4f} (Epoch {val_accs.index(max(val_accs)) + 1})\n")
        f.write(f"Actual Epochs: {len(train_losses)}\n")
        f.write(f"Early Stopped: {'Yes' if len(train_losses) < args.epochs else 'No'}\n")

        final_inference_time = val_inference_times[-1] if val_inference_times else 0.0
        best_epoch_idx = val_accs.index(max(val_accs))
        best_inference_time = val_inference_times[best_epoch_idx] if best_epoch_idx < len(val_inference_times) else 0.0
        avg_inference_time = sum(val_inference_times) / len(val_inference_times) if val_inference_times else 0.0

        single_sample_times = [s['single_sample_time'] for s in val_inference_stats if 'single_sample_time' in s]
        avg_single_sample_time = sum(single_sample_times) / len(single_sample_times) if single_sample_times else 0.0

        f.write(f"\n=== Inference Time ===\n")
        f.write(f"Batch Size: {val_loader.batch_size}\n")
        f.write(f"Final Batch Time: {final_inference_time:.4f}s\n")
        f.write(f"Final Per-Sample Time: {val_inference_stats[-1]['single_sample_time']:.6f}s\n")
        f.write(f"Best Epoch Batch Time: {best_inference_time:.4f}s (Epoch {best_epoch_idx + 1})\n")
        f.write(f"Best Epoch Per-Sample: {val_inference_stats[best_epoch_idx]['single_sample_time']:.6f}s\n")
        f.write(f"Avg Batch Time: {avg_inference_time:.4f}s\n")
        f.write(f"Avg Per-Sample Time: {avg_single_sample_time:.6f}s\n")
        f.write(f"Min Batch Time: {min(val_inference_times):.4f}s\n")
        f.write(f"Max Batch Time: {max(val_inference_times):.4f}s\n")
        f.write(f"Min Per-Sample Time: {min(single_sample_times):.6f}s\n")
        f.write(f"Max Per-Sample Time: {max(single_sample_times):.6f}s\n")

        f.write(f"\n=== Classification Metrics ===\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

        try:
            from sklearn.metrics import balanced_accuracy_score
            balanced_acc = balanced_accuracy_score(labels, predicted)
            f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
            print(f"[Extra Metrics] Balanced Accuracy={balanced_acc:.4f}")
        except Exception as e:
            print(f"[Extra Metrics - Balanced Accuracy skipped] {e}")

        try:
            from sklearn.metrics import cohen_kappa_score, confusion_matrix
            import numpy as np
            cm = confusion_matrix(labels, predicted)
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)

            specificity_per_class = TN / (TN + FP + 1e-8)
            specificity = np.mean(specificity_per_class)
            recall_macro = recall_score(labels, predicted, average='macro', zero_division=0)
            gmean = np.sqrt(recall_macro * specificity)
            kappa = cohen_kappa_score(labels, predicted)

            f.write(f"Cohen’s Kappa: {kappa:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write(f"G-Mean: {gmean:.4f}\n")
            print(f"[Extra Metrics] Kappa={kappa:.4f}, Specificity={specificity:.4f}, GMean={gmean:.4f}")
        except Exception as e:
            print(f"[Extra Metrics (skipped)] {e}")

        f.write(f"\n=== Detailed Classification Report ===\n")
        f.write(classification_rep)

    print(f"Results saved to: {experiment_dir}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Validation Accuracy: {max(val_accs):.4f}")

    print(f"\n=== Inference Time Summary ===")
    print(f"Batch Size: {val_loader.batch_size}")
    print(f"Final Batch Time: {final_inference_time:.4f}s")
    print(f"Final Per-Sample Time: {val_inference_stats[-1]['single_sample_time']:.6f}s")
    print(f"Best Epoch Batch Time: {best_inference_time:.4f}s (Epoch {best_epoch_idx + 1})")
    print(f"Best Epoch Per-Sample Time: {val_inference_stats[best_epoch_idx]['single_sample_time']:.6f}s")
    print(f"Avg Batch Time: {avg_inference_time:.4f}s")
    print(f"Avg Per-Sample Time: {avg_single_sample_time:.6f}s")
    print(f"Min Per-Sample Time: {min(single_sample_times):.6f}s")
    print(f"Max Per-Sample Time: {max(single_sample_times):.6f}s")

    model_path = f"{experiment_dir}/models/{args.experiment_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    from draw import plot_training_curves, plot_train_val_loss, plot_train_val_accuracy
    plot_training_curves(train_losses, train_accs, val_losses, val_accs,
                         f"{experiment_dir}/fig", args.experiment_name)
    plot_train_val_loss(train_losses, val_losses, f"{experiment_dir}/fig")
    plot_train_val_accuracy(train_accs, val_accs, f"{experiment_dir}/fig")

    drawScatter([labels, predicted], ['true', 'pred'], f"{experiment_dir}/fig", args.experiment_name)
    plot_confusion(labels, predicted, f"{experiment_dir}/fig", args.experiment_name)
    plot_confusion_raw(labels, predicted, f"{experiment_dir}/fig", args.experiment_name)

    from draw import plot_model_performance_summary
    plot_model_performance_summary(
        train_losses, train_accs, val_losses, val_accs,
        test_acc, predicted, labels, f"{experiment_dir}/fig"
    )

    print("All figures saved!")