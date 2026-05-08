# data.py
# Data loading, windowing, and splitting for the SSTF framework.
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from clip_integration import convert_text_to_features, TextEmbedder
from gpt_text_generator import GPTTextGenerator
from text_utils import load_texts_from_csv
import os


def load_or_generate_texts(features, labels, data_path, args):
    
    base_path = os.path.splitext(data_path)[0]
    text_csv_path = f"{base_path}_window_texts.csv"

    # Use cached descriptions if available
    if args.use_text_cache and not args.force_regenerate_text and os.path.exists(text_csv_path):
        print(f"Loading cached window text descriptions from: {text_csv_path}")
        return load_texts_from_csv(text_csv_path)

    if args.force_regenerate_text:
        print("Force regeneration of text descriptions ...")
    else:
        print("Cache not found, generating text descriptions ...")

    # Use GPT-OSS offline generation
    if args.use_gpt:
        # Check that the GPT-OSS model path is provided
        if not hasattr(args, 'gpt_oss_model_path') or not args.gpt_oss_model_path:
            raise ValueError("GPT-OSS model path is required when use_gpt=True")
        print(f"Using offline GPT-OSS mode to generate window-level text descriptions ...")
    else:
        print("Using rule-based text generation ...")

    generator = GPTTextGenerator(args)
    text_descriptions = generator.generate_texts_for_windows_batch(
        features, labels, args.window_size, args.stride,
        args.data_batch_size, text_csv_path
    )
    return text_descriptions


def load_data(file_path, window_size, stride, clip_model, device, text_embedder, args,
              raw_sensor_only=False, skip_text=False):
    
    print("=== Start data loading and processing ===")

    # 1. Read raw data
    print("1. Reading sensor data ...")
    data = pd.read_csv(file_path)
    features = data[["x", "y", "z"]].values
    labels = data["classification"].values
    cow_ids = data["cow_id"].values

    if skip_text:
        print("2. Skipping text processing (model does not use text features) ...")
        print("3. Windowing ...")
        sensor_seqs, text_feats, labels, cow_ids = window_sampling_raw_no_text(
            features, labels, cow_ids, window_size, stride, device
        )
    else:
        # 2. Generate or load text descriptions
        print("2. Loading or generating text descriptions ...")
        text_descriptions = load_or_generate_texts(features, labels, file_path, args)

        # 3. Encode text with frozen CLIP
        print("3. Encoding text with frozen CLIP ...")
        text_features = convert_text_to_features(text_descriptions, clip_model, device,
                                                 batch_size=args.data_batch_size)
        text_embedder = text_embedder.to(device)
        text_features = text_features.to(torch.float32).to(device)

        # 4. Windowing and project text features
        print("4. Windowing ...")
        sensor_seqs, text_feats, labels, cow_ids = window_sampling_raw(
            features, labels, cow_ids, text_features, window_size, stride,
            device, text_embedder
        )

    print("=== Data loading and processing completed ===")
    return sensor_seqs, text_feats, labels, cow_ids


def window_sampling_raw(features, labels, cow_ids, text_features, window_size, stride,
                        device, text_embedder=None):
    
    sensor_seqs = []
    text_feats = []
    labels_out = []
    cow_ids_out = []

    features = torch.tensor(features, dtype=torch.float32)
    if len(text_features) == (len(features) - window_size) // stride + 1:
        text_features = torch.tensor(text_features, dtype=torch.float32).to(device)
        print(f"Using window-level text features, total {len(text_features)} windows")

        if text_embedder is not None:
            print(f"Using TextEmbedder to reduce text features from "
                  f"{text_features.shape[-1]}D to {text_embedder.fc.out_features}D")
            with torch.no_grad():
                text_features = text_embedder(text_features)  # [num_windows, text_feat_dim]

    window_idx = 0
    for index in range(0, len(features) - window_size + 1, stride):
        window_features = features[index:index + window_size].clone().detach()
        window_labels = labels[index:index + window_size]
        window_cow_id = cow_ids[index + window_size // 2]

        # Majority vote for window label
        majority_label = Counter(window_labels).most_common(1)[0][0]

        sensor_seqs.append(window_features)
        text_feats.append(text_features[window_idx])
        labels_out.append(majority_label)
        cow_ids_out.append(window_cow_id)
        window_idx += 1

    sensor_seqs = torch.stack(sensor_seqs)          # [N, T, 3]
    text_feats = torch.stack(text_feats)            # [N, D_text]
    labels_out = torch.tensor(labels_out, dtype=torch.long)
    cow_ids_out = torch.tensor(cow_ids_out, dtype=torch.long)
    return sensor_seqs, text_feats, labels_out, cow_ids_out


def window_sampling_raw_no_text(features, labels, cow_ids, window_size, stride, device):
    """
    Window raw sensor data without any text features. Useful for
    unimodal (sensor-only) variants such as TFM or LST.

    Returns:
        sensor_seqs, empty text_feats, labels, cow_ids
    """
    sensor_seqs = []
    labels_out = []
    cow_ids_out = []

    features = torch.tensor(features, dtype=torch.float32)
    for index in range(0, len(features) - window_size + 1, stride):
        window_features = features[index:index + window_size].clone().detach()
        window_labels = labels[index:index + window_size]
        window_cow_id = cow_ids[index + window_size // 2]

        majority_label = Counter(window_labels).most_common(1)[0][0]

        sensor_seqs.append(window_features)
        labels_out.append(majority_label)
        cow_ids_out.append(window_cow_id)

    sensor_seqs = torch.stack(sensor_seqs)
    num_windows = len(sensor_seqs)
    text_feats = torch.zeros(num_windows, 1, device=device)  # dummy text features
    labels_out = torch.tensor(labels_out, dtype=torch.long)
    cow_ids_out = torch.tensor(cow_ids_out, dtype=torch.long)
    return sensor_seqs, text_feats, labels_out, cow_ids_out


def preprocess_data_by_cow(sensor_seqs, text_feats, labels, cow_ids, device):

    unique_cow_ids = np.unique(cow_ids)
    num_cows = len(unique_cow_ids)
    print(f"Total cows: {num_cows}")
    print(f"Cow IDs: {unique_cow_ids}")

    np.random.seed(42)
    num_test_cows = max(1, int(num_cows * 0.2))
    num_val_cows = max(1, int(num_cows * 0.2))

    test_cow_ids = np.random.choice(unique_cow_ids, num_test_cows, replace=False)
    remaining = np.setdiff1d(unique_cow_ids, test_cow_ids)
    val_cow_ids = np.random.choice(remaining, num_val_cows, replace=False)
    train_cow_ids = np.setdiff1d(remaining, val_cow_ids)

    print(f"Train cows: {train_cow_ids} ({len(train_cow_ids)} cows)")
    print(f"Val cows:   {val_cow_ids} ({len(val_cow_ids)} cows)")
    print(f"Test cows:  {test_cow_ids} ({len(test_cow_ids)} cows)")

    train_mask = np.isin(cow_ids, train_cow_ids)
    val_mask = np.isin(cow_ids, val_cow_ids)
    test_mask = np.isin(cow_ids, test_cow_ids)

    sensor_np = sensor_seqs.cpu().numpy()
    text_np = text_feats.cpu().numpy()
    labels_np = labels.cpu().numpy()

    X_train_sensor = sensor_np[train_mask]
    X_val_sensor = sensor_np[val_mask]
    X_test_sensor = sensor_np[test_mask]
    X_train_text = text_np[train_mask]
    X_val_text = text_np[val_mask]
    X_test_text = text_np[test_mask]
    y_train = labels_np[train_mask]
    y_val = labels_np[val_mask]
    y_test = labels_np[test_mask]

    print(f"Train windows: {len(X_train_sensor)}")
    print(f"Val windows:   {len(X_val_sensor)}")
    print(f"Test windows:  {len(X_test_sensor)}")

    # Standardize using training statistics
    scaler = StandardScaler()
    num_channels = X_train_sensor.shape[-1]
    X_train_sensor = scaler.fit_transform(X_train_sensor.reshape(-1, num_channels)) \
                         .reshape(X_train_sensor.shape)
    X_val_sensor = scaler.transform(X_val_sensor.reshape(-1, num_channels)) \
                       .reshape(X_val_sensor.shape)
    X_test_sensor = scaler.transform(X_test_sensor.reshape(-1, num_channels)) \
                        .reshape(X_test_sensor.shape)

    X_train_sensor = torch.tensor(X_train_sensor, dtype=torch.float32)
    X_val_sensor = torch.tensor(X_val_sensor, dtype=torch.float32)
    X_test_sensor = torch.tensor(X_test_sensor, dtype=torch.float32)
    X_train_text = torch.tensor(X_train_text, dtype=torch.float32)
    X_val_text = torch.tensor(X_val_text, dtype=torch.float32)
    X_test_text = torch.tensor(X_test_text, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return (X_train_sensor, X_train_text, y_train), \
           (X_val_sensor, X_val_text, y_val), \
           (X_test_sensor, X_test_text, y_test)


def preprocess_data_random(sensor_seqs, text_feats, labels, device):
    """Random stratified split (fallback when cow-level split is not possible)."""
    print("Using random split ...")
    sensor_np = sensor_seqs.cpu().numpy()
    text_np = text_feats.cpu().numpy()
    labels_np = labels.cpu().numpy()

    X_temp_sensor, X_test_sensor, X_temp_text, X_test_text, y_temp, y_test = train_test_split(
        sensor_np, text_np, labels_np, test_size=0.2, random_state=42, stratify=labels_np
    )
    X_train_sensor, X_val_sensor, X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_temp_sensor, X_temp_text, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    num_channels = X_train_sensor.shape[-1]
    X_train_sensor = scaler.fit_transform(X_train_sensor.reshape(-1, num_channels)) \
                         .reshape(X_train_sensor.shape)
    X_val_sensor = scaler.transform(X_val_sensor.reshape(-1, num_channels)) \
                       .reshape(X_val_sensor.shape)
    X_test_sensor = scaler.transform(X_test_sensor.reshape(-1, num_channels)) \
                        .reshape(X_test_sensor.shape)

    X_train_sensor = torch.tensor(X_train_sensor, dtype=torch.float32)
    X_val_sensor = torch.tensor(X_val_sensor, dtype=torch.float32)
    X_test_sensor = torch.tensor(X_test_sensor, dtype=torch.float32)
    X_train_text = torch.tensor(X_train_text, dtype=torch.float32)
    X_val_text = torch.tensor(X_val_text, dtype=torch.float32)
    X_test_text = torch.tensor(X_test_text, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return (X_train_sensor, X_train_text, y_train), \
           (X_val_sensor, X_val_text, y_val), \
           (X_test_sensor, X_test_text, y_test)


def preprocess_data(sensor_seqs, text_feats, labels, device, cow_ids=None, use_cow_split=False):

    if use_cow_split and cow_ids is not None:
        unique_cows = np.unique(cow_ids)
        num_cows = len(unique_cows)
        if num_cows >= 3:
            print(f"Using cow-level split ... ({num_cows} cows)")
            return preprocess_data_by_cow(sensor_seqs, text_feats, labels, cow_ids, device)
        else:
            print(f"Not enough cows ({num_cows}), falling back to random split ...")
    print("Using random split ...")
    return preprocess_data_random(sensor_seqs, text_feats, labels, device)


def create_dataloader(train_data, val_data, test_data, batch_size, device):
    """
    Create PyTorch DataLoaders for training/validation/test.
    """
    X_train_sensor, X_train_text, y_train = train_data
    X_val_sensor, X_val_text, y_val = val_data
    X_test_sensor, X_test_text, y_test = test_data

    train_dataset = TensorDataset(X_train_sensor, X_train_text, y_train)
    val_dataset = TensorDataset(X_val_sensor, X_val_text, y_val)
    test_dataset = TensorDataset(X_test_sensor, X_test_text, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    return train_loader, val_loader, test_loader