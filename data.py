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
from torch.utils.data import DataLoader, TensorDataset


def load_or_generate_texts(features, labels, data_path, args):
    """
    加载或生成文本描述
    
    Args:
        features: 传感器特征
        labels: 标签
        data_path: 数据文件路径
        args: 配置参数对象
    Returns:
        文本描述列表（每个窗口一个描述）
    """
    # 构建文本文件路径
    base_path = os.path.splitext(data_path)[0]
    text_csv_path = f"{base_path}_window_texts.csv"
    
    # 检查是否使用缓存
    if args.use_text_cache and not args.force_regenerate_text and os.path.exists(text_csv_path):
        print(f"加载已存在的窗口文本描述CSV文件: {text_csv_path}")
        return load_texts_from_csv(text_csv_path)
    
    # 强制重新生成或缓存文件不存在
    if args.force_regenerate_text:
        print("强制重新生成文本描述...")
    else:
        print("缓存文件不存在，开始生成文本描述...")
    
    # 生成窗口级文本
    if args.use_gpt:
        # GPT模式
        if not args.openai_api_key:
            print("GPT模式需要API密钥，将切换到普通模式")
            args.use_gpt = False
        else:
            print(f"使用GPT模式生成窗口级文本描述...")
    else:
        # 普通模式
        print("使用普通规则模式生成窗口级文本描述...")
    
    generator = GPTTextGenerator(args)
    # 使用分批生成方法
    text_descriptions = generator.generate_texts_for_windows_batch(features, labels, args.window_size, args.stride, args.data_batch_size, text_csv_path)
    
    return text_descriptions

def load_data(file_path, window_size, stride, clip_model, device, text_embedder, args, raw_sensor_only=False, skip_text=False):
    """
    数据加载函数，支持原始传感器数据
    raw_sensor_only: 如果为True，只返回原始x,y,z数据
    skip_text: 如果为True，跳过文本处理，返回空的文本特征
    """
    print("=== 开始数据加载和处理 ===")

    # 1. 读取原始数据
    print("1. 读取传感器数据...")
    data = pd.read_csv(file_path)
    features = data[["x", "y", "z"]].values
    labels = data["classification"].values
    cow_ids = data["cow_id"].values  # 获取牛ID

    if skip_text:
        # 跳过文本处理，直接进行窗口化
        print("2. 跳过文本处理（模型不使用文本特征）...")
        print("3. 窗口化处理...")
        sensor_seqs, text_feats, labels, cow_ids = window_sampling_raw_no_text(features, labels, cow_ids, window_size, stride, device)
    else:
        # 2. 使用现有的文本生成逻辑
        print("2. 加载或生成文本描述...")
        text_descriptions = load_or_generate_texts(features, labels, file_path, args)

        # 3. 文本特征提取
        print("3. 使用CLIP模型提取文本特征...")
        text_features = convert_text_to_features(text_descriptions, clip_model, device, batch_size=args.data_batch_size)
        text_embedder = text_embedder.to(device)
        text_features = text_features.to(torch.float32).to(device)
        text_features = text_features.to(device)

        # 4. 窗口化处理
        print("4. 窗口化处理...")
        # 返回原始传感器数据
        sensor_seqs, text_feats, labels, cow_ids = window_sampling_raw(features, labels, cow_ids, text_features, window_size, stride, device, text_embedder)

    print("=== 数据加载和处理完成 ===")
    return sensor_seqs, text_feats, labels, cow_ids


def window_sampling_raw(features, labels, cow_ids, text_features, window_size, stride, device, text_embedder=None):
    """
    原始传感器数据窗口采样
    返回：
      - sensor_seqs: [num_windows, window_size, 3] - 原始x,y,z数据
      - text_feats:  [num_windows, text_feat_dim]
      - labels:      [num_windows]
      - cow_ids:     [num_windows] - 每个窗口对应的牛ID
    """
    sensor_seqs = []
    text_feats = []
    labels_out = []
    cow_ids_out = []

    # features = torch.tensor(features, dtype=torch.float32).to(device)
    features = torch.tensor(features, dtype=torch.float32)
    if len(text_features) == (len(features) - window_size) // stride + 1:
        text_features = torch.tensor(text_features, dtype=torch.float32).to(device)
        print(f"使用窗口级文本特征，共 {len(text_features)} 个窗口")
        
        # 使用TextEmbedder将512维CLIP特征降维到指定维度
        if text_embedder is not None:
            print(f"使用TextEmbedder将文本特征从{text_features.shape[-1]}维降维到{text_embedder.fc.out_features}维")
            with torch.no_grad():
                text_features = text_embedder(text_features)  # [num_windows, text_feat_dim]

    window_idx = 0
    num_windows = (len(features) - window_size) // stride + 1
    for index in range(0, len(features) - window_size + 1, stride):
        # 直接使用原始传感器数据
        # window_features = features[index:index + window_size].clone().detach().to(device)
        window_features = features[index:index + window_size].clone().detach()
        window_labels = labels[index:index + window_size]
        window_cow_id = cow_ids[index + window_size // 2]  # 使用窗口中间位置的牛ID
        
        # 多数投票的方式提取窗口标签
        label_counts = Counter(window_labels)
        majority_label = label_counts.most_common(1)[0][0]
        
        sensor_seqs.append(window_features)
        text_feats.append(text_features[window_idx])
        labels_out.append(majority_label)
        cow_ids_out.append(window_cow_id)
        window_idx += 1
        
    sensor_seqs = torch.stack(sensor_seqs)  # [num_windows, window_size, 3]
    text_feats = torch.stack(text_feats)    # [num_windows, text_feat_dim]
    labels_out = torch.tensor(labels_out, dtype=torch.long)
    cow_ids_out = torch.tensor(cow_ids_out, dtype=torch.long)
    return sensor_seqs, text_feats, labels_out, cow_ids_out


def window_sampling_raw_no_text(features, labels, cow_ids, window_size, stride, device):
    """
    不使用文本的原始传感器数据窗口采样
    返回：
      - sensor_seqs: [num_windows, window_size, 3] - 原始x,y,z数据
      - text_feats:  [num_windows, 1] - 空的文本特征
      - labels:      [num_windows]
      - cow_ids:     [num_windows] - 每个窗口对应的牛ID
    """
    sensor_seqs = []
    labels_out = []
    cow_ids_out = []

    # features = torch.tensor(features, dtype=torch.float32).to(device)
    features = torch.tensor(features, dtype=torch.float32)

    for index in range(0, len(features) - window_size + 1, stride):
        # 直接使用原始传感器数据
        # window_features = features[index:index + window_size].clone().detach().to(device)
        window_features = features[index:index + window_size].clone().detach()
        window_labels = labels[index:index + window_size]
        window_cow_id = cow_ids[index + window_size // 2]  # 使用窗口中间位置的牛ID

        # 多数投票的方式提取窗口标签
        label_counts = Counter(window_labels)
        majority_label = label_counts.most_common(1)[0][0]

        sensor_seqs.append(window_features)
        labels_out.append(majority_label)
        cow_ids_out.append(window_cow_id)

    sensor_seqs = torch.stack(sensor_seqs)  # [num_windows, window_size, 3]
    # 创建空的文本特征
    num_windows = len(sensor_seqs)
    text_feats = torch.zeros(num_windows, 1, device=device)  # 空的文本特征
    labels_out = torch.tensor(labels_out, dtype=torch.long)
    cow_ids_out = torch.tensor(cow_ids_out, dtype=torch.long)
    return sensor_seqs, text_feats, labels_out, cow_ids_out


def preprocess_data_by_cow(sensor_seqs, text_feats, labels, cow_ids, device):
    """
    按牛ID划分训练、验证、测试集
    Args:
        sensor_seqs: [num_windows, window_size, sensor_dim]
        text_feats:  [num_windows, text_feat_dim]
        labels:      [num_windows]
        cow_ids:     [num_windows] - 每个窗口对应的牛ID
        device: 计算设备
        test_cow_ratio: 用于测试的牛的比例
        val_cow_ratio: 用于验证的牛的比例
    Returns:
        训练、验证、测试数据
    """
    # 获取所有唯一的牛ID
    unique_cow_ids = np.unique(cow_ids)
    num_cows = len(unique_cow_ids)
    
    print(f"总共有 {num_cows} 头牛")
    print(f"牛ID列表: {unique_cow_ids}")
    
    # 简单划分：测试20%，验证20%，训练60%
    # num_test_cows = max(1, int(num_cows * 0.2))
    # num_val_cows = max(1, int(num_cows * 0.2))

    # # 优化划分：测试17%，验证28%，训练55%
    # # 18头牛的分配：测试3头，验证5头，训练10头
    num_test_cows = max(1, int(num_cows * 0.17))
    num_val_cows = max(1, int(num_cows * 0.28))
    
    # 随机选择用于测试和验证的牛
    np.random.seed(42)  # 固定随机种子以确保可重复性
    test_cow_ids = np.random.choice(unique_cow_ids, num_test_cows, replace=False)
    remaining_cows = np.setdiff1d(unique_cow_ids, test_cow_ids)
    val_cow_ids = np.random.choice(remaining_cows, num_val_cows, replace=False)
    train_cow_ids = np.setdiff1d(remaining_cows, val_cow_ids)
    
    print(f"训练集牛ID: {train_cow_ids} ({len(train_cow_ids)}头)")
    print(f"验证集牛ID: {val_cow_ids} ({len(val_cow_ids)}头)")
    print(f"测试集牛ID: {test_cow_ids} ({len(test_cow_ids)}头)")
    
    # 根据牛ID划分数据
    train_mask = np.isin(cow_ids, train_cow_ids)
    val_mask = np.isin(cow_ids, val_cow_ids)
    test_mask = np.isin(cow_ids, test_cow_ids)
    
    # 转换为numpy数组进行划分
    sensor_np = sensor_seqs.cpu().numpy()
    text_np = text_feats.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 提取对应的数据
    X_train_sensor = sensor_np[train_mask]
    X_val_sensor = sensor_np[val_mask]
    X_test_sensor = sensor_np[test_mask]

    X_train_text = text_np[train_mask]
    X_val_text = text_np[val_mask]
    X_test_text = text_np[test_mask]
    
    y_train = labels_np[train_mask]
    y_val = labels_np[val_mask]
    y_test = labels_np[test_mask]
    
    print(f"训练集大小: {len(X_train_sensor)} 个窗口")
    print(f"验证集大小: {len(X_val_sensor)} 个窗口")
    print(f"测试集大小: {len(X_test_sensor)} 个窗口")

    # 基于训练集拟合标准化参数，并对验证/测试集做变换
    scaler = StandardScaler()
    num_channels = X_train_sensor.shape[-1]
    X_train_flat = X_train_sensor.reshape(-1, num_channels)
    X_val_flat = X_val_sensor.reshape(-1, num_channels)
    X_test_flat = X_test_sensor.reshape(-1, num_channels)

    scaler.fit(X_train_flat)
    X_train_sensor = scaler.transform(X_train_flat).reshape(X_train_sensor.shape)
    X_val_sensor = scaler.transform(X_val_flat).reshape(X_val_sensor.shape)
    X_test_sensor = scaler.transform(X_test_flat).reshape(X_test_sensor.shape)
    
    X_train_sensor = torch.tensor(X_train_sensor, dtype=torch.float32)
    X_val_sensor = torch.tensor(X_val_sensor, dtype=torch.float32)
    X_test_sensor = torch.tensor(X_test_sensor, dtype=torch.float32)

    X_train_text = torch.tensor(X_train_text, dtype=torch.float32)
    X_val_text = torch.tensor(X_val_text, dtype=torch.float32)
    X_test_text = torch.tensor(X_test_text, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    
    return (X_train_sensor, X_train_text, y_train), (X_val_sensor, X_val_text, y_val), (X_test_sensor, X_test_text, y_test)



def preprocess_data_random(sensor_seqs, text_feats, labels, device):
    """
    随机划分训练、验证、测试集
    """
    print("使用随机划分方式...")
    # 转换为numpy数组进行划分
    sensor_np = sensor_seqs.cpu().numpy()
    text_np = text_feats.cpu().numpy()
    labels_np = labels.cpu().numpy()
    # 划分数据集
    X_temp_sensor, X_test_sensor, X_temp_text, X_test_text, y_temp, y_test = train_test_split(
        sensor_np, text_np, labels_np, test_size=0.2, random_state=42, stratify=labels_np
    )
    X_train_sensor, X_val_sensor, X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_temp_sensor, X_temp_text, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    # 基于训练集拟合标准化参数，并对验证/测试集做变换
    scaler = StandardScaler()
    num_channels = X_train_sensor.shape[-1]
    X_train_flat = X_train_sensor.reshape(-1, num_channels)
    X_val_flat = X_val_sensor.reshape(-1, num_channels)
    X_test_flat = X_test_sensor.reshape(-1, num_channels)

    scaler.fit(X_train_flat)
    X_train_sensor = scaler.transform(X_train_flat).reshape(X_train_sensor.shape)
    X_val_sensor = scaler.transform(X_val_flat).reshape(X_val_sensor.shape)
    X_test_sensor = scaler.transform(X_test_flat).reshape(X_test_sensor.shape)

    # # 转换回PyTorch张量
    # X_train_sensor = torch.tensor(X_train_sensor, dtype=torch.float32).to(device)
    # X_val_sensor = torch.tensor(X_val_sensor, dtype=torch.float32).to(device)
    # X_test_sensor = torch.tensor(X_test_sensor, dtype=torch.float32).to(device)
    # X_train_text = torch.tensor(X_train_text, dtype=torch.float32).to(device)
    # X_val_text = torch.tensor(X_val_text, dtype=torch.float32).to(device)
    # X_test_text = torch.tensor(X_test_text, dtype=torch.float32).to(device)
    # y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    # y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    # y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    # 转换回PyTorch张量（保持在CPU）
    X_train_sensor = torch.tensor(X_train_sensor, dtype=torch.float32)
    X_val_sensor = torch.tensor(X_val_sensor, dtype=torch.float32)
    X_test_sensor = torch.tensor(X_test_sensor, dtype=torch.float32)

    X_train_text = torch.tensor(X_train_text, dtype=torch.float32)
    X_val_text = torch.tensor(X_val_text, dtype=torch.float32)
    X_test_text = torch.tensor(X_test_text, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    
    return (X_train_sensor, X_train_text, y_train), (X_val_sensor, X_val_text, y_val), (X_test_sensor, X_test_text, y_test)

def preprocess_data(sensor_seqs, text_feats, labels, device, cow_ids=None, use_cow_split=False):
    """
    数据预处理：划分训练、验证、测试集
    Args:
        sensor_seqs: [num_windows, window_size, sensor_dim]
        text_feats:  [num_windows, text_feat_dim]
        labels:      [num_windows]
        device: 计算设备
        cow_ids: 牛ID列表，如果提供则按牛划分
        use_cow_split: 是否使用按牛划分的方式
    Returns:
        训练、验证、测试数据
    """
    if use_cow_split and cow_ids is not None:
        # 检查牛的数量
        unique_cows = np.unique(cow_ids)
        num_cows = len(unique_cows)
        
        if num_cows >= 3:
            print(f"使用按牛ID划分数据集的方式... (共{num_cows}头牛)")
            return preprocess_data_by_cow(sensor_seqs, text_feats, labels, cow_ids, device)
        else:
            print(f"牛数量不足({num_cows}头)，使用随机划分方式...")
            return preprocess_data_random(sensor_seqs, text_feats, labels, device)
    else:
        print("使用随机划分数据集的方式...")
        return preprocess_data_random(sensor_seqs, text_feats, labels, device)
    

def create_dataloader(train_data, val_data, test_data, batch_size, device):
    """
    创建数据加载器，支持三元组输入
    Args:
        train_data, val_data, test_data: (sensor_seqs, text_feats, labels)
        batch_size: 批次大小
        device: 计算设备
    Returns:
        训练、验证、测试数据加载器
    """
    X_train_sensor, X_train_text, y_train = train_data
    X_val_sensor, X_val_text, y_val = val_data
    X_test_sensor, X_test_text, y_test = test_data

    # 创建 TensorDataset（数据保持在 CPU，不提前搬到 GPU）
    train_dataset = TensorDataset(X_train_sensor, X_train_text, y_train)
    val_dataset = TensorDataset(X_val_sensor, X_val_text, y_val)
    test_dataset = TensorDataset(X_test_sensor, X_test_text, y_test)

    # 高性能 DataLoader：多线程 + pin_memory + 持久工作线程
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader






# def preprocess_data_by_cow(sensor_seqs, text_feats, labels, cow_ids, device):
#     """
#     按牛ID划分训练、验证、测试集
#     Args:
#         sensor_seqs: [num_windows, window_size, sensor_dim]
#         text_feats:  [num_windows, text_feat_dim]
#         labels:      [num_windows]
#         cow_ids:     [num_windows] - 每个窗口对应的牛ID
#         device: 计算设备
#         test_cow_ratio: 用于测试的牛的比例
#         val_cow_ratio: 用于验证的牛的比例
#     Returns:
#         训练、验证、测试数据
#     """
#     # 获取所有唯一的牛ID
#     unique_cow_ids = np.unique(cow_ids)
#     num_cows = len(unique_cow_ids)
    
#     print(f"总共有 {num_cows} 头牛")
#     print(f"牛ID列表: {unique_cow_ids}")
    
#     # 简单划分：测试20%，验证20%，训练60%
#     # num_test_cows = max(1, int(num_cows * 0.2))
#     # num_val_cows = max(1, int(num_cows * 0.2))

#     # # 优化划分：测试17%，验证28%，训练55%
#     # # 18头牛的分配：测试3头，验证5头，训练10头
#     num_test_cows = max(1, int(num_cows * 0.17))
#     num_val_cows = max(1, int(num_cows * 0.28))
    
#     np.random.seed(42)  # 固定随机种子以确保可重复性
#     test_cow_ids = np.random.choice(unique_cow_ids, num_test_cows, replace=False)
#     remaining_cows = np.setdiff1d(unique_cow_ids, test_cow_ids)
#     val_cow_ids = np.random.choice(remaining_cows, num_val_cows, replace=False)
#     train_cow_ids = np.setdiff1d(remaining_cows, val_cow_ids)
    
#     print(f"训练集牛ID: {train_cow_ids} ({len(train_cow_ids)}头)")
#     print(f"验证集牛ID: {val_cow_ids} ({len(val_cow_ids)}头)")
#     print(f"测试集牛ID: {test_cow_ids} ({len(test_cow_ids)}头)")
    
#     # 根据牛ID划分数据
#     train_mask = np.isin(cow_ids, train_cow_ids)
#     val_mask = np.isin(cow_ids, val_cow_ids)
#     test_mask = np.isin(cow_ids, test_cow_ids)
    
#     # 转换为numpy数组进行划分
#     sensor_np = sensor_seqs.cpu().numpy()
#     text_np = text_feats.cpu().numpy()
#     labels_np = labels.cpu().numpy()
    
#     # 提取对应的数据
#     X_train_sensor = sensor_np[train_mask]
#     X_val_sensor = sensor_np[val_mask]
#     X_test_sensor = sensor_np[test_mask]

#     X_train_text = text_np[train_mask]
#     X_val_text   = text_np[val_mask]
#     X_test_text  = text_np[test_mask]
#     # # ===== Random TXT ablation: 临时打乱训练集文本 =====新增代码
#     # np.random.seed(42)                 # 固定随机种子，保证可复现
#     # perm = np.random.permutation(len(X_train_text))
#     # X_train_text = X_train_text[perm]
#     # # ==================================================

    
#     y_train = labels_np[train_mask]
#     y_val = labels_np[val_mask]
#     y_test = labels_np[test_mask]
    
#     print(f"训练集大小: {len(X_train_sensor)} 个窗口")
#     print(f"验证集大小: {len(X_val_sensor)} 个窗口")
#     print(f"测试集大小: {len(X_test_sensor)} 个窗口")

#     # 基于训练集拟合标准化参数，并对验证/测试集做变换
#     scaler = StandardScaler()
#     num_channels = X_train_sensor.shape[-1]
#     X_train_flat = X_train_sensor.reshape(-1, num_channels)
#     X_val_flat = X_val_sensor.reshape(-1, num_channels)
#     X_test_flat = X_test_sensor.reshape(-1, num_channels)

#     scaler.fit(X_train_flat)
#     X_train_sensor = scaler.transform(X_train_flat).reshape(X_train_sensor.shape)
#     X_val_sensor = scaler.transform(X_val_flat).reshape(X_val_sensor.shape)
#     X_test_sensor = scaler.transform(X_test_flat).reshape(X_test_sensor.shape)
    
#     X_train_sensor = torch.tensor(X_train_sensor, dtype=torch.float32)
#     X_val_sensor = torch.tensor(X_val_sensor, dtype=torch.float32)
#     X_test_sensor = torch.tensor(X_test_sensor, dtype=torch.float32)

#     X_train_text = torch.tensor(X_train_text, dtype=torch.float32)
#     X_val_text = torch.tensor(X_val_text, dtype=torch.float32)
#     X_test_text = torch.tensor(X_test_text, dtype=torch.float32)

#     y_train = torch.tensor(y_train, dtype=torch.long)
#     y_val = torch.tensor(y_val, dtype=torch.long)
#     y_test = torch.tensor(y_test, dtype=torch.long)

    
#     return (X_train_sensor, X_train_text, y_train), (X_val_sensor, X_val_text, y_val), (X_test_sensor, X_test_text, y_test)