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
    # 调用config
    args = parse_args()

    # 设置随机种子以确保结果可重现
    set_reproducible_training()

    # 解析 GPU 设备
    if args.use_gpu:
        device_list = args.devices.split(',')
        device_ids = [int(device) for device in device_list]
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device = torch.device('cpu')
        device_ids = []
    
    # 加载CLIP模型和文本嵌入模块
    clip_model, preprocess, text_embedder = load_clip_model(device, text_input_size=512, output_dim=args.text_feat_dim)
    clip_model = clip_model.to(device)
    text_embedder = text_embedder.to(device)

    # 加载数据和数据预处理
    if args.model in ['HybridFBM_Text_LSTM', 'HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive']:
        # 带文本特征的模型：使用原始传感器数据和文本特征
        sensor_seqs, text_feats, labels, cow_ids = load_data(args.data_path, args.window_size, args.stride, clip_model, device, text_embedder, args, raw_sensor_only=True)
    elif args.model in ['HybridFBM_LSTM']:
        # 无文本特征的模型：使用原始传感器数据，跳过文本处理
        sensor_seqs, text_feats, labels, cow_ids = load_data(args.data_path, args.window_size, args.stride, clip_model, device, text_embedder, args, raw_sensor_only=True, skip_text=True)
    else:
        # 其他模型类型不支持
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    # 根据配置选择数据集划分方式
    train_data, val_data, test_data = preprocess_data(sensor_seqs, text_feats, labels, device, cow_ids=cow_ids, use_cow_split=args.use_cow_split)
    print(train_data[0].shape, train_data[2].shape)
    print(val_data[0].shape, val_data[2].shape)
    print(test_data[0].shape, test_data[2].shape)

    # ========== 统计数据集分布 ==========
    print("\n" + "=" * 60)
    print("Table 1. Distribution of samples for the Standard Monitoring Scenario")
    print("=" * 60)

    label_names = {0: 'Other', 1: 'Rumination', 2: 'Eating'}

    # train_data[2] 是 labels
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
    # ========== 统计结束 ==========

    # 创建数据加载器，确保传感器数据和文本特征一起返回
    train_loader, val_loader, test_loader = create_dataloader(train_data, val_data, test_data, args.batch_size, device)

    # 实例化模型
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        
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
            raise ValueError(f"不支持的模型类型: {args.model}")

        if args.use_gpu and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
        else:
            model = model.to(device)
        print(model)

        # 计算并打印模型的参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters of {args.model}: {total_params}")

        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        exit(1)

    print("========== 推理显存测试（部署需要） ==========")
    model.eval()
    dummy_sensor = torch.randn(1, args.fbm_seq_len, args.raw_input_size).to(device)
    dummy_text   = torch.randn(1, args.text_feat_dim).to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_sensor, dummy_text)
    print("【推理显存峰值】", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("======================================================")
    
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练和验证<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # 定义损失函数（加入 label smoothing 提升稳定性）
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    # 定义优化器（提高权重衰减抑制抖动与过拟合）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_decay', 2e-4))

    # 学习率调度策略选择
    scheduler_type = "warmup_cosine"  # 可选: "plateau", "cosine", "warmup_cosine"
    
    if scheduler_type == "plateau":
        # ReduceLROnPlateau - 基于验证损失的自适应调度
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',           # 监控验证损失
            factor=0.7,           # 学习率衰减因子
            patience=5,           # 等待轮次8
            threshold=1e-3,       # 改进阈值
            cooldown=1,           # 冷却期3
            verbose=True, 
            min_lr=1e-6
        )
    elif scheduler_type == "cosine":
        # 余弦退火 - 平滑的学习率衰减
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif scheduler_type == "warmup_cosine":
        # 预热+余弦退火 - 先预热再余弦衰减
        warmup_epochs = max(1, min(10, args.epochs // 10))
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        # 默认使用ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8, threshold=1e-3, cooldown=3, verbose=True, min_lr=1e-6)

    # 初始化 Early Stopping 对象
    early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.005)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_inference_times = []  # 存储每轮的推理时间
    val_inference_stats = []  # 存储每轮的推理统计信息

    for epoch in range(args.epochs):
        train_losses, train_accs = train_model(model, loss_function, train_loader, optimizer, train_losses, train_accs, device)
        val_losses, val_accs, inference_stats = val_model(model, loss_function, val_loader, val_losses, val_accs, device)
        val_inference_times.append(inference_stats['avg_batch_time'])
        val_inference_stats.append(inference_stats)
        
        print(
            f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}')
        print(f'推理时间 - 批次: {inference_stats["avg_batch_time"]:.4f}s, 单样本: {inference_stats["single_sample_time"]:.6f}s, 总时间: {inference_stats["total_time"]:.4f}s')
        
        # 更新学习率
        if scheduler_type == "plateau":
            scheduler.step(val_losses[-1])  # ReduceLROnPlateau需要验证损失作为参数
        else:
            scheduler.step()  # 其他调度器不需要参数
        print("Learning rate:", optimizer.param_groups[0]['lr'])
        # # 检查是否早停
        if early_stopping(val_losses[-1]):
            print("Early stopping")
            break
    # 测试模型
    # print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # test_accs_list, predicted = test_model(model, test_loader)
    # test_acc = test_accs_list[-1] if test_accs_list else 0.0  # 取最后一个值作为测试准确率
    experiment_dir = args.save_dir
    os.makedirs(f"{experiment_dir}/models", exist_ok=True)
    os.makedirs(f"{experiment_dir}/fig", exist_ok=True)
    # 测试模型
    # print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # test_acc, test_f1, predicted = test_model(model, test_loader, save_dir=f"{experiment_dir}/fig")
    # print(f"测试准确率: {test_acc:.4f}")
    # print(f"测试F1分数: {test_f1:.4f}")

    print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    (
        test_acc, test_f1, predicted,
        y_all, fbm_raw, cnn_raw, text_raw,
        fbm_proj, cnn_proj, text_proj
    ) = test_model(model, test_loader)   # 不画图，只收集特征

    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试F1分数: {test_f1:.4f}")

    print("🔹 提取动态融合特征用于可视化 ...")
    fused_all = []
    model.eval()
    with torch.no_grad():
        for batch_sensor, batch_text, _ in tqdm(test_loader, desc="提取融合特征"):
            batch_sensor = batch_sensor.to(device, non_blocking=True)
            batch_text = batch_text.to(device, non_blocking=True)
            fused = model(batch_sensor, batch_text, return_x="fused")
            fused_all.append(fused.cpu())

    fused_all = torch.cat(fused_all, dim=0)


    # 获取真实标签
    labels = test_data[2].cpu().numpy()

    # 计算详细的分类报告
    from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
    classification_rep = classification_report(labels, predicted, zero_division=0)
    precision = precision_score(labels, predicted, average='weighted', zero_division=0)
    recall = recall_score(labels, predicted, average='weighted', zero_division=0)
    f1 = f1_score(labels, predicted, average='weighted', zero_division=0)

    # 打印分类指标到控制台
    print(f"=== 分类指标 ===")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1 Score): {f1:.4f}")

    if val_inference_stats:
             print(f"FLOPs (per sample): {val_inference_stats[-1].get('flops_G', 0):.4f} G")

    # 打印详细分类报告到控制台
    print(f"\n=== 详细分类报告 ===")
    print(classification_rep)

    # 保存实验结果摘要
    with open(f"{experiment_dir}/{args.experiment_name}_results.txt", 'w', encoding='utf-8') as f:
        f.write(f"实验名称: {args.experiment_name}\n")
        f.write(f"模型类型: {args.model}\n")
        f.write(f"训练轮次: {args.epochs}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"输入维度: {args.raw_input_size}\n")
        f.write(f"LSTM隐藏层: {args.lstm_hidden}\n")
        f.write(f"MLP维度: {args.mlp_dim}\n")
        f.write(f"FBM序列长度: {args.fbm_seq_len}\n")
        f.write(f"FBM隐藏层维度: {args.fbm_hidden_dim}\n")
        f.write(f"\n=== 训练结果 ===\n")
        f.write(f"测试准确率: {test_acc:.4f}\n")
        f.write(f"测试F1分数: {test_f1:.4f}\n")


        # 获取最后一次验证的统计信息
        last_stats = val_inference_stats[-1] if val_inference_stats else {}
        flops_val = last_stats.get('flops_G', 0.0)
        params_val = last_stats.get('params_M', 0.0)
        
        f.write(f"\n=== 模型复杂度 ===\n")
        f.write(f"参数量 (Params): {params_val:.2f} M\n")
        f.write(f"计算量 (FLOPs/sample): {flops_val:.4f} GFLOPs\n")
        # ----------------------------------------------------

        f.write(f"最终训练损失: {train_losses[-1]:.4f}\n")
        f.write(f"最终训练准确率: {train_accs[-1]:.4f}\n")
        f.write(f"最终验证损失: {val_losses[-1]:.4f}\n")
        f.write(f"最终验证准确率: {val_accs[-1]:.4f}\n")
        f.write(f"测试准确率: {test_acc:.4f}\n")
        f.write(f"最佳验证准确率: {max(val_accs):.4f}\n")
        f.write(f"最佳验证准确率轮次: {val_accs.index(max(val_accs)) + 1}\n")
        f.write(f"实际训练轮次: {len(train_losses)}\n")
        f.write(f"是否早停: {'是' if len(train_losses) < args.epochs else '否'}\n")


        # 计算推理时间统计
        final_inference_time = val_inference_times[-1] if val_inference_times else 0.0
        best_epoch_idx = val_accs.index(max(val_accs))
        best_inference_time = val_inference_times[best_epoch_idx] if best_epoch_idx < len(val_inference_times) else 0.0
        avg_inference_time = sum(val_inference_times) / len(val_inference_times) if val_inference_times else 0.0
        
        # 计算单样本推理时间统计
        single_sample_times = [stats['single_sample_time'] for stats in val_inference_stats if 'single_sample_time' in stats]
        avg_single_sample_time = sum(single_sample_times) / len(single_sample_times) if single_sample_times else 0.0
        
        # 保存推理时间
        f.write(f"\n=== 推理时间统计 ===\n")
        f.write(f"批次大小: {val_loader.batch_size}\n")
        f.write(f"最终批次推理时间: {final_inference_time:.4f}s\n")
        f.write(f"最终单样本推理时间: {val_inference_stats[-1]['single_sample_time']:.6f}s\n")
        f.write(f"最佳准确率轮次批次时间: {best_inference_time:.4f}s (第{best_epoch_idx + 1}轮)\n")
        f.write(f"最佳准确率轮次单样本时间: {val_inference_stats[best_epoch_idx]['single_sample_time']:.6f}s\n")
        f.write(f"平均批次推理时间: {avg_inference_time:.4f}s\n")
        f.write(f"平均单样本推理时间: {avg_single_sample_time:.6f}s\n")
        f.write(f"最快批次推理时间: {min(val_inference_times):.4f}s\n")
        f.write(f"最慢批次推理时间: {max(val_inference_times):.4f}s\n")
        f.write(f"最快单样本推理时间: {min(single_sample_times):.6f}s\n")
        f.write(f"最慢单样本推理时间: {max(single_sample_times):.6f}s\n")
        
        # 保存分类指标
        f.write(f"\n=== 分类指标 ===\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall): {recall:.4f}\n")
        f.write(f"F1分数 (F1 Score): {f1:.4f}\n")
        # 新增
        try:
            from sklearn.metrics import balanced_accuracy_score
            balanced_acc = balanced_accuracy_score(labels, predicted)
            f.write(f"平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}\n")
            print(f"[Extra Metrics] Balanced Accuracy={balanced_acc:.4f}")
        except Exception as e:
            print(f"[Extra Metrics - Balanced Accuracy skipped] {e}")

        # === 附加分类指标（Cohen’s Kappa、Specificity、G-Mean）===
        try:
            from sklearn.metrics import cohen_kappa_score, confusion_matrix
            import numpy as np

            # 计算混淆矩阵并求出各类指标
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

            # 写入结果文件
            f.write(f"Cohen’s Kappa 系数: {kappa:.4f}\n")
            f.write(f"平均特异度 (Specificity): {specificity:.4f}\n")
            f.write(f"几何平均值 (G-Mean): {gmean:.4f}\n")

            # 控制台打印
            print(f"[Extra Metrics] Kappa={kappa:.4f}, Specificity={specificity:.4f}, GMean={gmean:.4f}")

        except Exception as e:
            print(f"[Extra Metrics (skipped)] {e}")
        
        # 保存详细分类报告
        f.write(f"\n=== 详细分类报告 ===\n")
        f.write(classification_rep)

    print(f"实验结果已保存到: {experiment_dir}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"最佳验证准确率: {max(val_accs):.4f}")
    
    # 输出推理时间统计
    print(f"\n=== 推理时间统计 ===")
    print(f"批次大小: {val_loader.batch_size}")
    print(f"最终批次推理时间: {final_inference_time:.4f}s")
    print(f"最终单样本推理时间: {val_inference_stats[-1]['single_sample_time']:.6f}s")
    print(f"最佳准确率轮次批次时间: {best_inference_time:.4f}s (第{best_epoch_idx + 1}轮)")
    print(f"最佳准确率轮次单样本时间: {val_inference_stats[best_epoch_idx]['single_sample_time']:.6f}s")
    print(f"平均批次推理时间: {avg_inference_time:.4f}s")
    print(f"平均单样本推理时间: {avg_single_sample_time:.6f}s")
    print(f"最快单样本推理时间: {min(single_sample_times):.6f}s")
    print(f"最慢单样本推理时间: {max(single_sample_times):.6f}s")
    

    # 保存模型
    model_path = f"{experiment_dir}/models/{args.experiment_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    # 1. 训练曲线 - 修改保存路径
    from draw import plot_training_curves, plot_train_val_loss, plot_train_val_accuracy
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, f"{experiment_dir}/fig", args.experiment_name)
    plot_train_val_loss(train_losses, val_losses, f"{experiment_dir}/fig")
    plot_train_val_accuracy(train_accs, val_accs, f"{experiment_dir}/fig")

    # 2. 混淆矩阵和散点图 - 修改保存路径
    drawScatter([labels, predicted], ['true', 'pred'], f"{experiment_dir}/fig", args.experiment_name)
    plot_confusion(labels, predicted, f"{experiment_dir}/fig", args.experiment_name)

    plot_confusion_raw(labels, predicted, f"{experiment_dir}/fig", args.experiment_name)



    # 3. 模型性能总结 - 修改保存路径
    from draw import plot_model_performance_summary
    plot_model_performance_summary(train_losses, train_accs, val_losses, val_accs,
                                 test_acc, predicted, labels, f"{experiment_dir}/fig")

    # 绘制和保存所有结果图
    print("正在生成结果图表...")
    # fig_dir = f"{experiment_dir}/fig"
    # plot_tsne_no_contrastive(fbm_raw, cnn_raw, text_raw, y_all, save_dir=fig_dir)
    # plot_tsne(fbm_proj, cnn_proj, text_proj, y_all, save_dir=fig_dir)
    # plot_modal_similarity(fbm_proj, cnn_proj, text_proj, save_dir=fig_dir)
    # plot_avg_modal_similarity(fbm_proj, cnn_proj, text_proj, save_dir=fig_dir)
    # plot_fusion_weight_distribution(model, fbm_raw, cnn_raw, text_raw, save_dir=fig_dir)
    # plot_classwise_modal_similarity(fbm_proj, cnn_proj, text_proj, y_all, save_dir=fig_dir)
    # plot_tsne_modal_comparison_dynamic(fbm_raw, cnn_raw, text_raw, fused_all, y_all, save_dir=fig_dir)
    
    print("所有图表已保存完成!")


    
