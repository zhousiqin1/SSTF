import argparse
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal Time Series classification')
# HybridFBM_Text_LSTM 
    parser.add_argument('-model', type=str, default='HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive',
                       choices=['HybridFBM_LSTM', 'HybridFBM_LSTM_Attention', 'HybridFBM_LSTM_Simple'])
    
    # 数据路径     filtered_data_rounded_g     small_test_data  Dataset/blackcow_filtered_only.csv  blackcow_aligned_10hz
    parser.add_argument('-data_path', type=str, default='Dataset/filtered_data_rounded_g.csv', help="数据集路径")

    parser.add_argument('-pretrained_model', type=str, default='5_Contrastive_model.pt', 
                       help='Path to pretrained model weights')

    # 实验设置 0_FBM_LSTM_CNN_2D_Text_Dynamic_Contrastive Random_TXT  6_Contrastive
    parser.add_argument('-experiment_name', type=str, default='6_Contrastive')

    # 时间窗口相关参数 
    parser.add_argument('-window_size', type=int, default=1200, help="时间窗口大小")
    parser.add_argument('-stride', type=int, default=1200, help="窗口步长")

    
    # 学习率等训练参数
    parser.add_argument('-lr', type=float, default=0.0002, help="学习率")  # 0.0002
    parser.add_argument('-epochs', type=int, default=50, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=64, help="训练批次大小")  # 64
    parser.add_argument('-weight_decay', type=float, default=0.0001, help='权重衰减系数')
    
    # 数据处理批次大小
    parser.add_argument('-data_batch_size', type=int, default=1024, help="数据处理批次大小（用于文本生成和CLIP编码）")

    # ==================== 模型架构参数 ====================
    # 基础模型参数
    parser.add_argument('-raw_input_size', type=int, default=3, help="原始传感器输入维度（通常是3：x,y,z加速度）")
    parser.add_argument('-text_feat_dim', type=int, default=32, help='文本特征的维度（经过TextEmbedder降维后的维度）')
    parser.add_argument('-output_size', type=int, default=3, help='分类类别数量')
    parser.add_argument('-lstm_hidden', type=int, default=128, help='LSTM隐藏层单元数')   # 128
    parser.add_argument('-mlp_dim', type=int, default=256, help='MLP隐藏层单元数')
    
    # FBM组件参数
    parser.add_argument('-fbm_seq_len', type=int, default=1200, help='FBM处理的序列长度，应该与window_size一致')
    parser.add_argument('-fbm_block_size', type=int, default=128, help="FBM组件的块大小")
    parser.add_argument('-fbm_hidden_dim', type=int, default=128, help="FBM组件的隐藏层维度")  # 128
    parser.add_argument('-attention_heads', type=int, default=4, help="注意力机制的头数")
    parser.add_argument('-dropout', type=float, default=0.6, help="模型dropout率")  # 0.6

    # 新模型特有参数
    parser.add_argument('-cnn_hidden_dim', type=int, default=128, help="CNN分支隐藏层维度")  # 128
    parser.add_argument('-conv_time_kernel', type=int, default=31, help='2D CNN 时间维卷积核宽度')  # 31

    # 数据集划分控制
    parser.add_argument('-use_cow_split', type=bool, default=True, 
                       help="是否使用按牛ID划分数据集的方式（True=按牛划分，False=随机划分）")
    
    # 设备设置
    parser.add_argument('-use_gpu', type=bool, default=True, help="是否使用GPU")
    parser.add_argument('-devices', type=str, default="0", help="使用哪些 GPU 设备，逗号分隔的列表，例如 0,1,2,3")


    # GPT文本生成设置
    parser.add_argument('-use_gpt', type=bool, default=True, help="是否使用GPT生成文本描述")
    parser.add_argument('-openai_api_key', type=str, default="sk-ElTR04jgHgVmTBCT8sTrMdjdPRBk2TlexyDnjzdAqIsMersQ", help="OpenAI API密钥")
    parser.add_argument('-gpt_base_url', type=str, default="http://chatapi.littlewheat.com/v1", help="OpenAI代理地址")
    parser.add_argument('-gpt_model', type=str, default="gpt-4o-mini", help="使用的GPT模型名称")
    parser.add_argument('-gpt_max_tokens', type=int, default=70, help="GPT生成文本的最大token数量")
    parser.add_argument('-gpt_temperature', type=float, default=0.6, help="GPT生成文本的温度参数")
    parser.add_argument('-gpt_timeout', type=int, default=30, help="GPT API调用的超时时间（秒）")
    parser.add_argument('-gpt_max_retries', type=int, default=3, help="GPT API调用的最大重试次数")
    
    parser.add_argument('-gpt_system_prompt', type=str, 
                       default="""You are an animal behavior expert specializing in movement pattern analysis.

Your task is to analyze animal movement patterns from sensor data and provide behavioral insights that complement the numerical measurements.

Context: The data comes from accelerometers attached to animals, measuring movement in three dimensions. The indicators include:
- Rhythmic patterns (periodicity, autocorrelation)
- Head position (pitch, roll angles)
- Movement intensity and variability
- Temporal patterns (peak frequency, intervals)
- Movement complexity

Analysis Process:
1. First analyze the movement features and identify key behavioral indicators
2. Then generate concise behavioral descriptions based on your analysis
3. Focus on behavioral interpretation and context
4. Provide semantic descriptions that go beyond raw numbers
5. Use clear, descriptive language
6. Aim for behavioral insights rather than measurement repetition
7. Generate distinctive descriptions that differentiate between different movement patterns""", 
                       help="GPT system prompt")
    
    
    # 文本缓存控制
    parser.add_argument('-force_regenerate_text', type=bool, default=False, 
                       help="是否强制重新生成文本描述（忽略已存在的缓存文件）")
    parser.add_argument('-use_text_cache', type=bool, default=True, 
                       help="是否使用文本缓存文件（如果存在）")
    
    # 结果保存设置
    parser.add_argument('-save_dir', type=str, default='', 
                       help="结果保存目录（留空则自动生成时间戳目录）")

    # 解析并返回参数
    args = parser.parse_args()
    
    # 自动生成保存目录
    if not args.save_dir:
        # 查找下一个可用的编号
        base_dir = "results"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # 使用模型名作为基础目录名
        base_model_dir = f"{base_dir}/{args.experiment_name}"
        
        # 检查是否已存在同名目录
        if not os.path.exists(base_model_dir):
            args.save_dir = base_model_dir
        else:
            # 查找已存在的编号
            existing_numbers = []
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)):
                    if item.startswith(f"{args.experiment_name}_"):
                        try:
                            # 提取编号
                            num_str = item.split(f"{args.experiment_name}_")[1]
                            num = int(num_str)
                            existing_numbers.append(num)
                        except:
                            pass
            
            # 生成下一个编号
            next_number = 1
            if existing_numbers:
                next_number = max(existing_numbers) + 1
            
            args.save_dir = f"{base_dir}/{args.experiment_name}_{next_number}"
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/fig", exist_ok=True)
    
    return args