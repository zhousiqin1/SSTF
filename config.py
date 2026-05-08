import argparse
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal Time Series classification')

    parser.add_argument('-model', type=str, default='HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive',
                        choices=['HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive'])
    
    # Data path
    parser.add_argument('-data_path', type=str, default='Dataset', help="Dataset path")

    parser.add_argument('-pretrained_model', type=str, default='', 
                        help='Path to pretrained model weights')

    # Experiment settings
    parser.add_argument('-experiment_name', type=str, default='baseline')

    # Time window parameters 
    parser.add_argument('-window_size', type=int, default=1200, help="Window size")
    parser.add_argument('-stride', type=int, default=1200, help="Window stride")
    
    # Training hyperparameters
    parser.add_argument('-lr', type=float, default=0.0002, help="Learning rate")
    parser.add_argument('-epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('-batch_size', type=int, default=64, help="Training batch size")
    parser.add_argument('-weight_decay', type=float, default=0.0001, help='Weight decay coefficient')
    

    parser.add_argument('-data_batch_size', type=int, default=1024, help="Data processing batch size (for text generation and CLIP encoding)")


    parser.add_argument('-raw_input_size', type=int, default=3, help="Raw sensor input dimension (typically 3: x,y,z acceleration)")
    parser.add_argument('-text_feat_dim', type=int, default=32, help='Text feature dimension (after TextEmbedder reduction)')
    parser.add_argument('-output_size', type=int, default=3, help='Number of classification categories')
    parser.add_argument('-lstm_hidden', type=int, default=128, help='LSTM hidden units')
    parser.add_argument('-mlp_dim', type=int, default=256, help='MLP hidden units')
    
    # FBM component parameters
    parser.add_argument('-fbm_seq_len', type=int, default=1200, help='FBM sequence length, should match window_size')
    parser.add_argument('-fbm_block_size', type=int, default=128, help="FBM component block size")
    parser.add_argument('-fbm_hidden_dim', type=int, default=128, help="FBM component hidden dimension")
    parser.add_argument('-attention_heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('-dropout', type=float, default=0.6, help="Dropout rate")
    
    # New model specific parameters
    parser.add_argument('-cnn_hidden_dim', type=int, default=128, help="CNN branch hidden dimension")
    parser.add_argument('-conv_time_kernel', type=int, default=31, help='2D CNN temporal kernel width')

    # Dataset split control
    parser.add_argument('-use_cow_split', type=bool, default=True, 
                        help="Whether to split dataset by cow ID (True = by cow, False = random split)")
    
    # Device settings
    parser.add_argument('-use_gpu', type=bool, default=True, help="Whether to use GPU")
    parser.add_argument('-devices', type=str, default="0", help="GPU devices to use, comma-separated list, e.g. 0,1,2,3")

    # GPT-OSS text generation settings (local offline model)
    parser.add_argument('-use_gpt', type=bool, default=True, help="Whether to use GPT-OSS for text generation")
    parser.add_argument('-gpt_oss_model_path', type=str, default='./models/gpt-oss', help="Path to the local GPT-OSS model")
    parser.add_argument('-gpt_oss_tokenizer_path', type=str, default='./models/gpt-oss-tokenizer', help="Path to the GPT-OSS tokenizer")
    parser.add_argument('-gpt_max_tokens', type=int, default=70, help="Maximum number of tokens for generated text")
    parser.add_argument('-gpt_temperature', type=float, default=0.5, help="Temperature for text generation")
    parser.add_argument('-gpt_system_prompt', type=str, 
                        default="You are an animal behavior expert specializing in movement pattern analysis. Analyze the following sensor-derived movement indicators and generate a concise behavioral description. Focus on rhythmic patterns, head posture, movement intensity, and complexity. Provide semantic interpretations rather than numerical summaries.",
                        help="System prompt for GPT-OSS")
    

    parser.add_argument('-force_regenerate_text', type=bool, default=False, 
                        help="Force regeneration of text descriptions (ignore existing cache files)")
    parser.add_argument('-use_text_cache', type=bool, default=True, 
                        help="Use text cache files if they exist")
    

    parser.add_argument('-save_dir', type=str, default='', 
                        help="Result save directory (leave empty for auto-generated timestamp directory)")


    args = parser.parse_args()
    

    if not args.save_dir:
        base_dir = "results"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        base_model_dir = f"{base_dir}/{args.experiment_name}"
        
        if not os.path.exists(base_model_dir):
            args.save_dir = base_model_dir
        else:
            existing_numbers = []
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)):
                    if item.startswith(f"{args.experiment_name}_"):
                        try:
                            num_str = item.split(f"{args.experiment_name}_")[1]
                            num = int(num_str)
                            existing_numbers.append(num)
                        except:
                            pass
            next_number = 1
            if existing_numbers:
                next_number = max(existing_numbers) + 1
            args.save_dir = f"{base_dir}/{args.experiment_name}_{next_number}"
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/fig", exist_ok=True)
    
    return args