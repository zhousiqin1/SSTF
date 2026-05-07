# clip_integration.py
import torch
import torch.nn as nn
import clip

class TextEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextEmbedder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # 将文本特征从512维映射到12维
        self.relu = nn.ReLU()

    def forward(self, text_features):
        # 将文本特征传入全连接层，然后使用ReLU进行非线性变换
        x = self.fc(text_features)
        x = self.relu(x)
        return x  # 返回经过 ReLU 激活后的文本特征


def safe_clip_tokenize(text, max_length=70):
    tokens = clip.tokenize([text])[0]
    nonzero = (tokens != 0).nonzero(as_tuple=True)[0]
    if len(nonzero) > max_length:
        tokens[max_length:] = 0
    return tokens


def convert_text_to_features(text_data, clip_model, device, batch_size=512):
    """
    使用CLIP模型将文本数据分批转换为特征向量，减少显存占用
    
    Args:
        text_data: 文本数据列表
        clip_model: CLIP模型
        device: 计算设备
        batch_size: 批处理大小
        
    Returns:
        文本特征张量
    """
    all_features = []
    clip_model = clip_model.cpu().float()  # 保证CLIP模型在CPU且为float32
    
    print(f"开始CLIP编码，总文本数: {len(text_data)}, 批次大小: {batch_size}")
    
    with torch.no_grad():
        for i in range(0, len(text_data), batch_size):
            batch_text = text_data[i:i+batch_size]
            batch_end = min(i + batch_size, len(text_data))
            
            print(f"处理CLIP批次 {i//batch_size + 1}/{(len(text_data) + batch_size - 1)//batch_size}")
            
            # 兜底截断，保证每条文本不会超过77个CLIP token
            text_inputs = torch.stack([safe_clip_tokenize(t, max_length=70) for t in batch_text])
            # 确保输入为long类型（tokenizer输出），CLIP模型会自动处理，不需要转float32
            text_features = clip_model.encode_text(text_inputs)
            text_features = text_features.float()  # 保证输出为float32
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_features.append(text_features.cpu())  # 保证在CPU
            
            # 清理内存
            del text_inputs, text_features
            import gc
            gc.collect()
    
    all_features = torch.cat(all_features, dim=0)
    print(f"CLIP编码完成，特征形状: {all_features.shape}")
    return all_features


def load_clip_model(device='cpu', text_input_size=512, output_dim=12):
    clip_model, preprocess = clip.load('ViT-B/16', device)
    text_embedder = TextEmbedder(text_input_size, output_dim).to(device)  # 确保 text_embedder 在指定设备上
    return clip_model, preprocess, text_embedder