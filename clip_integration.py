import torch
import torch.nn as nn
import clip
import gc


class TextEmbedder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TextEmbedder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, text_features):
        x = self.fc(text_features)
        x = self.relu(x)
        return x


def safe_clip_tokenize(text, max_length=70):

    tokens = clip.tokenize([text])[0]
    nonzero = (tokens != 0).nonzero(as_tuple=True)[0]
    if len(nonzero) > max_length:
        tokens[max_length:] = 0
    return tokens


def convert_text_to_features(text_data, clip_model, device, batch_size=512):

    all_features = []
    clip_model = clip_model.cpu().float()

    print(f"Starting CLIP encoding, total texts: {len(text_data)}, batch size: {batch_size}")

    with torch.no_grad():
        for i in range(0, len(text_data), batch_size):
            batch_text = text_data[i:i + batch_size]

            print(f"Processing CLIP batch {i // batch_size + 1}/{(len(text_data) + batch_size - 1) // batch_size}")

            # Truncate each text to max 70 tokens before CLIP tokenization
            text_inputs = torch.stack([safe_clip_tokenize(t, max_length=70) for t in batch_text])
            text_features = clip_model.encode_text(text_inputs)
            text_features = text_features.float()
            # L2 normalization for stable cross-modal alignment
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_features.append(text_features.cpu())

            del text_inputs, text_features
            gc.collect()

    all_features = torch.cat(all_features, dim=0)
    print(f"CLIP encoding completed, feature shape: {all_features.shape}")
    return all_features


def load_clip_model(device='cpu', text_input_size=512, output_dim=32):
    clip_model, preprocess = clip.load('ViT-B/16', device)
    text_embedder = TextEmbedder(text_input_size, output_dim).to(device)
    return clip_model, preprocess, text_embedder