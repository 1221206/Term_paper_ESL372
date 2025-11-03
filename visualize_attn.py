import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# WeightedAttentionLayer
class WeightedAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(WeightedAttentionLayer, self).__init__()
        self.attn_weights = nn.Parameter(torch.randn(input_dim))  # attn wt

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        # output weighted out and attn scores
        attn_scores = torch.matmul(x, self.attn_weights)  # [batch_size, seq_len]
        attn_scores = torch.softmax(attn_scores, dim=1)  # norm
        attn_scores = attn_scores.unsqueeze(-1)  # [batch_size, seq_len, 1]
        output = x * attn_scores  
        return output, attn_scores  

# model architecture
class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionModel, self).__init__()
        self.attn = WeightedAttentionLayer(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.attention_weights_cache = []

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        attn_output, attn_weights = self.attn(x)  # get attn out and wt
        self.attention_weights_cache.append(attn_weights.detach().cpu())  # chache weights
        x = attn_output.mean(dim=1)  # mean pooling
        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        return output

def visualize_attention_weights(attn_weights_list, sample_idx, save_dir='attn_plots'):
    """
    attn_weights_list: List of attention weight tensors [batch_size, seq_len, 1]
    sample_idx: Which sample in which batch (batch_id, sample_id)
    """
    os.makedirs(save_dir, exist_ok=True)

    batch_id, sample_id = sample_idx
    if batch_id >= len(attn_weights_list):
        raise IndexError("batch_id is out of range")
    if sample_id >= attn_weights_list[batch_id].shape[0]:
        raise IndexError("sample_id is out of range")

    attn_weights = attn_weights_list[batch_id][sample_id].squeeze()  # [seq_len]
    plt.figure(figsize=(10, 2))
    sns.heatmap(attn_weights.unsqueeze(0).numpy(), cmap='viridis', cbar=True, annot=True, xticklabels=False)
    plt.title(f'Attention Weights (Batch {batch_id}, Sample {sample_id})')
    plt.xlabel('Feature Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'attn_batch{batch_id}_sample{sample_id}.png'))
    plt.close()

# def hyper params
if __name__ == "__main__":
    batch_size = 32
    seq_len = 16  # Assuming 16 features
    input_dim = 1  # One feature value at each position (can also be multi-dimensional)
    hidden_dim = 64
    output_dim = 1

    model = AttentionModel(input_dim, hidden_dim, output_dim)
    dummy_input = torch.rand(batch_size, seq_len, input_dim)

    # forward pass for different batches
    for _ in range(5):  #  Assume 5 batches
        output = model(dummy_input)

    # attn wt for 3rd sample in 2nd batch
    visualize_attention_weights(model.attention_weights_cache, sample_idx=(1, 2))