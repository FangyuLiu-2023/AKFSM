# title: A Key Feature Screening Method for Human Activity Recognition Based on Multi-head Attention Mechanism
# create: Hao Wang, Fangyu Liu, Xiang Li, Ye Li, and Fangmin Sun
# date: March 2025

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from thop import profile

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear transformation
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculating attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Output linear transformation
        return self.out(attn_output), attn_weights

class IndependentLinearLayersEfficient(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_channels, num_heads=8, drop=0.5, classes_num=18):
        super(IndependentLinearLayersEfficient, self).__init__()
        self.num_channels = num_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mha = MultiHeadAttention(self.output_dim, num_heads)
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(156 * 6 * 16, classes_num),
        )
        # Linear 1
        # Adjust the weight shape to (input_dim, C, hidden_dim)
        self.weights1 = nn.Parameter(torch.randn(input_dim, num_channels, hidden_dim))  # Shape: (input_dim, C, hidden_dim)
        self.biases1 = nn.Parameter(torch.randn(1, num_channels, hidden_dim))           # Shape: (1, C, hidden_dim)

        # Linear 2
        # Adjust the weight shape to (hidden_dim, C, output_dim)
        self.weights2 = nn.Parameter(torch.randn(hidden_dim, num_channels, output_dim)) # Shape: (hidden_dim, C, output_dim)
        self.biases2 = nn.Parameter(torch.randn(1, num_channels, output_dim))           # Shape: (1, C, output_dim)
    def forward(self, x):
        # Input Shape: (B, C, input_dim)
        # Use matrix operations to calculate the output of all channels at once
        outputs = torch.einsum('bci,ich->bch', x, self.weights1) + self.biases1
        # Current shape: (B, C, hidden_dim)
        # Use matrix operations to calculate the output of all channels at once
        outputs = torch.einsum('bch,hco->bco', outputs, self.weights2) + self.biases2
        # Current shape: [batch_size, num_channels, output_dim]
        # Multi-head attention mechanism
        query = outputs
        key = outputs
        value = outputs
        attn_output, attn_weights = self.mha(query, key, value)
        # Calculate the attention weights of each attention head for N=C * output_dim features
        attn_weights_per_head = attn_weights.mean(dim=-1)  # Shape: (B, num_heads, C)
        attn_output = self.out(attn_output)
        return attn_output, attn_weights_per_head


if __name__ == '__main__':
    # N = C * D_dim
    input = torch.randn(1, 6 * 156, 1)
    
    test_net = IndependentLinearLayersEfficient(input_dim=1, hidden_dim=32, output_dim=16, num_channels=156 * 6, drop=0.5, classes_num=18)

    y_pred, y_attn = test_net(input)
    # print(y_pred)
    print(y_pred.shape)
    print(y_attn.shape)

    print('--------------------------------------------------------------------------------')
    total = sum([param.nelement() for param in test_net.parameters()])
    # 1048576 == 1024 * 1024
    # 1073741824 == 1024 * 1024 * 1024
    # %.2f: Keep 2 decimal places
    print("Number of parameter: %.2fM" % (total / 1048576))

    flops, params = profile(test_net, inputs=(input,))
    print("Number of flops: %.2fG" % (flops / 1073741824))
    print("Number of parameters: %.2fM" % (params / 1048576))
    print("Number of flops: %.2f" % (flops / 1))
    print("Number of parameters: %.2f" % (params / 1))