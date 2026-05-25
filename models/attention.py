"""
Attention Mechanisms for Temporal and Spatial Focus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism
    Learns to focus on emotionally important frames
    """
    
    def __init__(self, feature_dim, attention_dim=128):
        """
        Initialize temporal attention
        
        Args:
            feature_dim (int): Input feature dimension
            attention_dim (int): Attention hidden dimension
        """
        super(TemporalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, features, mask=None):
        """
        Apply temporal attention
        
        Args:
            features (torch.Tensor): Input features (B, T, feature_dim)
            mask (torch.Tensor): Optional mask for padding (B, T)
        
        Returns:
            tuple: (attended_features, attention_weights, attention_map)
        """
        # Compute attention scores
        scores = self.attention_net(features)  # (B, T, 1)
        scores = scores.squeeze(-1)  # (B, T)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=1)  # (B, T)
        
        # Apply attention to features
        attended = (features * attention_weights.unsqueeze(-1)).sum(dim=1)  # (B, feature_dim)
        
        return attended, attention_weights, scores


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism
    Focuses on significant motion regions in the frame
    """
    
    def __init__(self, input_channels):
        """
        Initialize spatial attention
        
        Args:
            input_channels (int): Number of input channels
        """
        super(SpatialAttention, self).__init__()
        
        # Channel attention
        self.fc1 = nn.Linear(input_channels, max(input_channels // 16, 1))
        self.fc2 = nn.Linear(max(input_channels // 16, 1), input_channels)
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    
    def channel_attention(self, x):
        """Channel attention branch"""
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        avg_att = self.fc2(F.relu(self.fc1(avg_pool)))
        
        # Global max pooling
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        max_att = self.fc2(F.relu(self.fc1(max_pool)))
        
        channel_att = (avg_att + max_att).sigmoid()
        return x * channel_att.view(x.size(0), x.size(1), 1, 1)
    
    def spatial_attention(self, x):
        """Spatial attention branch"""
        # Channel average and max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        x_cat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv(x_cat).sigmoid()
        
        return x * spatial_att
    
    def forward(self, x):
        """
        Apply spatial-channel attention
        
        Args:
            x (torch.Tensor): Input (B, C, H, W)
        
        Returns:
            torch.Tensor: Attended output
        """
        x_ch = self.channel_attention(x)
        x_sp = self.spatial_attention(x_ch)
        return x_sp


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for temporal feature fusion
    """
    
    def __init__(self, feature_dim, num_heads=4):
        """
        Initialize multi-head attention
        
        Args:
            feature_dim (int): Feature dimension
            num_heads (int): Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, values, keys, query, mask=None):
        """
        Forward pass
        
        Args:
            values (torch.Tensor): (B, T, feature_dim)
            keys (torch.Tensor): (B, T, feature_dim)
            query (torch.Tensor): (B, T, feature_dim)
            mask (torch.Tensor): Optional mask
        
        Returns:
            torch.Tensor: Attention output
        """
        N = query.shape[0]
        
        # Linear transformations
        Q = self.query(query)  # (B, T, feature_dim)
        K = self.key(keys)
        V = self.value(values)
        
        # Split into multiple heads
        Q = Q.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = K.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)  # (B, num_heads, T, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(N, -1, self.num_heads * self.head_dim)
        
        out = self.fc_out(out)
        
        return out
