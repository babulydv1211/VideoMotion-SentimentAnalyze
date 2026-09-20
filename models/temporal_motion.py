"""
Temporal Motion Model using LSTM and ConvLSTM
"""

import torch
import torch.nn as nn


class TemporalMotionLSTM(nn.Module):
    """
    LSTM-based temporal motion model
    Processes optical flow features and frame embeddings over time
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        """
        Initialize temporal motion LSTM
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): LSTM hidden dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
        """
        super(TemporalMotionLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(self, motion_features):
        """
        Process motion features through LSTM
        
        Args:
            motion_features (torch.Tensor): Shape (B, T, input_dim)
                - B: batch size
                - T: sequence length
                - input_dim: feature dimension
        
        Returns:
            torch.Tensor: LSTM outputs (B, T, output_dim)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(motion_features)
        
        return lstm_out


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for processing spatial-temporal features"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size):
        """
        Initialize ConvLSTM cell
        
        Args:
            input_dim (int): Number of input channels
            hidden_dim (int): Number of hidden channels
            kernel_size (int): Kernel size for convolution
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Convolution combining input and hidden state
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )
    
    def forward(self, x, h, c):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input (B, input_dim, H, W)
            h (torch.Tensor): Hidden state (B, hidden_dim, H, W)
            c (torch.Tensor): Cell state (B, hidden_dim, H, W)
        
        Returns:
            tuple: (h_new, c_new)
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Convolution
        gates = self.conv(combined)
        
        # Split into input, forget, cell, output gates
        i_t, f_t, g_t, o_t = torch.chunk(gates, 4, dim=1)
        
        # Apply activations
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)
        
        # Update cell and hidden states
        c_new = f_t * c + i_t * g_t
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new


class TemporalMotionConvLSTM(nn.Module):
    """
    ConvLSTM-based temporal motion model
    Useful for processing spatial-temporal motion maps
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, kernel_size=3):
        """
        Initialize ConvLSTM
        
        Args:
            input_dim (int): Number of input channels
            hidden_dim (int): Number of hidden channels
            num_layers (int): Number of ConvLSTM layers
            kernel_size (int): Kernel size
        """
        super(TemporalMotionConvLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Build ConvLSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size))
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input (B, T, C, H, W)
        
        Returns:
            torch.Tensor: Output (B, T, hidden_dim, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Initialize hidden and cell states
        h = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) 
             for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) 
             for _ in range(self.num_layers)]
        
        outputs = []
        
        # Process sequence
        for t in range(T):
            x_t = x[:, t, :, :, :]  # (B, C, H, W)
            
            for layer_idx in range(self.num_layers):
                x_t, c[layer_idx] = self.layers[layer_idx](x_t, h[layer_idx], c[layer_idx])
                h[layer_idx] = x_t
            
            outputs.append(x_t.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)  # (B, T, hidden_dim, H, W)
        
        return outputs
