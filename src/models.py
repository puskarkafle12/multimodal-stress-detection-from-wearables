"""
Modeling Framework
=================

This module contains multiple neural network architectures for stress prediction:
- 1D CNN/TCN
- Transformer encoder
- Multimodal fusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TemporalBlock(nn.Module):
    """Temporal Convolutional Network block"""
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, 
                 dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """Remove padding from the right side"""
    
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class StressCNN(nn.Module):
    """1D CNN model for stress prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, 
                 dropout: float = 0.2, window_length: int = 600):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_length = window_length
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
        
        # Batch normalization and dropout
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Improved output layer with intermediate layer for better capacity
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim // 2)
        self.fc1_dropout = nn.Dropout(dropout * 0.5)
        # Output layer: predict sequence instead of single value
        self.fc2 = nn.Linear(hidden_dim // 2, self.window_length)  # [batch_size, window_length]
        
    def forward(self, x):
        # x shape: [batch_size, window_length, input_dim]
        x = x.transpose(1, 2)  # [batch_size, input_dim, window_length]
        
        # Apply convolutional layers with residual connections for better gradient flow
        for i, (conv, bn, dropout) in enumerate(zip(self.conv_layers, self.batch_norms, self.dropouts)):
            residual = x if i > 0 and x.shape[1] == self.hidden_dim else None
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            # Add residual connection if dimensions match
            if residual is not None:
                x = x + residual  # Residual connection
            x = dropout(x)
        
        # For sequence prediction, preserve temporal structure
        # x shape: [batch_size, hidden_dim, window_length]
        # Transpose to [batch_size, window_length, hidden_dim] for per-timestep processing
        x = x.transpose(1, 2)  # [batch_size, window_length, hidden_dim]
        
        # Reshape to apply FC layers: [batch_size * window_length, hidden_dim]
        batch_size, window_length, hidden_dim = x.shape
        x = x.reshape(batch_size * window_length, hidden_dim)
        
        # Apply FC layers
        x = self.fc1(x)  # [batch_size * window_length, hidden_dim // 2]
        x = self.fc1_bn(x)  # Batch norm
        x = F.relu(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)  # [batch_size * window_length, window_length]
        
        # Reshape back: [batch_size, window_length, window_length]
        x = x.reshape(batch_size, window_length, self.window_length)
        
        # Extract diagonal: for timestep t, take prediction at position t
        # This gives [batch_size, window_length] where each position corresponds to its timestep
        indices = torch.arange(window_length, device=x.device)
        x = x[:, indices, indices]  # [batch_size, window_length]
        
        return x

class StressTCN(nn.Module):
    """Temporal Convolutional Network for stress prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3,
                 dropout: float = 0.2, kernel_size: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # TCN layers
        self.tcn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            self.tcn_layers.append(
                TemporalBlock(hidden_dim, hidden_dim, kernel_size, stride=1,
                             dilation=dilation, padding=padding, dropout=dropout)
            )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [batch_size, window_length, input_dim]
        x = x.transpose(1, 2)  # [batch_size, input_dim, window_length]
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply TCN layers
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)
        
        # Global pooling
        x = self.global_pool(x)  # [batch_size, hidden_dim, 1]
        x = x.squeeze(-1)  # [batch_size, hidden_dim]
        
        # Output
        x = self.fc(x)  # [batch_size, 1]
        
        return x

class StressLSTM(nn.Module):
    """LSTM model for stress prediction (as per TA discussion)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension (2x if bidirectional)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Enhanced output layer: predict sequence from LSTM outputs
        # Use all LSTM outputs (not just last hidden state) for sequence prediction
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.fc1_dropout = nn.Dropout(dropout * 0.5)
        # Output sequence: one value per timestep
        self.fc2 = nn.Linear(lstm_output_dim // 2, 1)  # Will be applied per timestep
        
    def forward(self, x):
        # x shape: [batch_size, window_length, input_dim]
        # Input projection
        x = self.input_proj(x)  # [batch_size, window_length, hidden_dim]
        
        # LSTM forward pass - get outputs for all timesteps
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch_size, window_length, lstm_output_dim]
        
        # Apply FC layers to each timestep to predict stress sequence
        batch_size, window_length, hidden_dim = lstm_out.shape
        # Reshape to apply FC layers: [batch_size * window_length, hidden_dim]
        lstm_flat = lstm_out.reshape(batch_size * window_length, hidden_dim)
        
        x = self.fc1(lstm_flat)  # [batch_size * window_length, hidden_dim // 2]
        x = F.relu(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)  # [batch_size * window_length, 1]
        
        # Reshape back to sequence: [batch_size, window_length]
        x = x.reshape(batch_size, window_length)
        
        return x

class StressTransformer(nn.Module):
    """Transformer encoder model for stress prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3,
                 num_heads: int = 8, dropout: float = 0.2, window_length: int = 12):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_length = window_length
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, window_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Enhanced output layer: predict sequence from transformer outputs
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc1_dropout = nn.Dropout(dropout * 0.5)
        # Output sequence: one value per timestep (excluding CLS token)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)  # Will be applied per timestep
        
    def forward(self, x):
        # x shape: [batch_size, window_length, input_dim]
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, window_length, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, window_length+1, hidden_dim]
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, window_length+1, hidden_dim]
        
        # Use all timestep outputs (excluding CLS token) for sequence prediction
        sequence_output = x[:, 1:, :]  # [batch_size, window_length, hidden_dim]
        
        # Apply FC layers to each timestep
        batch_size, window_length, hidden_dim = sequence_output.shape
        # Reshape to apply FC layers: [batch_size * window_length, hidden_dim]
        seq_flat = sequence_output.reshape(batch_size * window_length, hidden_dim)
        
        x = self.fc1(seq_flat)  # [batch_size * window_length, hidden_dim // 2]
        x = F.relu(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)  # [batch_size * window_length, 1]
        
        # Reshape back to sequence: [batch_size, window_length]
        x = x.reshape(batch_size, window_length)
        
        return x

class ModalityEncoder(nn.Module):
    """Individual encoder for each modality"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, window_length, input_dim]
        x = x.transpose(1, 2)  # [batch_size, input_dim, window_length]
        x = self.encoder(x)  # [batch_size, hidden_dim, 1]
        return x.squeeze(-1)  # [batch_size, hidden_dim]

class ModalityCNNEncoder(nn.Module):
    """CNN encoder that returns embedding before final FC layer (for late fusion)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, window_length: int = 12):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_length = window_length
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
        
        # Batch normalization and dropout
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: [batch_size, window_length, input_dim]
        x = x.transpose(1, 2)  # [batch_size, input_dim, window_length]
        
        # Apply convolutional layers
        for conv, bn, dropout in zip(self.conv_layers, self.batch_norms, self.dropouts):
            x = F.relu(bn(conv(x)))
            x = dropout(x)
        
        # Global pooling
        x = self.global_pool(x)  # [batch_size, hidden_dim, 1]
        x = x.squeeze(-1)  # [batch_size, hidden_dim]
        
        return x  # Return embedding without final FC layer

class EarlyFusionModel(nn.Module):
    """Early fusion model: V1+V2+..V5 -> model (as per TA discussion)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3,
                 dropout: float = 0.2, window_length: int = 12, encoder_type: str = "cnn",
                 bidirectional: bool = False, num_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim  # Should be 5 signals * 2 (value + mask) = 10
        self.hidden_dim = hidden_dim
        self.window_length = window_length
        self.encoder_type = encoder_type
        
        # Shared encoder (treats all signals as one input)
        if encoder_type == "cnn":
            self.encoder = StressCNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                window_length=window_length
            )
        elif encoder_type == "lstm":
            self.encoder = StressLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif encoder_type == "transformer":
            self.encoder = StressTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                window_length=window_length
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'cnn', 'lstm', or 'transformer'")
        
    def forward(self, x):
        # x shape: [batch_size, window_length, input_dim]
        # Input already contains all 5 signals concatenated
        return self.encoder(x)

class AttentionFusion(nn.Module):
    """Attention-based fusion for late fusion (as per TA discussion)"""
    
    def __init__(self, hidden_dim: int, num_modalities: int, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Attention mechanism
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, modality_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_embeddings: List of tensors, each of shape [batch_size, hidden_dim]
        
        Returns:
            Fused embedding of shape [batch_size, hidden_dim]
        """
        # Stack embeddings: [batch_size, num_modalities, hidden_dim]
        stacked = torch.stack(modality_embeddings, dim=1)
        batch_size = stacked.size(0)
        
        # Compute attention
        query = self.query(stacked.mean(dim=1))  # [batch_size, hidden_dim]
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        key = self.key(stacked)  # [batch_size, num_modalities, hidden_dim]
        value = self.value(stacked)  # [batch_size, num_modalities, hidden_dim]
        
        # Attention scores
        scores = torch.bmm(query, key.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # [batch_size, 1, num_modalities]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        fused = torch.bmm(attn_weights, value)  # [batch_size, 1, hidden_dim]
        fused = fused.squeeze(1)  # [batch_size, hidden_dim]
        
        # Output projection
        fused = self.output_proj(fused)
        
        return fused

class LateFusionModel(nn.Module):
    """Late fusion model: V1->model->h1, v2->model->h2 separately, fuse h1_h2…h5 (as per TA discussion)"""
    
    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int = 64,
                 fusion_dim: int = 128, dropout: float = 0.2, window_length: int = 12,
                 fusion_type: str = "attention"):  # "attention" or "concatenate"
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_type = fusion_type
        
        # Individual modality encoders (each signal gets its own encoder)
        self.modality_encoders = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            # Use CNN encoder for each modality (returns embedding before FC layer)
            self.modality_encoders[modality] = ModalityCNNEncoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
                window_length=window_length
            )
        
        # Fusion layer
        if fusion_type == "attention":
            self.fusion = AttentionFusion(hidden_dim, len(modality_dims), dropout)
            fusion_input_dim = hidden_dim
        else:  # concatenate
            fusion_input_dim = len(modality_dims) * hidden_dim
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            fusion_input_dim = fusion_dim // 2
        
        # Output layer: predict sequence instead of single value
        # Need window_length for sequence output
        self.window_length = window_length
        if fusion_type == "attention":
            self.fc = nn.Linear(hidden_dim, 1)  # Will be applied per timestep
        else:
            self.fc = nn.Linear(fusion_dim // 2, 1)  # Will be applied per timestep
        
    def forward(self, x_dict: Dict[str, torch.Tensor]):
        # Encode each modality separately for each timestep
        batch_size = next(iter(x_dict.values())).size(0)
        window_length = next(iter(x_dict.values())).size(1)
        device = next(iter(x_dict.values())).device
        
        # Process each timestep separately, then fuse
        timestep_outputs = []
        for t in range(window_length):
            modality_embeddings = []
            for modality, x in x_dict.items():
                if modality in self.modality_encoders:
                    # Extract timestep t: [batch_size, 2] -> [batch_size, 1, 2]
                    x_t = x[:, t:t+1, :]  # [batch_size, 1, 2]
                    # Get embedding for this timestep
                    embedding = self.modality_encoders[modality](x_t)  # [batch_size, hidden_dim]
                    modality_embeddings.append(embedding)
            
            if modality_embeddings:
                # Fuse embeddings for this timestep
                if self.fusion_type == "attention":
                    fused_t = self.fusion(modality_embeddings)  # [batch_size, hidden_dim]
                else:
                    fused_t = torch.cat(modality_embeddings, dim=1)  # [batch_size, num_modalities * hidden_dim]
                    fused_t = self.fusion(fused_t)  # [batch_size, fusion_dim // 2]
                
                # Apply output layer to this timestep
                output_t = self.fc(fused_t)  # [batch_size, 1]
                timestep_outputs.append(output_t)
            else:
                # Handle case where no modalities are available
                output_t = torch.zeros(batch_size, 1).to(device)
                timestep_outputs.append(output_t)
        
        # Stack timestep outputs: [batch_size, window_length, 1]
        output = torch.cat(timestep_outputs, dim=1)  # [batch_size, window_length]
        
        return output

class MultimodalStressModel(nn.Module):
    """Multimodal model with separate encoders for each modality (legacy - kept for compatibility)"""
    
    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int = 64,
                 fusion_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.modality_dims = modality_dims
        
        # Individual modality encoders
        self.modality_encoders = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.modality_encoders[modality] = ModalityEncoder(dim, hidden_dim, dropout)
        
        # Fusion layer
        total_dim = len(modality_dims) * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.fc = nn.Linear(fusion_dim // 2, 1)
        
    def forward(self, x_dict: Dict[str, torch.Tensor]):
        # Encode each modality
        modality_embeddings = []
        for modality, x in x_dict.items():
            if modality in self.modality_encoders:
                embedding = self.modality_encoders[modality](x)
                modality_embeddings.append(embedding)
        
        # Concatenate embeddings
        if modality_embeddings:
            fused = torch.cat(modality_embeddings, dim=1)
        else:
            # Handle case where no modalities are available
            batch_size = next(iter(x_dict.values())).size(0)
            fused = torch.zeros(batch_size, 0).to(next(iter(x_dict.values())).device)
        
        # Fusion and output
        fused = self.fusion(fused)
        output = self.fc(fused)
        
        return output

class ModelFactory:
    """Factory class for creating different model architectures"""
    
    @staticmethod
    def create_model(model_type: str, input_dim: int, config) -> nn.Module:
        """Create a model based on the specified type"""
        
        if model_type == "cnn":
            return StressCNN(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                window_length=int(config.window_length_min * config.sampling_rate * 60)
            )
        
        elif model_type == "tcn":
            return StressTCN(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
        
        elif model_type == "lstm":
            bidirectional = getattr(config, 'bidirectional', False)
            return StressLSTM(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                bidirectional=bidirectional
            )
        
        elif model_type == "transformer":
            num_heads = getattr(config, 'num_heads', 8)
            return StressTransformer(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=num_heads,
                dropout=config.dropout,
                window_length=int(config.window_length_min / 5)  # samples at 5-min intervals
            )
        
        elif model_type == "early_fusion":
            # Early fusion: all 5 signals concatenated (10 features: 5 signals * 2 channels each)
            # Support encoder_type: "cnn", "lstm", or "transformer"
            encoder_type = getattr(config, 'encoder_type', 'cnn')
            bidirectional = getattr(config, 'bidirectional', False)
            num_heads = getattr(config, 'num_heads', 8)
            
            return EarlyFusionModel(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                window_length=int(config.window_length_min / 5),  # samples at 5-min intervals
                encoder_type=encoder_type,
                bidirectional=bidirectional,
                num_heads=num_heads
            )
        
        elif model_type == "late_fusion":
            # Late fusion: Use selected modalities from config
            # Get selected modalities from config, or use all 5 by default
            all_modalities = ['heart_rate', 'sleep', 'cgm', 'oxygen_saturation', 'respiratory_rate']
            selected_modalities = getattr(config, 'selected_modalities', all_modalities)
            
            # Validate and filter
            selected_modalities = [m for m in selected_modalities if m in all_modalities]
            if not selected_modalities:
                selected_modalities = all_modalities
            
            # Create modality_dims dict for selected modalities only
            modality_dims = {mod: 2 for mod in selected_modalities}  # value + mask
            
            return LateFusionModel(
                modality_dims=modality_dims,
                hidden_dim=config.hidden_dim // 2,
                fusion_dim=config.hidden_dim,
                dropout=config.dropout,
                window_length=int(config.window_length_min / 5),  # samples at 5-min intervals
                fusion_type="attention"  # or "concatenate"
            )
        
        elif model_type == "multimodal":
            # Define modality dimensions (value + mask for each)
            modality_dims = {
                'heart_rate': 2,
                'oxygen_saturation': 2,
                'respiratory_rate': 2,
                'physical_activity': 2,
                'physical_activity_calorie': 2,
                'sleep': 2,
                'cgm': 2
            }
            
            return MultimodalStressModel(
                modality_dims=modality_dims,
                hidden_dim=config.hidden_dim // 2,  # Smaller per-modality encoders
                fusion_dim=config.hidden_dim,
                dropout=config.dropout
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class LossFunctions:
    """Custom loss functions for stress prediction"""
    
    @staticmethod
    def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Huber loss for robust regression"""
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=delta)
        linear = abs_diff - quadratic
        return 0.5 * quadratic ** 2 + delta * linear
    
    @staticmethod
    def temporal_smoothness_loss(pred: torch.Tensor, window_ids: List[str], 
                                lambda_smooth: float = 0.1) -> torch.Tensor:
        """Encourage temporal smoothness in predictions"""
        if len(pred) < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Group predictions by participant
        participant_preds = {}
        for i, window_id in enumerate(window_ids):
            participant_id = window_id.split('_')[0]
            if participant_id not in participant_preds:
                participant_preds[participant_id] = []
            participant_preds[participant_id].append(pred[i])
        
        smoothness_loss = 0.0
        for participant_id, preds in participant_preds.items():
            if len(preds) > 1:
                preds_tensor = torch.stack(preds)
                diff = torch.diff(preds_tensor, dim=0)
                smoothness_loss += torch.mean(diff ** 2)
        
        return lambda_smooth * smoothness_loss
    
    @staticmethod
    def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, 
                   gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        # Convert to binary classification
        target_binary = (target > 0.5).float()
        
        # Compute cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred, target_binary, reduction='none')
        
        # Compute p_t
        p_t = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
        
        return focal_loss.mean()
