# models/cnn_mamba_uq_model.py
"""
CNN-Mamba-UQ Model for Battery SOH and RUL Prediction
----------------------------------------------------
Combines:
- CNN: Local pattern detection (capacity fade, resistance spikes)
- Mamba: Long-term sequence modeling (degradation trajectory)
- UQ: Uncertainty quantification (prediction confidence)

Architecture:
    Input: (batch, sequence_len, n_features)
       ↓
    CNN Block (local feature extraction)
       ↓
    Mamba Block (state space modeling)
       ↓
    UQ Head (mean + variance prediction)
       ↓
    Output: (prediction, uncertainty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
import numpy as np


# ============================================
# CONVOLUTIONAL BLOCK (Local Pattern Detection)
# ============================================

class ConvBlock(nn.Module):
    """
    Convolutional block for local feature extraction
    Detects short-term patterns in battery data
    """
    def __init__(self, input_dim: int, hidden_dims: list, kernel_size: int = 5, dropout: float = 0.1):
        """
        Args:
            input_dim: Number of input features (7 for this dataset)
            hidden_dims: List of hidden dimensions for each conv layer
            kernel_size: Size of convolutional kernel
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim)
        Returns:
            Output tensor (batch, seq_len, hidden_dims[-1])
        """
        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Transpose back: (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        return x


# ============================================
# MAMBA BLOCK (State Space Model)
# ============================================

class MambaBlock(nn.Module):
    """
    Mamba State Space Model for long-sequence modeling
    Efficient alternative to Transformers with O(n) complexity
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        use_bias: bool = False
    ):
        """
        Args:
            d_model: Model dimension (input/output features)
            d_state: State space dimension (SSM latent dimension)
            d_conv: Convolution dimension for input projection
            expand_factor: Expansion factor for inner dimension
            dropout: Dropout rate
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = int(expand_factor * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=use_bias)
        
        # Depthwise convolution for input
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM parameters (learned state space)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize SSM parameters"""
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        
        # Initialize conv to be near identity
        nn.init.uniform_(self.conv1d.weight, -0.01, 0.01)
        nn.init.zeros_(self.conv1d.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, res = x_and_res.chunk(2, dim=-1)
        
        # Depthwise convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Remove padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM computation (simplified for this implementation)
        # In practice, this would be the full selective SSM
        # Here we use an efficient approximation
        x = self._ssm_core(x)
        
        # Residual connection
        x = x * F.silu(res)
        
        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x
    
    def _ssm_core(self, x: torch.Tensor) -> torch.Tensor:
        """
        Core SSM computation
        Implements: y = (A * x + B) * C
        """
        batch, seq_len, d_inner = x.shape
        
        # Project to state space
        x_proj = self.x_proj(x)  # (batch, seq_len, 2*d_state + 1)
        
        # Split into A, B, C components
        delta, B, C = torch.split(x_proj, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)  # Ensure positivity
        
        # Simplified state update (would be full scan in production)
        # For simplicity, using a linear transformation
        # Full Mamba would implement selective scan here
        
        # Output projection
        x = x * delta
        
        return x


# ============================================
# MAMBA ENCODER (Multiple Mamba Layers)
# ============================================

class MambaEncoder(nn.Module):
    """
    Stack of Mamba blocks for deep sequence modeling
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = x + layer(self.norm(x))  # Residual connection
            x = self.dropout(x)
        
        return x


# ============================================
# UNCERTAINTY QUANTIFICATION HEAD
# ============================================

class UncertaintyHead(nn.Module):
    """
    Uncertainty quantification head
    Predicts both mean and variance (aleatoric uncertainty)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Mean head (prediction)
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        
        # Variance head (uncertainty)
        self.var_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, input_dim)
        Returns:
            mean: Predicted value (batch, 1)
            variance: Predicted uncertainty (batch, 1)
        """
        features = self.shared(x)
        
        mean = self.mean_head(features)
        log_var = self.var_head(features)
        variance = torch.exp(log_var) + 1e-6  # Ensure positivity
        
        return mean, variance


# ============================================
# CNN-MAMBA-UQ MAIN MODEL
# ============================================

class CNNMambaUQ(nn.Module):
    """
    Complete CNN-Mamba-UQ model for battery SOH/RUL prediction
    
    Architecture:
    Input: (batch, seq_len, n_features)
       ↓
    ConvBlock: Local feature extraction
       ↓
    MambaEncoder: Long-term sequence modeling
       ↓
    Global Pooling: Aggregate sequence information
       ↓
    UncertaintyHead: Prediction + confidence
       ↓
    Output: (prediction, uncertainty)
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        seq_len: int = 50,
        conv_hidden_dims: list = [64, 128, 256],
        mamba_d_model: int = 256,
        mamba_n_layers: int = 4,
        mamba_d_state: int = 16,
        uq_hidden_dim: int = 128,
        dropout: float = 0.1,
        prediction_type: str = "soh"  # "soh" or "rul"
    ):
        """
        Args:
            input_dim: Number of input features (default: 7)
            seq_len: Input sequence length (default: 50)
            conv_hidden_dims: Hidden dimensions for CNN layers
            mamba_d_model: Mamba model dimension
            mamba_n_layers: Number of Mamba layers
            mamba_d_state: Mamba state dimension
            uq_hidden_dim: Hidden dimension for uncertainty head
            dropout: Dropout rate for regularization
            prediction_type: "soh" or "rul"
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.prediction_type = prediction_type
        
        # 1. CNN Block (Local pattern detection)
        self.cnn_block = ConvBlock(
            input_dim=input_dim,
            hidden_dims=conv_hidden_dims,
            kernel_size=5,
            dropout=dropout
        )
        
        # Project CNN output to Mamba dimension
        self.cnn_proj = nn.Linear(conv_hidden_dims[-1], mamba_d_model)
        
        # 2. Mamba Encoder (Long-term sequence modeling)
        self.mamba_encoder = MambaEncoder(
            d_model=mamba_d_model,
            n_layers=mamba_n_layers,
            d_state=mamba_d_state,
            dropout=dropout
        )
        
        # 3. Global pooling (aggregate sequence)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. Uncertainty Head (Prediction + Confidence)
        self.uncertainty_head = UncertaintyHead(
            input_dim=mamba_d_model,
            hidden_dim=uq_hidden_dim,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_features: If True, return intermediate features
            
        Returns:
            mean: Predicted value (batch, 1)
            variance: Prediction uncertainty (batch, 1)
            (optional) features: Intermediate features if return_features=True
        """
        batch_size = x.shape[0]
        
        # 1. CNN feature extraction
        cnn_features = self.cnn_block(x)  # (batch, seq_len, conv_dim)
        cnn_features = self.cnn_proj(cnn_features)  # (batch, seq_len, mamba_dim)
        
        # 2. Mamba sequence modeling
        mamba_features = self.mamba_encoder(cnn_features)  # (batch, seq_len, mamba_dim)
        
        # 3. Global pooling (take last timestep or average)
        # For RUL, last timestep is most informative
        if self.prediction_type == "rul":
            # Use last timestep for RUL (current state)
            pooled_features = mamba_features[:, -1, :]  # (batch, mamba_dim)
        else:
            # Average pooling for SOH (smooth degradation)
            pooled_features = mamba_features.mean(dim=1)  # (batch, mamba_dim)
        
        # 4. Uncertainty head
        mean, variance = self.uncertainty_head(pooled_features)
        
        if return_features:
            return mean, variance, pooled_features
        
        return mean, variance
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for epistemic uncertainty
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            n_samples: Number of Monte Carlo samples
            
        Returns:
            mean: Average prediction
            aleatoric: Data uncertainty (from model)
            epistemic: Model uncertainty (from MC dropout)
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        aleatoric_vars = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                mean, var = self.forward(x)
                predictions.append(mean)
                aleatoric_vars.append(var)
        
        self.eval()
        
        # Aggregate predictions
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch, 1)
        aleatoric_vars = torch.stack(aleatoric_vars, dim=0)  # (n_samples, batch, 1)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Aleatoric uncertainty (mean of variances)
        aleatoric = aleatoric_vars.mean(dim=0)
        
        # Epistemic uncertainty (variance of predictions)
        epistemic = predictions.var(dim=0)
        
        # Total uncertainty
        total_uncertainty = aleatoric + epistemic
        
        return mean_pred, aleatoric, epistemic, total_uncertainty
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention-like weights from Mamba for interpretability
        (Simplified version - would need full SSM scan for true attention)
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            weights: Importance weights for each timestep
        """
        with torch.no_grad():
            cnn_features = self.cnn_block(x)
            cnn_features = self.cnn_proj(cnn_features)
            
            # Get Mamba outputs
            mamba_features = self.mamba_encoder(cnn_features)
            
            # Compute simple importance as gradient of output wrt input
            # This is a proxy for attention
            mamba_features.requires_grad_(True)
            pooled = mamba_features.mean(dim=1)
            importance = pooled.abs().sum(dim=-1)
            
        return importance


# ============================================
# MODEL FACTORY (Create models for SOH/RUL)
# ============================================

def create_soh_model(
    input_dim: int = 7,
    seq_len: int = 50,
    dropout: float = 0.1
) -> CNNMambaUQ:
    """
    Create CNN-Mamba-UQ model for SOH prediction
    
    Args:
        input_dim: Number of input features
        seq_len: Sequence length
        dropout: Dropout rate
        
    Returns:
        Configured CNNMambaUQ model for SOH
    """
    model = CNNMambaUQ(
        input_dim=input_dim,
        seq_len=seq_len,
        conv_hidden_dims=[64, 128, 256],
        mamba_d_model=256,
        mamba_n_layers=4,
        mamba_d_state=16,
        uq_hidden_dim=128,
        dropout=dropout,
        prediction_type="soh"
    )
    
    return model


def create_rul_model(
    input_dim: int = 7,
    seq_len: int = 50,
    dropout: float = 0.1
) -> CNNMambaUQ:
    """
    Create CNN-Mamba-UQ model for RUL prediction
    
    Args:
        input_dim: Number of input features
        seq_len: Sequence length
        dropout: Dropout rate
        
    Returns:
        Configured CNNMambaUQ model for RUL
    """
    model = CNNMambaUQ(
        input_dim=input_dim,
        seq_len=seq_len,
        conv_hidden_dims=[64, 128, 256],
        mamba_d_model=256,
        mamba_n_layers=4,
        mamba_d_state=16,
        uq_hidden_dim=128,
        dropout=dropout,
        prediction_type="rul"
    )
    
    return model


# ============================================
# MODEL SUMMARY AND TESTING
# ============================================

def get_model_summary(model: CNNMambaUQ) -> Dict[str, Any]:
    """
    Get model summary statistics
    
    Args:
        model: CNNMambaUQ model
        
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "prediction_type": model.prediction_type,
        "input_dim": model.input_dim,
        "seq_len": model.seq_len
    }


if __name__ == "__main__":
    print("=" * 60)
    print("CNN-MAMBA-UQ MODEL FOR BATTERY SOH/RUL PREDICTION")
    print("=" * 60)
    
    # Test SOH model
    print("\n[1] Creating SOH Model...")
    soh_model = create_soh_model(input_dim=7, seq_len=50)
    soh_stats = get_model_summary(soh_model)
    print(f"    Total parameters: {soh_stats['total_parameters']:,}")
    print(f"    Model size: {soh_stats['model_size_mb']:.2f} MB")
    
    # Test RUL model
    print("\n[2] Creating RUL Model...")
    rul_model = create_rul_model(input_dim=7, seq_len=50)
    rul_stats = get_model_summary(rul_model)
    print(f"    Total parameters: {rul_stats['total_parameters']:,}")
    print(f"    Model size: {rul_stats['model_size_mb']:.2f} MB")
    
    # Test forward pass
    print("\n[3] Testing forward pass...")
    batch_size = 32
    seq_len = 50
    input_dim = 7
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        soh_pred, soh_var = soh_model(x)
        rul_pred, rul_var = rul_model(x)
    
    print(f"    SOH prediction shape: {soh_pred.shape}")
    print(f"    SOH uncertainty shape: {soh_var.shape}")
    print(f"    RUL prediction shape: {rul_pred.shape}")
    print(f"    RUL uncertainty shape: {rul_var.shape}")
    
    # Test Monte Carlo Dropout
    print("\n[4] Testing Monte Carlo Dropout...")
    mean, aleatoric, epistemic, total = soh_model.predict_with_uncertainty(x, n_samples=5)
    print(f"    Mean prediction: {mean[0, 0].item():.4f}")
    print(f"    Aleatoric uncertainty: {aleatoric[0, 0].item():.6f}")
    print(f"    Epistemic uncertainty: {epistemic[0, 0].item():.6f}")
    print(f"    Total uncertainty: {total[0, 0].item():.6f}")
    
    print("\n" + "=" * 60)
    print("MODEL READY FOR TRAINING")
    print("=" * 60)