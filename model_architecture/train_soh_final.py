# train_soh_final.py
"""
SOH Prediction with CNN-Mamba-UQ - FINAL OPTIMIZED VERSION
Target: MAE < 0.7%, R² > 0.97
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# CONFIGURATION - OPTIMIZED FOR SOH
# ============================================================

CFG = {
    # Paths
    'data_path': r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv",
    'save_path': r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\soh_best.pt",
    
    # Features (6 features, cycle_index removed as it's implicit in sequence)
    'feat_cols': [
        "dc_internal_resistance", "temperature_avg",
        "charge_capacity", "charge_energy",
        "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    ],
    
    # Sequence parameters
    'window_size': 40,        # 40 cycles input (increased for better context)
    'stride': 1,
    
    # Model architecture
    'input_dim': 6,
    'cnn_channels': [64, 128, 256],   # Larger CNN
    'cnn_kernels': [5, 9, 15],        # Multi-scale kernels
    'd_model': 256,                    # Larger Mamba hidden dim
    'd_state': 32,                     # Larger state space
    'd_conv': 4,
    'expand': 2,
    'n_mamba_layers': 4,               # More Mamba layers
    'dropout': 0.1,
    
    # Training
    'epochs': 100,
    'batch_size': 128,                 # Balanced batch size
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    
    # Loss weights
    'tail_weight': 3.0,                # Weight for low SOH samples
    
    # Early stopping
    'patience': 20,
    'min_delta': 1e-6,
}

# ============================================================
# DATA PREPROCESSING
# ============================================================

class BatteryPreprocessor:
    def __init__(self, feat_cols):
        self.feat_cols = feat_cols
        self.scaler = StandardScaler()
    
    def fit(self, df_train):
        self.scaler.fit(df_train[self.feat_cols].values)
        return self
    
    def transform(self, df):
        df = df.copy()
        df[self.feat_cols] = self.scaler.transform(df[self.feat_cols].values)
        return df


def load_and_preprocess(data_path, feat_cols):
    """Load data and preprocess with proper split"""
    df = pd.read_csv(data_path)
    
    # Check if split column exists
    if 'split' not in df.columns:
        raise ValueError("'split' column not found in CSV! Please recreate CSV from pickle.")
    
    print(f"\nSplit distribution:")
    for split in ['train', 'val', 'test']:
        count = (df['split'] == split).sum()
        cells = df[df['split'] == split]['cell_id'].nunique() if 'cell_id' in df.columns else 'N/A'
        print(f"  {split}: {count} rows, {cells} cells")
    
    # Fit preprocessor on training data only
    train_df = df[df['split'] == 'train']
    preprocessor = BatteryPreprocessor(feat_cols)
    preprocessor.fit(train_df)
    
    # Transform all data
    df = preprocessor.transform(df)
    
    return df, preprocessor

# ============================================================
# DATASET
# ============================================================

class SOHSequenceDataset(Dataset):
    """Sliding window dataset for SOH prediction"""
    
    def __init__(self, df, feat_cols, window_size, stride=1, split=None):
        self.window_size = window_size
        self.samples = []
        self.weights = []
        
        # Filter by split
        if split is not None:
            df = df[df['split'] == split]
        
        # Group by cell_id
        for cell_id, cell_df in df.groupby('cell_id'):
            cell_df = cell_df.sort_values('cycle_index').reset_index(drop=True)
            X = cell_df[feat_cols].values.astype(np.float32)
            y = cell_df['soh'].values.astype(np.float32)
            
            # Create sliding windows
            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                x_win = X[start:end]
                y_last = y[end - 1]
                
                self.samples.append((x_win, y_last))
                
                # Weight: more weight on degraded samples
                weight = CFG['tail_weight'] if y_last < 0.90 else 1.0
                self.weights.append(weight)
        
        self.weights = np.array(self.weights, dtype=np.float32)
        print(f"  {split if split else 'all'}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.FloatTensor(x),
            torch.FloatTensor([y]),
            torch.FloatTensor([self.weights[idx]])
        )

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, channels, kernels, dropout=0.1):
        super().__init__()
        assert len(channels) == len(kernels)
        self.branches = nn.ModuleList()
        
        for ch, k in zip(channels, kernels):
            self.branches.append(nn.Sequential(
                nn.Conv1d(input_dim, ch, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            ))
        
        self.out_dim = sum(channels)
        self.proj = nn.Linear(self.out_dim, self.out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, W, C) -> (B, C, W)
        x = x.permute(0, 2, 1)
        branch_outs = [b(x) for b in self.branches]
        x = torch.cat(branch_outs, dim=1)
        x = x.permute(0, 2, 1)
        x = self.dropout(F.gelu(self.proj(x)))
        return x


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv-1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def ssm(self, x):
        B, L, D = x.shape
        N = self.d_state
        
        delta_BC = self.x_proj(x)
        delta_raw = delta_BC[..., :1]
        B_ssm = delta_BC[..., 1:N+1]
        C_ssm = delta_BC[..., N+1:]
        
        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB_u = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)
        
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB_u[:, t]
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_.permute(0, 2, 1))[..., :x_.shape[1]].permute(0, 2, 1)
        x_conv = F.silu(x_conv)
        
        y = self.ssm(x_conv)
        y = y * F.silu(z)
        y = self.dropout(self.out_proj(y))
        return y + residual


class MambaEncoder(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class CNNMambaUQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg
        
        self.cnn = MultiScaleCNN(C['input_dim'], C['cnn_channels'], C['cnn_kernels'], C['dropout'])
        cnn_out_dim = sum(C['cnn_channels'])
        
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, C['d_model']),
            nn.LayerNorm(C['d_model']),
            nn.GELU(),
            nn.Dropout(C['dropout']),
        )
        
        self.mamba = MambaEncoder(C['d_model'], C['d_state'], C['d_conv'], 
                                   C['expand'], C['n_mamba_layers'], C['dropout'])
        
        self.attn_pool = nn.Linear(C['d_model'], 1)
        
        # SOH Head
        self.soh_head = nn.Sequential(
            nn.Linear(C['d_model'], 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(C['dropout']),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(C['dropout']),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
    
    def encode(self, x):
        z = self.cnn(x)
        z = self.cnn_proj(z)
        z = self.mamba(z)
        attn = F.softmax(self.attn_pool(z), dim=1)
        z = (z * attn).sum(dim=1)
        return z
    
    def forward(self, x):
        z = self.encode(x)
        return self.soh_head(z).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# TRAINING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            return self.counter >= self.patience
        return False
    
    def restore(self, model):
        model.load_state_dict(self.best_state)


def train_epoch(model, loader, optimizer, cfg):
    model.train()
    total_loss = 0
    
    for x, y, w in loader:
        x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = (w * (pred - y.squeeze()) ** 2).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    
    with torch.no_grad():
        for x, y, w in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y.squeeze().cpu().numpy())
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    mae = mean_absolute_error(all_true, all_pred) * 100
    r2 = r2_score(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred)) * 100
    
    return mae, r2, rmse


def train(model, train_loader, val_loader, cfg):
    print("\n" + "="*60)
    print("TRAINING CNN-MAMBA-UQ FOR SOH PREDICTION")
    print("="*60)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    early_stopping = EarlyStopping(patience=cfg['patience'], min_delta=cfg['min_delta'])
    
    history = {'train_loss': [], 'val_mae': [], 'val_r2': [], 'val_rmse': []}
    best_val_mae = float('inf')
    
    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Learning rate: {cfg['learning_rate']}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, cfg)
        val_mae, val_r2, val_rmse = validate(model, val_loader)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['val_rmse'].append(val_rmse)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), cfg['save_path'])
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"Loss: {train_loss:.5f} | "
                  f"Val MAE: {val_mae:.4f}% | "
                  f"Val RMSE: {val_rmse:.4f}% | "
                  f"R²: {val_r2:.4f}")
        
        if early_stopping(train_loss, model):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    early_stopping.restore(model)
    return history

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, test_loader, cfg):
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    model.eval()
    all_pred, all_true = [], []
    
    with torch.no_grad():
        for x, y, w in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y.squeeze().cpu().numpy())
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    # Overall metrics
    mae = mean_absolute_error(all_true, all_pred) * 100
    rmse = np.sqrt(mean_squared_error(all_true, all_pred)) * 100
    r2 = r2_score(all_true, all_pred)
    
    # Error by SOH region
    regions = [
        ('SOH < 0.85', all_true < 0.85),
        ('0.85 ≤ SOH < 0.90', (all_true >= 0.85) & (all_true < 0.90)),
        ('0.90 ≤ SOH < 0.95', (all_true >= 0.90) & (all_true < 0.95)),
        ('SOH ≥ 0.95', all_true >= 0.95),
    ]
    
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall Metrics:")
    print(f"  MAE:  {mae:.4f}%  (target: <0.70%)")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  R²:   {r2:.4f}   (target: >0.97)")
    
    print(f"\nError by SOH Region:")
    for name, mask in regions:
        if mask.sum() > 0:
            region_mae = mean_absolute_error(all_true[mask], all_pred[mask]) * 100
            print(f"  {name}: MAE = {region_mae:.4f}% (n={mask.sum()})")
    
    # Check targets
    targets_met = (mae < 0.70) and (r2 > 0.97)
    print(f"\n{'='*60}")
    print(f"TARGETS MET: {'✓ YES' if targets_met else '✗ NO'}")
    print(f"{'='*60}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': all_pred, 'targets': all_true}

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("CNN-MAMBA-UQ FOR BATTERY SOH PREDICTION")
    print("="*60)
    
    # Load data
    print("\n[1] Loading and preprocessing data...")
    df, preprocessor = load_and_preprocess(CFG['data_path'], CFG['feat_cols'])
    
    # Create datasets
    print("\n[2] Creating sequences...")
    train_ds = SOHSequenceDataset(df, CFG['feat_cols'], CFG['window_size'], CFG['stride'], 'train')
    val_ds = SOHSequenceDataset(df, CFG['feat_cols'], CFG['window_size'], CFG['stride'], 'val')
    test_ds = SOHSequenceDataset(df, CFG['feat_cols'], CFG['window_size'], CFG['stride'], 'test')
    
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False, num_workers=0)
    
    # Create model
    print("\n[3] Building model...")
    model = CNNMambaUQ(CFG).to(DEVICE)
    print(f"  Total parameters: {count_parameters(model):,}")
    
    # Train
    history = train(model, train_loader, val_loader, CFG)
    
    # Evaluate
    results = evaluate(model, test_loader, CFG)
    
    # Save preprocessor
    import joblib
    joblib.dump(preprocessor.scaler, 'soh_scaler.pkl')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {CFG['save_path']}")
    print(f"Scaler saved to: soh_scaler.pkl")
    print("="*60)
    
    return model, history, results

if __name__ == "__main__":
    model, history, results = main()