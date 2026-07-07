"""
BiLSTM Baseline for Battery SOH Estimation
-------------------------------------------
- Same data, features, and split as CNN-Mamba-UQ
- Bidirectional LSTM with 2 layers
- Captures past and future context within the sequence
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"
SAVE_DIR = Path(__file__).parent
SAVE_DIR.mkdir(exist_ok=True)

# Feature columns (same as CNN-Mamba-UQ)
FEAT_COLS = [
    "dc_internal_resistance", "temperature_avg",
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "ir_rel", "cycle_pos",
]

# Training config
CFG = {
    "window_size": 50,
    "stride": 2,
    "batch_size": 256,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.15,
    "learning_rate": 1e-3,
    "epochs": 120,
    "patience": 25,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

print(f"Device: {CFG['device']}")

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def add_relative_features(df):
    """Add per-cell relative features (same as CNN-Mamba-UQ)"""
    df = df.copy()
    cap_rel_list, en_rel_list, ir_rel_list, cycle_pos_list = [], [], [], []

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle_index")
        early = cell_df.iloc[:10]

        nom_cap = early["charge_capacity"].mean()
        nom_energy = early["charge_energy"].mean()
        nom_ir = early["dc_internal_resistance"].mean()
        min_cycle = cell_df["cycle_index"].min()
        max_cycle = cell_df["cycle_index"].max()
        cyc_range = max(max_cycle - min_cycle, 1)

        cap_rel_list.append((cell_df["charge_capacity"] - nom_cap) / (nom_cap + 1e-9))
        en_rel_list.append((cell_df["charge_energy"] - nom_energy) / (nom_energy + 1e-9))
        ir_rel_list.append((cell_df["dc_internal_resistance"] - nom_ir) / (nom_ir + 1e-9))
        cycle_pos_list.append((cell_df["cycle_index"] - min_cycle) / cyc_range)

    df["cap_rel"] = pd.concat(cap_rel_list)
    df["energy_rel"] = pd.concat(en_rel_list)
    df["ir_rel"] = pd.concat(ir_rel_list)
    df["cycle_pos"] = pd.concat(cycle_pos_list)
    return df


def create_sequences(df, window_size=50, stride=2):
    """Create sliding window sequences"""
    X_list, y_list, split_list, cell_list = [], [], [], []
    
    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
        X = cell_df[FEAT_COLS].values.astype(np.float32)
        y = cell_df["soh"].values.astype(np.float32)
        split = cell_df["split"].values
        cell = cell_df["barcode"].values[0]
        
        for end in range(window_size, len(X) + 1, stride):
            start = end - window_size
            X_list.append(X[start:end])
            y_list.append(y[end - 1])
            split_list.append(split[end - 1])
            cell_list.append(cell)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    splits = np.array(split_list)
    cells = np.array(cell_list)
    
    return X, y, splits, cells


def normalize_features(X_train, X_val, X_test):
    """Normalize features per feature dimension (not per timestep)"""
    n_train, seq_len, n_feat = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feat)
    X_val_flat = X_val.reshape(-1, n_feat)
    X_test_flat = X_test.reshape(-1, n_feat)
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    X_train = X_train_flat.reshape(-1, seq_len, n_feat)
    X_val = X_val_flat.reshape(-1, seq_len, n_feat)
    X_test = X_test_flat.reshape(-1, seq_len, n_feat)
    
    return X_train, X_val, X_test, scaler


def load_data():
    """Load and preprocess data"""
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape: {df.shape}")
    
    df = add_relative_features(df)
    print(f"  Features added: {df.shape}")
    
    X, y, splits, cells = create_sequences(df, CFG["window_size"], CFG["stride"])
    print(f"  Sequences created: {X.shape}")
    
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# ============================================================
# BiLSTM MODEL
# ============================================================

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out.squeeze(-1)

# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(CFG["device"]), y.to(CFG["device"])
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(CFG["device"]), y.to(CFG["device"])
            pred = model(x)
            loss = criterion(pred, y.squeeze())
            total_loss += loss.item()
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y.squeeze().cpu().numpy())
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    return total_loss / len(loader), all_pred, all_true


def train(model, train_loader, val_loader, cfg):
    print("\n" + "="*60)
    print("TRAINING BiLSTM BASELINE")
    print("="*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_pred, val_true = validate(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch % 10 == 0) or (epoch == 1):
            mae = mean_absolute_error(val_true, val_pred) * 100
            r2 = r2_score(val_true, val_pred)
            print(f"  Epoch {epoch:3d}/{cfg['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {mae:.4f}% | R²: {r2:.4f}")
        
        if patience_counter >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    print(f"\n  Best val loss: {best_val_loss:.6f}")
    return model

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, label="Test"):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(CFG["device"])
            pred = model(x)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y.squeeze().cpu().numpy())
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    mae = mean_absolute_error(all_true, all_pred) * 100
    rmse = np.sqrt(mean_squared_error(all_true, all_pred)) * 100
    r2 = r2_score(all_true, all_pred)
    
    print(f"\n  ── {label} Results ──────────────────────────────")
    print(f"  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  R²   : {r2:.5f}")
    
    return {"mae_pct": mae, "rmse_pct": rmse, "r2": r2, "y_true": all_true, "y_pred": all_pred}


def plot_results(y_true, y_pred, save_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=5)
    plt.plot([0.7, 1.0], [0.7, 1.0], "r--", label="Perfect Prediction")
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"BiLSTM: Predictions vs True SOH\nMAE: {mae:.4f}%, R²: {r2:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved: {save_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("BiLSTM BASELINE FOR BATTERY SOH ESTIMATION")
    print("="*60)
    
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data()
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CFG["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=CFG["batch_size"], shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=CFG["batch_size"], shuffle=False)
    
    print("\n[2] Building BiLSTM model...")
    model = BiLSTMModel(
        input_size=10,
        hidden_size=CFG["hidden_size"],
        num_layers=CFG["num_layers"],
        dropout=CFG["dropout"]
    ).to(CFG["device"])
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    
    print("\n[3] Training...")
    model = train(model, train_loader, val_loader, CFG)
    
    print("\n[4] Saving model...")
    torch.save(model.state_dict(), SAVE_DIR / "bilstm_model.pt")
    print(f"  Model saved to: {SAVE_DIR / 'bilstm_model.pt'}")
    
    print("\n[5] Evaluating...")
    train_results = evaluate(model, train_loader, "Train")
    val_results = evaluate(model, val_loader, "Validation")
    test_results = evaluate(model, test_loader, "Test")
    
    # JSON-serializable config
    config_serializable = CFG.copy()
    config_serializable["device"] = str(config_serializable["device"])
    
    results = {
        "train_mae_pct": train_results["mae_pct"],
        "train_rmse_pct": train_results["rmse_pct"],
        "train_r2": train_results["r2"],
        "val_mae_pct": val_results["mae_pct"],
        "val_rmse_pct": val_results["rmse_pct"],
        "val_r2": val_results["r2"],
        "test_mae_pct": test_results["mae_pct"],
        "test_rmse_pct": test_results["rmse_pct"],
        "test_r2": test_results["r2"],
        "n_params": n_params,
        "config": config_serializable,
    }
    
    with open(SAVE_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {SAVE_DIR / 'results.json'}")
    
    print("\n[6] Creating plots...")
    plot_results(test_results["y_true"], test_results["y_pred"], SAVE_DIR / "predictions.png")
    
    print("\n" + "="*60)
    print("BiLSTM RESULTS SUMMARY")
    print("="*60)
    print(f"  Train MAE: {train_results['mae_pct']:.4f}%")
    print(f"  Val MAE:   {val_results['mae_pct']:.4f}%")
    print(f"  Test MAE:  {test_results['mae_pct']:.4f}%")
    print(f"  Test R²:   {test_results['r2']:.5f}")
    print("="*60)

if __name__ == "__main__":
    main()