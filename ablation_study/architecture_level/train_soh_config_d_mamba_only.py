# train_soh_config_d_mamba_only.py
#
# ARCHITECTURE ABLATION - CONFIG D
# Remove CNN entirely - Mamba only
#
# Configuration: Full Model but WITHOUT CNN
# - Multi-Scale CNN: ✗ (completely removed)
# - Mamba Encoder: ✓ (processes raw features directly)
# - Attention Pooling: ✓

import os, math, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Config - SOH ONLY (Same as Full Model)
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Paths
    soh_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv",
    save_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\soh_config_d_mamba_only.pt",
    
    # Features - SAME 10 features as full model
    input_dim = 10,
    window_size = 50,
    soh_stride = 2,
    
    # Model - NO CNN
    d_model = 128,
    d_state = 16,
    d_conv = 4,
    expand = 2,
    n_mamba_layers = 3,
    dropout = 0.15,
    
    # Training - IDENTICAL to full model
    soh_epochs = 120,
    soh_lr = 2e-4,
    soh_batch = 256,
    soh_wd = 1e-4,
    soh_patience = 25,
    tail_weight = 3.0,
    warmup_epochs = 10,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loading & preprocessing (SAME as full model)
# ─────────────────────────────────────────────────────────────────────────────

def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-cell relative features"""
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


FEAT_COLS = [
    "dc_internal_resistance", "temperature_avg",
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "ir_rel", "cycle_pos",
]


def load_soh_data(soh_path):
    """Load and preprocess SOH data only"""
    soh = pd.read_csv(soh_path)
    soh = add_relative_features(soh)
    
    # Fit scaler on train split
    scaler = StandardScaler()
    scaler.fit(soh[soh.split == "train"][FEAT_COLS].values)
    soh[FEAT_COLS] = scaler.transform(soh[FEAT_COLS].values)
    
    return soh, scaler


class SequenceDataset(Dataset):
    """Sliding-window dataset for SOH"""
    def __init__(self, df, window_size, stride=1, split=None,
                 weighted=False, tail_thr=0.90, tail_weight=1.0):
        self.samples = []
        self.weights = []
        subset = df if split is None else df[df.split == split]

        for _, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[FEAT_COLS].values.astype(np.float32)
            y = cell_df["soh"].values.astype(np.float32)

            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                y_last = y[end - 1]
                self.samples.append((X[start:end], y_last))
                if weighted:
                    w = tail_weight if y_last < tail_thr else 1.0
                else:
                    w = 1.0
                self.weights.append(w)

        self.weights = np.array(self.weights, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(self.weights[idx])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Model - CONFIG D: Mamba Only (No CNN)
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=d_conv, padding=d_conv-1,
                                groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A.expand(self.d_inner, -1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def ssm(self, x):
        B, L, D = x.shape
        N = self.d_state
        dBC = self.x_proj(x)
        delta = F.softplus(self.dt_proj(dBC[..., :1]))
        B_ssm = dBC[..., 1:N+1]
        C_ssm = dBC[..., N+1:]
        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB_u = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB_u[:, t]
            ys.append((h * C_ssm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        x_c = self.conv1d(x_.permute(0,2,1))[..., :x_.shape[1]].permute(0,2,1)
        y = self.ssm(F.silu(x_c)) * F.silu(z)
        return self.dropout(self.out_proj(y)) + res


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


class MambaOnlySOH(nn.Module):
    """
    CONFIG D: Mamba Only - No CNN
    Processes raw features directly through Mamba
    """
    def __init__(self, cfg):
        super().__init__()
        C = cfg
        
        # ⭐ CONFIG D: NO CNN - Project input directly to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(C["input_dim"], C["d_model"]),
            nn.LayerNorm(C["d_model"]), nn.GELU(),
            nn.Dropout(C["dropout"]),
        )
        
        # Mamba (SAME as full model)
        self.mamba = MambaEncoder(C["d_model"], C["d_state"], C["d_conv"],
                                  C["expand"], C["n_mamba_layers"], C["dropout"])
        
        # Attention Pooling (SAME as full model)
        self.attn_pool = nn.Linear(C["d_model"], 1)
        
        # SOH head (SAME as full model)
        self.soh_head = nn.Sequential(
            nn.Linear(C["d_model"], 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(64, 2),  # [mean, log_var]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def encode(self, x):
        # ⭐ CONFIG D: Input goes directly to Mamba (no CNN)
        z = self.input_proj(x)  # (batch, seq_len, d_model)
        z = self.mamba(z)
        attn = F.softmax(self.attn_pool(z), dim=1)
        return (z * attn).sum(dim=1)

    def forward(self, x):
        z = self.encode(x)
        out = self.soh_head(z)
        mu = torch.sigmoid(out[:, 0])  # SOH ∈ (0,1)
        log_var = out[:, 1].clamp(-10, 5)
        return mu, log_var


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Loss, Training, Evaluation (IDENTICAL to full model)
# ─────────────────────────────────────────────────────────────────────────────

def soh_loss(mu, log_var, target, weight=None):
    """MSE loss for SOH"""
    mu = torch.clamp(mu, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    loss = (mu - target) ** 2
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def cosine_lr(optimizer, epoch, warmup, total_epochs, base_lr):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


class EarlyStopping:
    def __init__(self, patience=20, delta=1e-5):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best = None
        self.stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best is None or val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_soh(model, train_ds, val_ds, cfg):
    print("\n" + "="*60)
    print("  TRAINING SOH - CONFIG D (Mamba Only, No CNN)")
    print("="*60)
    
    batch = cfg["soh_batch"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["soh_lr"], weight_decay=cfg["soh_wd"])
    es = EarlyStopping(patience=cfg["soh_patience"])
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_r2": []}
    
    for epoch in range(cfg["soh_epochs"]):
        cosine_lr(opt, epoch, cfg["warmup_epochs"], cfg["soh_epochs"], cfg["soh_lr"])
        
        # Train
        model.train()
        train_loss = 0.0
        for x, y, w in train_loader:
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            opt.zero_grad()
            mu, log_var = model(x)
            loss = soh_loss(mu, log_var, y, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        all_pred, all_true = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y, w in val_loader:
                x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
                mu, log_var = model(x)
                val_loss += soh_loss(mu, log_var, y, weight=w).item()
                all_pred.extend(mu.cpu().numpy())
                all_true.extend(y.cpu().numpy())
        val_loss /= len(val_loader)
        
        # Metrics
        all_pred, all_true = np.array(all_pred), np.array(all_true)
        mae = mean_absolute_error(all_true, all_pred) * 100
        r2 = r2_score(all_true, all_pred)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(mae)
        history["val_r2"].append(r2)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['soh_epochs']} | "
                  f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                  f"MAE: {mae:.4f}% | R²: {r2:.4f}")
        
        es(val_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    es.restore(model)
    print(f"\n  Best val loss: {es.best:.6f}")
    return history


def evaluate_soh(model, test_ds, cfg):
    print("\n" + "="*60)
    print("  SOH EVALUATION — TEST SET (Config D - Mamba Only)")
    print("="*60)
    
    test_loader = DataLoader(test_ds, batch_size=cfg["soh_batch"], shuffle=False)
    model.eval()
    
    all_mu, all_y, all_lo, all_hi = [], [], [], []
    
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mu, log_var = model(x)
            sigma = torch.exp(0.5 * log_var)
            z = 1.645
            all_mu.extend(mu.cpu().numpy())
            all_y.extend(y.cpu().numpy())
            all_lo.extend((mu - z*sigma).cpu().numpy())
            all_hi.extend((mu + z*sigma).cpu().numpy())
    
    y_true, y_pred = np.array(all_y), np.array(all_mu)
    y_lo, y_hi = np.array(all_lo), np.array(all_hi)
    
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred)**2)) * 100
    r2 = r2_score(y_true, y_pred)
    picp = np.mean((y_true >= y_lo) & (y_true <= y_hi))
    pinw = np.mean(y_hi - y_lo) / (y_true.max() - y_true.min() + 1e-8)
    
    print(f"\n  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  R²   : {r2:.5f}")
    print(f"  PICP : {picp:.4f}")
    print(f"  PINW : {pinw:.4f}")
    
    return {"mae": mae, "rmse": rmse, "r2": r2, "picp": picp, "pinw": pinw}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def print_model_summary(model, cfg):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total:,} total | {trainable:,} trainable")
    print(f"  Input shape: ({cfg['window_size']}, {cfg['input_dim']})")
    print(f"  d_model: {cfg['d_model']}  | Mamba layers: {cfg['n_mamba_layers']}")
    print(f"  ⚙️  Config D: NO CNN (Mamba only)")


if __name__ == "__main__":
    print("="*60)
    print("  CONFIG D: REMOVE CNN ENTIRELY")
    print("  (Mamba Only - No CNN)")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load data
    print("\nLoading SOH data...")
    soh_df, scaler = load_soh_data(CFG["soh_path"])
    print(f"  SOH: {soh_df.shape}")
    print(f"  Cells: {soh_df['barcode'].nunique()}")
    
    # Build datasets
    W = CFG["window_size"]
    train_ds = SequenceDataset(soh_df, W, CFG["soh_stride"], "train",
                               weighted=True, tail_weight=CFG["tail_weight"])
    val_ds = SequenceDataset(soh_df, W, CFG["soh_stride"], "val",
                             weighted=True, tail_weight=CFG["tail_weight"])
    test_ds = SequenceDataset(soh_df, W, CFG["soh_stride"], "test")
    
    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences:   {len(val_ds):,}")
    print(f"  Test sequences:  {len(test_ds):,}")
    
    # Build model - CONFIG D
    print("\nBuilding Mamba-Only model - Config D (No CNN)...")
    model = MambaOnlySOH(CFG).to(DEVICE)
    print_model_summary(model, CFG)
    
    # Train
    history = train_soh(model, train_ds, val_ds, CFG)
    
    # Evaluate
    results = evaluate_soh(model, test_ds, CFG)
    
    # Save model
    os.makedirs(os.path.dirname(CFG["save_path"]), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": CFG,
        "feat_cols": FEAT_COLS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "results": results,
        "history": history,
        "config": "Config D - Mamba Only (No CNN)"
    }, CFG["save_path"])
    print(f"\n  Model saved → {CFG['save_path']}")
    
    # Final summary
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY - CONFIG D")
    print("="*60)
    print(f"  SOH  MAE: {results['mae']:.4f}%  "
          f"R²: {results['r2']:.5f}  "
          f"PICP: {results['picp']:.4f}  "
          f"PINW: {results['pinw']:.4f}")
    print("="*60)