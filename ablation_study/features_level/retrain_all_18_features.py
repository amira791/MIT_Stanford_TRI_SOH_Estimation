# train_soh_full_features.py
#
# Retrains the CNN-Mamba-UQ SOH model on ALL legitimate, non-leaky features
# from the fuller feature CSV, so permutation_importance_soh.py can rank
# every candidate feature -- not just the 10 the original model happened
# to use.
#
# EXCLUDED FEATURES AND WHY:
#   - discharge_capacity   : leakage score 1.0 -- SOH = discharge_capacity / nominal_capacity
#   - discharge_energy     : leakage score 0.8 -- strongly tied to discharge_capacity
#   - eol_cycle             : NEW LEAKAGE CATCH. This is a per-cell label computed
#                              from that cell's ENTIRE future trajectory (cycle at
#                              which it crossed end-of-life). Using it as a per-cycle
#                              input tells the model how the cell's story ends before
#                              it happens -- the same failure mode as discharge_capacity,
#                              just less obvious. Excluded.
#   - cell_id, split, protocol : identifiers / categorical metadata, not numeric
#                                 per-cycle sensor features. protocol could be added
#                                 later via one-hot/embedding if desired.
#   - soh                   : the target itself
#
# INCLUDED (18 features):
#   cycle_index, charge_capacity, charge_energy, dc_internal_resistance,
#   temperature_maximum, temperature_average, temperature_minimum,
#   date_time_iso_numeric, coulombic_efficiency_lagged_1/2,
#   cap_rel, energy_rel, ir_rel, cycle_pos, dc_ir_norm, cycle_norm,
#   initial_capacity, initial_resistance

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
# 1.  Feature list (18 features, leakage excluded)
# ─────────────────────────────────────────────────────────────────────────────

FEAT_COLS = [
    "cycle_index",
    "charge_capacity",
    "charge_energy",
    "dc_internal_resistance",
    "temperature_maximum",
    "temperature_average",
    "temperature_minimum",
    "date_time_iso_numeric",
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2",
    "cap_rel",
    "energy_rel",
    "ir_rel",
    "cycle_pos",
    "dc_ir_norm",
    "cycle_norm",
    "initial_capacity",
    "initial_resistance",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    soh_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results2\soh_full_with_split.csv",
    save_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\soh_full_features_best.pt",

    input_dim = len(FEAT_COLS),  # 18 -- dynamically matches FEAT_COLS above
    window_size = 50,
    soh_stride = 2,

    cnn_channels = [32, 64, 128],
    cnn_kernels = [3, 7, 15],
    d_model = 128,
    d_state = 16,
    d_conv = 4,
    expand = 2,
    n_mamba_layers = 3,
    dropout = 0.15,

    soh_epochs = 120,
    soh_lr = 2e-4,
    soh_batch = 256,
    soh_wd = 1e-4,
    soh_patience = 25,
    tail_weight = 3.0,
    warmup_epochs = 10,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Data loading (fuller CSV already has cap_rel/energy_rel/ir_rel/cycle_pos
#     precomputed, so no need to recompute them here)
# ─────────────────────────────────────────────────────────────────────────────


def load_soh_data(soh_path):
    df = pd.read_csv(soh_path)
    print(f"  Raw shape: {df.shape}")

    missing = [f for f in FEAT_COLS if f not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    df = df.dropna(subset=FEAT_COLS + ["soh", "cycle_index", "cell_id", "split"])
    print(f"  After NaN drop: {df.shape}")

    scaler = StandardScaler()
    scaler.fit(df[df.split == "train"][FEAT_COLS].values)
    df[FEAT_COLS] = scaler.transform(df[FEAT_COLS].values)

    return df, scaler


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
# 4.  Model (identical architecture to train_soh_only.py; only input_dim differs)
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, channels, kernels, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for ch, k in zip(channels, kernels):
            self.branches.append(nn.Sequential(
                nn.Conv1d(input_dim, ch, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm1d(ch), nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm1d(ch), nn.GELU(),
            ))
        self.out_dim = sum(channels)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1).permute(0, 2, 1)
        return self.dropout(F.gelu(self.proj(x)))


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


class CNNMambaSOH(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg
        self.cnn = MultiScaleCNN(C["input_dim"], C["cnn_channels"],
                                 C["cnn_kernels"], C["dropout"])
        cnn_out = sum(C["cnn_channels"])
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out, C["d_model"]),
            nn.LayerNorm(C["d_model"]), nn.GELU(),
            nn.Dropout(C["dropout"]),
        )
        self.mamba = MambaEncoder(C["d_model"], C["d_state"], C["d_conv"],
                                  C["expand"], C["n_mamba_layers"], C["dropout"])
        self.attn_pool = nn.Linear(C["d_model"], 1)
        self.soh_head = nn.Sequential(
            nn.Linear(C["d_model"], 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(64, 2),
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
        z = self.cnn_proj(self.cnn(x))
        z = self.mamba(z)
        attn = F.softmax(self.attn_pool(z), dim=1)
        return (z * attn).sum(dim=1)

    def forward(self, x):
        z = self.encode(x)
        out = self.soh_head(z)
        mu = torch.sigmoid(out[:, 0])
        log_var = out[:, 1].clamp(-10, 5)
        return mu, log_var


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Loss
# ─────────────────────────────────────────────────────────────────────────────

def soh_loss(mu, log_var, target, weight=None):
    mu = torch.clamp(mu, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    loss = (mu - target) ** 2
    if weight is not None:
        loss = loss * weight
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training
# ─────────────────────────────────────────────────────────────────────────────

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
    print("  TRAINING SOH -- FULL 18-FEATURE SET")
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


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_soh(model, test_ds, cfg):
    print("\n" + "="*60)
    print("  SOH EVALUATION — TEST SET (full 18-feature model)")
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
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def print_model_summary(model, cfg):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total:,} total | {trainable:,} trainable")
    print(f"  Input shape: ({cfg['window_size']}, {cfg['input_dim']})")
    print(f"  Features ({len(FEAT_COLS)}): {FEAT_COLS}")


if __name__ == "__main__":
    print("="*60)
    print("  TRAINING SOH MODEL -- ALL 18 LEAKAGE-FREE FEATURES")
    print("="*60)
    print(f"Device: {DEVICE}")

    print("\nLoading data...")
    soh_df, scaler = load_soh_data(CFG["soh_path"])
    print(f"  Cells: {soh_df['cell_id'].nunique()}")

    W = CFG["window_size"]
    train_ds = SequenceDataset(soh_df, W, CFG["soh_stride"], "train",
                               weighted=True, tail_weight=CFG["tail_weight"])
    val_ds = SequenceDataset(soh_df, W, CFG["soh_stride"], "val",
                             weighted=True, tail_weight=CFG["tail_weight"])
    test_ds = SequenceDataset(soh_df, W, CFG["soh_stride"], "test")

    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences:   {len(val_ds):,}")
    print(f"  Test sequences:  {len(test_ds):,}")

    print("\nBuilding CNN-Mamba-UQ model (18 features)...")
    model = CNNMambaSOH(CFG).to(DEVICE)
    print_model_summary(model, CFG)

    history = train_soh(model, train_ds, val_ds, CFG)
    results = evaluate_soh(model, test_ds, CFG)

    os.makedirs(os.path.dirname(CFG["save_path"]), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": CFG,
        "feat_cols": FEAT_COLS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "results": results,
        "history": history
    }, CFG["save_path"])
    print(f"\n  Model saved → {CFG['save_path']}")

    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY (18-feature model)")
    print("="*60)
    print(f"  SOH  MAE: {results['mae']:.4f}%  "
          f"R²: {results['r2']:.5f}  "
          f"PICP: {results['picp']:.4f}  "
          f"PINW: {results['pinw']:.4f}")
    print("="*60)