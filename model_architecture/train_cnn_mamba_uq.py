"""
Battery SOH & RUL Estimation
Architecture: CNN-Mamba-UQ  (dual-head, semi-supervised RUL)
Targets: SOH MAE < 0.7%  R² > 0.97
         RUL MAE < 30 cycles  R² > 0.90
         PICP ≥ 90%  PINW minimal
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports & reproducibility
# ─────────────────────────────────────────────────────────────────────────────
import os, math, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config  (all hyper-parameters in one place)
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # paths
    soh_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv",
    rul_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\rul\rul_full.csv",
    
    # Save model locally:
    save_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\cnn_mamba_uq_battery.pt",
    

    # features
    feat_cols = [
        "dc_internal_resistance", "temperature_avg",
        "charge_capacity", "charge_energy",
        "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    ],

    # sequence
    window_size   = 32,   # W cycles fed as a sequence
    stride        = 1,    # slide 1 cycle at a time

    # model
    input_dim     = 6,
    cnn_channels  = [32, 64, 128],   # 3 conv blocks
    cnn_kernels   = [3, 7, 15],      # multi-scale kernels
    d_model       = 128,             # Mamba hidden dim
    d_state       = 16,              # SSM state expansion
    d_conv        = 4,               # Mamba inner conv
    expand        = 2,               # Mamba channel expand
    n_mamba_layers= 3,
    dropout       = 0.15,

    # training – SOH
    soh_epochs    = 80,
    soh_lr        = 3e-4,
    soh_batch     = 256,
    soh_weight_decay = 1e-4,
    lambda_mono   = 0.1,   # monotonicity penalty weight
    tail_weight   = 5.0,   # extra weight for SOH < 0.90

    # training – RUL
    rul_epochs    = 80,
    rul_lr        = 3e-4,
    rul_batch     = 256,
    rul_weight_decay = 1e-4,
    huber_delta   = 50.0,
    lambda_rul    = 1.0,
    lambda_pseudo = 0.3,   # pseudo-label loss weight
    pseudo_conf_thr = 0.15, # accept pseudo-label if PINW/range < threshold

    # UQ
    mc_samples    = 50,    # forward passes for MC-Dropout
    ci_alpha      = 0.90,  # 90% prediction interval
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class BatteryPreprocessor:
    """
    Per-cell z-score normalisation of features.
    Fit only on training cells, apply to all.
    """
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


def load_and_preprocess(soh_path, rul_path, feat_cols):
    soh = pd.read_csv(soh_path)
    rul = pd.read_csv(rul_path)

    preprocessor = BatteryPreprocessor(feat_cols)
    preprocessor.fit(soh[soh.split == "train"])

    soh = preprocessor.transform(soh)
    rul = preprocessor.transform(rul)

    return soh, rul, preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# 3. Datasets
# ─────────────────────────────────────────────────────────────────────────────

class SOHSequenceDataset(Dataset):
    """
    Sliding window over each cell's cycle trajectory.
    Each sample = (W × 6) input, scalar SOH target at the last cycle.
    """
    def __init__(self, df, feat_cols, window_size, stride=1, split=None):
        self.window_size = window_size
        self.samples = []   # (features_np, soh_scalar)
        self.weights = []   # per-sample loss weight

        subset = df if split is None else df[df.split == split]

        for cell_id, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[feat_cols].values.astype(np.float32)   # (T, 6)
            y = cell_df["soh"].values.astype(np.float32)        # (T,)

            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                x_win = X[start:end]           # (W, 6)
                y_last = y[end - 1]            # predict SOH at last cycle

                self.samples.append((x_win, y_last))

                # Up-weight degraded samples
                w = CFG["tail_weight"] if y_last < 0.90 else 1.0
                self.weights.append(w)

        self.weights = np.array(self.weights, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x),           # (W, 6)
            torch.tensor(y),           # scalar
            torch.tensor(self.weights[idx]),
        )


class RULSequenceDataset(Dataset):
    """
    Sliding window for RUL.
    Returns labeled flag so the training loop can apply different loss terms.
    """
    def __init__(self, df, feat_cols, window_size, stride=1, split=None):
        self.window_size = window_size
        self.samples = []  # (features_np, rul_scalar, has_label)

        subset = df if split is None else df[df.split == split]

        for cell_id, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X   = cell_df[feat_cols].values.astype(np.float32)
            y   = cell_df["rul"].values.astype(np.float32)
            lbl = cell_df["has_label"].values.astype(np.int8)

            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                x_win     = X[start:end]
                y_last    = y[end - 1]
                lbl_last  = int(lbl[end - 1])
                self.samples.append((x_win, y_last, lbl_last))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, lbl = self.samples[idx]
        return (
            torch.tensor(x),
            torch.tensor(y),
            torch.tensor(lbl, dtype=torch.int8),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model components
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleCNN(nn.Module):
    """
    Three parallel Conv1D branches with kernels 3, 7, 15.
    Input:  (B, W, input_dim)
    Output: (B, W, sum(cnn_channels))
    """
    def __init__(self, input_dim, channels, kernels, dropout=0.1):
        super().__init__()
        assert len(channels) == len(kernels)
        self.branches = nn.ModuleList()
        for ch, k in zip(channels, kernels):
            self.branches.append(nn.Sequential(
                nn.Conv1d(input_dim, ch, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            ))
        self.out_dim = sum(channels)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x):
        # x: (B, W, C) → (B, C, W) for Conv1d
        x = x.permute(0, 2, 1)
        branch_outs = [b(x) for b in self.branches]
        x = torch.cat(branch_outs, dim=1)  # (B, total_ch, W)
        x = x.permute(0, 2, 1)             # (B, W, total_ch)
        x = self.dropout(F.gelu(self.proj(x)))
        return x


class MambaBlock(nn.Module):
    """
    Pure-PyTorch Mamba-style Selective State Space Model block.
    No CUDA-specific kernels — fully compatible with CPU and any GPU.
    Reference: Gu & Dao 2023 (S6 / Mamba)

    Input/Output shape: (B, L, d_model)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.d_inner  = int(expand * d_model)

        # Input projection (x-branch and z-branch)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise causal conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSM parameters projections
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A: log-spaced negative values  (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def ssm(self, x):
        """
        Selective scan (simplified recurrent form — O(L·N)).
        x: (B, L, d_inner)
        """
        B, L, D = x.shape
        N = self.d_state

        # Compute Δ, B_ssm, C_ssm from input
        delta_BC = self.x_proj(x)                       # (B, L, N+N+1)
        delta_raw = delta_BC[..., :1]                   # (B, L, 1)
        B_ssm     = delta_BC[..., 1:N+1]               # (B, L, N)
        C_ssm     = delta_BC[..., N+1:]                 # (B, L, N)

        # Δ softplus
        delta = F.softplus(self.dt_proj(delta_raw))     # (B, L, d_inner)

        # Discretised A  (ZOH)
        A = -torch.exp(self.A_log)                      # (d_inner, N)
        # dA: (B, L, d_inner, N)
        dA = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )

        # dB * u: (B, L, d_inner, N)
        dB_u = (delta.unsqueeze(-1) * B_ssm.unsqueeze(2)
                * x.unsqueeze(-1))

        # Recurrent scan
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB_u[:, t]              # (B, d_inner, N)
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(-1) # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                      # (B, L, d_inner)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x):
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)                     # (B, L, d_inner) each

        # Causal conv
        x_conv = self.conv1d(
            x_.permute(0, 2, 1)
        )[..., :x_.shape[1]].permute(0, 2, 1)
        x_conv = F.silu(x_conv)

        # SSM
        y = self.ssm(x_conv)

        # Gate
        y = y * F.silu(z)

        y = self.dropout(self.out_proj(y))
        return y + residual


class MambaEncoder(nn.Module):
    """Stack of N Mamba blocks."""
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
    """
    Full CNN-Mamba-UQ backbone with dual SOH + RUL heads.

    SOH head : MLP → sigmoid  (output in [0,1])
    RUL head : MLP → ReLU     (output ≥ 0 cycles)
    MC-Dropout applied during inference for UQ.
    """
    def __init__(self, cfg):
        super().__init__()
        C = cfg

        # ── CNN encoder ──────────────────────────────────────────────────
        self.cnn = MultiScaleCNN(
            C["input_dim"],
            C["cnn_channels"],
            C["cnn_kernels"],
            C["dropout"],
        )
        cnn_out_dim = sum(C["cnn_channels"])  # 224

        # Project CNN output to d_model
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, C["d_model"]),
            nn.LayerNorm(C["d_model"]),
            nn.GELU(),
            nn.Dropout(C["dropout"]),
        )

        # ── Mamba encoder ────────────────────────────────────────────────
        self.mamba = MambaEncoder(
            C["d_model"],
            C["d_state"],
            C["d_conv"],
            C["expand"],
            C["n_mamba_layers"],
            C["dropout"],
        )

        # ── Shared pooling ───────────────────────────────────────────────
        # Attention pooling over the sequence  (learn which cycles matter)
        self.attn_pool = nn.Linear(C["d_model"], 1)

        # ── SOH head ─────────────────────────────────────────────────────
        self.soh_head = nn.Sequential(
            nn.Linear(C["d_model"], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ── RUL head ─────────────────────────────────────────────────────
        self.rul_head = nn.Sequential(
            nn.Linear(C["d_model"], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(C["dropout"]),
            nn.Linear(64, 1),
            nn.ReLU(),            # RUL ≥ 0
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def encode(self, x):
        """
        x : (B, W, 6)
        returns : (B, d_model)  — attended latent representation
        """
        z = self.cnn(x)                         # (B, W, cnn_out)
        z = self.cnn_proj(z)                    # (B, W, d_model)
        z = self.mamba(z)                       # (B, W, d_model)

        # Attention pooling
        attn = F.softmax(self.attn_pool(z), dim=1)  # (B, W, 1)
        z = (z * attn).sum(dim=1)                   # (B, d_model)
        return z

    def forward(self, x, task="soh"):
        z = self.encode(x)
        if task == "soh":
            return self.soh_head(z).squeeze(-1)
        elif task == "rul":
            return self.rul_head(z).squeeze(-1)
        else:
            return self.soh_head(z).squeeze(-1), self.rul_head(z).squeeze(-1)

    def predict_with_uncertainty(self, x, task="soh", n_samples=50, ci=0.90):
        """
        Monte Carlo Dropout inference.
        Keeps dropout ACTIVE during inference.
        Returns: mean, lower_ci, upper_ci
        """
        self.train()   # activates dropout
        with torch.no_grad():
            preds = torch.stack(
                [self.forward(x, task=task) for _ in range(n_samples)],
                dim=0,
            )  # (n_samples, B)
        self.eval()

        lo = (1 - ci) / 2
        hi = 1 - lo
        mean  = preds.mean(0)
        lower = preds.quantile(lo, dim=0)
        upper = preds.quantile(hi, dim=0)
        return mean, lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# 5. Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def weighted_mse(pred, target, weight):
    return (weight * (pred - target) ** 2).mean()


def monotonicity_penalty(pred_seq):
    """
    Encourage predicted SOH to be non-increasing within each window.
    pred_seq : (B, W)  — predictions for all W positions in window
    """
    diff = pred_seq[:, 1:] - pred_seq[:, :-1]   # should be ≤ 0
    violations = F.relu(diff)
    return violations.mean()


def soh_loss(pred, target, weight, lambda_mono=0.1):
    mse = weighted_mse(pred, target, weight)
    return mse


def huber_loss(pred, target, delta=50.0):
    return F.huber_loss(pred, target, delta=delta)


def rul_semi_supervised_loss(pred, target, has_label,
                             pseudo_pred=None, pseudo_conf=None,
                             lambda_pseudo=0.3, huber_delta=50.0):
    """
    Labeled rows   → Huber loss
    Unlabeled rows → pseudo-label loss weighted by teacher confidence
    """
    labeled_mask = has_label.bool()

    # Labeled loss
    if labeled_mask.sum() > 0:
        loss_labeled = huber_loss(
            pred[labeled_mask],
            target[labeled_mask],
            huber_delta,
        )
    else:
        loss_labeled = torch.tensor(0.0, device=pred.device)

    # Pseudo loss
    loss_pseudo = torch.tensor(0.0, device=pred.device)
    if pseudo_pred is not None and (~labeled_mask).sum() > 0:
        unlabeled_pred   = pred[~labeled_mask]
        unlabeled_pseudo = pseudo_pred[~labeled_mask]
        if pseudo_conf is not None:
            conf_w = pseudo_conf[~labeled_mask].clamp(0, 1)
        else:
            conf_w = torch.ones_like(unlabeled_pred)
        loss_pseudo = (conf_w * (unlabeled_pred - unlabeled_pseudo) ** 2).mean()

    return loss_labeled + lambda_pseudo * loss_pseudo


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training helpers
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=12, delta=1e-5):
        self.patience = patience
        self.delta    = delta
        self.counter  = 0
        self.best     = None
        self.stop     = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best is None or val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model):
        model.load_state_dict(self.best_state)


def make_loaders(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SOH training
# ─────────────────────────────────────────────────────────────────────────────

def train_soh(model, soh_df, cfg):
    print("\n" + "="*60)
    print("  TRAINING SOH HEAD")
    print("="*60)

    train_ds = SOHSequenceDataset(soh_df, cfg["feat_cols"],
                                  cfg["window_size"], cfg["stride"], "train")
    val_ds   = SOHSequenceDataset(soh_df, cfg["feat_cols"],
                                  cfg["window_size"], cfg["stride"], "val")

    train_loader = make_loaders(train_ds, cfg["soh_batch"], shuffle=True)
    val_loader   = make_loaders(val_ds,   cfg["soh_batch"], shuffle=False)

    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["soh_lr"],
        weight_decay=cfg["soh_weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    es = EarlyStopping(patience=12)

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_r2": []}

    for epoch in range(1, cfg["soh_epochs"] + 1):
        # ── Train ───────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for x, y, w in train_loader:
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x, task="soh")
            loss = weighted_mse(pred, y, w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()
        t_loss /= len(train_loader)

        # ── Validate ────────────────────────────────────────
        model.eval()
        all_pred, all_true = [], []
        v_loss = 0.0
        with torch.no_grad():
            for x, y, w in val_loader:
                x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
                pred = model(x, task="soh")
                loss = weighted_mse(pred, y, w)
                v_loss += loss.item()
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(y.cpu().numpy())
        v_loss /= len(val_loader)

        all_pred = np.array(all_pred)
        all_true = np.array(all_true)
        mae = mean_absolute_error(all_true, all_pred) * 100   # as %
        r2  = r2_score(all_true, all_pred)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_mae"].append(mae)
        history["val_r2"].append(r2)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg['soh_epochs']} | "
                  f"Train: {t_loss:.5f} | Val: {v_loss:.5f} | "
                  f"MAE: {mae:.4f}% | R²: {r2:.4f}")

        es(v_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    es.restore(model)
    print(f"\n  Best val loss: {es.best:.6f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 8. RUL training  (semi-supervised with teacher-student pseudo-labeling)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pseudo_labels(teacher, unlabeled_loader, cfg):
    """
    Run teacher model with MC-Dropout on unlabeled windows.
    Returns pseudo-labels and confidence weights (based on PINW).
    """
    teacher.eval()
    all_mean, all_conf = [], []

    with torch.no_grad():
        for batch in unlabeled_loader:
            x   = batch[0].to(DEVICE)
            mean, lo, hi = teacher.predict_with_uncertainty(
                x, task="rul",
                n_samples=cfg["mc_samples"],
                ci=cfg["ci_alpha"],
            )
            pinw = (hi - lo).cpu().numpy()            # interval width
            # Confidence weight: narrow interval → high confidence
            conf = np.exp(-pinw / (pinw.mean() + 1e-8))
            all_mean.extend(mean.cpu().numpy())
            all_conf.extend(conf)

    return np.array(all_mean), np.array(all_conf)


def train_rul(model, rul_df, cfg, soh_history=None):
    print("\n" + "="*60)
    print("  TRAINING RUL HEAD  (semi-supervised)")
    print("="*60)

    train_ds = RULSequenceDataset(rul_df, cfg["feat_cols"],
                                  cfg["window_size"], cfg["stride"], "train")
    val_ds   = RULSequenceDataset(rul_df, cfg["feat_cols"],
                                  cfg["window_size"], cfg["stride"], "val")

    train_loader = make_loaders(train_ds, cfg["rul_batch"], shuffle=True)
    val_loader   = make_loaders(val_ds,   cfg["rul_batch"], shuffle=False)

    n_labeled   = sum(1 for _, _, lbl in train_ds if lbl == 1)
    n_unlabeled = len(train_ds) - n_labeled
    print(f"  Train samples: {len(train_ds):,}  "
          f"(labeled: {n_labeled:,} | unlabeled: {n_unlabeled:,})")
    print(f"  Val samples: {len(val_ds):,}")

    # Freeze backbone initially — only train RUL head
    # (backbone already trained on SOH — fine-tune after warmup)
    for p in model.cnn.parameters():   p.requires_grad = False
    for p in model.mamba.parameters(): p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["rul_lr"],
        weight_decay=cfg["rul_weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    es = EarlyStopping(patience=12)

    # Pre-compute pseudo-labels (teacher = current model after SOH training)
    print("  Generating initial pseudo-labels from teacher model...")
    pseudo_means, pseudo_confs = generate_pseudo_labels(model, train_loader, cfg)
    pseudo_idx = 0

    history = {"train_loss": [], "val_loss": [],
               "val_mae": [], "val_r2": [], "val_picp": [], "val_pinw": []}

    # Unfreeze backbone at epoch 20 for end-to-end fine-tuning
    UNFREEZE_EPOCH = 20

    for epoch in range(1, cfg["rul_epochs"] + 1):

        if epoch == UNFREEZE_EPOCH:
            print(f"  Epoch {epoch}: unfreezing backbone for end-to-end tuning")
            for p in model.cnn.parameters():   p.requires_grad = True
            for p in model.mamba.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["rul_lr"] * 0.3,
                weight_decay=cfg["rul_weight_decay"],
            )

        # Refresh pseudo-labels every 15 epochs
        if epoch > 1 and (epoch - 1) % 15 == 0:
            print(f"  Epoch {epoch}: refreshing pseudo-labels...")
            pseudo_means, pseudo_confs = generate_pseudo_labels(
                model, train_loader, cfg)

        # ── Train ───────────────────────────────────────────
        model.train()
        t_loss = 0.0
        pseudo_idx = 0

        for x, y, lbl in train_loader:
            x, y, lbl = x.to(DEVICE), y.to(DEVICE), lbl.to(DEVICE)
            bs = x.shape[0]

            # Gather pseudo-labels for this batch
            batch_pseudo = torch.tensor(
                pseudo_means[pseudo_idx: pseudo_idx + bs],
                dtype=torch.float32, device=DEVICE,
            )
            batch_conf = torch.tensor(
                pseudo_confs[pseudo_idx: pseudo_idx + bs],
                dtype=torch.float32, device=DEVICE,
            )
            pseudo_idx += bs

            # Threshold: only use high-confidence pseudo-labels
            conf_mask = batch_conf > (1 - cfg["pseudo_conf_thr"])
            if not conf_mask.any():
                batch_pseudo = None
                batch_conf   = None

            optimizer.zero_grad()
            pred = model(x, task="rul")

            # Replace rul = -1 with pseudo-label for unlabeled rows
            y_eff = y.clone()
            unlabeled = (lbl == 0)
            if batch_pseudo is not None:
                y_eff[unlabeled] = batch_pseudo[unlabeled]

            loss = rul_semi_supervised_loss(
                pred, y_eff, lbl.float(),
                pseudo_pred=batch_pseudo,
                pseudo_conf=batch_conf if batch_conf is not None else None,
                lambda_pseudo=cfg["lambda_pseudo"],
                huber_delta=cfg["huber_delta"],
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        scheduler.step()
        t_loss /= len(train_loader)

        # ── Validate (labeled only) ──────────────────────────
        model.eval()
        all_pred, all_true = [], []
        all_lo, all_hi     = [], []
        v_loss = 0.0

        with torch.no_grad():
            for x, y, lbl in val_loader:
                x, y, lbl = x.to(DEVICE), y.to(DEVICE), lbl.to(DEVICE)
                labeled_mask = lbl.bool()
                if labeled_mask.sum() == 0:
                    continue

                pred = model(x, task="rul")
                loss = huber_loss(pred[labeled_mask],
                                  y[labeled_mask], cfg["huber_delta"])
                v_loss += loss.item()

                # Point predictions
                all_pred.extend(pred[labeled_mask].cpu().numpy())
                all_true.extend(y[labeled_mask].cpu().numpy())

                # MC-Dropout intervals for labeled validation
                x_lab = x[labeled_mask]
                m, lo, hi = model.predict_with_uncertainty(
                    x_lab, task="rul",
                    n_samples=cfg["mc_samples"],
                    ci=cfg["ci_alpha"],
                )
                all_lo.extend(lo.cpu().numpy())
                all_hi.extend(hi.cpu().numpy())

        if len(all_pred) == 0:
            continue

        v_loss /= max(len(val_loader), 1)
        all_pred = np.array(all_pred)
        all_true = np.array(all_true)
        all_lo   = np.array(all_lo)
        all_hi   = np.array(all_hi)

        mae  = mean_absolute_error(all_true, all_pred)
        r2   = r2_score(all_true, all_pred)
        picp = np.mean((all_true >= all_lo) & (all_true <= all_hi))
        pinw = np.mean(all_hi - all_lo) / (all_true.max() - all_true.min() + 1e-8)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_mae"].append(mae)
        history["val_r2"].append(r2)
        history["val_picp"].append(picp)
        history["val_pinw"].append(pinw)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg['rul_epochs']} | "
                  f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | "
                  f"MAE: {mae:.2f} | R²: {r2:.4f} | "
                  f"PICP: {picp:.3f} | PINW: {pinw:.4f}")

        es(v_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    es.restore(model)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_picp_pinw(true, lower, upper, target_range=None):
    coverage = np.mean((true >= lower) & (true <= upper))
    width    = np.mean(upper - lower)
    t_range  = target_range or (true.max() - true.min() + 1e-8)
    pinw     = width / t_range
    return coverage, pinw


def evaluate_soh(model, soh_df, cfg):
    print("\n" + "="*60)
    print("  SOH EVALUATION — TEST SET")
    print("="*60)

    test_ds = SOHSequenceDataset(soh_df, cfg["feat_cols"],
                                 cfg["window_size"], cfg["stride"], "test")
    test_loader = make_loaders(test_ds, cfg["soh_batch"], shuffle=False)

    model.eval()
    all_pred, all_true = [], []
    all_lo,   all_hi   = [], []

    for x, y, w in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mean, lo, hi = model.predict_with_uncertainty(
            x, task="soh",
            n_samples=cfg["mc_samples"],
            ci=cfg["ci_alpha"],
        )
        all_pred.extend(mean.cpu().numpy())
        all_true.extend(y.cpu().numpy())
        all_lo.extend(lo.cpu().numpy())
        all_hi.extend(hi.cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_lo   = np.array(all_lo)
    y_hi   = np.array(all_hi)

    mae  = mean_absolute_error(y_true, y_pred) * 100   # as %
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2   = r2_score(y_true, y_pred)
    picp, pinw = compute_picp_pinw(y_true, y_lo, y_hi)

    # Error by SOH region
    mask_low  = y_true < 0.90
    mask_mid  = (y_true >= 0.90) & (y_true < 0.95)
    mask_high = y_true >= 0.95

    print(f"\n  ── Overall ──────────────────────────────")
    print(f"  MAE  : {mae:.4f}%   (target < 0.70%)")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  R²   : {r2:.5f}   (target > 0.97)")
    print(f"  PICP : {picp:.4f}  (target ≥ 0.90)")
    print(f"  PINW : {pinw:.4f}  (lower = better)")

    print(f"\n  ── MAE by SOH region ────────────────────")
    for label, mask in [("SOH < 0.90", mask_low),
                        ("0.90–0.95",  mask_mid),
                        ("SOH > 0.95", mask_high)]:
        if mask.sum() > 0:
            region_mae = mean_absolute_error(y_true[mask], y_pred[mask]) * 100
            print(f"  {label}: MAE = {region_mae:.4f}%  (n={mask.sum()})")

    targets_met = (mae < 0.70) and (r2 > 0.97) and (picp >= 0.90)
    status = "✓ ALL TARGETS MET" if targets_met else "✗ targets not fully met"
    print(f"\n  {status}")

    return {
        "mae_pct": mae, "rmse_pct": rmse, "r2": r2,
        "picp": picp, "pinw": pinw,
        "y_true": y_true, "y_pred": y_pred,
        "y_lo": y_lo, "y_hi": y_hi,
    }


def evaluate_rul(model, rul_df, cfg):
    print("\n" + "="*60)
    print("  RUL EVALUATION — TEST SET (labeled only)")
    print("="*60)

    test_ds = RULSequenceDataset(rul_df, cfg["feat_cols"],
                                 cfg["window_size"], cfg["stride"], "test")
    test_loader = make_loaders(test_ds, cfg["rul_batch"], shuffle=False)

    model.eval()
    all_pred, all_true = [], []
    all_lo,   all_hi   = [], []

    for x, y, lbl in test_loader:
        x, y, lbl = x.to(DEVICE), y.to(DEVICE), lbl
        labeled_mask = lbl.bool()
        if labeled_mask.sum() == 0:
            continue
        x_lab = x[labeled_mask]
        y_lab = y[labeled_mask]

        mean, lo, hi = model.predict_with_uncertainty(
            x_lab, task="rul",
            n_samples=cfg["mc_samples"],
            ci=cfg["ci_alpha"],
        )
        all_pred.extend(mean.cpu().numpy())
        all_true.extend(y_lab.cpu().numpy())
        all_lo.extend(lo.cpu().numpy())
        all_hi.extend(hi.cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_lo   = np.array(all_lo)
    y_hi   = np.array(all_hi)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = r2_score(y_true, y_pred)
    picp, pinw = compute_picp_pinw(y_true, y_lo, y_hi)

    # Error by RUL region
    mask_near = y_true < 50
    mask_mid  = (y_true >= 50)  & (y_true < 200)
    mask_far  = y_true >= 200

    print(f"\n  ── Overall ──────────────────────────────")
    print(f"  MAE  : {mae:.2f} cycles  (target < 30)")
    print(f"  RMSE : {rmse:.2f} cycles")
    print(f"  R²   : {r2:.5f}   (target > 0.90)")
    print(f"  PICP : {picp:.4f}  (target ≥ 0.90)")
    print(f"  PINW : {pinw:.4f}  (lower = better)")

    print(f"\n  ── MAE by RUL region ────────────────────")
    for label, mask in [("RUL < 50",    mask_near),
                        ("RUL 50–200",  mask_mid),
                        ("RUL > 200",   mask_far)]:
        if mask.sum() > 0:
            region_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            print(f"  {label}: MAE = {region_mae:.2f} cycles  (n={mask.sum()})")

    targets_met = (mae < 30) and (r2 > 0.90) and (picp >= 0.90)
    status = "✓ ALL TARGETS MET" if targets_met else "✗ targets not fully met"
    print(f"\n  {status}")

    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "picp": picp, "pinw": pinw,
        "y_true": y_true, "y_pred": y_pred,
        "y_lo": y_lo, "y_hi": y_hi,
    }


def print_model_summary(model, cfg):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total:,} total | {trainable:,} trainable")
    print(f"  Input shape: ({cfg['window_size']}, {cfg['input_dim']})")
    print(f"  CNN output dim: {sum(cfg['cnn_channels'])}")
    print(f"  d_model: {cfg['d_model']}  | Mamba layers: {cfg['n_mamba_layers']}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main — run everything
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading & preprocessing data...")
    soh_df, rul_df, preprocessor = load_and_preprocess(
        CFG["soh_path"], CFG["rul_path"], CFG["feat_cols"]
    )
    print(f"  SOH: {soh_df.shape}  |  RUL: {rul_df.shape}")

    print("\nBuilding CNN-Mamba-UQ model...")
    model = CNNMambaUQ(CFG).to(DEVICE)
    print_model_summary(model, CFG)

    # ── Phase 1: Train SOH ───────────────────────────────────────────────
    soh_history = train_soh(model, soh_df, CFG)

    # ── Phase 2: Train RUL (semi-supervised) ────────────────────────────
    rul_history = train_rul(model, rul_df, CFG)

    # ── Evaluation ──────────────────────────────────────────────────────
    soh_results = evaluate_soh(model, soh_df, CFG)
    rul_results = evaluate_rul(model, rul_df, CFG)

    # ── Save model ──────────────────────────────────────────────────────
    save_path = "/mnt/user-data/outputs/cnn_mamba_uq_battery.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": CFG,
        "soh_results": {k: v for k, v in soh_results.items()
                        if not isinstance(v, np.ndarray)},
        "rul_results": {k: v for k, v in rul_results.items()
                        if not isinstance(v, np.ndarray)},
    }, save_path)
    print(f"\n  Model saved → {save_path}")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  SOH  MAE : {soh_results['mae_pct']:.4f}%  R²: {soh_results['r2']:.5f}  "
          f"PICP: {soh_results['picp']:.4f}  PINW: {soh_results['pinw']:.4f}")
    print(f"  RUL  MAE : {rul_results['mae']:.2f} cyc  R²: {rul_results['r2']:.5f}  "
          f"PICP: {rul_results['picp']:.4f}  PINW: {rul_results['pinw']:.4f}")
    print("="*60)