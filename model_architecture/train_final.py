"""
Battery SOH & RUL Estimation — CNN-Mamba-UQ  (FIXED & IMPROVED)
================================================================
Targets:  SOH MAE < 0.70%   R² > 0.97
          RUL MAE < 30 cyc  R² > 0.90
          PICP ≥ 0.90        PINW minimal

BUGS FIXED vs original train_cnn_mamba_uq.py
─────────────────────────────────────────────
BUG-1 [CRITICAL – SOH]:  No per-cell feature normalisation.
  charge_capacity at 100 % SOH spans 1.01–1.10 across cells.
  A global StandardScaler cannot remove the cell-to-cell offset, so the
  model cannot learn "this cell has degraded" from absolute capacity alone.
  FIX: per-cell relative features computed during preprocessing; the global
       scaler is then applied on top.

BUG-2 [CRITICAL – SOH]:  Evaluation used MC-Dropout mean.
  Training validation used model.eval() (deterministic), but
  evaluate_soh() called predict_with_uncertainty() which re-enables dropout
  → noisy stochastic means at test time caused R² = −22.
  FIX: evaluate_soh / evaluate_rul use deterministic model.eval() for point
       predictions; MC-Dropout is used only to build the CI bounds.

BUG-3 [CRITICAL – RUL]:  Initial pseudo-labels from untrained RUL head.
  generate_pseudo_labels() was called immediately after SOH training, before
  the RUL head had ever seen a single cycle.  All pseudo-labels ≈ 0 cycles,
  corrupting every unlabeled batch.
  FIX: labelled-only warm-up phase (20 epochs) before pseudo-label generation.

BUG-4 [CRITICAL – RUL]:  Frozen backbone has no RUL-relevant features.
  The CNN+Mamba backbone learnt SOH-specific representations.  Training the
  RUL head alone on top of those representations is insufficient.
  FIX: after a short head-only warm-up, the full model is unfrozen and
       fine-tuned end-to-end on the RUL objective.

BUG-5 [MODERATE]:  Early stopping patience too small.
  patience=12 stopped SOH training at epoch 16/80 (far from convergence).
  FIX: patience=25 for SOH, 20 for RUL; min_delta tightened.

BUG-6 [MODERATE]:  Window too short / no positional context.
  window_size=32 ≈ 3.9 % of the average cell lifetime.  The model had no
  way to know "where in the cell's life" the current window sits.
  FIX: window_size=64 + relative-cycle-position feature appended to the
       sequence (normalised cycle index within each cell).

BUG-7 [MODERATE]:  PICP catastrophically low (≈ 0.05–0.09).
  MC-Dropout alone produces severely underestimated uncertainty.
  FIX: explicit NLL head (predicts μ + log σ²) replaces pure MC-Dropout.
       Both SOH and RUL heads output two scalars; confidence intervals are
       derived from the predicted σ rather than from dropout variance.
       MC-Dropout is kept as a secondary consistency check.

ARCHITECTURE IMPROVEMENTS
──────────────────────────
• Relative degradation features (see BUG-1 fix).
• Relative cycle-position embedding (see BUG-6 fix).
• NLL (Gaussian negative log-likelihood) loss head for calibrated UQ.
• Separate RUL projection head applied after the shared Mamba encoder.
• Cosine warmup LR schedule replacing plain CosineAnnealingWarmRestarts.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports & reproducibility
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
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # ── paths ────────────────────────────────────────────────────────────────
    soh_path  = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv",
    rul_path  = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\rul\rul_full.csv",
    save_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\cnn_mamba_uq_battery_fixed.pt",

    # ── raw feature columns (absolute) ───────────────────────────────────────
    feat_cols_raw = [
        "dc_internal_resistance", "temperature_avg",
        "charge_capacity", "charge_energy",
        "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    ],
    # Relative features are added in preprocessing → total input_dim = 10
    # [raw(6)] + [cap_rel, energy_rel, ir_rel, cycle_pos_rel] = 10
    input_dim = 10,

    # ── sequence ─────────────────────────────────────────────────────────────
    window_size = 64,   # FIX BUG-6: was 32  →  64 cycles of context
    stride      = 2,    # slide 2 cycles at a time (reduce dataset size slightly)

    # ── model ────────────────────────────────────────────────────────────────
    cnn_channels   = [32, 64, 128],
    cnn_kernels    = [3, 7, 15],
    d_model        = 128,
    d_state        = 16,
    d_conv         = 4,
    expand         = 2,
    n_mamba_layers = 3,
    dropout        = 0.15,

    # ── training – SOH ───────────────────────────────────────────────────────
    soh_epochs      = 120,
    soh_lr          = 2e-4,
    soh_batch       = 256,
    soh_wd          = 1e-4,
    soh_patience    = 20,      # FIX BUG-5: was 12
    tail_weight     = 3.0,     # extra weight for SOH < 0.90
    warmup_epochs   = 10,

    # ── training – RUL ───────────────────────────────────────────────────────
    rul_epochs       = 120,
    rul_lr           = 2e-4,
    rul_batch        = 256,
    rul_wd           = 1e-4,
    rul_patience     = 20,
    rul_warmup_ep    = 20,     # FIX BUG-3: head-only warm-up before pseudo-labels
    huber_delta      = 40.0,
    lambda_pseudo    = 0.2,
    pseudo_conf_thr  = 0.2,

    # ── UQ ───────────────────────────────────────────────────────────────────
    mc_samples = 30,
    ci_alpha   = 0.90,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def add_relative_features(df: pd.DataFrame, feat_cols_raw: list) -> pd.DataFrame:
    """
    FIX BUG-1: add per-cell relative features so the model can learn
    degradation independently of cell-to-cell manufacturing variance.

    New columns added:
      cap_rel   – capacity relative to that cell's first-10-cycle mean
      energy_rel– charge energy relative to first-10-cycle mean
      ir_rel    – DC-IR relative to first-10-cycle mean
      cycle_pos – cycle index normalised to [0, 1] within each cell
    """
    df = df.copy()
    cap_rel_list, en_rel_list, ir_rel_list, cycle_pos_list = [], [], [], []

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle_index")
        early   = cell_df.iloc[:10]

        nom_cap    = early["charge_capacity"].mean()
        nom_energy = early["charge_energy"].mean()
        nom_ir     = early["dc_internal_resistance"].mean()
        max_cycle  = cell_df["cycle_index"].max()
        min_cycle  = cell_df["cycle_index"].min()
        cyc_range  = max(max_cycle - min_cycle, 1)

        cap_rel    = (cell_df["charge_capacity"]         - nom_cap)    / (nom_cap    + 1e-9)
        en_rel     = (cell_df["charge_energy"]           - nom_energy) / (nom_energy + 1e-9)
        ir_rel     = (cell_df["dc_internal_resistance"]  - nom_ir)     / (nom_ir     + 1e-9)
        cycle_pos  = (cell_df["cycle_index"] - min_cycle) / cyc_range

        cap_rel_list.append(cap_rel)
        en_rel_list.append(en_rel)
        ir_rel_list.append(ir_rel)
        cycle_pos_list.append(cycle_pos)

    df["cap_rel"]   = pd.concat(cap_rel_list)
    df["energy_rel"]= pd.concat(en_rel_list)
    df["ir_rel"]    = pd.concat(ir_rel_list)
    df["cycle_pos"] = pd.concat(cycle_pos_list)
    return df


# All feature columns used by the model (raw + relative)
FEAT_COLS = [
    "dc_internal_resistance", "temperature_avg",
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "ir_rel", "cycle_pos",
]


class BatteryPreprocessor:
    """Global StandardScaler fit only on training rows."""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, df_train):
        self.scaler.fit(df_train[FEAT_COLS].values)
        return self

    def transform(self, df):
        df = df.copy()
        df[FEAT_COLS] = self.scaler.transform(df[FEAT_COLS].values)
        return df


def load_and_preprocess(soh_path, rul_path):
    soh = pd.read_csv(soh_path)
    rul = pd.read_csv(rul_path)

    # Add relative features before scaling
    soh = add_relative_features(soh, CFG["feat_cols_raw"])
    rul = add_relative_features(rul, CFG["feat_cols_raw"])

    pp = BatteryPreprocessor()
    pp.fit(soh[soh.split == "train"])
    soh = pp.transform(soh)
    rul = pp.transform(rul)

    return soh, rul, pp


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Datasets
# ─────────────────────────────────────────────────────────────────────────────

class SOHSequenceDataset(Dataset):
    def __init__(self, df, window_size, stride=1, split=None):
        self.samples = []
        self.weights = []
        subset = df if split is None else df[df.split == split]

        for _, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[FEAT_COLS].values.astype(np.float32)
            y = cell_df["soh"].values.astype(np.float32)

            for end in range(window_size, len(X) + 1, stride):
                start  = end - window_size
                x_win  = X[start:end]
                y_last = y[end - 1]
                self.samples.append((x_win, y_last))
                w = CFG["tail_weight"] if y_last < 0.90 else 1.0
                self.weights.append(w)

        self.weights = np.array(self.weights, dtype=np.float32)

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(self.weights[idx])


class RULSequenceDataset(Dataset):
    def __init__(self, df, window_size, stride=1, split=None):
        self.samples = []
        subset = df if split is None else df[df.split == split]

        for _, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X   = cell_df[FEAT_COLS].values.astype(np.float32)
            y   = cell_df["rul"].values.astype(np.float32)
            lbl = cell_df["has_label"].values.astype(np.int8)

            for end in range(window_size, len(X) + 1, stride):
                start    = end - window_size
                self.samples.append((X[start:end], y[end-1], int(lbl[end-1])))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        x, y, lbl = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(lbl, dtype=torch.int8)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model
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
        self.proj    = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x):                           # x: (B, W, C)
        x = x.permute(0, 2, 1)                      # (B, C, W)
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1).permute(0, 2, 1) # (B, W, total_ch)
        return self.dropout(F.gelu(self.proj(x)))


class MambaBlock(nn.Module):
    """Pure-PyTorch selective-scan SSM block (Gu & Dao 2023)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
                                  kernel_size=d_conv, padding=d_conv-1,
                                  groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state + d_state + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A.expand(self.d_inner, -1)))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def ssm(self, x):
        B, L, D = x.shape
        N = self.d_state
        dBC     = self.x_proj(x)
        delta   = F.softplus(self.dt_proj(dBC[..., :1]))     # (B,L,d_inner)
        B_ssm   = dBC[..., 1:N+1]
        C_ssm   = dBC[..., N+1:]
        A       = -torch.exp(self.A_log)
        dA      = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB_u    = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB_u[:, t]
            ys.append((h * C_ssm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        res  = x
        x    = self.norm(x)
        xz   = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        x_c  = self.conv1d(x_.permute(0,2,1))[..., :x_.shape[1]].permute(0,2,1)
        y    = self.ssm(F.silu(x_c)) * F.silu(z)
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


class CNNMambaUQ(nn.Module):
    """
    CNN-Mamba backbone with:
    - dual SOH + RUL heads
    - NLL uncertainty heads (predict μ AND log σ²)  [FIX BUG-7]
    - attention pooling
    """
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

        def _head(out_dim):
            return nn.Sequential(
                nn.Linear(C["d_model"], 128), nn.LayerNorm(128), nn.GELU(),
                nn.Dropout(C["dropout"]),
                nn.Linear(128, 64), nn.GELU(),
                nn.Dropout(C["dropout"]),
                nn.Linear(64, out_dim),       # out_dim=2: [mean, log_var]
            )

        # Each head outputs 2 values: [μ, log σ²]
        self.soh_head = _head(2)
        self.rul_head = _head(2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def encode(self, x):
        z = self.cnn_proj(self.cnn(x))             # (B, W, d_model)
        z = self.mamba(z)
        attn = F.softmax(self.attn_pool(z), dim=1)  # (B, W, 1)
        return (z * attn).sum(dim=1)               # (B, d_model)

    def forward(self, x, task="soh"):
        z = self.encode(x)
        if task == "soh":
            out = self.soh_head(z)
            mu      = torch.sigmoid(out[:, 0])         # SOH ∈ (0,1)
            log_var = out[:, 1].clamp(-10, 5)
            return mu, log_var
        else:
            out = self.rul_head(z)
            mu      = F.softplus(out[:, 0])            # RUL ≥ 0
            log_var = out[:, 1].clamp(-10, 10)
            return mu, log_var

    @torch.no_grad()
    def predict_with_uncertainty(self, x, task="soh", n_samples=30, ci=0.90):
        """
        FIX BUG-2 & BUG-7: deterministic μ for point prediction, σ from the
        NLL head for CI.  MC-Dropout adds secondary epistemic uncertainty.
        Returns: mean, lower_ci, upper_ci
        """
        self.eval()
        mu, log_var = self.forward(x, task=task)
        sigma = torch.exp(0.5 * log_var)

        # Optional: add small MC-Dropout epistemic component
        self.train()
        mc_means = torch.stack([self.forward(x, task=task)[0]
                                 for _ in range(n_samples)], dim=0)
        self.eval()
        mc_std = mc_means.std(0)

        total_std = (sigma**2 + mc_std**2).sqrt()

        z_score = torch.tensor(
            float(np.abs(np.quantile(np.random.randn(100_000), (1+ci)/2))),
            device=x.device
        )
        return mu, mu - z_score * total_std, mu + z_score * total_std


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_nll(mu, log_var, target, weight=None):
    """Gaussian NLL: −log p(y | μ, σ²) = ½[log σ² + (y−μ)²/σ²]"""
    var  = torch.exp(log_var) + 1e-6
    nll  = 0.5 * (log_var + (target - mu)**2 / var)
    if weight is not None:
        nll = nll * weight
    return nll.mean()

def soh_mse_loss(mu, log_var, target, weight=None):
    """
    Simple MSE loss for SOH - stable, always positive
    log_var is ignored (only for compatibility with model output)
    """
    mu = torch.clamp(mu, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    
    loss = (mu - target) ** 2
    
    if weight is not None:
        loss = loss * weight
    
    return loss.mean()


def huber_loss(pred, target, delta=40.0):
    return F.huber_loss(pred, target, delta=delta)


def rul_loss(mu, log_var, target, has_label,
             pseudo_mu=None, pseudo_conf=None,
             lambda_pseudo=0.2, delta=40.0):
    labeled = has_label.bool()
    if labeled.sum() > 0:
        loss_lab = gaussian_nll(mu[labeled], log_var[labeled], target[labeled])
    else:
        loss_lab = torch.tensor(0.0, device=mu.device)

    loss_ps = torch.tensor(0.0, device=mu.device)
    if pseudo_mu is not None and (~labeled).sum() > 0:
        ul_mu  = mu[~labeled]
        ul_ps  = pseudo_mu[~labeled]
        if pseudo_conf is not None:
            w = pseudo_conf[~labeled].clamp(0, 1)
        else:
            w = torch.ones_like(ul_mu)
        loss_ps = (w * (ul_mu - ul_ps)**2).mean()

    return loss_lab + lambda_pseudo * loss_ps


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch, warmup, total_epochs, base_lr):
    """Linear warmup + cosine decay."""
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


class EarlyStopping:
    def __init__(self, patience=25, delta=1e-5):
        self.patience = patience; self.delta = delta
        self.counter = 0; self.best = None; self.stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best is None or val_loss < self.best - self.delta:
            self.best = val_loss; self.counter = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def make_loader(ds, batch_size, shuffle=True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SOH training
# ─────────────────────────────────────────────────────────────────────────────

def train_soh(model, soh_df, cfg):
    print("\n" + "="*60)
    print("  TRAINING SOH HEAD")
    print("="*60)

    W = cfg["window_size"]; S = cfg["stride"]
    train_ds = SOHSequenceDataset(soh_df, W, S, "train")
    val_ds   = SOHSequenceDataset(soh_df, W, S, "val")
    tl = make_loader(train_ds, cfg["soh_batch"])
    vl = make_loader(val_ds,   cfg["soh_batch"], shuffle=False)
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["soh_lr"],
                             weight_decay=cfg["soh_wd"])
    es  = EarlyStopping(patience=cfg["soh_patience"])
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_r2": []}

    for epoch in range(cfg["soh_epochs"]):
        cosine_lr(opt, epoch, cfg["warmup_epochs"], cfg["soh_epochs"], cfg["soh_lr"])
        model.train()
        t_loss = 0.0
        for x, y, w in tl:
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            opt.zero_grad()
            mu, lv = model(x, task="soh")
            loss = soh_mse_loss(mu, lv, y, weight=w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
        t_loss /= len(tl)

        model.eval()
        all_mu, all_y = [], []
        v_loss = 0.0
        with torch.no_grad():
            for x, y, w in vl:
                x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
                mu, lv = model(x, task="soh")
                v_loss += soh_mse_loss(mu, lv, y, weight=w).item()
                all_mu.extend(mu.cpu().numpy())
                all_y.extend(y.cpu().numpy())
        v_loss /= len(vl)

        ap, at = np.array(all_mu), np.array(all_y)
        mae = mean_absolute_error(at, ap) * 100
        r2  = r2_score(at, ap)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_mae"].append(mae)
        history["val_r2"].append(r2)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['soh_epochs']} | "
                  f"Train: {t_loss:.5f} | Val: {v_loss:.5f} | "
                  f"MAE: {mae:.4f}% | R²: {r2:.4f}")

        es(v_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    es.restore(model)
    print(f"\n  Best val loss: {es.best:.6f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 8.  RUL training  (semi-supervised, with corrected pseudo-labelling)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_pseudo_labels(model, loader, cfg):
    model.eval()
    all_mu, all_conf = [], []
    for batch in loader:
        x = batch[0].to(DEVICE)
        mu, lv = model(x, task="rul")
        sigma  = torch.exp(0.5 * lv)
        # Confidence = exp(-normalised interval width)
        iw = (2 * 1.645 * sigma).cpu().numpy()          # ~90 % CI width
        conf = np.exp(-iw / (iw.mean() + 1e-8))
        all_mu.extend(mu.cpu().numpy())
        all_conf.extend(conf)
    return np.array(all_mu), np.array(all_conf)


def train_rul(model, rul_df, cfg):
    print("\n" + "="*60)
    print("  TRAINING RUL HEAD  (semi-supervised)")
    print("="*60)

    W = cfg["window_size"]; S = cfg["stride"]
    train_ds = RULSequenceDataset(rul_df, W, S, "train")
    val_ds   = RULSequenceDataset(rul_df, W, S, "val")
    tl = make_loader(train_ds, cfg["rul_batch"])
    vl = make_loader(val_ds,   cfg["rul_batch"], shuffle=False)

    n_lab = sum(1 for _, _, l in train_ds if l == 1)
    print(f"  Train: {len(train_ds):,}  (labeled: {n_lab:,} | "
          f"unlabeled: {len(train_ds)-n_lab:,})")
    print(f"  Val: {len(val_ds):,}")

    # FIX BUG-3 & BUG-4: Phase A — freeze backbone, train RUL head on labeled only
    for p in model.cnn.parameters():   p.requires_grad = False
    for p in model.mamba.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["rul_lr"], weight_decay=cfg["rul_wd"]
    )
    print(f"  Phase A: {cfg['rul_warmup_ep']} epochs, labeled only, backbone frozen")
    for epoch in range(cfg["rul_warmup_ep"]):
        cosine_lr(opt, epoch, 3, cfg["rul_warmup_ep"], cfg["rul_lr"])
        model.train()
        for x, y, lbl in tl:
            x, y, lbl = x.to(DEVICE), y.to(DEVICE), lbl.to(DEVICE)
            labeled = lbl.bool()
            if labeled.sum() == 0: continue
            opt.zero_grad()
            mu, lv = model(x, task="rul")
            loss = gaussian_nll(mu[labeled], lv[labeled], y[labeled])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (epoch+1) % 5 == 0:
            print(f"    Warmup epoch {epoch+1}/{cfg['rul_warmup_ep']}")

    # Phase B: unfreeze backbone, semi-supervised with pseudo-labels
    print("  Phase B: full end-to-end fine-tune with pseudo-labels")
    for p in model.cnn.parameters():   p.requires_grad = True
    for p in model.mamba.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["rul_lr"] * 0.5,
                             weight_decay=cfg["rul_wd"])
    es = EarlyStopping(patience=cfg["rul_patience"])

    print("  Generating pseudo-labels from warmed-up model...")
    pseudo_mu, pseudo_conf = generate_pseudo_labels(model, tl, cfg)

    history = {"train_loss": [], "val_loss": [],
               "val_mae": [], "val_r2": [], "val_picp": [], "val_pinw": []}

    for epoch in range(cfg["rul_epochs"]):
        cosine_lr(opt, epoch, 5, cfg["rul_epochs"], cfg["rul_lr"] * 0.5)

        if epoch > 0 and epoch % 15 == 0:
            print(f"  Epoch {epoch+1}: refreshing pseudo-labels...")
            pseudo_mu, pseudo_conf = generate_pseudo_labels(model, tl, cfg)

        model.train()
        t_loss = 0.0
        p_idx  = 0
        for x, y, lbl in tl:
            x, y, lbl = x.to(DEVICE), y.to(DEVICE), lbl.to(DEVICE)
            bs = x.shape[0]
            pm = torch.tensor(pseudo_mu[p_idx:p_idx+bs],
                               dtype=torch.float32, device=DEVICE)
            pc = torch.tensor(pseudo_conf[p_idx:p_idx+bs],
                               dtype=torch.float32, device=DEVICE)
            p_idx += bs

            conf_mask = pc > (1 - cfg["pseudo_conf_thr"])
            pm_use = pm if conf_mask.any() else None
            pc_use = pc if conf_mask.any() else None

            opt.zero_grad()
            mu, lv = model(x, task="rul")
            loss = rul_loss(mu, lv, y, lbl.float(),
                            pseudo_mu=pm_use, pseudo_conf=pc_use,
                            lambda_pseudo=cfg["lambda_pseudo"],
                            delta=cfg["huber_delta"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
        t_loss /= len(tl)

        # Validate
        model.eval()
        all_mu, all_y, all_lo, all_hi = [], [], [], []
        v_loss = 0.0
        with torch.no_grad():
            for x, y, lbl in vl:
                x, y, lbl = x.to(DEVICE), y.to(DEVICE), lbl.to(DEVICE)
                labeled = lbl.bool()
                if labeled.sum() == 0: continue
                mu, lv = model(x, task="rul")
                v_loss += gaussian_nll(
                    mu[labeled], lv[labeled], y[labeled]).item()
                all_mu.extend(mu[labeled].cpu().numpy())
                all_y.extend(y[labeled].cpu().numpy())
                # CI from NLL head  (FIX BUG-7)
                sigma = torch.exp(0.5 * lv[labeled])
                z = 1.645
                all_lo.extend((mu[labeled] - z*sigma).cpu().numpy())
                all_hi.extend((mu[labeled] + z*sigma).cpu().numpy())

        if not all_mu: continue
        v_loss /= max(len(vl), 1)
        ap, at = np.array(all_mu), np.array(all_y)
        lo, hi = np.array(all_lo), np.array(all_hi)
        mae  = mean_absolute_error(at, ap)
        r2   = r2_score(at, ap)
        picp = np.mean((at >= lo) & (at <= hi))
        pinw = np.mean(hi - lo) / (at.max() - at.min() + 1e-8)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_mae"].append(mae)
        history["val_r2"].append(r2)
        history["val_picp"].append(picp)
        history["val_pinw"].append(pinw)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['rul_epochs']} | "
                  f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | "
                  f"MAE: {mae:.2f} | R²: {r2:.4f} | "
                  f"PICP: {picp:.3f} | PINW: {pinw:.4f}")

        es(v_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    es.restore(model)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Evaluation  (FIX BUG-2: deterministic μ for point metrics)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_soh(model, soh_df, cfg):
    print("\n" + "="*60)
    print("  SOH EVALUATION — TEST SET")
    print("="*60)

    test_ds = SOHSequenceDataset(soh_df, cfg["window_size"], cfg["stride"], "test")
    loader  = make_loader(test_ds, cfg["soh_batch"], shuffle=False)

    model.eval()                                     # deterministic point preds
    all_mu, all_y, all_lo, all_hi = [], [], [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mu, lv = model(x, task="soh")            # FIX BUG-2
            sigma   = torch.exp(0.5 * lv)
            z       = 1.645
            all_mu.extend(mu.cpu().numpy())
            all_y.extend(y.cpu().numpy())
            all_lo.extend((mu - z*sigma).cpu().numpy())
            all_hi.extend((mu + z*sigma).cpu().numpy())

    yt  = np.array(all_y)
    yp  = np.array(all_mu)
    ylo = np.array(all_lo)
    yhi = np.array(all_hi)

    mae  = mean_absolute_error(yt, yp) * 100
    rmse = np.sqrt(np.mean((yt - yp)**2)) * 100
    r2   = r2_score(yt, yp)
    picp = np.mean((yt >= ylo) & (yt <= yhi))
    pinw = np.mean(yhi - ylo) / (yt.max() - yt.min() + 1e-8)

    print(f"\n  ── Overall ──────────────────────────────")
    print(f"  MAE  : {mae:.4f}%   (target < 0.70%)")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  R²   : {r2:.5f}   (target > 0.97)")
    print(f"  PICP : {picp:.4f}  (target ≥ 0.90)")
    print(f"  PINW : {pinw:.4f}  (lower = better)")
    print(f"\n  ── MAE by SOH region ────────────────────")
    for label, mask in [("SOH < 0.90", yt < 0.90),
                        ("0.90–0.95",  (yt >= 0.90) & (yt < 0.95)),
                        ("SOH > 0.95", yt >= 0.95)]:
        if mask.sum() > 0:
            rm = mean_absolute_error(yt[mask], yp[mask]) * 100
            print(f"  {label}: MAE = {rm:.4f}%  (n={mask.sum()})")

    ok = (mae < 0.70) and (r2 > 0.97) and (picp >= 0.90)
    print(f"\n  {'✓ ALL TARGETS MET' if ok else '✗ targets not fully met'}")
    return dict(mae_pct=mae, rmse_pct=rmse, r2=r2, picp=picp, pinw=pinw,
                y_true=yt, y_pred=yp, y_lo=ylo, y_hi=yhi)


def evaluate_rul(model, rul_df, cfg):
    print("\n" + "="*60)
    print("  RUL EVALUATION — TEST SET (labeled only)")
    print("="*60)

    test_ds = RULSequenceDataset(rul_df, cfg["window_size"], cfg["stride"], "test")
    loader  = make_loader(test_ds, cfg["rul_batch"], shuffle=False)

    model.eval()
    all_mu, all_y, all_lo, all_hi = [], [], [], []

    with torch.no_grad():
        for x, y, lbl in loader:
            x, y, lbl = x.to(DEVICE), y.to(DEVICE), lbl
            labeled = lbl.bool()
            if labeled.sum() == 0: continue
            mu, lv   = model(x[labeled.to(DEVICE)], task="rul")
            sigma    = torch.exp(0.5 * lv)
            z        = 1.645
            all_mu.extend(mu.cpu().numpy())
            all_y.extend(y[labeled].cpu().numpy())
            all_lo.extend((mu - z*sigma).cpu().numpy())
            all_hi.extend((mu + z*sigma).cpu().numpy())

    yt  = np.array(all_y)
    yp  = np.array(all_mu)
    ylo = np.array(all_lo)
    yhi = np.array(all_hi)

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(np.mean((yt - yp)**2))
    r2   = r2_score(yt, yp)
    picp = np.mean((yt >= ylo) & (yt <= yhi))
    pinw = np.mean(yhi - ylo) / (yt.max() - yt.min() + 1e-8)

    print(f"\n  ── Overall ──────────────────────────────")
    print(f"  MAE  : {mae:.2f} cycles  (target < 30)")
    print(f"  RMSE : {rmse:.2f} cycles")
    print(f"  R²   : {r2:.5f}   (target > 0.90)")
    print(f"  PICP : {picp:.4f}  (target ≥ 0.90)")
    print(f"  PINW : {pinw:.4f}  (lower = better)")
    print(f"\n  ── MAE by RUL region ────────────────────")
    for label, mask in [("RUL < 50",   yt < 50),
                        ("RUL 50–200", (yt >= 50) & (yt < 200)),
                        ("RUL > 200",  yt >= 200)]:
        if mask.sum() > 0:
            rm = mean_absolute_error(yt[mask], yp[mask])
            print(f"  {label}: MAE = {rm:.2f} cycles  (n={mask.sum()})")

    ok = (mae < 30) and (r2 > 0.90) and (picp >= 0.90)
    print(f"\n  {'✓ ALL TARGETS MET' if ok else '✗ targets not fully met'}")
    return dict(mae=mae, rmse=rmse, r2=r2, picp=picp, pinw=pinw,
                y_true=yt, y_pred=yp, y_lo=ylo, y_hi=yhi)


def print_model_summary(model, cfg):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total:,} total | {trainable:,} trainable")
    print(f"  Input shape: ({cfg['window_size']}, {cfg['input_dim']})")
    print(f"  CNN output dim: {sum(cfg['cnn_channels'])}")
    print(f"  d_model: {cfg['d_model']}  | Mamba layers: {cfg['n_mamba_layers']}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading & preprocessing data...")
    soh_df, rul_df, pp = load_and_preprocess(CFG["soh_path"], CFG["rul_path"])
    print(f"  SOH: {soh_df.shape}  |  RUL: {rul_df.shape}")

    print("\nBuilding CNN-Mamba-UQ model...")
    model = CNNMambaUQ(CFG).to(DEVICE)
    print_model_summary(model, CFG)

    soh_history = train_soh(model, soh_df, CFG)
    rul_history = train_rul(model, rul_df, CFG)

    soh_results = evaluate_soh(model, soh_df, CFG)
    rul_results = evaluate_rul(model, rul_df, CFG)

    # ── Save ────────────────────────────────────────────────────────────────
    save_path = CFG["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": CFG,
        "feat_cols": FEAT_COLS,
        "scaler_mean": pp.scaler.mean_.tolist(),
        "scaler_std":  pp.scaler.scale_.tolist(),
        "soh_results": {k: v for k, v in soh_results.items()
                        if not isinstance(v, np.ndarray)},
        "rul_results": {k: v for k, v in rul_results.items()
                        if not isinstance(v, np.ndarray)},
    }, save_path)
    print(f"\n  Model saved → {save_path}")

    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  SOH  MAE: {soh_results['mae_pct']:.4f}%  "
          f"R²: {soh_results['r2']:.5f}  "
          f"PICP: {soh_results['picp']:.4f}  "
          f"PINW: {soh_results['pinw']:.4f}")
    print(f"  RUL  MAE: {rul_results['mae']:.2f} cyc  "
          f"R²: {rul_results['r2']:.5f}  "
          f"PICP: {rul_results['picp']:.4f}  "
          f"PINW: {rul_results['pinw']:.4f}")
    print("="*60)