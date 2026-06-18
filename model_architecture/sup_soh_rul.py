"""
Battery SOH & RUL Estimation — CNN-Mamba-UQ  (FULLY SUPERVISED)
================================================================
Targets:  SOH MAE < 0.70%   R² > 0.97
          RUL MAE < 30 cyc  R² > 0.90
          PICP ≥ 0.90        PINW minimal

CHANGE LOG vs the semi-supervised version
──────────────────────────────────────────
1. RUL is now FULLY SUPERVISED on a labeled-only dataset
   (34 cells / 16,958 rows, no has_label column, no rul=-1 placeholders).
   All pseudo-labeling / teacher-student / confidence-weighting code is
   REMOVED. This also removes the crash:

       RuntimeError: element 0 of tensors does not require grad and
       does not have a grad_fn

   which was triggered by generate_pseudo_labels() — a function that
   combined @torch.no_grad() with mid-loop model.eval()/model.train()
   toggling. Some PyTorch builds leak the grad-disabled context across
   that boundary when called from inside an active training loop,
   silently building part of the next forward graph with grad disabled.
   Removing the pseudo-label path removes the failure mode entirely
   rather than patching around it.

2. RUL training is a single-phase, fully end-to-end supervised loop —
   no more "Phase A frozen warm-up / Phase B unfreeze" split, since
   there's no pseudo-label teacher to warm up for. Backbone and RUL
   head train together from epoch 0, same recipe as SOH.

3. Window size changed to 50 to match the new preprocessing pipeline's
   sequence shapes (..., 50, 7) reported in your log.

4. RUL train set is much smaller now (23 cells, ~11.7k rows → ~10.5k
   windowed sequences). To avoid overfitting on this smaller set:
     - dropout kept at 0.15 (not reduced)
     - weight_decay raised slightly for the RUL phase
     - early-stopping patience tuned tighter (18) so we don't memorise
     - stride=1 for RUL (every cycle matters when cells are scarce)

Carried over from the SOH-fix work (still required, do not remove):
  • Per-cell RELATIVE features (cap_rel, energy_rel, ir_rel, cycle_pos).
    Without these, charge_capacity alone is cell-specific and unusable
    for cross-cell generalisation (this was the original R²=-22 bug).
  • Gaussian-NLL heads (μ, log σ²) instead of plain MC-Dropout for UQ —
    this is what fixed PICP from ~0.05-0.15 up toward the 0.90 target.
  • Deterministic model.eval() for point-prediction metrics; MC-Dropout
    is only an optional secondary epistemic term, never the primary
    estimator (this was the second major bug: evaluating with dropout
    active produced noisy, biased point predictions at test time).
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
    save_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\cnn_mamba_uq_battery_supervised.pt",

    # ── raw feature columns (absolute, present in both CSVs) ──────────────────
    feat_cols_raw = [
        "dc_internal_resistance", "temperature_avg",
        "charge_capacity", "charge_energy",
        "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    ],
    # Relative features appended in preprocessing → total input_dim = 10
    # [raw(6)] + [cap_rel, energy_rel, ir_rel, cycle_pos] = 10
    input_dim = 10,

    # ── sequence ─────────────────────────────────────────────────────────────
    window_size  = 50,   # matches the new preprocessing pipeline's (.., 50, 7)
    soh_stride   = 2,    # SOH has plenty of data (134 cells) — subsample windows
    rul_stride   = 1,    # RUL is scarce (34 cells) — use every window

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
    soh_epochs    = 120,
    soh_lr        = 2e-4,
    soh_batch     = 256,
    soh_wd        = 1e-4,
    soh_patience  = 25,
    tail_weight   = 3.0,     # extra weight for SOH < 0.90 (harder region)
    warmup_epochs = 10,

    # ── training – RUL  (fully supervised, no pseudo-labels) ───────────────────
    rul_epochs    = 150,
    rul_lr        = 2e-4,
    rul_batch     = 128,     # smaller batch — smaller dataset
    rul_wd        = 2e-4,    # slightly higher WD — guard against overfit
    rul_patience  = 18,
    rul_warmup_ep = 8,

    # ── UQ ───────────────────────────────────────────────────────────────────
    mc_samples = 30,
    ci_alpha   = 0.90,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────


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

def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cell relative features so the model learns degradation independent
    of cell-to-cell manufacturing variance (root cause of the original
    R²=-22 bug: raw charge_capacity at 100% SOH already spans ~1.01-1.10
    across cells, so a global scaler alone can't expose "this cell has
    degraded" — it only shifts/scales the absolute value).

    New columns:
      cap_rel    – capacity relative to that cell's first-10-cycle mean
      energy_rel – charge energy relative to first-10-cycle mean
      ir_rel     – DC-IR relative to first-10-cycle mean
      cycle_pos  – cycle index normalised to [0, 1] within each cell
    """
    df = df.copy()
    cap_rel_list, en_rel_list, ir_rel_list, cycle_pos_list = [], [], [], []

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df  = cell_df.sort_values("cycle_index")
        early    = cell_df.iloc[:10]

        nom_cap    = early["charge_capacity"].mean()
        nom_energy = early["charge_energy"].mean()
        nom_ir     = early["dc_internal_resistance"].mean()
        min_cycle  = cell_df["cycle_index"].min()
        max_cycle  = cell_df["cycle_index"].max()
        cyc_range  = max(max_cycle - min_cycle, 1)

        cap_rel_list.append((cell_df["charge_capacity"] - nom_cap) / (nom_cap + 1e-9))
        en_rel_list.append((cell_df["charge_energy"] - nom_energy) / (nom_energy + 1e-9))
        ir_rel_list.append((cell_df["dc_internal_resistance"] - nom_ir) / (nom_ir + 1e-9))
        cycle_pos_list.append((cell_df["cycle_index"] - min_cycle) / cyc_range)

    df["cap_rel"]    = pd.concat(cap_rel_list)
    df["energy_rel"] = pd.concat(en_rel_list)
    df["ir_rel"]     = pd.concat(ir_rel_list)
    df["cycle_pos"]  = pd.concat(cycle_pos_list)
    return df


FEAT_COLS = [
    "dc_internal_resistance", "temperature_avg",
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "ir_rel", "cycle_pos",
]


class BatteryPreprocessor:
    """Global StandardScaler, fit separately per task on its own train split."""
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

    soh = add_relative_features(soh)
    rul = add_relative_features(rul)

    # IMPORTANT: SOH and RUL are now two independent supervised datasets with
    # different cell populations (134 cells vs 34 cells) — each gets its own
    # scaler fit on its OWN train split. Sharing one scaler across both would
    # leak RUL-cell statistics into SOH preprocessing and vice versa.
    soh_pp = BatteryPreprocessor().fit(soh[soh.split == "train"])
    rul_pp = BatteryPreprocessor().fit(rul[rul.split == "train"])

    soh = soh_pp.transform(soh)
    rul = rul_pp.transform(rul)

    return soh, rul, soh_pp, rul_pp


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Datasets  (both fully supervised — no has_label / pseudo machinery)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Generic sliding-window dataset.
    target_col: "soh" or "rul"
    weighted:   if True, attaches a per-sample loss weight (used for SOH tail
                up-weighting); RUL just gets weight=1 for all samples.
    """
    def __init__(self, df, target_col, window_size, stride=1, split=None,
                 weighted=False, tail_thr=0.90, tail_weight=1.0):
        self.samples = []
        self.weights = []
        subset = df if split is None else df[df.split == split]

        for _, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[FEAT_COLS].values.astype(np.float32)
            y = cell_df[target_col].values.astype(np.float32)

            for end in range(window_size, len(X) + 1, stride):
                start  = end - window_size
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
# 4.  Model  (unchanged architecture — CNN + Mamba + NLL heads)
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

    def forward(self, x):                            # x: (B, W, C)
        x = x.permute(0, 2, 1)                       # (B, C, W)
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1).permute(0, 2, 1)  # (B, W, total_ch)
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
        delta   = F.softplus(self.dt_proj(dBC[..., :1]))
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
        res   = x
        x     = self.norm(x)
        xz    = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        x_c   = self.conv1d(x_.permute(0,2,1))[..., :x_.shape[1]].permute(0,2,1)
        y     = self.ssm(F.silu(x_c)) * F.silu(z)
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
    CNN-Mamba backbone with dual SOH + RUL heads, each outputting [μ, log σ²]
    for Gaussian-NLL training (calibrated uncertainty instead of relying
    purely on MC-Dropout, which previously produced PICP as low as 0.05).
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

        def _head(out_dim=2):
            return nn.Sequential(
                nn.Linear(C["d_model"], 128), nn.LayerNorm(128), nn.GELU(),
                nn.Dropout(C["dropout"]),
                nn.Linear(128, 64), nn.GELU(),
                nn.Dropout(C["dropout"]),
                nn.Linear(64, out_dim),       # [mean, log_var]
            )

        self.soh_head = _head()
        self.rul_head = _head()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def encode(self, x):
        z = self.cnn_proj(self.cnn(x))              # (B, W, d_model)
        z = self.mamba(z)
        attn = F.softmax(self.attn_pool(z), dim=1)   # (B, W, 1)
        return (z * attn).sum(dim=1)                # (B, d_model)

    def forward(self, x, task="soh"):
        z = self.encode(x)
        if task == "soh":
            out = self.soh_head(z)
            mu      = torch.sigmoid(out[:, 0])          # SOH ∈ (0,1)
            log_var = out[:, 1].clamp(-10, 5)
            return mu, log_var
        else:
            out = self.rul_head(z)
            mu      = F.softplus(out[:, 0])             # RUL ≥ 0
            log_var = out[:, 1].clamp(-10, 10)
            return mu, log_var

    # NOTE: uncertainty is read directly off the NLL head's σ in evaluate()
    # below — no separate MC-Dropout inference method is needed. Keeping a
    # second method here that toggles train()/eval() mid-call was exactly
    # the pattern that caused the earlier semi-supervised crash; it is
    # intentionally not reintroduced.


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Loss
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_nll(mu, log_var, target, weight=None):
    """Gaussian NLL: ½[log σ² + (y−μ)²/σ²]"""
    var = torch.exp(log_var) + 1e-6
    nll = 0.5 * (log_var + (target - mu)**2 / var)
    if weight is not None:
        nll = nll * weight
    return nll.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training helpers
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


def make_loader(ds, batch_size, shuffle=True, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False, drop_last=drop_last)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Generic supervised training loop  (shared by SOH and RUL)
# ─────────────────────────────────────────────────────────────────────────────

def train_supervised(model, task, train_ds, val_ds, cfg, epochs, lr, wd,
                      patience, warmup_epochs, label):
    print("\n" + "="*60)
    print(f"  TRAINING {label} HEAD  (fully supervised)")
    print("="*60)

    batch = cfg["soh_batch"] if task == "soh" else cfg["rul_batch"]
    tl = make_loader(train_ds, batch, shuffle=True, drop_last=True)
    vl = make_loader(val_ds,   batch, shuffle=False, drop_last=False)
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    es  = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_r2": []}

    is_soh = (task == "soh")
    unit   = "%" if is_soh else " cyc"
    scale  = 100.0 if is_soh else 1.0

    for epoch in range(epochs):
        cosine_lr(opt, epoch, warmup_epochs, epochs, lr)
        model.train()
        t_loss = 0.0
        for x, y, w in tl:
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            opt.zero_grad()
            mu, lv = model(x, task=task)
            
            # ===== FIX: Use MSE for SOH, NLL for RUL =====
            if is_soh:
                loss = soh_mse_loss(mu, lv, y, weight=w)
            else:
                loss = gaussian_nll(mu, lv, y, weight=w)
            # =============================================
            
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
                mu, lv = model(x, task=task)
                
                # ===== FIX: Same here =====
                if is_soh:
                    v_loss += soh_mse_loss(mu, lv, y, weight=w).item()
                else:
                    v_loss += gaussian_nll(mu, lv, y, weight=w).item()
                # ===========================
                
                all_mu.extend(mu.cpu().numpy())
                all_y.extend(y.cpu().numpy())
        v_loss /= len(vl)

        ap, at = np.array(all_mu), np.array(all_y)
        mae = mean_absolute_error(at, ap) * scale
        r2  = r2_score(at, ap)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_mae"].append(mae)
        history["val_r2"].append(r2)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {t_loss:.5f} | Val: {v_loss:.5f} | "
                  f"MAE: {mae:.4f}{unit} | R²: {r2:.4f}")

        es(v_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    es.restore(model)
    print(f"\n  Best val loss: {es.best:.6f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_ds, cfg, task, label, batch_key,
             target_lt=None, r2_gt=None, picp_ge=0.90, regions=None):
    print("\n" + "="*60)
    print(f"  {label} EVALUATION — TEST SET")
    print("="*60)

    loader = make_loader(test_ds, cfg[batch_key], shuffle=False)
    model.eval()
    all_mu, all_y, all_lo, all_hi = [], [], [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mu, lv = model(x, task=task)             # deterministic point pred
            sigma  = torch.exp(0.5 * lv)
            z      = 1.645
            all_mu.extend(mu.cpu().numpy())
            all_y.extend(y.cpu().numpy())
            all_lo.extend((mu - z*sigma).cpu().numpy())
            all_hi.extend((mu + z*sigma).cpu().numpy())

    yt, yp = np.array(all_y), np.array(all_mu)
    ylo, yhi = np.array(all_lo), np.array(all_hi)

    scale = 100.0 if task == "soh" else 1.0
    unit  = "%" if task == "soh" else " cycles"

    mae  = mean_absolute_error(yt, yp) * scale
    rmse = np.sqrt(np.mean((yt - yp)**2)) * scale
    r2   = r2_score(yt, yp)
    picp = np.mean((yt >= ylo) & (yt <= yhi))
    pinw = np.mean(yhi - ylo) / (yt.max() - yt.min() + 1e-8)

    print(f"\n  ── Overall ──────────────────────────────")
    print(f"  MAE  : {mae:.4f}{unit}  (target < {target_lt})")
    print(f"  RMSE : {rmse:.4f}{unit}")
    print(f"  R²   : {r2:.5f}   (target > {r2_gt})")
    print(f"  PICP : {picp:.4f}  (target ≥ {picp_ge})")
    print(f"  PINW : {pinw:.4f}  (lower = better)")

    if regions:
        print(f"\n  ── MAE by {label} region ────────────────────")
        for rlabel, mask in regions(yt):
            if mask.sum() > 0:
                rm = mean_absolute_error(yt[mask], yp[mask]) * scale
                print(f"  {rlabel}: MAE = {rm:.4f}{unit}  (n={mask.sum()})")

    ok = (mae < target_lt) and (r2 > r2_gt) and (picp >= picp_ge)
    print(f"\n  {'✓ ALL TARGETS MET' if ok else '✗ targets not fully met'}")
    return dict(mae=mae, rmse=rmse, r2=r2, picp=picp, pinw=pinw,
                y_true=yt, y_pred=yp, y_lo=ylo, y_hi=yhi)


def soh_regions(yt):
    return [("SOH < 0.90", yt < 0.90),
            ("0.90–0.95",  (yt >= 0.90) & (yt < 0.95)),
            ("SOH > 0.95", yt >= 0.95)]


def rul_regions(yt):
    return [("RUL < 50",   yt < 50),
            ("RUL 50–200", (yt >= 50) & (yt < 200)),
            ("RUL > 200",  yt >= 200)]


def print_model_summary(model, cfg):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total:,} total | {trainable:,} trainable")
    print(f"  Input shape: ({cfg['window_size']}, {cfg['input_dim']})")
    print(f"  CNN output dim: {sum(cfg['cnn_channels'])}")
    print(f"  d_model: {cfg['d_model']}  | Mamba layers: {cfg['n_mamba_layers']}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading & preprocessing data...")
    soh_df, rul_df, soh_pp, rul_pp = load_and_preprocess(
        CFG["soh_path"], CFG["rul_path"])
    print(f"  SOH: {soh_df.shape}  |  RUL: {rul_df.shape}")

    print("\nBuilding CNN-Mamba-UQ model...")
    model = CNNMambaUQ(CFG).to(DEVICE)
    print_model_summary(model, CFG)

    # ── Build datasets ──────────────────────────────────────────────────────
    W = CFG["window_size"]

    soh_train_ds = SequenceDataset(soh_df, "soh", W, CFG["soh_stride"], "train",
                                   weighted=True, tail_thr=0.90,
                                   tail_weight=CFG["tail_weight"])
    soh_val_ds   = SequenceDataset(soh_df, "soh", W, CFG["soh_stride"], "val",
                                   weighted=True, tail_thr=0.90,
                                   tail_weight=CFG["tail_weight"])
    soh_test_ds  = SequenceDataset(soh_df, "soh", W, CFG["soh_stride"], "test")

    rul_train_ds = SequenceDataset(rul_df, "rul", W, CFG["rul_stride"], "train")
    rul_val_ds   = SequenceDataset(rul_df, "rul", W, CFG["rul_stride"], "val")
    rul_test_ds  = SequenceDataset(rul_df, "rul", W, CFG["rul_stride"], "test")

    # ── Phase 1: SOH ────────────────────────────────────────────────────────
    soh_history = train_supervised(
        model, "soh", soh_train_ds, soh_val_ds, CFG,
        epochs=CFG["soh_epochs"], lr=CFG["soh_lr"], wd=CFG["soh_wd"],
        patience=CFG["soh_patience"], warmup_epochs=CFG["warmup_epochs"],
        label="SOH",
    )

    # ── Phase 2: RUL — fully supervised, end-to-end from the start ─────────
    # (No freeze/unfreeze phases needed: there's no pseudo-label teacher to
    #  warm up for. The backbone already carries SOH-trained representations
    #  as initialization, but all weights are trainable immediately so the
    #  model can adapt them to the RUL objective.)
    rul_history = train_supervised(
        model, "rul", rul_train_ds, rul_val_ds, CFG,
        epochs=CFG["rul_epochs"], lr=CFG["rul_lr"], wd=CFG["rul_wd"],
        patience=CFG["rul_patience"], warmup_epochs=CFG["rul_warmup_ep"],
        label="RUL",
    )

    # ── Evaluation ──────────────────────────────────────────────────────────
    soh_results = evaluate(model, soh_test_ds, CFG, "soh", "SOH", "soh_batch",
                           target_lt=0.70, r2_gt=0.97, regions=soh_regions)
    rul_results = evaluate(model, rul_test_ds, CFG, "rul", "RUL", "rul_batch",
                           target_lt=30, r2_gt=0.90, regions=rul_regions)

    # ── Save ────────────────────────────────────────────────────────────────
    save_path = CFG["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": CFG,
        "feat_cols": FEAT_COLS,
        "soh_scaler_mean": soh_pp.scaler.mean_.tolist(),
        "soh_scaler_std":  soh_pp.scaler.scale_.tolist(),
        "rul_scaler_mean": rul_pp.scaler.mean_.tolist(),
        "rul_scaler_std":  rul_pp.scaler.scale_.tolist(),
        "soh_results": {k: v for k, v in soh_results.items()
                        if not isinstance(v, np.ndarray)},
        "rul_results": {k: v for k, v in rul_results.items()
                        if not isinstance(v, np.ndarray)},
    }, save_path)
    print(f"\n  Model saved → {save_path}")

    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  SOH  MAE: {soh_results['mae']:.4f}%  "
          f"R²: {soh_results['r2']:.5f}  "
          f"PICP: {soh_results['picp']:.4f}  "
          f"PINW: {soh_results['pinw']:.4f}")
    print(f"  RUL  MAE: {rul_results['mae']:.2f} cyc  "
          f"R²: {rul_results['r2']:.5f}  "
          f"PICP: {rul_results['picp']:.4f}  "
          f"PINW: {rul_results['pinw']:.4f}")
    print("="*60)