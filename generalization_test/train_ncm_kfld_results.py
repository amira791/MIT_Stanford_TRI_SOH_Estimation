# train_snl_ncm_kfold_v2.py
# K-Fold by cell training on SNL NCM dataset (9 features)
# v2 changes vs the version you ran:
#   1. assign_kfold_splits() is now PROTOCOL-STRATIFIED: cells are assigned
#      to test folds (and to train/val within each fold's remaining cells)
#      round-robin WITHIN each protocol group, instead of one global random
#      shuffle. This doesn't change what the model can do - it reduces how
#      much of the fold-to-fold variance you observed (R2 0.81-0.95, cal.
#      PICP 0.79-0.99) was "this fold's test/val cells happened to be a
#      skewed protocol mix" vs genuine generalization variance. Reported
#      std should tighten and become more trustworthy as a result.
#   2. Pooled out-of-fold (OOF) metrics are now computed IN ADDITION to the
#      existing mean-of-folds +/- std. Since every cell is test exactly
#      once across the 5 folds, concatenating all 5 folds' predictions
#      gives one set of metrics computed over the FULL 14,632-row dataset,
#      each row predicted by a model that never saw that row's cell. This
#      is the other conventional way CV results get reported and is
#      commonly expected alongside mean+-std.
# Everything else (model, losses, training loop, per-fold evaluation logic)
# is unchanged from your version.

import os
import math
import time
import warnings
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── 9 FEATURES (from v3 preprocessing) ───
NCM_FEAT_COLS = [
    "charge_capacity",
    "charge_energy",
    "coulombic_efficiency_lagged_1",
    "coulombic_efficiency_lagged_2",
    "cap_rel",
    "energy_rel",
    "cycle_pos",
    "temperature_avg",
    "voltage_range",
]

CFG = dict(
    data_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results\ncm_with_temp_processed_v3.csv",
    save_dir = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints",

    input_dim  = 9,
    window_size = 50,
    soh_stride  = 2,

    cnn_channels = [32, 64, 128],
    cnn_kernels  = [3, 7, 15],
    d_model      = 128,
    d_state      = 16,
    d_conv       = 4,
    expand       = 2,
    n_mamba_layers = 3,
    dropout      = 0.15,

    bidirectional = True,
    evidential    = True,
    calibrate     = True,

    # ─── K-Fold Settings ───
    n_folds = 5,
    kfold_seed = 42,
    fold_val_frac = 0.2,   # fraction of each fold's non-test cells used for val,
                             # allocated per-protocol (see assign_kfold_splits)

    # Training (same as MIT)
    soh_epochs   = 120,
    soh_lr       = 2e-4,
    soh_batch    = 256,
    soh_wd       = 1e-4,
    soh_patience = 25,
    tail_weight  = 3.0,
    warmup_epochs = 10,

    nig_mse_warmup_epochs = 10,
    evid_lambda   = 0.01,
    evid_lambda_max = 0.05,
    evid_mse_weight = 1.0,

    latency_batch_sizes = [1, 32, 256],
    latency_reps = 100,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ncm_data(data_path):
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} rows, {df['cell_id'].nunique()} cells")
    print(f"  SOH range: [{df['soh'].min():.4f}, {df['soh'].max():.4f}]")
    if "protocol" not in df.columns:
        raise KeyError("Expected a 'protocol' column for stratified fold assignment - "
                        "check this CSV was produced by the v3 (or later) prepare script.")
    print(f"  Protocol groups: {sorted(df['protocol'].unique())}")
    return df


class NCMSequenceDataset(Dataset):
    def __init__(self, df, window_size, stride=1, split=None,
                 weighted=False, tail_thr=0.90, tail_weight=1.0):
        self.samples = []
        self.weights = []
        self.cell_ids = []
        subset = df if split is None else df[df.split == split]

        for cid, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[NCM_FEAT_COLS].values.astype(np.float32)
            y = cell_df["soh"].values.astype(np.float32)

            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                y_last = y[end - 1]
                self.samples.append((X[start:end], y_last))
                self.cell_ids.append(cid)
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
# 3. Model (FULL BEM-SOH) - unchanged
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, channels, kernels, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for ch, k in zip(channels, kernels):
            self.branches.append(nn.Sequential(
                nn.Conv1d(input_dim, ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(ch), nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=k, padding=k // 2, bias=False),
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
                                 kernel_size=d_conv, padding=d_conv - 1,
                                 groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
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
        B_ssm = dBC[..., 1:N + 1]
        C_ssm = dBC[..., N + 1:]
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
        x_c = self.conv1d(x_.permute(0, 2, 1))[..., :x_.shape[1]].permute(0, 2, 1)
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


class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, n_layers, dropout):
        super().__init__()
        self.fwd = MambaEncoder(d_model, d_state, d_conv, expand, n_layers, dropout)
        self.bwd = MambaEncoder(d_model, d_state, d_conv, expand, n_layers, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        z_f = self.fwd(x)
        z_b = self.bwd(torch.flip(x, dims=[1]))
        z_b = torch.flip(z_b, dims=[1])
        return self.fuse(torch.cat([z_f, z_b], dim=-1))


class EvidentialHead(nn.Module):
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),
        )

    def forward(self, z):
        out = self.net(z)
        gamma = torch.sigmoid(out[:, 0])
        nu = F.softplus(out[:, 1]).clamp(max=50.0) + 1e-6
        alpha = F.softplus(out[:, 2]).clamp(max=50.0) + 1.0 + 1e-6
        beta = F.softplus(out[:, 3]) + 1e-6
        return gamma, nu, alpha, beta


class GaussianHead(nn.Module):
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, z):
        out = self.net(z)
        mu = torch.sigmoid(out[:, 0])
        log_var = out[:, 1].clamp(-10, 5)
        return mu, log_var


class BEM_SOH(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg
        self.cfg = cfg

        self.cnn = MultiScaleCNN(C["input_dim"], C["cnn_channels"],
                                  C["cnn_kernels"], C["dropout"])
        cnn_out = sum(C["cnn_channels"])

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out, C["d_model"]),
            nn.LayerNorm(C["d_model"]), nn.GELU(),
            nn.Dropout(C["dropout"]),
        )

        if C["bidirectional"]:
            self.encoder = BiMambaEncoder(C["d_model"], C["d_state"], C["d_conv"],
                                           C["expand"], C["n_mamba_layers"], C["dropout"])
        else:
            self.encoder = MambaEncoder(C["d_model"], C["d_state"], C["d_conv"],
                                         C["expand"], C["n_mamba_layers"], C["dropout"])

        self.attn_pool = nn.Linear(C["d_model"], 1)

        if C["evidential"]:
            self.head = EvidentialHead(C["d_model"], C["dropout"])
        else:
            self.head = GaussianHead(C["d_model"], C["dropout"])

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
        z = self.cnn_proj(self.cnn(x))
        z = self.encoder(z)
        attn = F.softmax(self.attn_pool(z), dim=1)
        return (z * attn).sum(dim=1), attn

    def forward(self, x):
        z, attn = self.encode(x)
        return self.head(z), attn


# ─────────────────────────────────────────────────────────────────────────────
# 4. Losses - unchanged
# ─────────────────────────────────────────────────────────────────────────────

def mse_warmup_loss(pred_mean, target, weight=None):
    pred_mean = torch.clamp(pred_mean, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    loss = (pred_mean - target) ** 2
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def nig_nll(gamma, nu, alpha, beta, y):
    two_b_lambda = 2 * beta * (1 + nu)
    nll = 0.5 * torch.log(math.pi / nu) \
        - alpha * torch.log(two_b_lambda) \
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_b_lambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll


def evidential_regularizer(gamma, nu, alpha, y):
    error = torch.abs(y - gamma)
    evidence = 2 * nu + alpha
    return error * evidence


def evidential_loss(gamma, nu, alpha, beta, y, weight=None, lam=0.01, mse_weight=1.0):
    y_c = torch.clamp(y, 0.0, 1.0)
    gamma_c = torch.clamp(gamma, 0.0, 1.0)
    mse = (gamma_c - y_c) ** 2
    loss = nig_nll(gamma, nu, alpha, beta, y) \
        + lam * evidential_regularizer(gamma, nu, alpha, y) \
        + mse_weight * mse
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def forward_and_loss(model, x, y, w, cfg, epoch):
    (out, attn) = model(x)
    gamma, nu, alpha, beta = out

    if epoch < cfg["nig_mse_warmup_epochs"]:
        loss = mse_warmup_loss(gamma, y, weight=w)
    else:
        prog = min(1.0, (epoch - cfg["nig_mse_warmup_epochs"]) /
                   max(1, cfg["soh_epochs"] - cfg["nig_mse_warmup_epochs"]))
        lam = cfg["evid_lambda"] + prog * (cfg["evid_lambda_max"] - cfg["evid_lambda"])
        loss = evidential_loss(gamma, nu, alpha, beta, y, weight=w, lam=lam,
                                mse_weight=cfg["evid_mse_weight"])

    return loss, gamma


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training Utilities - unchanged
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch, warmup, total_epochs, base_lr):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


class EarlyStopping:
    def __init__(self, patience=25, delta=1e-5):
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


def train_fold(model, train_ds, val_ds, cfg, fold_idx):
    print("\n" + "=" * 60)
    print(f"  TRAINING FOLD {fold_idx+1}/{cfg['n_folds']}")
    print("=" * 60)

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
            loss, _ = forward_and_loss(model, x, y, w, cfg, epoch)
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
                loss, point_pred = forward_and_loss(model, x, y, w, cfg, epoch)
                val_loss += loss.item()
                all_pred.extend(point_pred.cpu().numpy())
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
            with torch.no_grad():
                xb, yb, wb = next(iter(val_loader))
                (g, nu_b, a_b, _), _ = model(xb.to(DEVICE))
                mean_evidence = (2 * nu_b + a_b).mean().item()
            print(f"  Epoch {epoch + 1:3d}/{cfg['soh_epochs']} | "
                  f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                  f"MAE: {mae:.4f}% | R2: {r2:.4f} | evidence: {mean_evidence:.2f}")

        es(val_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    es.restore(model)
    print(f"\n  Best val loss: {es.best:.6f}")
    return history, es.best


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation Functions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, cfg):
    model.eval()
    all_y, all_mu, all_sigma = [], [], []
    all_aleatoric, all_epistemic = [], []

    for x, y, w in loader:
        x = x.to(DEVICE)
        out, _ = model(x)
        gamma, nu, alpha, beta = out
        aleatoric = (beta / (alpha - 1)).cpu().numpy()
        epistemic = (beta / (nu * (alpha - 1))).cpu().numpy()
        sigma = np.sqrt(aleatoric + epistemic)
        mu = gamma.cpu().numpy()
        all_aleatoric.extend(aleatoric)
        all_epistemic.extend(epistemic)
        all_mu.extend(mu)
        all_sigma.extend(sigma)
        all_y.extend(y.numpy())

    return (np.array(all_y), np.array(all_mu), np.array(all_sigma),
            np.array(all_aleatoric), np.array(all_epistemic))


def fit_isotonic_calibrator(y_true, mu, sigma, n_q=20):
    quantiles = np.linspace(0.05, 0.95, n_q)
    empirical = []
    for q in quantiles:
        z = norm.ppf(q)
        covered = (y_true <= mu + z * sigma).mean()
        empirical.append(covered)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(quantiles, empirical)
    return iso


def calibrated_interval(mu, sigma, iso_calibrator, conf=0.90):
    target_q_lo, target_q_hi = (1 - conf) / 2, 1 - (1 - conf) / 2
    grid = np.linspace(0.001, 0.999, 400)
    mapped = iso_calibrator.predict(grid)
    q_lo = grid[np.argmin(np.abs(mapped - target_q_lo))]
    q_hi = grid[np.argmin(np.abs(mapped - target_q_hi))]
    z_lo, z_hi = norm.ppf(q_lo), norm.ppf(q_hi)
    return mu + z_lo * sigma, mu + z_hi * sigma


def evaluate_fold(model, val_loader, test_loader, cfg):
    """Unchanged per-fold metrics, PLUS the raw arrays needed for pooled
    OOF reporting are now returned alongside the summary dict."""
    print("\n  Evaluating fold...")

    y_val, mu_val, sigma_val, _, _ = get_predictions(model, val_loader, cfg)
    y_true, y_pred, sigma_test, aleatoric, epistemic = get_predictions(model, test_loader, cfg)

    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(np.clip(y_true, 1e-6, None), y_pred) * 100

    z = 1.645
    y_lo_raw = y_pred - z * sigma_test
    y_hi_raw = y_pred + z * sigma_test
    picp_raw = np.mean((y_true >= y_lo_raw) & (y_true <= y_hi_raw))
    pinw_raw = np.mean(y_hi_raw - y_lo_raw) / (y_true.max() - y_true.min() + 1e-8)

    results = {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
               "picp_raw": picp_raw, "pinw_raw": pinw_raw}

    y_lo_cal, y_hi_cal = None, None
    if cfg["calibrate"]:
        iso = fit_isotonic_calibrator(y_val, mu_val, sigma_val)
        y_lo_cal, y_hi_cal = calibrated_interval(y_pred, sigma_test, iso, conf=0.90)
        picp_cal = np.mean((y_true >= y_lo_cal) & (y_true <= y_hi_cal))
        pinw_cal = np.mean(y_hi_cal - y_lo_cal) / (y_true.max() - y_true.min() + 1e-8)
        results.update({"picp_calibrated": picp_cal, "pinw_calibrated": pinw_cal})

    results.update({"mean_aleatoric": float(np.nanmean(aleatoric)),
                     "mean_epistemic": float(np.nanmean(epistemic))})

    raw = {
        "y_true": y_true, "y_pred": y_pred,
        "y_lo_raw": y_lo_raw, "y_hi_raw": y_hi_raw,
        "y_lo_cal": y_lo_cal, "y_hi_cal": y_hi_cal,
    }
    return results, raw


def compute_pooled_metrics(all_raw, calibrate):
    """Pools every fold's test predictions (each row predicted by a model
    that never saw that row's cell) into ONE set of metrics computed over
    the full dataset, as opposed to averaging 5 independently-computed
    per-fold metrics. Reported alongside, not instead of, mean+-std."""
    y_true = np.concatenate([r["y_true"] for r in all_raw])
    y_pred = np.concatenate([r["y_pred"] for r in all_raw])
    y_lo_raw = np.concatenate([r["y_lo_raw"] for r in all_raw])
    y_hi_raw = np.concatenate([r["y_hi_raw"] for r in all_raw])

    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(np.clip(y_true, 1e-6, None), y_pred) * 100
    picp_raw = np.mean((y_true >= y_lo_raw) & (y_true <= y_hi_raw))
    pinw_raw = np.mean(y_hi_raw - y_lo_raw) / (y_true.max() - y_true.min() + 1e-8)

    pooled = {"n_rows": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
              "picp_raw": picp_raw, "pinw_raw": pinw_raw}

    if calibrate and all(r["y_lo_cal"] is not None for r in all_raw):
        y_lo_cal = np.concatenate([r["y_lo_cal"] for r in all_raw])
        y_hi_cal = np.concatenate([r["y_hi_cal"] for r in all_raw])
        picp_cal = np.mean((y_true >= y_lo_cal) & (y_true <= y_hi_cal))
        pinw_cal = np.mean(y_hi_cal - y_lo_cal) / (y_true.max() - y_true.min() + 1e-8)
        pooled.update({"picp_calibrated": picp_cal, "pinw_calibrated": pinw_cal})

    return pooled


# ─────────────────────────────────────────────────────────────────────────────
# 7. K-Fold Assignment - NOW PROTOCOL-STRATIFIED
# ─────────────────────────────────────────────────────────────────────────────

def assign_kfold_splits(df, n_folds=5, seed=42, val_frac=0.2, protocol_col="protocol"):
    """Assign K-Fold by cell, stratified by protocol group.

    Cells within EACH protocol group are shuffled and dealt round-robin
    across the n_folds test slots, instead of shuffling all cells globally
    once. This guarantees every fold's test set gets (as close to) a
    representative mix of protocols as the group sizes allow, rather than
    risking one fold's test set landing almost entirely in one temperature/
    C-rate bucket purely by chance.

    The same round-robin-per-protocol idea is then applied to split each
    fold's REMAINING (non-test) cells into train/val, so the isotonic
    calibrator (fit on val) also sees a representative protocol mix rather
    than whatever a plain 80/20 index slice happens to hand it.

    This does not change what the model can learn - every cell is still
    test exactly once across the 5 folds, same as before. It only reduces
    how much of the fold-to-fold variance reflects "unlucky protocol mix"
    rather than genuine generalization variance.
    """
    rng = np.random.default_rng(seed)
    cell_protocol = df.groupby("cell_id")[protocol_col].first()

    # round-robin deal each protocol's cells across the n_folds test slots
    test_slots = [[] for _ in range(n_folds)]
    for proto, group in cell_protocol.groupby(cell_protocol):
        cells = sorted(group.index.tolist())
        rng.shuffle(cells)
        for i, c in enumerate(cells):
            test_slots[i % n_folds].append(c)

    fold_assignments = {}
    for fold_idx in range(n_folds):
        test_cells = test_slots[fold_idx]
        remaining_cells = [c for c in cell_protocol.index if c not in test_cells]
        remaining_protocol = cell_protocol.loc[remaining_cells]

        # stratified train/val split of the remaining cells
        val_cells, train_cells = [], []
        for proto, group in remaining_protocol.groupby(remaining_protocol):
            cells = sorted(group.index.tolist())
            rng.shuffle(cells)
            n_val = max(1, int(round(val_frac * len(cells)))) if len(cells) > 1 else 0
            val_cells.extend(cells[:n_val])
            train_cells.extend(cells[n_val:])

        fold_assignments[fold_idx] = {
            "train": set(train_cells),
            "val": set(val_cells),
            "test": set(test_cells),
        }
        print(f"  Fold {fold_idx+1}: train={len(train_cells)}, val={len(val_cells)}, test={len(test_cells)}")

    # protocol coverage report - confirms the stratification actually worked
    print("\n  Protocol coverage per fold's test set (cell counts):")
    proto_list = sorted(cell_protocol.unique())
    header = "  " + f"{'protocol':<24}" + "".join(f"fold{f+1:<5}" for f in range(n_folds))
    print(header)
    for proto in proto_list:
        row = f"  {proto:<24}"
        for fold_idx in range(n_folds):
            n = sum(1 for c in fold_assignments[fold_idx]["test"] if cell_protocol[c] == proto)
            row += f"{n:<9}"
        print(row)

    return fold_assignments


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main K-Fold Training
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  K-FOLD TRAINING ON SNL NCM (9 Features) - v2 (stratified + pooled OOF)")
    print("=" * 60)
    print(f"  Folds: {CFG['n_folds']}")
    print(f"  Device: {DEVICE}")

    # ─── Load data ───
    print("\nLoading NCM data...")
    df = load_ncm_data(CFG["data_path"])
    print(f"  Features: {len(NCM_FEAT_COLS)} (including temperature and voltage_range)")

    # ─── Assign K-Fold splits (protocol-stratified) ───
    print("\nAssigning K-Fold splits (protocol-stratified)...")
    fold_assignments = assign_kfold_splits(df, n_folds=CFG["n_folds"], seed=CFG["kfold_seed"],
                                            val_frac=CFG["fold_val_frac"])

    # ─── Store results ───
    all_results = []
    all_raw = []

    for fold_idx in range(CFG["n_folds"]):
        print("\n" + "=" * 60)
        print(f"  FOLD {fold_idx+1}/{CFG['n_folds']}")
        print("=" * 60)

        train_cells = fold_assignments[fold_idx]["train"]
        val_cells = fold_assignments[fold_idx]["val"]
        test_cells = fold_assignments[fold_idx]["test"]

        def label_cell(cid):
            if cid in train_cells:
                return "train"
            if cid in val_cells:
                return "val"
            return "test"

        df_fold = df.copy()
        df_fold["split"] = df_fold["cell_id"].map(label_cell)

        # Normalize
        scaler = StandardScaler()
        scaler.fit(df_fold[df_fold.split == "train"][NCM_FEAT_COLS].values)
        df_fold[NCM_FEAT_COLS] = scaler.transform(df_fold[NCM_FEAT_COLS].values)

        # Create datasets
        W = CFG["window_size"]
        train_ds = NCMSequenceDataset(df_fold, W, CFG["soh_stride"], "train",
                                       weighted=True, tail_weight=CFG["tail_weight"])
        val_ds = NCMSequenceDataset(df_fold, W, CFG["soh_stride"], "val",
                                     weighted=True, tail_weight=CFG["tail_weight"])
        test_ds = NCMSequenceDataset(df_fold, W, CFG["soh_stride"], "test")

        val_loader = DataLoader(val_ds, batch_size=CFG["soh_batch"], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=CFG["soh_batch"], shuffle=False)

        print(f"\n  Train windows: {len(train_ds):,}")
        print(f"  Val windows:   {len(val_ds):,}")
        print(f"  Test windows:  {len(test_ds):,}")

        # Build model
        print("\n  Building BEM-SOH model...")
        model = BEM_SOH(CFG).to(DEVICE)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        history, best_loss = train_fold(model, train_ds, val_ds, CFG, fold_idx)

        # Evaluate
        results, raw = evaluate_fold(model, val_loader, test_loader, CFG)
        all_results.append(results)
        all_raw.append(raw)

        print(f"\n  Fold {fold_idx+1} Results:")
        print(f"    MAE: {results['mae']:.4f}%")
        print(f"    R\u00b2:  {results['r2']:.4f}")
        print(f"    PICP (cal): {results.get('picp_calibrated', results['picp_raw']):.4f}")

    # ─── Mean +/- std summary (as before) ───
    print("\n" + "=" * 60)
    print("  K-FOLD SUMMARY (Mean \u00b1 Std across folds)")
    print("=" * 60)

    metrics = ["mae", "rmse", "r2", "picp_raw", "pinw_raw"]
    if "picp_calibrated" in all_results[0]:
        metrics.extend(["picp_calibrated", "pinw_calibrated"])

    summary = {}
    for metric in metrics:
        values = [r[metric] for r in all_results if metric in r]
        if values:
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values
            }
            print(f"  {metric}: {np.mean(values):.4f} \u00b1 {np.std(values):.4f}")

    print("\n  Individual fold results:")
    for i, r in enumerate(all_results):
        print(f"    Fold {i+1}: MAE={r['mae']:.4f}%, R\u00b2={r['r2']:.4f}")

    # ─── NEW: pooled out-of-fold summary ───
    print("\n" + "=" * 60)
    print("  POOLED OUT-OF-FOLD SUMMARY (all folds concatenated)")
    print("=" * 60)
    pooled = compute_pooled_metrics(all_raw, CFG["calibrate"])
    print(f"  Rows pooled: {pooled['n_rows']:,} "
          f"(every row predicted by a model that never saw its cell)")
    for k, v in pooled.items():
        if k != "n_rows":
            print(f"  {k}: {v:.4f}")
    print("\n  NOTE: this is the same 5 fold-models' predictions as the mean+-std table "
          "above, aggregated differently (pooled residuals vs. mean-of-per-fold-metrics). "
          "The two won't be identical - report both.")

    # ─── Save results ───
    os.makedirs(CFG["save_dir"], exist_ok=True)
    save_path = os.path.join(CFG["save_dir"], "snl_ncm_kfold_results_v2.json")
    with open(save_path, "w") as f:
        json.dump({
            "summary_mean_std": summary,
            "pooled_oof": pooled,
            "individual_folds": all_results,
            "cfg": CFG,
        }, f, indent=2, default=str)

    print(f"\n  Results saved -> {save_path}")

    # ─── Final table ───
    print("\n" + "=" * 60)
    print("  FINAL RESULTS TABLE")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Mean':<12} {'Std':<12} {'Pooled OOF':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    for metric, data in summary.items():
        pooled_val = pooled.get(metric)
        pooled_str = f"{pooled_val:.4f}" if pooled_val is not None else "n/a"
        print(f"  {metric:<20} {data['mean']:<12.4f} \u00b1{data['std']:<11.4f} {pooled_str:<12}")

    print("\n" + "=" * 60)
    print("  K-FOLD TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()