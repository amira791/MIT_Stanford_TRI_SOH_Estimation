# config_C.py
# Configuration C: Causal + Evidential
# 
# Purpose: Tests the effect of evidential UQ alone
# Compares to Config A to isolate the evidential contribution
#
# Architecture:
#   - CNN Backbone: Multi-scale (3 branches: 32, 64, 128 channels)
#   - Pooling: Attention pooling (learned weights)
#   - Encoder: Causal (forward-only) Mamba x3 layers
#   - UQ Head: Evidential (NIG: gamma, nu, alpha, beta)
#   - Calibration: None
#
# What it isolates:
#   - Effect of evidential regression (vs Config A: Gaussian-NLL)
#   - Same encoder (Causal), same CNN, same pooling

import os, math, time, warnings, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Config
# ─────────────────────────────────────────────────────────────────────────────

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

CFG = dict(
    # Paths
    soh_path  = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv",
    save_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\config_C_causal_evidential.pt",

    # Features
    input_dim  = 10,
    window_size = 50,
    soh_stride  = 2,

    # Model
    cnn_channels = [32, 64, 128],
    cnn_kernels  = [3, 7, 15],
    d_model      = 128,
    d_state      = 16,
    d_conv       = 4,
    expand       = 2,
    n_mamba_layers = 3,
    dropout      = 0.15,

    # ─── ARCHITECTURE SWITCHES ───
    # CONFIG C: Causal + Evidential
    bidirectional = False,   # ← CAUSAL (forward-only)
    evidential    = True,    # ← EVIDENTIAL (NIG head)
    calibrate     = False,   # ← NO calibration

    # Training
    soh_epochs   = 120,
    soh_lr       = 2e-4,
    soh_batch    = 256,
    soh_wd       = 1e-4,
    soh_patience = 25,
    tail_weight  = 3.0,
    warmup_epochs = 10,

    # Evidential-specific
    nig_mse_warmup_epochs = 10,
    evid_lambda   = 0.01,
    evid_lambda_max = 0.05,
    evid_mse_weight = 1.0,

    # Deployment-metric measurement
    latency_batch_sizes = [1, 32, 256],
    latency_reps = 100,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-cell relative features."""
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
    soh = pd.read_csv(soh_path)
    soh = add_relative_features(soh)

    scaler = StandardScaler()
    scaler.fit(soh[soh.split == "train"][FEAT_COLS].values)
    soh[FEAT_COLS] = scaler.transform(soh[FEAT_COLS].values)

    return soh, scaler


class SequenceDataset(Dataset):
    """Sliding-window dataset for SOH."""
    def __init__(self, df, window_size, stride=1, split=None,
                 weighted=False, tail_thr=0.90, tail_weight=1.0):
        self.samples = []
        self.weights = []
        self.cell_ids = []
        subset = df if split is None else df[df.split == split]

        for cid, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[FEAT_COLS].values.astype(np.float32)
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
# 3.  Model (SAME as original, just with config switches)
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


class GaussianHead(nn.Module):
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # [mean, log_var]
        )

    def forward(self, z):
        out = self.net(z)
        mu = torch.sigmoid(out[:, 0])
        log_var = out[:, 1].clamp(-10, 5)
        return mu, log_var


class EvidentialHead(nn.Module):
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),  # gamma, log_nu, log_alpha, log_beta
        )

    def forward(self, z):
        out = self.net(z)
        gamma = torch.sigmoid(out[:, 0])
        nu    = F.softplus(out[:, 1]).clamp(max=50.0) + 1e-6
        alpha = F.softplus(out[:, 2]).clamp(max=50.0) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        return gamma, nu, alpha, beta


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
# 4.  Losses
# ─────────────────────────────────────────────────────────────────────────────

def mse_warmup_loss(pred_mean, target, weight=None):
    pred_mean = torch.clamp(pred_mean, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    loss = (pred_mean - target) ** 2
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def gaussian_nll_loss(mu, log_var, target, weight=None):
    mu = torch.clamp(mu, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    var = torch.exp(log_var)
    nll = 0.5 * log_var + 0.5 * (target - mu) ** 2 / var
    if weight is not None:
        nll = nll * weight
    return nll.mean()


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


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training
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


def forward_and_loss(model, x, y, w, cfg, epoch):
    (out, attn) = model(x)
    evidential = cfg["evidential"]

    if evidential:
        gamma, nu, alpha, beta = out
        if epoch < cfg["nig_mse_warmup_epochs"]:
            loss = mse_warmup_loss(gamma, y, weight=w)
        else:
            prog = min(1.0, (epoch - cfg["nig_mse_warmup_epochs"]) /
                       max(1, cfg["soh_epochs"] - cfg["nig_mse_warmup_epochs"]))
            lam = cfg["evid_lambda"] + prog * (cfg["evid_lambda_max"] - cfg["evid_lambda"])
            loss = evidential_loss(gamma, nu, alpha, beta, y, weight=w, lam=lam,
                                    mse_weight=cfg["evid_mse_weight"])
        point_pred = gamma
    else:
        mu, log_var = out
        if epoch < cfg["nig_mse_warmup_epochs"]:
            loss = mse_warmup_loss(mu, y, weight=w)
        else:
            loss = gaussian_nll_loss(mu, log_var, y, weight=w)
        point_pred = mu

    return loss, point_pred


def train_soh(model, train_ds, val_ds, cfg):
    print("\n" + "=" * 60)
    print(f"  TRAINING CONFIG C: Causal + Evidential")
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

        # Log evidence diagnostic for evidential models
        if (epoch + 1) % 10 == 0 or epoch == 0:
            evid_note = ""
            if cfg["evidential"]:
                with torch.no_grad():
                    xb, yb, wb = next(iter(val_loader))
                    (g, nu_b, a_b, _), _ = model(xb.to(DEVICE))
                    mean_evidence = (2 * nu_b + a_b).mean().item()
                evid_note = f" | mean_evidence: {mean_evidence:.2f}"
            print(f"  Epoch {epoch + 1:3d}/{cfg['soh_epochs']} | "
                  f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                  f"MAE: {mae:.4f}% | R2: {r2:.4f}{evid_note}")

        es(val_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    es.restore(model)
    print(f"\n  Best val loss: {es.best:.6f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, cfg):
    model.eval()
    all_y, all_mu, all_sigma = [], [], []
    all_aleatoric, all_epistemic = [], []

    for x, y, _ in loader:
        x = x.to(DEVICE)
        out, _ = model(x)
        if cfg["evidential"]:
            gamma, nu, alpha, beta = out
            aleatoric = (beta / (alpha - 1)).cpu().numpy()
            epistemic = (beta / (nu * (alpha - 1))).cpu().numpy()
            sigma = np.sqrt(aleatoric + epistemic)
            mu = gamma.cpu().numpy()
            all_aleatoric.extend(aleatoric)
            all_epistemic.extend(epistemic)
        else:
            mu, log_var = out
            sigma = torch.exp(0.5 * log_var).cpu().numpy()
            mu = mu.cpu().numpy()
            all_aleatoric.extend([np.nan] * len(mu))
            all_epistemic.extend([np.nan] * len(mu))

        all_mu.extend(mu)
        all_sigma.extend(sigma)
        all_y.extend(y.numpy())

    return (np.array(all_y), np.array(all_mu), np.array(all_sigma),
            np.array(all_aleatoric), np.array(all_epistemic))


def fit_isotonic_calibrator(y_true, mu, sigma, n_q=20):
    from scipy.stats import norm
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
    from scipy.stats import norm
    target_q_lo, target_q_hi = (1 - conf) / 2, 1 - (1 - conf) / 2
    grid = np.linspace(0.001, 0.999, 400)
    mapped = iso_calibrator.predict(grid)
    q_lo = grid[np.argmin(np.abs(mapped - target_q_lo))]
    q_hi = grid[np.argmin(np.abs(mapped - target_q_hi))]
    z_lo, z_hi = norm.ppf(q_lo), norm.ppf(q_hi)
    return mu + z_lo * sigma, mu + z_hi * sigma


def evaluate_soh(model, val_loader, test_loader, cfg):
    print("\n" + "=" * 60)
    print("  SOH EVALUATION - TEST SET (Config C: Causal + Evidential)")
    print("=" * 60)

    y_val, mu_val, sigma_val, _, _ = get_predictions(model, val_loader, cfg)
    y_true, y_pred, sigma_test, aleatoric, epistemic = get_predictions(model, test_loader, cfg)

    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(np.clip(y_true, 1e-6, None), y_pred) * 100

    z = 1.645  # nominal 90% Gaussian z
    y_lo_raw = y_pred - z * sigma_test
    y_hi_raw = y_pred + z * sigma_test
    picp_raw = np.mean((y_true >= y_lo_raw) & (y_true <= y_hi_raw))
    pinw_raw = np.mean(y_hi_raw - y_lo_raw) / (y_true.max() - y_true.min() + 1e-8)

    print(f"\n  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  MAPE : {mape:.4f}%")
    print(f"  R2   : {r2:.5f}")
    print(f"\n  -- UNCALIBRATED intervals (nominal 90% Gaussian) --")
    print(f"  PICP : {picp_raw:.4f}")
    print(f"  PINW : {pinw_raw:.4f}")

    results = {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
               "picp_raw": picp_raw, "pinw_raw": pinw_raw}

    if cfg["calibrate"]:
        iso = fit_isotonic_calibrator(y_val, mu_val, sigma_val)
        y_lo_cal, y_hi_cal = calibrated_interval(y_pred, sigma_test, iso, conf=0.90)
        picp_cal = np.mean((y_true >= y_lo_cal) & (y_true <= y_hi_cal))
        pinw_cal = np.mean(y_hi_cal - y_lo_cal) / (y_true.max() - y_true.min() + 1e-8)
        print(f"\n  -- CALIBRATED intervals (isotonic, fit on val) --")
        print(f"  PICP : {picp_cal:.4f}  (target ~0.90)")
        print(f"  PINW : {pinw_cal:.4f}")
        results.update({"picp_calibrated": picp_cal, "pinw_calibrated": pinw_cal})

    if cfg["evidential"]:
        print(f"\n  -- Uncertainty decomposition (mean over test set) --")
        print(f"  Mean aleatoric var  : {np.nanmean(aleatoric):.6f}")
        print(f"  Mean epistemic var  : {np.nanmean(epistemic):.6f}")
        results.update({"mean_aleatoric": float(np.nanmean(aleatoric)),
                         "mean_epistemic": float(np.nanmean(epistemic))})

    print(f"\n  -- MAE by SOH region --")
    for label, mask in [
        ("SOH < 0.90", y_true < 0.90),
        ("0.90-0.95", (y_true >= 0.90) & (y_true < 0.95)),
        ("SOH > 0.95", y_true >= 0.95)
    ]:
        if mask.sum() > 0:
            rm = mean_absolute_error(y_true[mask], y_pred[mask]) * 100
            print(f"  {label}: MAE = {rm:.4f}%  (n={mask.sum()})")

    return results, (y_true, y_pred, sigma_test, aleatoric, epistemic)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Deployment Metrics
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model):
    tmp_path = "_tmp_size_check.pt"
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    os.remove(tmp_path)
    return size_mb


def analytical_complexity(cfg):
    D = cfg["d_model"]
    d_inner = int(cfg["expand"] * D)
    N = cfg["d_state"]
    L = cfg["window_size"]
    n_layers = cfg["n_mamba_layers"]
    dirs = 2 if cfg["bidirectional"] else 1

    ssm_flops_per_layer = L * d_inner * N * 4
    ssm_total = ssm_flops_per_layer * n_layers * dirs

    cnn_flops = 0
    for ch, k in zip(cfg["cnn_channels"], cfg["cnn_kernels"]):
        cnn_flops += 2 * L * ch * ch * k

    return {
        "ssm_recurrence_flops_estimate": int(ssm_total),
        "cnn_flops_estimate": int(cnn_flops),
        "sequence_length": L,
        "mamba_time_complexity": "O(L * d_inner * d_state) -- linear in L",
        "attention_alternative_would_be": "O(L^2 * d_model) -- quadratic in L",
        "directions": dirs,
        "mamba_layers": n_layers,
    }


@torch.no_grad()
def measure_inference_latency(model, cfg, warmup=10):
    model.eval()
    results = {}
    L, D_in = cfg["window_size"], cfg["input_dim"]

    for bs in cfg["latency_batch_sizes"]:
        dummy = torch.randn(bs, L, D_in, device=DEVICE)

        for _ in range(warmup):
            _ = model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(cfg["latency_reps"]):
            _ = model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_time = t1 - t0
        per_batch_ms = (total_time / cfg["latency_reps"]) * 1000
        per_sample_ms = per_batch_ms / bs
        throughput = bs * cfg["latency_reps"] / total_time

        results[f"batch_{bs}"] = {
            "ms_per_batch": round(per_batch_ms, 4),
            "ms_per_sample": round(per_sample_ms, 4),
            "throughput_samples_per_sec": round(throughput, 1),
        }
    return results


def print_deployment_report(model, cfg):
    print("\n" + "=" * 60)
    print("  MODEL COMPLEXITY & DEPLOYMENT METRICS (Config C)")
    print("=" * 60)

    total, trainable = count_parameters(model)
    size_mb = model_size_mb(model)
    complexity = analytical_complexity(cfg)
    latency = measure_inference_latency(model, cfg)

    print(f"\n  Parameters       : {total:,} total | {trainable:,} trainable")
    print(f"  Model size (fp32): {size_mb:.3f} MB")
    print(f"  Input shape      : ({cfg['window_size']}, {cfg['input_dim']})")
    print(f"  Encoder          : {'Bidirectional' if cfg['bidirectional'] else 'Causal'} "
          f"Mamba x{cfg['n_mamba_layers']} layers")
    print(f"  UQ head          : {'Evidential (NIG)' if cfg['evidential'] else 'Gaussian-NLL'}")

    print(f"\n  -- Analytical complexity --")
    for k, v in complexity.items():
        print(f"  {k:35s}: {v}")

    print(f"\n  -- Measured inference latency ({DEVICE}) --")
    for bs, stats in latency.items():
        print(f"  {bs:10s} -> {stats['ms_per_sample']:.4f} ms/sample | "
              f"{stats['throughput_samples_per_sec']:.1f} samples/sec")

    return {
        "total_params": total,
        "trainable_params": trainable,
        "model_size_mb": size_mb,
        "complexity": complexity,
        "latency": latency,
        "device": str(DEVICE),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CONFIGURATION C: Causal + Evidential")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    print("\nLoading SOH data...")
    soh_df, scaler = load_soh_data(CFG["soh_path"])
    print(f"  SOH: {soh_df.shape}")
    print(f"  Cells: {soh_df['barcode'].nunique()}")

    W = CFG["window_size"]
    train_ds = SequenceDataset(soh_df, W, CFG["soh_stride"], "train",
                                weighted=True, tail_weight=CFG["tail_weight"])
    val_ds = SequenceDataset(soh_df, W, CFG["soh_stride"], "val",
                              weighted=True, tail_weight=CFG["tail_weight"])
    test_ds = SequenceDataset(soh_df, W, CFG["soh_stride"], "test")

    cells_train = set(train_ds.cell_ids)
    cells_val = set(val_ds.cell_ids)
    cells_test = set(test_ds.cell_ids)
    overlap = (cells_train & cells_val) | (cells_train & cells_test) | (cells_val & cells_test)
    print(f"\n  Cell-level split check -> overlapping cells across splits: {len(overlap)} "
          f"({'OK' if len(overlap) == 0 else 'LEAKAGE DETECTED -- FIX SPLIT'})")

    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences:   {len(val_ds):,}")
    print(f"  Test sequences:  {len(test_ds):,}")

    val_loader = DataLoader(val_ds, batch_size=CFG["soh_batch"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CFG["soh_batch"], shuffle=False)

    print("\nBuilding BEM-SOH model (Config C: Causal + Evidential)...")
    model = BEM_SOH(CFG).to(DEVICE)
    total, trainable = count_parameters(model)
    print(f"  Parameters: {total:,} total | {trainable:,} trainable")

    history = train_soh(model, train_ds, val_ds, CFG)

    results, raw_preds = evaluate_soh(model, val_loader, test_loader, CFG)
    deploy = print_deployment_report(model, CFG)

    os.makedirs(os.path.dirname(CFG["save_path"]), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": CFG,
        "feat_cols": FEAT_COLS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "results": results,
        "deploy": deploy,
        "history": history,
    }, CFG["save_path"])
    print(f"\n  Model saved -> {CFG['save_path']}")

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY (Config C)")
    print("=" * 60)
    print(json.dumps({**results, **{"total_params": deploy["total_params"],
                                     "model_size_mb": deploy["model_size_mb"]}},
                      indent=2, default=str))
    print("=" * 60)