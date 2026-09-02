# train_ncm_transfer.py
# Transfer learning experiment: MIT-pretrained BEM-SOH -> NCM chemistry.
#
# Solves the input-dimension mismatch (MIT=10 features, NCM=8) by mapping
# NCM data into the SAME 10-channel input the pretrained model expects:
#   - 6 features map directly (charge_capacity, charge_energy, CE_lag1/2,
#     cap_rel, energy_rel, cycle_pos), scaled with the ORIGINAL MIT scaler's
#     mean/std (not a fresh NCM scaler) - this is what makes pretrained
#     filters meaningfully applicable to the new data.
#   - ir_rel's slot is filled with voltage_range (a resistance-related
#     proxy), scaled with its own NCM statistics (no MIT equivalent exists).
#   - dc_internal_resistance and temperature_avg slots are zero-padded
#     (genuinely unavailable in NCM) - equivalent to "mean value" under a
#     standard scaler, but the model cannot use real information from these
#     channels on NCM data. State this explicitly as a limitation.
#
# Runs THREE conditions on the identical 10-channel padded input, so the
# comparison isolates the effect of the training/init regime, not the input
# representation:
#   1. from_scratch  - random init, train on NCM only            (baseline)
#   2. frozen        - load MIT weights, freeze CNN+encoder+pool,
#                       fine-tune ONLY the evidential head          (feature-extraction)
#   3. finetune       - load MIT weights, fine-tune ALL weights,
#                       reduced LR                                  (full fine-tuning)

import os, math, time, warnings, json, copy
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.isotonic import IsotonicRegression
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
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    mit_checkpoint_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\bem_soh_best.pt",
    ncm_data_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\generalization_test\generalization_results\ncm_processed.csv",
    save_dir = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\transfer",

    window_size = 50,
    soh_stride  = 2,
    tail_weight = 3.0,

    # from_scratch: LR/epochs matched to original MIT training regime
    scratch_epochs = 120, scratch_lr = 2e-4, scratch_patience = 25,
    # frozen feature-extraction: fewer epochs needed (only head trains), can afford higher LR
    frozen_epochs = 60, frozen_lr = 1e-3, frozen_patience = 15,
    # full fine-tune: LOWER LR than from-scratch (standard transfer-learning practice -
    # avoids catastrophically overwriting pretrained representations), fewer epochs
    finetune_epochs = 60, finetune_lr = 2e-5, finetune_patience = 15,

    soh_batch = 128,   # smaller than MIT (256) - NCM has far fewer sequences
    soh_wd = 1e-4,
    warmup_epochs = 5,

    nig_mse_warmup_epochs = 8,
    evid_lambda = 0.01, evid_lambda_max = 0.05, evid_mse_weight = 1.0,

    latency_batch_sizes = [1, 32], latency_reps = 50,
)

# The original MIT feature order - MUST match FEAT_COLS in train_soh_bem.py exactly
MIT_FEAT_COLS = [
    "dc_internal_resistance", "temperature_avg",
    "charge_capacity", "charge_energy",
    "coulombic_efficiency_lagged_1", "coulombic_efficiency_lagged_2",
    "cap_rel", "energy_rel", "ir_rel", "cycle_pos",
]

# Positions that are genuinely unavailable in NCM -> zero-padded
ZERO_PAD_POSITIONS = {0: "dc_internal_resistance", 1: "temperature_avg"}
# Position 8 (ir_rel) is substituted with NCM's voltage_range (proxy, own scaling)
PROXY_SUBSTITUTION = {8: ("ir_rel", "voltage_range")}
# Everything else maps 1:1 by name, scaled with the ORIGINAL MIT scaler stats


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model (identical to train_soh_bem.py - MUST match for weight loading)
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
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv - 1, groups=self.d_inner, bias=True)
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
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand, dropout)
                                      for _ in range(n_layers)])
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
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model), nn.GELU())

    def forward(self, x):
        z_f = self.fwd(x)
        z_b = self.bwd(torch.flip(x, dims=[1]))
        z_b = torch.flip(z_b, dims=[1])
        return self.fuse(torch.cat([z_f, z_b], dim=-1))


class EvidentialHead(nn.Module):
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 4),
        )

    def forward(self, z):
        out = self.net(z)
        gamma = torch.sigmoid(out[:, 0])
        nu = F.softplus(out[:, 1]).clamp(max=50.0) + 1e-6
        alpha = F.softplus(out[:, 2]).clamp(max=50.0) + 1.0 + 1e-6
        beta = F.softplus(out[:, 3]) + 1e-6
        return gamma, nu, alpha, beta


class BEM_SOH(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg
        self.cnn = MultiScaleCNN(C["input_dim"], C["cnn_channels"], C["cnn_kernels"], C["dropout"])
        cnn_out = sum(C["cnn_channels"])
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out, C["d_model"]), nn.LayerNorm(C["d_model"]), nn.GELU(), nn.Dropout(C["dropout"]))
        if C["bidirectional"]:
            self.encoder = BiMambaEncoder(C["d_model"], C["d_state"], C["d_conv"], C["expand"], C["n_mamba_layers"], C["dropout"])
        else:
            self.encoder = MambaEncoder(C["d_model"], C["d_state"], C["d_conv"], C["expand"], C["n_mamba_layers"], C["dropout"])
        self.attn_pool = nn.Linear(C["d_model"], 1)
        self.head = EvidentialHead(C["d_model"], C["dropout"])
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
# 3. Losses (identical to train_soh_bem.py)
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
    nll = 0.5 * torch.log(math.pi / nu) - alpha * torch.log(two_b_lambda) \
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_b_lambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll


def evidential_regularizer(gamma, nu, alpha, y):
    return torch.abs(y - gamma) * (2 * nu + alpha)


def evidential_loss(gamma, nu, alpha, beta, y, weight=None, lam=0.01, mse_weight=1.0):
    y_c = torch.clamp(y, 0.0, 1.0)
    gamma_c = torch.clamp(gamma, 0.0, 1.0)
    mse = (gamma_c - y_c) ** 2
    loss = nig_nll(gamma, nu, alpha, beta, y) + lam * evidential_regularizer(gamma, nu, alpha, y) + mse_weight * mse
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def forward_and_loss(model, x, y, w, cfg, epoch):
    (gamma, nu, alpha, beta), attn = model(x)
    if epoch < cfg["nig_mse_warmup_epochs"]:
        loss = mse_warmup_loss(gamma, y, weight=w)
    else:
        prog = min(1.0, (epoch - cfg["nig_mse_warmup_epochs"]) / max(1, cfg["_epochs"] - cfg["nig_mse_warmup_epochs"]))
        lam = cfg["evid_lambda"] + prog * (cfg["evid_lambda_max"] - cfg["evid_lambda"])
        loss = evidential_loss(gamma, nu, alpha, beta, y, weight=w, lam=lam, mse_weight=cfg["evid_mse_weight"])
    return loss, gamma


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training utilities
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
    """NOTE: tracks val_mae directly (not composite val_loss) - this is the
    fix flagged in earlier discussion of the MIT/NCM runs, where val_loss
    and val_mae could diverge under the evidential objective."""
    def __init__(self, patience=20, delta=1e-5):
        self.patience, self.delta = patience, delta
        self.counter, self.best, self.stop, self.best_state = 0, None, False, None

    def __call__(self, val_mae, model):
        if self.best is None or val_mae < self.best - self.delta:
            self.best = val_mae
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def run_training(model, train_ds, val_ds, cfg, lr, epochs, patience, freeze_backbone=False):
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Frozen mode: training {sum(p.numel() for p in trainable):,} / "
              f"{sum(p.numel() for p in model.parameters()):,} params (head only)")
    else:
        for p in model.parameters():
            p.requires_grad = True

    cfg = dict(cfg)
    cfg["_epochs"] = epochs
    batch = cfg["soh_batch"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=lr, weight_decay=cfg["soh_wd"])
    es = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_r2": []}

    for epoch in range(epochs):
        cosine_lr(opt, epoch, min(cfg["warmup_epochs"], epochs // 4 or 1), epochs, lr)

        model.train()
        if freeze_backbone:
            model.cnn.eval(); model.cnn_proj.eval(); model.encoder.eval()  # keep BN/dropout frozen-mode stable
        train_loss = 0.0
        for x, y, w in train_loader:
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            opt.zero_grad()
            loss, _ = forward_and_loss(model, x, y, w, cfg, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.5f} | "
                  f"Val: {val_loss:.5f} | MAE: {mae:.4f}% | R2: {r2:.4f}")

        es(mae, model)  # tracks val_mae, not val_loss
        if es.stop:
            print(f"    Early stopping at epoch {epoch+1} (best val_mae={es.best:.4f}%)")
            break

    es.restore(model)
    return history, es.best


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build the 10-channel padded NCM feature table
# ─────────────────────────────────────────────────────────────────────────────

def build_padded_ncm_table(ncm_df, mit_scaler_mean, mit_scaler_std):
    """Returns ncm_df with 10 new columns 'padded_0'...'padded_9' matching
    MIT_FEAT_COLS positions, ready to feed the pretrained-shape model."""
    df = ncm_df.copy()
    mean = np.asarray(mit_scaler_mean, dtype=np.float64)
    std = np.asarray(mit_scaler_std, dtype=np.float64)

    # fit a small local scaler for the proxy feature (voltage_range) - no MIT
    # equivalent exists, so we standardize it against NCM's own train split
    from sklearn.preprocessing import StandardScaler
    proxy_col = PROXY_SUBSTITUTION[8][1]
    proxy_scaler = StandardScaler()
    proxy_scaler.fit(df[df.split == "train"][[proxy_col]].values)
    proxy_scaled = proxy_scaler.transform(df[[proxy_col]].values).flatten()

    for pos, mit_name in enumerate(MIT_FEAT_COLS):
        col = f"padded_{pos}"
        if pos in ZERO_PAD_POSITIONS:
            df[col] = 0.0
        elif pos in PROXY_SUBSTITUTION:
            df[col] = proxy_scaled
        else:
            if mit_name not in df.columns:
                raise KeyError(
                    f"Expected NCM column '{mit_name}' (mapped from MIT position "
                    f"{pos}) not found. Check NCM_FEAT_COLS / preprocessing output."
                )
            df[col] = (df[mit_name].values - mean[pos]) / (std[pos] + 1e-9)

    return df, proxy_scaler


PADDED_COLS = [f"padded_{i}" for i in range(10)]


class PaddedSequenceDataset(Dataset):
    def __init__(self, df, window_size, stride=1, split=None,
                 weighted=False, tail_thr=0.90, tail_weight=1.0):
        self.samples, self.weights, self.cell_ids = [], [], []
        subset = df if split is None else df[df.split == split]

        for cid, cell_df in subset.groupby("cell_id"):
            cell_df = cell_df.sort_values("cycle_index").reset_index(drop=True)
            X = cell_df[PADDED_COLS].values.astype(np.float32)
            y = cell_df["soh"].values.astype(np.float32)
            for end in range(window_size, len(X) + 1, stride):
                start = end - window_size
                y_last = y[end - 1]
                self.samples.append((X[start:end], y_last))
                self.cell_ids.append(cid)
                w = tail_weight if (weighted and y_last < tail_thr) else 1.0
                self.weights.append(w)
        self.weights = np.array(self.weights, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(self.weights[idx])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation (identical structure to train_soh_bem.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_y, all_mu, all_sigma, all_al, all_ep = [], [], [], [], []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        (gamma, nu, alpha, beta), _ = model(x)
        aleatoric = (beta / (alpha - 1)).cpu().numpy()
        epistemic = (beta / (nu * (alpha - 1))).cpu().numpy()
        sigma = np.sqrt(aleatoric + epistemic)
        all_mu.extend(gamma.cpu().numpy()); all_sigma.extend(sigma)
        all_al.extend(aleatoric); all_ep.extend(epistemic); all_y.extend(y.numpy())
    return (np.array(all_y), np.array(all_mu), np.array(all_sigma),
            np.array(all_al), np.array(all_ep))


def fit_isotonic_calibrator(y_true, mu, sigma, n_q=20):
    from scipy.stats import norm
    quantiles = np.linspace(0.05, 0.95, n_q)
    empirical = [(y_true <= mu + norm.ppf(q) * sigma).mean() for q in quantiles]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(quantiles, empirical)
    return iso


def calibrated_interval(mu, sigma, iso, conf=0.90):
    from scipy.stats import norm
    q_lo_t, q_hi_t = (1 - conf) / 2, 1 - (1 - conf) / 2
    grid = np.linspace(0.001, 0.999, 400)
    mapped = iso.predict(grid)
    q_lo = grid[np.argmin(np.abs(mapped - q_lo_t))]
    q_hi = grid[np.argmin(np.abs(mapped - q_hi_t))]
    return mu + norm.ppf(q_lo) * sigma, mu + norm.ppf(q_hi) * sigma


def evaluate(model, val_loader, test_loader, tag):
    y_val, mu_val, sigma_val, _, _ = get_predictions(model, val_loader)
    y_true, y_pred, sigma_test, aleatoric, epistemic = get_predictions(model, test_loader)

    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 100
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(np.clip(y_true, 1e-6, None), y_pred) * 100

    z = 1.645
    lo_raw, hi_raw = y_pred - z * sigma_test, y_pred + z * sigma_test
    picp_raw = np.mean((y_true >= lo_raw) & (y_true <= hi_raw))
    pinw_raw = np.mean(hi_raw - lo_raw) / (y_true.max() - y_true.min() + 1e-8)

    iso = fit_isotonic_calibrator(y_val, mu_val, sigma_val)
    lo_cal, hi_cal = calibrated_interval(y_pred, sigma_test, iso)
    picp_cal = np.mean((y_true >= lo_cal) & (y_true <= hi_cal))
    pinw_cal = np.mean(hi_cal - lo_cal) / (y_true.max() - y_true.min() + 1e-8)

    print(f"\n  [{tag}]  MAE={mae:.4f}%  RMSE={rmse:.4f}%  MAPE={mape:.4f}%  R2={r2:.5f}")
    print(f"  [{tag}]  PICP raw/cal = {picp_raw:.4f}/{picp_cal:.4f}   PINW raw/cal = {pinw_raw:.4f}/{pinw_cal:.4f}")

    return {"tag": tag, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
            "picp_raw": picp_raw, "pinw_raw": pinw_raw,
            "picp_calibrated": picp_cal, "pinw_calibrated": pinw_cal,
            "mean_aleatoric": float(np.nanmean(aleatoric)),
            "mean_epistemic": float(np.nanmean(epistemic))}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main - runs all three conditions on identical padded input
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading pretrained MIT checkpoint...")
    ckpt = torch.load(CFG["mit_checkpoint_path"], map_location=DEVICE)
    
    # Debug: inspect what's in the checkpoint
    print(f"Checkpoint keys: {ckpt.keys()}")
    
    # Get the config from checkpoint
    mit_cfg = ckpt["cfg"]
    print(f"Config type: {type(mit_cfg)}")
    
    # Handle both dict and other types
    if isinstance(mit_cfg, dict):
        model_cfg = mit_cfg.copy()
    else:
        # If it's not a dict, try to convert it
        try:
            model_cfg = dict(mit_cfg)
        except:
            # If it's a custom object with __dict__
            model_cfg = {k: v for k, v in mit_cfg.__dict__.items() if not k.startswith('_')}
    
    # Ensure input_dim is set correctly
    model_cfg["input_dim"] = 10
    
    # Verify required keys exist and set defaults if missing
    required_keys = ["cnn_channels", "cnn_kernels", "d_model", "d_state", 
                     "d_conv", "expand", "n_mamba_layers", "dropout", "bidirectional"]
    for key in required_keys:
        if key not in model_cfg:
            print(f"Warning: '{key}' not found in checkpoint config. Using default from original training.")
            # Set defaults from the original training script
            if key == "cnn_channels": model_cfg[key] = [32, 64, 128]
            elif key == "cnn_kernels": model_cfg[key] = [3, 7, 15]
            elif key == "d_model": model_cfg[key] = 128
            elif key == "d_state": model_cfg[key] = 16
            elif key == "d_conv": model_cfg[key] = 4
            elif key == "expand": model_cfg[key] = 2
            elif key == "n_mamba_layers": model_cfg[key] = 3
            elif key == "dropout": model_cfg[key] = 0.15
            elif key == "bidirectional": model_cfg[key] = True
    
    # Check feature columns match
    assert ckpt["feat_cols"] == MIT_FEAT_COLS, \
        f"Checkpoint feat_cols don't match MIT_FEAT_COLS.\nCheckpoint: {ckpt['feat_cols']}\nExpected: {MIT_FEAT_COLS}"

    print("Loading + padding NCM data to 10-channel MIT-compatible input...")
    ncm_raw = pd.read_csv(CFG["ncm_data_path"])
    ncm_df, proxy_scaler = build_padded_ncm_table(ncm_raw, ckpt["scaler_mean"], ckpt["scaler_std"])
    print(f"  NCM: {ncm_df.shape[0]:,} rows, {ncm_df['cell_id'].nunique()} cells")

    W = CFG["window_size"]
    train_ds = PaddedSequenceDataset(ncm_df, W, CFG["soh_stride"], "train", weighted=True, tail_weight=CFG["tail_weight"])
    val_ds   = PaddedSequenceDataset(ncm_df, W, CFG["soh_stride"], "val", weighted=True, tail_weight=CFG["tail_weight"])
    test_ds  = PaddedSequenceDataset(ncm_df, W, CFG["soh_stride"], "test")
    val_loader = DataLoader(val_ds, batch_size=CFG["soh_batch"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CFG["soh_batch"], shuffle=False)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    os.makedirs(CFG["save_dir"], exist_ok=True)
    all_results = []

    # ── Condition 1: from-scratch, padded input (controlled baseline) ──
    print("\n" + "=" * 60 + "\n  CONDITION 1: FROM-SCRATCH (padded input)\n" + "=" * 60)
    m_scratch = BEM_SOH(model_cfg).to(DEVICE)
    run_training(m_scratch, train_ds, val_ds, CFG, CFG["scratch_lr"], CFG["scratch_epochs"], CFG["scratch_patience"])
    res = evaluate(m_scratch, val_loader, test_loader, "from_scratch")
    torch.save(m_scratch.state_dict(), os.path.join(CFG["save_dir"], "ncm_from_scratch.pt"))
    all_results.append(res)

    # ── Condition 2: frozen feature-extraction ──
    print("\n" + "=" * 60 + "\n  CONDITION 2: FROZEN FEATURE-EXTRACTION\n" + "=" * 60)
    m_frozen = BEM_SOH(model_cfg).to(DEVICE)
    m_frozen.load_state_dict(ckpt["model_state_dict"])
    run_training(m_frozen, train_ds, val_ds, CFG, CFG["frozen_lr"], CFG["frozen_epochs"],
                 CFG["frozen_patience"], freeze_backbone=True)
    res = evaluate(m_frozen, val_loader, test_loader, "frozen_feature_extraction")
    torch.save(m_frozen.state_dict(), os.path.join(CFG["save_dir"], "ncm_frozen.pt"))
    all_results.append(res)

    # ── Condition 3: full fine-tuning ──
    print("\n" + "=" * 60 + "\n  CONDITION 3: FULL FINE-TUNING\n" + "=" * 60)
    m_finetune = BEM_SOH(model_cfg).to(DEVICE)
    m_finetune.load_state_dict(ckpt["model_state_dict"])
    run_training(m_finetune, train_ds, val_ds, CFG, CFG["finetune_lr"], CFG["finetune_epochs"], CFG["finetune_patience"])
    res = evaluate(m_finetune, val_loader, test_loader, "full_finetune")
    torch.save(m_finetune.state_dict(), os.path.join(CFG["save_dir"], "ncm_finetuned.pt"))
    all_results.append(res)

    print("\n" + "=" * 60 + "\n  SUMMARY: TRANSFER LEARNING COMPARISON\n" + "=" * 60)
    summary_df = pd.DataFrame(all_results).set_index("tag")
    print(summary_df.to_string())
    summary_df.to_csv(os.path.join(CFG["save_dir"], "transfer_comparison.csv"))
    print(f"\n  Saved -> {os.path.join(CFG['save_dir'], 'transfer_comparison.csv')}")