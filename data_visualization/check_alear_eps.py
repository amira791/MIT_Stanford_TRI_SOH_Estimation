# diagnose_uncertainty.py
# Full diagnostic to determine whether the uncertainty decomposition issue
# is in the MODEL or in the FIGURE.

import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\bem_soh_best.pt"
SOH_DATA_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\data_preprocessing\final_dataset\soh\soh_full.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
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
# Model definitions
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
            nn.Linear(64, 2),
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
            nn.Linear(64, 4),
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
# Prediction function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions_with_attention(model, loader, cfg):
    """Returns predictions, uncertainties, and attention weights."""
    model.eval()
    all_y, all_mu, all_sigma = [], [], []
    all_aleatoric, all_epistemic = [], []
    all_nu, all_alpha, all_beta = [], [], []
    all_attn = []

    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(DEVICE)
        out, attn = model(x)
        gamma, nu, alpha, beta = out
        aleatoric = (beta / (alpha - 1)).cpu().numpy()
        epistemic = (beta / (nu * (alpha - 1))).cpu().numpy()
        sigma = np.sqrt(aleatoric + epistemic)
        mu = gamma.cpu().numpy()
        all_aleatoric.extend(aleatoric)
        all_epistemic.extend(epistemic)
        all_nu.extend(nu.cpu().numpy())
        all_alpha.extend(alpha.cpu().numpy())
        all_beta.extend(beta.cpu().numpy())
        all_mu.extend(mu)
        all_sigma.extend(sigma)
        all_y.extend(y.numpy())
        all_attn.append(attn.cpu().numpy())

    attn_all = np.concatenate(all_attn, axis=0) if all_attn else None
    return (np.array(all_y), np.array(all_mu), np.array(all_sigma),
            np.array(all_aleatoric), np.array(all_epistemic),
            np.array(all_nu), np.array(all_alpha), np.array(all_beta),
            attn_all)


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  UNCERTAINTY DECOMPOSITION DIAGNOSTIC")
    print("=" * 60)

    # ─── Load checkpoint ───
    print("\nLoading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["cfg"]

    # ─── Load data ───
    print("Loading data...")
    soh_df, scaler = load_soh_data(SOH_DATA_PATH)
    W = cfg["window_size"]

    test_ds = SequenceDataset(soh_df, W, cfg["soh_stride"], "test")
    test_loader = DataLoader(test_ds, batch_size=cfg["soh_batch"], shuffle=False)

    # ─── Build model ───
    print("Building model...")
    model = BEM_SOH(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ─── Get predictions ───
    print("Getting predictions...")
    (y_true, y_pred, sigma_test, aleatoric, epistemic,
     nu_vals, alpha_vals, beta_vals, attn) = get_predictions_with_attention(
        model, test_loader, cfg
    )

    # ─── 1. Check raw distributions ───
    print("\n" + "=" * 60)
    print("  1. RAW PARAMETER DISTRIBUTIONS")
    print("=" * 60)

    for name, arr in [("nu", nu_vals), ("alpha", alpha_vals), ("beta", beta_vals)]:
        print(f"\n  {name}:")
        print(f"    min:    {arr.min():.6e}")
        print(f"    max:    {arr.max():.6e}")
        print(f"    mean:   {arr.mean():.6e}")
        print(f"    std:    {arr.std():.6e}")
        print(f"    median: {np.median(arr):.6e}")

    # ─── 2. Check uncertainty distributions ───
    print("\n" + "=" * 60)
    print("  2. UNCERTAINTY DISTRIBUTIONS")
    print("=" * 60)

    for name, arr in [("Aleatoric", aleatoric), ("Epistemic", epistemic)]:
        print(f"\n  {name}:")
        print(f"    min:    {arr.min():.6e}")
        print(f"    max:    {arr.max():.6e}")
        print(f"    mean:   {arr.mean():.6e}")
        print(f"    std:    {arr.std():.6e}")
        print(f"    median: {np.median(arr):.6e}")

    # ─── 3. Correlation ───
    print("\n" + "=" * 60)
    print("  3. CORRELATION BETWEEN ALEATORIC AND EPISTEMIC")
    print("=" * 60)

    corr_pearson = np.corrcoef(aleatoric, epistemic)[0, 1]
    corr_spearman = np.corrcoef(np.argsort(np.argsort(aleatoric)),
                                 np.argsort(np.argsort(epistemic)))[0, 1]
    print(f"  Pearson correlation:  {corr_pearson:.6f}")
    print(f"  Spearman correlation: {corr_spearman:.6f}")

    # ─── 4. Ratio aleatoric/epistemic = nu ───
    print("\n" + "=" * 60)
    print("  4. RATIO ALEATORIC/EPISTEMIC (should equal nu)")
    print("=" * 60)

    ratio = aleatoric / (epistemic + 1e-12)
    print(f"  Ratio:")
    print(f"    min:    {ratio.min():.4f}")
    print(f"    max:    {ratio.max():.4f}")
    print(f"    mean:   {ratio.mean():.4f}")
    print(f"    std:    {ratio.std():.4f}")
    print(f"\n  nu (from model):")
    print(f"    min:    {nu_vals.min():.4f}")
    print(f"    max:    {nu_vals.max():.4f}")
    print(f"    mean:   {nu_vals.mean():.4f}")
    print(f"    std:    {nu_vals.std():.4f}")

    # ─── 5. Check if nu is constant ───
    print("\n" + "=" * 60)
    print("  5. IS NU CONSTANT?")
    print("=" * 60)

    nu_std = nu_vals.std()
    nu_range = nu_vals.max() - nu_vals.min()
    nu_cv = nu_std / (nu_vals.mean() + 1e-12)  # coefficient of variation

    print(f"  nu std:                {nu_std:.6f}")
    print(f"  nu range:              {nu_range:.6f}")
    print(f"  nu coefficient of var: {nu_cv:.6f}")

    if nu_cv < 0.05:
        print("\n  ⚠️ nu is APPROXIMATELY CONSTANT (CV < 5%)")
        print("  → This is a MODEL issue: the model is not learning to vary nu")
        print("  → Consequently, aleatoric/epistemic = nu ≈ constant")
        print("  → The 1:1 line in the figure is caused by this")
    elif nu_cv < 0.2:
        print("\n  ⚠️ nu has LOW variation (CV < 20%)")
        print("  → The model is learning some variation but not enough")
    else:
        print("\n  ✅ nu has MODERATE/HIGH variation (CV > 20%)")
        print("  → The model is learning to vary nu across samples")
        print("  → The figure may look like a line due to log-log scaling")

    # ─── 6. Check aleatoric/epistemic dominance ───
    print("\n" + "=" * 60)
    print("  6. WHICH UNCERTAINTY DOMINATES?")
    print("=" * 60)

    n_aleatoric_dominates = (aleatoric > epistemic).sum()
    n_epistemic_dominates = (epistemic > aleatoric).sum()
    total = len(aleatoric)

    print(f"  Aleatoric > Epistemic: {n_aleatoric_dominates}/{total} "
          f"({100*n_aleatoric_dominates/total:.1f}%)")
    print(f"  Epistemic > Aleatoric: {n_epistemic_dominates}/{total} "
          f"({100*n_epistemic_dominates/total:.1f}%)")

    # ─── 7. Sample values ───
    print("\n" + "=" * 60)
    print("  7. SAMPLE VALUES (first 10)")
    print("=" * 60)
    print(f"  {'Sample':<8} {'Aleatoric':<14} {'Epistemic':<14} {'Ratio':<10} {'nu':<10}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*10} {'-'*10}")
    for i in range(min(10, len(aleatoric))):
        print(f"  {i:<8} {aleatoric[i]:<14.6e} {epistemic[i]:<14.6e} "
              f"{aleatoric[i]/epistemic[i]:<10.4f} {nu_vals[i]:<10.4f}")

    # ─── 8. Conclusion ───
    print("\n" + "=" * 60)
    print("  8. CONCLUSION")
    print("=" * 60)

    if nu_cv < 0.05:
        print("""
  The problem is in the MODEL, not the FIGURE.

  The model learned to output approximately constant nu across all samples.
  Since aleatoric/epistemic = nu by definition, the ratio is also constant,
  which makes aleatoric and epistemic perfectly proportional.

  This produces the 1:1 line in the log-log scatter plot.

  To fix this:
    1. Remove or increase the clamp on nu (currently max=50.0)
    2. Add a diversity regularization on nu (encourage it to vary)
    3. Reduce the MSE anchor weight (currently 1.0) to let nu learn
    4. Increase d_state (currently 16) for more expressive SSM
        """)
    else:
        print("""
  The model is learning to vary nu across samples.

  The figure may look like a line due to:
    1. Log-log scaling compressing the spread
    2. Most samples having similar uncertainty levels
    3. A strong but not perfect correlation

  This is NORMAL behavior and not a problem.
        """)

    print("\n" + "=" * 60)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 60)