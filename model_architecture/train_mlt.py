"""
step3_train_mlt.py
==================
Key changes vs train_50:
  1. Huber loss (robust to the 2 anomalous cells dragging R² down)
  2. Multi-task loss: future_soh + current_soh + eol_prob
  3. current_soh target built from the last cycle of each window
  4. eol_prob target = 1 if predicted future SOH < 0.80 else 0
"""

import sys, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_training_mlt import (
    RESULTS_DIR, MODEL_SAVE_DIR, SCALER_PATH,
    FEATURE_COLS, TARGET_COL,
    TRAIN_FRAC, VAL_FRAC, RANDOM_SEED,
    SEQ_LEN, PREDICTION_HORIZON,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, LR_PATIENCE,
    LOSS_TYPE, HUBER_DELTA, MTL_WEIGHTS,
    SOH_EOL_THRESHOLD,
)
from cnn_mamba_uq_mlt import CNNMambaUQ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  Loss: {LOSS_TYPE}  |  Horizon: {PREDICTION_HORIZON}")


# ── SPLIT ──────────────────────────────────────────────────────────────────

def cell_out_split(df):
    rng   = np.random.default_rng(RANDOM_SEED)
    cells = df["cell_id"].unique()
    idx   = rng.permutation(len(cells))
    n_tr  = int(len(cells) * TRAIN_FRAC)
    n_val = int(len(cells) * VAL_FRAC)
    tr = cells[idx[:n_tr]]
    va = cells[idx[n_tr:n_tr+n_val]]
    te = cells[idx[n_tr+n_val:]]
    print(f"  Split: train={len(tr)} val={len(va)} test={len(te)} cells")
    return (df[df.cell_id.isin(tr)].copy(),
            df[df.cell_id.isin(va)].copy(),
            df[df.cell_id.isin(te)].copy(), te)


# ── SEQUENCE BUILDER ───────────────────────────────────────────────────────

def build_sequences(df, scaler, seq_len=SEQ_LEN, horizon=PREDICTION_HORIZON):
    """
    Returns TensorDataset with:
      X          : (N, seq_len, n_features)
      y_future   : (N,)  SOH at t+horizon  ← primary target
      y_current  : (N,)  SOH at t+seq_len-1 ← auxiliary target
      y_eol      : (N,)  1.0 if y_future < 0.80 else 0.0
    """
    X_list, yf_list, yc_list, ye_list = [], [], [], []

    for _, grp in df.groupby("cell_id"):
        grp   = grp.sort_values("cycle_index").reset_index(drop=True)
        Xc    = scaler.transform(grp[FEATURE_COLS].values).astype(np.float32)
        soh   = grp[TARGET_COL].values.astype(np.float32)
        n     = len(grp)
        if n <= seq_len + horizon:
            continue
        for i in range(n - seq_len - horizon + 1):
            X_list.append(Xc[i:i + seq_len])
            yf = soh[i + seq_len + horizon - 1]
            yc = soh[i + seq_len - 1]             # last cycle in window
            ye = 1.0 if yf < SOH_EOL_THRESHOLD else 0.0
            yf_list.append(yf); yc_list.append(yc); ye_list.append(ye)

    X  = torch.tensor(np.array(X_list),  dtype=torch.float32)
    yf = torch.tensor(np.array(yf_list), dtype=torch.float32)
    yc = torch.tensor(np.array(yc_list), dtype=torch.float32)
    ye = torch.tensor(np.array(ye_list), dtype=torch.float32)
    return TensorDataset(X, yf, yc, ye)


# ── LOSS FUNCTION ──────────────────────────────────────────────────────────

def build_loss():
    if LOSS_TYPE == "huber":
        base = nn.HuberLoss(delta=HUBER_DELTA)
    else:
        base = nn.MSELoss()
    bce = nn.BCELoss()

    def multi_task_loss(preds, yf, yc, ye):
        l_fut  = base(preds["future_soh"].squeeze(),  yf)
        l_cur  = base(preds["current_soh"].squeeze(), yc)
        l_eol  = bce(preds["eol_prob"].squeeze(),     ye)
        return (MTL_WEIGHTS["future_soh"]  * l_fut +
                MTL_WEIGHTS["current_soh"] * l_cur +
                MTL_WEIGHTS["eol_prob"]    * l_eol)
    return multi_task_loss


# ── TRAIN / EVAL EPOCHS ───────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, loss_fn, device):
    model.train()
    total = 0.0
    for X, yf, yc, ye in loader:
        X, yf, yc, ye = X.to(device), yf.to(device), yc.to(device), ye.to(device)
        optimiser.zero_grad()
        loss = loss_fn(model(X), yf, yc, ye)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for X, yf, yc, ye in loader:
        X, yf, yc, ye = X.to(device), yf.to(device), yc.to(device), ye.to(device)
        loss = loss_fn(model(X), yf, yc, ye)
        total += loss.item() * len(X)
    return total / len(loader.dataset)


# ── TRAINING LOOP ─────────────────────────────────────────────────────────

def train_model(model, train_dl, val_dl, device):
    loss_fn   = build_loss()
    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", patience=LR_PATIENCE, factor=0.5)

    best_val, wait = float("inf"), 0
    ckpt = MODEL_SAVE_DIR / "cnn_mamba_uq_mlt_best.pt"
    history = []

    print(f"\n{'='*65}")
    print(f"  Training CNN-Mamba-UQ mlt  |  {PREDICTION_HORIZON}-cycle horizon  |  {LOSS_TYPE} loss")
    print(f"{'='*65}")
    print(f"  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}  {'Time':>7}")
    print(f"  {'─'*58}")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr = train_epoch(model, train_dl, optimiser, loss_fn, device)
        va = eval_epoch(model, val_dl, loss_fn, device)
        scheduler.step(va)
        lr_now = optimiser.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": tr, "val": va, "lr": lr_now})
        mark = " ***" if va < best_val else ""
        print(f"  {epoch:>6}  {tr:>10.6f}  {va:>10.6f}  {lr_now:>10.2e}  {time.time()-t0:>6.1f}s{mark}")

        if va < best_val:
            best_val = va; wait = 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
        if wait >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n  Best val loss : {best_val:.6f}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, pd.DataFrame(history)


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    # Load dataset from mlt results dir
    pkl = RESULTS_DIR / "soh_dataset_mlt.pkl"
    if not pkl.exists():
        # fall back to shared results
        pkl = Path(__file__).parent.parent / "results" / "soh_dataset.pkl"
    print(f"Loading {pkl} …")
    df = pd.read_pickle(pkl)
    print(f"  {len(df):,} rows | {df['cell_id'].nunique()} cells")

    df_tr, df_va, df_te, test_cells = cell_out_split(df)

    scaler = StandardScaler()
    scaler.fit(df_tr[FEATURE_COLS].values)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    with open(RESULTS_DIR / "test_cells_mlt.pkl", "wb") as f:
        pickle.dump(test_cells, f)

    print("Building sequences …")
    tr_ds = build_sequences(df_tr, scaler)
    va_ds = build_sequences(df_va, scaler)
    te_ds = build_sequences(df_te, scaler)
    print(f"  train={len(tr_ds):,}  val={len(va_ds):,}  test={len(te_ds):,}")
    torch.save(te_ds, RESULTS_DIR / "test_dataset_mlt.pt")

    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = CNNMambaUQ().to(DEVICE)
    model, hist = train_model(model, tr_dl, va_dl, DEVICE)

    hist.to_csv(RESULTS_DIR / "training_history_mlt.csv", index=False)
    print(f"Done. All outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
