# model_architecture/train.py
"""
TRAIN / VAL / TEST SPLIT + MODEL TRAINING
-------------------------------------------
1. Loads soh_dataset.pkl (from step2)
2. Applies CELL-OUT split -> no data leakage
3. Builds sliding-window sequence datasets + DataLoaders
4. Trains CNN-Mamba-UQ with MSE loss + AdamW + Early stopping
5. Saves best checkpoint + training history
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from configurations.config_training import (
    RESULTS_DIR, FEATURE_COLS, TARGET_COL,
    TRAIN_FRAC, VAL_FRAC, RANDOM_SEED,
    SEQ_LEN, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, LR_PATIENCE
)

from cnn_mamba_uq import CNNMambaUQ

MODEL_SAVE_DIR = RESULTS_DIR / "checkpoints"
SCALER_PATH = RESULTS_DIR / "scaler.pkl"
MODEL_SAVE_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def cell_out_split(df: pd.DataFrame):
    """Split at the CELL level to prevent data leakage."""
    rng = np.random.default_rng(RANDOM_SEED)
    cells = df["cell_id"].unique()
    n = len(cells)
    idx = rng.permutation(n)

    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_cells = cells[idx[:n_train]]
    val_cells = cells[idx[n_train:n_train + n_val]]
    test_cells = cells[idx[n_train + n_val:]]

    print(f"\n  Cell-out split:")
    print(f"    Train : {len(train_cells)} cells")
    print(f"    Val   : {len(val_cells)} cells")
    print(f"    Test  : {len(test_cells)} cells")

    df_train = df[df["cell_id"].isin(train_cells)].copy()
    df_val = df[df["cell_id"].isin(val_cells)].copy()
    df_test = df[df["cell_id"].isin(test_cells)].copy()

    print(f"    Train samples : {len(df_train):,}")
    print(f"    Val   samples : {len(df_val):,}")
    print(f"    Test  samples : {len(df_test):,}")

    return df_train, df_val, df_test, test_cells

def build_sequences(df: pd.DataFrame, scaler: StandardScaler, seq_len: int = SEQ_LEN) -> TensorDataset:
    """Build sliding window sequences for each cell."""
    X_list, y_list = [], []

    for cell_id, grp in df.groupby("cell_id"):
        grp = grp.sort_values("cycle_index").reset_index(drop=True)
        X_cell = scaler.transform(grp[FEATURE_COLS].values).astype(np.float32)
        y_cell = grp[TARGET_COL].values.astype(np.float32)

        n = len(grp)
        if n <= seq_len:
            continue

        for i in range(n - seq_len):
            X_list.append(X_cell[i:i + seq_len])
            y_list.append(y_cell[i + seq_len])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return TensorDataset(X, y)

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        optimiser.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)

def train_model(model, train_dl, val_dl, device):
    criterion = nn.MSELoss()
    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", patience=LR_PATIENCE, factor=0.5)
    #scheduler = ReduceLROnPlateau(optimiser, mode="min", patience=LR_PATIENCE, factor=0.5, verbose=True)

    best_val = float("inf")
    patience_counter = 0
    history = []
    ckpt_path = MODEL_SAVE_DIR / "cnn_mamba_uq_best.pt"

    print(f"\n{'='*60}")
    print(f"  Training CNN-Mamba-UQ on {device}")
    print(f"{'='*60}")
    print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}  {'Time':>7}")
    print(f"  {'─'*55}")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, optimiser, criterion, device)
        val_loss = eval_epoch(model, val_dl, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        lr_now = optimiser.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "lr": lr_now})

        print(f"  {epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  {lr_now:>10.2e}  {elapsed:>6.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n  Best val MSE : {best_val:.6f}")
    print(f"  Checkpoint   : {ckpt_path}\n")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, pd.DataFrame(history)

def main():
    print("="*60)
    print("TRAINING SCRIPT STARTED")
    print("="*60)

    pkl_path = RESULTS_DIR / "soh_dataset.pkl"
    print(f"\nLoading dataset from {pkl_path} ...")
    df = pd.read_pickle(pkl_path)
    print(f"  {len(df):,} rows  |  {df['cell_id'].nunique()} cells")

    df_train, df_val, df_test, test_cells = cell_out_split(df)

    scaler = StandardScaler()
    scaler.fit(df_train[FEATURE_COLS].values)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Scaler saved -> {SCALER_PATH}")

    with open(RESULTS_DIR / "test_cells.pkl", "wb") as f:
        pickle.dump(test_cells, f)

    print("\nBuilding sliding-window sequences ...")
    train_ds = build_sequences(df_train, scaler)
    val_ds = build_sequences(df_val, scaler)
    test_ds = build_sequences(df_test, scaler)

    print(f"  Train sequences : {len(train_ds):,}")
    print(f"  Val   sequences : {len(val_ds):,}")
    print(f"  Test  sequences : {len(test_ds):,}")

    torch.save(test_ds, RESULTS_DIR / "test_dataset.pt")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("\nInitialising CNN-Mamba-UQ ...")
    model = CNNMambaUQ().to(DEVICE)

    model, history = train_model(model, train_dl, val_dl, DEVICE)

    history.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    print(f"\n  Training history -> {RESULTS_DIR / 'training_history.csv'}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()