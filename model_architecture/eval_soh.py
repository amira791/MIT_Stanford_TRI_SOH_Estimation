# Run this directly in your open Python terminal
# (where the error occurred)

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from train_final import SOHSequenceDataset, make_loader, CFG, FEAT_COLS, DEVICE

print("="*60)
print("SAVING BEST SOH MODEL FROM MEMORY")
print("="*60)

# Save the model
save_path = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\soh_best_final.pt"
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to: {save_path}")

# Evaluate on test set
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

# Reload data (already loaded, but just in case)
from train_final import soh_df

test_ds = SOHSequenceDataset(soh_df, CFG["window_size"], CFG["stride"], "test")
test_loader = make_loader(test_ds, CFG["soh_batch"], shuffle=False)

model.eval()
all_pred, all_true = [], []

with torch.no_grad():
    for x, y, w in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mu, _ = model(x, task="soh")
        all_pred.extend(mu.cpu().numpy())
        all_true.extend(y.cpu().numpy())

y_true = np.array(all_true)
y_pred = np.array(all_pred)

mae = mean_absolute_error(y_true, y_pred) * 100
rmse = np.sqrt(np.mean((y_true - y_pred)**2)) * 100
r2 = r2_score(y_true, y_pred)

print(f"\n{'='*60}")
print("TEST RESULTS")
print(f"{'='*60}")
print(f"  MAE:  {mae:.4f}%  (target: <0.70%)")
print(f"  RMSE: {rmse:.4f}%")
print(f"  R²:   {r2:.4f}   (target: >0.97)")
print(f"{'='*60}")

if mae < 0.70 and r2 > 0.97:
    print("\n🎉 TARGETS MET! Model is ready for deployment.")
else:
    print(f"\n⚠️ MAE {mae:.4f}% / R² {r2:.4f}")

# Optional: Save predictions
import pandas as pd
results_df = pd.DataFrame({
    'true_soh': y_true,
    'predicted_soh': y_pred,
    'absolute_error_pct': np.abs(y_true - y_pred) * 100
})
results_df.to_csv("soh_test_results_final.csv", index=False)
print(f"\n✅ Predictions saved to: soh_test_results_final.csv")