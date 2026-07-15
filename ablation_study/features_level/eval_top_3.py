# extract_top3_results.py
"""
Extract and display results from the saved Top-3 features model checkpoint.
"""

import torch
import json
import os
from pathlib import Path

# Path to the saved model checkpoint
MODEL_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\checkpoints\soh_top3_best.pt"
OUTPUT_PATH = r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation\results\soh_top3_results.json"

def extract_results():
    """Load the checkpoint and extract results."""
    print("=" * 60)
    print("  EXTRACTING TOP-3 FEATURES RESULTS")
    print("=" * 60)
    
    # Load checkpoint with weights_only=False (trusted source)
    print(f"\n[1] Loading checkpoint from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    
    # Check what's in the checkpoint
    print(f"\n[2] Checkpoint contents: {list(checkpoint.keys())}")
    
    # Extract results
    if "results" in checkpoint:
        results = checkpoint["results"]
        print("\n[3] Results found in checkpoint:")
        print(f"  MAE  : {results['mae']:.4f}%")
        print(f"  RMSE : {results['rmse']:.4f}%")
        print(f"  R²   : {results['r2']:.5f}")
        print(f"  PICP : {results['picp']:.4f}")
        print(f"  PINW : {results['pinw']:.4f}")
        
        # Also extract training history if available
        if "history" in checkpoint:
            history = checkpoint["history"]
            print(f"\n[4] Training history:")
            print(f"  Best validation loss: {history['val_loss'][-1]:.6f}")
            print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
            print(f"  Final validation MAE: {history['val_mae'][-1]:.4f}%")
            print(f"  Final validation R²: {history['val_r2'][-1]:.5f}")
        
        # Save results to JSON
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[5] Results saved to: {OUTPUT_PATH}")
        
    else:
        print("\n[3] No 'results' key found in checkpoint.")
        print("Available keys:", list(checkpoint.keys()))
        
        # Check if results are stored elsewhere
        if "cfg" in checkpoint:
            print("\n  Config found in checkpoint")
        if "feat_cols" in checkpoint:
            print("  Features found in checkpoint")
        if "scaler_mean" in checkpoint:
            print("  Scaler parameters found in checkpoint")

if __name__ == "__main__":
    extract_results()