# train_soh.py
"""
Training Script for SOH Prediction using CNN-Mamba-UQ
---------------------------------------------------
Trains the CNN-Mamba-UQ model for State of Health estimation
Supports:
- Supervised learning on all 134 cells
- Uncertainty quantification
- Checkpoint saving
- Early stopping
- Learning rate scheduling
- TensorBoard logging
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cnn_mamba_uq_model import create_soh_model, get_model_summary

# ============================================
# CONFIGURATION - FIXED FOR STABLE TRAINING
# ============================================

class Config:
    # Paths
    BASE_DIR = Path(r"C:\Users\admin\Desktop\DR2\16 Contributions\Contr03\MIT_Stanford_TRI_SOH_Estimation")

    DATA_DIR = BASE_DIR / "data_preprocessing" / "final_dataset" / "soh"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "soh"
    LOG_DIR = BASE_DIR / "logs_model" / "soh"
    RESULTS_DIR = BASE_DIR / "results_model" / "soh"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Data parameters
    SEQUENCE_LENGTH = 50
    N_FEATURES = 7
    BATCH_SIZE = 64
    
    # Model parameters
    DROPOUT = 0.1
    
    # Training parameters - CHANGED
    EPOCHS = 100
    LEARNING_RATE = 3e-4  # CHANGED: was 1e-3 (reduced by 3x)
    WEIGHT_DECAY = 1e-4   # CHANGED: was 1e-5 (increased by 10x)
    
    # Learning rate scheduling - CHANGED
    LR_PATIENCE = 5       # CHANGED: was 10 (reduced)
    LR_FACTOR = 0.5
    LR_MIN = 1e-7         # CHANGED: was 1e-6 (lower minimum)
    
    # Early stopping - CHANGED
    EARLY_STOPPING_PATIENCE = 15  # CHANGED: was 20 (reduced)
    EARLY_STOPPING_MIN_DELTA = 1e-5  # CHANGED: was 1e-4 (more sensitive)
    
    # Loss weights
    UNCERTAINTY_WEIGHT = 0.05  # CHANGED: was 0.1 (reduced to prevent instability)
    
    # Gradient clipping - CHANGED (added new parameter)
    GRAD_CLIP_NORM = 0.5  # NEW: was hardcoded 1.0 (reduced)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random seed
    SEED = 42

# Set random seed for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ============================================
# DATA LOADING
# ============================================

class SOHDataLoader:
    """Load SOH sequence data for training"""
    
    def __init__(self, data_dir: Path, sequence_length: int = 50):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        
    def load_data(self):
        """Load pre-computed sequence data"""
        print("\n[1] Loading SOH data...")
        
        # Load sequences
        X_train = np.load(self.data_dir / f"X_soh_train_seq{self.sequence_length}.npy")
        y_train = np.load(self.data_dir / f"y_soh_train_seq{self.sequence_length}.npy")
        X_val = np.load(self.data_dir / f"X_soh_val_seq{self.sequence_length}.npy")
        y_val = np.load(self.data_dir / f"y_soh_val_seq{self.sequence_length}.npy")
        X_test = np.load(self.data_dir / f"X_soh_test_seq{self.sequence_length}.npy")
        y_test = np.load(self.data_dir / f"y_soh_test_seq{self.sequence_length}.npy")
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        
        print(f"    Train: {X_train.shape}, {y_train.shape}")
        print(f"    Val:   {X_val.shape}, {y_val.shape}")
        print(f"    Test:  {X_test.shape}, {y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Create DataLoaders for training"""
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0,
            drop_last=False  # Keep all data
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader

# ============================================
# LOSS FUNCTIONS - CHANGED (Added numerical stability)
# ============================================

class SOHLoss(nn.Module):
    """
    Loss function for SOH prediction with uncertainty
    Loss = MSE(pred, target) + λ * uncertainty
    """
    def __init__(self, uncertainty_weight: float = 0.05):
        super().__init__()
        self.uncertainty_weight = uncertainty_weight
        
    def forward(self, pred, target, uncertainty):
        # Ensure values are finite
        pred = torch.clamp(pred, 0.0, 1.0)  # CHANGED: Clamp SOH predictions
        target = torch.clamp(target, 0.0, 1.0)
        uncertainty = torch.clamp(uncertainty, 1e-6, 1.0)  # CHANGED: Prevent zero/negative
        
        # MSE loss
        mse_loss = nn.functional.mse_loss(pred, target)
        
        # Check for NaN
        if torch.isnan(mse_loss):
            return torch.tensor(1.0, device=pred.device, requires_grad=True), mse_loss, torch.tensor(0.0)
        
        # Uncertainty regularization
        uncertainty_loss = uncertainty.mean()
        
        # Total loss
        total_loss = mse_loss + self.uncertainty_weight * uncertainty_loss
        
        return total_loss, mse_loss, uncertainty_loss


# ============================================
# TRAINER - CHANGED (Added stability improvements)
# ============================================

class SOHTrainer:
    """Trainer for SOH model"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Optimizer - CHANGED (added betas)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)  # Added explicit betas
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.LR_MIN,
            verbose=True  # CHANGED: Enabled to see LR changes
        )
        
        # Loss function
        self.criterion = SOHLoss(uncertainty_weight=config.UNCERTAINTY_WEIGHT)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []
        self.val_mse = []
        self.learning_rates = []  # NEW: Track LR
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_uncertainty = 0
        valid_batches = 0
        
        for X_batch, y_batch in tqdm(self.train_loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.config.DEVICE)
            y_batch = y_batch.to(self.config.DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred, uncertainty = self.model(X_batch)
            
            # Check for NaN in model outputs
            if torch.isnan(pred).any() or torch.isnan(uncertainty).any():
                print("Warning: NaN detected in model output, skipping batch")
                continue
            
            # Compute loss
            loss, mse_loss, uncertainty_loss = self.criterion(pred, y_batch, uncertainty)
            
            # Skip if loss is NaN
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
            
            # Backward pass with gradient clipping
            loss.backward()
            
            # CHANGED: Use configurable gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_NORM)
            
            # Check for exploding gradients
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > 10:  # Exploding gradient detected
                print(f"Warning: Gradient norm {total_norm:.2f} > 10, skipping update")
                continue
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_uncertainty += uncertainty_loss.item()
            valid_batches += 1
        
        if valid_batches == 0:
            return float('nan'), float('nan'), float('nan')
        
        return (total_loss / valid_batches, 
                total_mse / valid_batches, 
                total_uncertainty / valid_batches)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_uncertainty = 0
        valid_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.config.DEVICE)
                y_batch = y_batch.to(self.config.DEVICE)
                
                pred, uncertainty = self.model(X_batch)
                
                # Clamp predictions
                pred = torch.clamp(pred, 0.0, 1.0)
                uncertainty = torch.clamp(uncertainty, 1e-6, 1.0)
                
                loss, mse_loss, uncertainty_loss = self.criterion(pred, y_batch, uncertainty)
                
                if torch.isnan(loss):
                    continue
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_uncertainty += uncertainty_loss.item()
                valid_batches += 1
        
        if valid_batches == 0:
            return float('nan'), float('nan'), float('nan')
        
        return (total_loss / valid_batches, 
                total_mse / valid_batches, 
                total_uncertainty / valid_batches)
    
    def train(self):
        """Main training loop"""
        print("\n[3] Starting training...")
        print(f"    Device: {self.config.DEVICE}")
        print(f"    Epochs: {self.config.EPOCHS}")
        print(f"    Learning rate: {self.config.LEARNING_RATE}")
        print(f"    Batch size: {self.config.BATCH_SIZE}")
        print(f"    Gradient clip norm: {self.config.GRAD_CLIP_NORM}")
        
        for epoch in range(1, self.config.EPOCHS + 1):
            try:
                # Train
                train_loss, train_mse, train_unc = self.train_epoch()
                
                # Check for NaN
                if np.isnan(train_loss):
                    print(f"Epoch {epoch}: NaN training loss, reducing learning rate...")
                    # Reduce LR and continue
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    continue
                
                # Validate
                val_loss, val_mse, val_unc = self.validate()
                
                if np.isnan(val_loss):
                    print(f"Epoch {epoch}: NaN validation loss, skipping...")
                    continue
                
                # Store losses
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_mse.append(train_mse)
                self.val_mse.append(val_mse)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Print progress
                print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
                print(f"  Train - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, Unc: {train_unc:.6f}")
                print(f"  Val   - Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, Unc: {val_unc:.6f}")
                print(f"  LR: {current_lr:.2e}")
                
                # Early stopping check
                if val_loss < self.best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                    print(f"  ✓ New best model saved!")
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        print(f"\nEarly stopping triggered after {epoch} epochs")
                        break
                
                # Save checkpoint every 10 epochs
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, val_loss, is_best=False)
                    
            except Exception as e:
                print(f"Error at epoch {epoch}: {e}")
                continue
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': {
                'sequence_length': self.config.SEQUENCE_LENGTH,
                'n_features': self.config.N_FEATURES,
                'dropout': self.config.DROPOUT,
                'learning_rate': self.config.LEARNING_RATE,
                'weight_decay': self.config.WEIGHT_DECAY
            }
        }
        
        if is_best:
            path = self.config.CHECKPOINT_DIR / "best_model.pth"
        else:
            path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, path)
    
    def load_best_model(self):
        """Load the best saved model"""
        path = self.config.CHECKPOINT_DIR / "best_model.pth"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model from epoch {checkpoint['epoch']}")
            return checkpoint['epoch']
        return None
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n[4] Evaluating on test set...")
        self.model.eval()
        
        predictions = []
        targets = []
        uncertainties = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.config.DEVICE)
                pred, uncertainty = self.model(X_batch)
                
                # Clamp predictions
                pred = torch.clamp(pred, 0.0, 1.0)
                
                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(y_batch.cpu().numpy().flatten())
                uncertainties.extend(uncertainty.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        uncertainties = np.array(uncertainties)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        results = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties))
        }
        
        print(f"\nTest Results:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        print(f"  Mean Uncertainty: {results['mean_uncertainty']:.6f}")
        
        return results, predictions, targets, uncertainties
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss plot
        ax1 = axes[0]
        ax1.plot(self.train_losses, label='Train Loss', alpha=0.7)
        ax1.plot(self.val_losses, label='Val Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MSE plot
        ax2 = axes[1]
        ax2.plot(self.train_mse, label='Train MSE', alpha=0.7)
        ax2.plot(self.val_mse, label='Val MSE', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.set_title('MSE History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot - NEW
        ax3 = axes[2]
        ax3.plot(self.learning_rates, alpha=0.7)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.RESULTS_DIR / "training_history.png", dpi=150)
        plt.show()
        print(f"\nSaved: {self.config.RESULTS_DIR / 'training_history.png'}")
    
    def plot_predictions(self, predictions, targets, uncertainties):
        """Plot predictions vs targets with uncertainty"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        ax1 = axes[0]
        scatter = ax1.scatter(targets, predictions, c=uncertainties, cmap='viridis', alpha=0.5, s=10)
        ax1.plot([0.7, 1.0], [0.7, 1.0], 'r--', label='Perfect Prediction')
        ax1.set_xlabel('True SOH')
        ax1.set_ylabel('Predicted SOH')
        ax1.set_title('Predictions vs Targets (color = uncertainty)')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Uncertainty')
        
        # Uncertainty vs error
        ax2 = axes[1]
        errors = np.abs(predictions - targets)
        ax2.scatter(uncertainties, errors, alpha=0.5, s=10)
        ax2.set_xlabel('Predicted Uncertainty')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Uncertainty vs Error')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.RESULTS_DIR / "predictions.png", dpi=150)
        plt.show()
        print(f"Saved: {self.config.RESULTS_DIR / 'predictions.png'}")
    
    def plot_error_distribution(self, predictions, targets):
        """Plot error distribution"""
        errors = predictions - targets
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--')
        ax1.set_xlabel('Prediction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Box plot by SOH range
        ax2 = axes[1]
        soh_bins = [(0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.00)]
        errors_by_bin = []
        labels = []
        
        for low, high in soh_bins:
            mask = (targets >= low) & (targets < high)
            if mask.any():
                errors_by_bin.append(errors[mask])
                labels.append(f'{low:.2f}-{high:.2f}')
        
        ax2.boxplot(errors_by_bin, labels=labels)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_xlabel('SOH Range')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title('Error by SOH Range')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.RESULTS_DIR / "error_distribution.png", dpi=150)
        plt.show()
        print(f"Saved: {self.config.RESULTS_DIR / 'error_distribution.png'}")


# ============================================
# MAIN TRAINING PIPELINE
# ============================================

def main():
    print("=" * 60)
    print("TRAINING CNN-MAMBA-UQ FOR SOH PREDICTION")
    print("=" * 60)
    
    print(f"\nDevice: {Config.DEVICE}")
    
    # Load data
    data_loader = SOHDataLoader(Config.DATA_DIR, Config.SEQUENCE_LENGTH)
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_data()
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Create model
    print("\n[2] Creating model...")
    model = create_soh_model(
        input_dim=Config.N_FEATURES,
        seq_len=Config.SEQUENCE_LENGTH,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # Model summary
    model_summary = get_model_summary(model)
    print(f"    Total parameters: {model_summary['total_parameters']:,}")
    print(f"    Trainable parameters: {model_summary['trainable_parameters']:,}")
    print(f"    Model size: {model_summary['model_size_mb']:.2f} MB")
    
    # Save model architecture summary
    with open(Config.RESULTS_DIR / "model_summary.json", "w") as f:
        json.dump(model_summary, f, indent=2)
    
    # Train
    trainer = SOHTrainer(model, train_loader, val_loader, test_loader, Config)
    trainer.train()
    
    # Load best model
    best_epoch = trainer.load_best_model()
    
    # Evaluate
    results, predictions, targets, uncertainties = trainer.evaluate()
    
    # Save results
    with open(Config.RESULTS_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_soh': targets,
        'predicted_soh': predictions,
        'uncertainty': uncertainties
    })
    predictions_df.to_csv(Config.RESULTS_DIR / "predictions.csv", index=False)
    
    # Plotting
    trainer.plot_training_history()
    trainer.plot_predictions(predictions, targets, uncertainties)
    trainer.plot_error_distribution(predictions, targets)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {Config.RESULTS_DIR}")
    print(f"Model saved to: {Config.CHECKPOINT_DIR}")
    print(f"Logs saved to: {Config.LOG_DIR}")


if __name__ == "__main__":
    main()