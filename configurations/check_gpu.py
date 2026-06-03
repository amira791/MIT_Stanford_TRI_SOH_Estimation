import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
# from config import TRAINING_DEVICE, USE_MIXED_PRECISION, BATCH_SIZE

def check_gpu_setup():
    """Verify GPU is properly configured"""
    print("=" * 50)
    print("GPU CONFIGURATION CHECK")
    print("=" * 50)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device properties:")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  - Compute capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        
        # Test GPU memory allocation
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"\n✓ GPU memory allocation test: PASSED")
        del test_tensor
        torch.cuda.empty_cache()
        
    # print(f"\nTraining device: {TRAINING_DEVICE}")
    # print(f"Mixed precision training: {USE_MIXED_PRECISION}")
    # print(f"Batch size: {BATCH_SIZE}")
    
    # Recommendation
    if torch.cuda.is_available():
        print("\n✓ GPU is ready for training!")
        if BATCH_SIZE < 128:
            print("  💡 Consider increasing BATCH_SIZE to 128 or 256 for faster training")
    else:
        print("\n⚠ GPU not detected. Check:")
        print("  1. NVIDIA drivers installed? Run 'nvidia-smi'")
        print("  2. CUDA toolkit installed?")
        print("  3. PyTorch with CUDA support: 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'")

if __name__ == "__main__":
    check_gpu_setup()