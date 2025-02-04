import os
import torch
from train_script import main as train_main

def start_training():
    """Start the training process with the existing environment"""
    try:
        # Create necessary directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Verify CUDA is available
        if torch.cuda.is_available():
            print(f"\nUsing GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("\nNo GPU available. Using CPU.")
            
        print("\nStarting training process...")
        train_main()
        
        # Verify checkpoint was created
        checkpoint_path = "checkpoints/smol-lm2-final.ckpt"
        if os.path.exists(checkpoint_path):
            print(f"\nCheckpoint saved successfully at: {checkpoint_path}")
            print(f"Checkpoint size: {os.path.getsize(checkpoint_path) / 1e6:.2f} MB")
        else:
            print("\nWarning: Checkpoint was not created!")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    start_training() 