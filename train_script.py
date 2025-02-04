import os
import torch
from config import SmolLM2Config
from model import SmolLM2Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
import codecs
import csv
import sys

# Set CUDA environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Force UTF-8 encoding for stdout/stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

class CustomCSVLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def log_metrics(self, metrics, step=None):
        # Clean and encode metrics
        cleaned_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, str):
                # Clean string values
                v = v.encode('ascii', 'ignore').decode('ascii')
            cleaned_metrics[k] = v
        
        super().log_metrics(cleaned_metrics, step)

class GenerationMonitorCallback(Callback):
    def __init__(self, prompts, sample_every_n_steps=200):
        super().__init__()
        self.prompts = prompts
        self.sample_every_n_steps = sample_every_n_steps
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.sample_every_n_steps == 0:
            pl_module.eval()
            print(f"\n=== Generation samples at step {trainer.global_step + 1} ===")
            
            for prompt in self.prompts:
                try:
                    inputs = pl_module.tokenizer(
                        prompt, 
                        return_tensors="pt",
                        truncation=True,
                        max_length=pl_module.config.model.max_position_embeddings,
                        padding=True
                    ).to(pl_module.device)
                    
                    outputs = pl_module.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=200,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True
                    )
                    
                    generated_text = pl_module.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {generated_text}\n")
                    
                    # Log metrics with cleaned text
                    trainer.logger.log_metrics({
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'step': trainer.global_step
                    })
                    
                except Exception as e:
                    print(f"Generation error for prompt '{prompt}': {str(e)}")
            
            pl_module.train()

def main():
    try:
        # Create logging directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Load config
        config = SmolLM2Config("config.yaml")
        
        # Initialize model
        model = SmolLM2Lightning(config)
        
        # Setup custom CSV logger
        logger = CustomCSVLogger(
            save_dir=config.training.logging_dir,
            name="training_logs",
            version=None,
            flush_logs_every_n_steps=100
        )
        
        # Setup checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.training.checkpoint_dir,
            filename="smol-lm2-final",
            save_top_k=1,
            verbose=True,
            monitor="train_loss",
            mode="min",
            save_last=True
        )
        
        # Setup generation monitoring callback
        generation_callback = GenerationMonitorCallback(
            prompts=config.training.sample_prompts,
            sample_every_n_steps=config.training.sample_frequency
        )
        
        # Initialize trainer with CSV logger
        trainer = pl.Trainer(
            max_steps=config.training.max_steps,
            accelerator=config.hardware.accelerator,
            devices=config.hardware.devices,
            precision=config.hardware.precision,
            logger=logger,
            callbacks=[checkpoint_callback, generation_callback],
            gradient_clip_val=config.hardware.gradient_clip,
            accumulate_grad_batches=config.training.gradient_accumulation_steps,
            log_every_n_steps=config.training.logging_steps,
            deterministic=False,
            benchmark=True,
            strategy='auto'
        )
        
        # Train
        print("\n=== Starting Training ===")
        trainer.fit(model)
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(config.training.checkpoint_dir, "smol-lm2-final.ckpt")
        trainer.save_checkpoint(final_checkpoint_path)
        print(f"\nTraining completed. Model saved to {final_checkpoint_path}")
        
        # Verify checkpoint was saved
        if os.path.exists(final_checkpoint_path):
            print(f"Checkpoint size: {os.path.getsize(final_checkpoint_path) / 1e6:.2f} MB")
        else:
            raise Exception("Checkpoint was not saved!")
            
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        if torch.cuda.is_available():
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        raise

if __name__ == "__main__":
    main() 