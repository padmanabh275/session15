import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os
from moe import MoELayer

def _init_weights(module, std=0.02):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = float(eps)  # Ensure eps is a float
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = int(max_position_embeddings)  # Convert to int
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(self.max_position_embeddings).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        # Convert seq_len to int and ensure it's a valid value
        seq_len = int(seq_len) if seq_len is not None else x.size(1)
        if seq_len > self.max_position_embeddings:
            seq_len = self.max_position_embeddings
            
        return (
            self.cos_cached[:,:,:seq_len,:],
            self.sin_cached[:,:,:seq_len,:]
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Ensure proper broadcasting
    cos = cos[:, :, :q.size(2), :]  # [batch, 1, seq_len, dim]
    sin = sin[:, :, :q.size(2), :]  # [batch, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MLHAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        # Multi-query attention
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Repeat k/v for multi-query attention
        if self.num_key_value_heads != self.num_attention_heads:
            k = k.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        
        return self.o_proj(output)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attention = MLHAttention(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Use MoE for specified layers
        if layer_idx in config.moe_layers:
            self.feed_forward = MoELayer(config)
        else:
            self.feed_forward = MLP(config)
            
    def forward(self, x, attention_mask=None):
        # Pre-norm
        normed = self.norm1(x)
        # Attention
        x = x + self.attention(normed, attention_mask)
        # FFN/MoE
        normed = self.norm2(x)
        x = x + self.feed_forward(normed)
        return x

class SmolLM2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Initialize transformer layers
        self.layers = nn.ModuleList([
            DeepSeekBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))
        
        # Initialize rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings
        )
        
        # Initialize weights
        self.apply(lambda p: _init_weights(p, std=config.initializer_range))
        
    def forward(self, input_ids, attention_mask=None):
        try:
            # Ensure inputs are on the correct device
            device = input_ids.device
            batch_size, seq_length = input_ids.shape
            
            # Input validation
            if seq_length > self.config.max_position_embeddings:
                raise ValueError(f"Input sequence length {seq_length} exceeds maximum position embeddings {self.config.max_position_embeddings}")
            
            # Get embeddings
            hidden_states = self.embed_tokens(input_ids)
            
            # Get position embeddings
            cos, sin = self.rotary_emb(hidden_states, seq_length)
            
            # Generate attention mask if none provided
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length),
                    dtype=torch.bool,
                    device=device
                )
            else:
                # Convert to boolean if it's not already and ensure contiguous memory
                attention_mask = attention_mask.bool().contiguous()
            
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), device=device),
                diagonal=1
            ).bool()
            
            # Create attention mask [batch_size, 1, seq_length, seq_length]
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
            
            # Prepare causal mask
            causal_mask = causal_mask.view(1, 1, seq_length, seq_length)
            
            # Combine masks
            mask = attention_mask & ~causal_mask
            
            # Convert boolean mask to float mask
            mask = mask.to(dtype=hidden_states.dtype)
            mask = (1.0 - mask) * torch.finfo(hidden_states.dtype).min
            
            # Apply transformer layers
            for layer in self.layers:
                hidden_states = layer(hidden_states, mask)
            
            # Apply final normalization
            hidden_states = self.norm(hidden_states)
            
            # Project back to vocabulary
            logits = F.linear(hidden_states, self.embed_tokens.weight)
            
            return logits
            
        except Exception as e:
            print(f"\nForward pass error:")
            print(f"Input shape: {input_ids.shape}")
            print(f"Device: {input_ids.device}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Error: {str(e)}")
            raise

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None
    ):
        try:
            batch_size = input_ids.shape[0]
            current_length = input_ids.shape[1]
            device = input_ids.device
            
            # Input validation
            if current_length >= self.config.max_position_embeddings:
                raise ValueError(f"Input sequence length {current_length} exceeds maximum position embeddings {self.config.max_position_embeddings}")
            
            # Ensure we don't exceed maximum position embeddings
            max_length = min(max_length, self.config.max_position_embeddings)
            
            # Initialize attention mask if None
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
            
            for _ in range(max_length - current_length):
                # Forward pass
                outputs = self(input_ids, attention_mask)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Append new tokens
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens.unsqueeze(-1))], dim=-1)
                
                # Stop if we've hit special tokens
                if (pad_token_id is not None and (next_tokens == pad_token_id).all()) or \
                   (eos_token_id is not None and (next_tokens == eos_token_id).all()):
                    break
            
            return input_ids
            
        except Exception as e:
            print(f"\nGeneration error:")
            print(f"Input shape: {input_ids.shape}")
            print(f"Device: {input_ids.device}")
            print(f"Error: {str(e)}")
            raise

class TextDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        
        # Load dataset from HuggingFace
        full_dataset = load_dataset(
            config.data.datasets[0].path,
            config.data.datasets[0].subset,
            split=split
        )
        
        # Apply split ratio if less than 1
        if config.data.datasets[0].split_ratio < 1.0:
            num_samples = int(len(full_dataset) * config.data.datasets[0].split_ratio)
            self.dataset = full_dataset.select(range(num_samples))
        else:
            self.dataset = full_dataset
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get text from dataset
        text = self.dataset[idx]["text"]
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.model.max_position_embeddings,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings.input_ids.squeeze(),
            "attention_mask": encodings.attention_mask.squeeze(),
            "labels": encodings.input_ids.squeeze()
        }

class SmolLM2Lightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize the base model
        self.model = SmolLM2(config.model)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)
    
    def training_step(self, batch, batch_idx):
        try:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask", None)
            
            # Ensure tensors are contiguous and on the correct device
            inputs = input_ids[..., :-1].contiguous()
            labels = input_ids[..., 1:].contiguous()
            
            if attention_mask is not None:
                attention_mask = attention_mask[..., :-1].contiguous()
            
            # Forward pass
            logits = self(inputs, attention_mask)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.model.vocab_size),
                labels.view(-1),
                ignore_index=self.config.model.pad_token_id if self.config.model.pad_token_id is not None else -100,
                reduction='mean'
            )
            
            # Detach loss for logging
            loss_value = loss.detach().float()
            
            # Log metrics
            self.log('train_loss', loss_value, prog_bar=True, on_step=True, sync_dist=True)
            
            return loss
            
        except Exception as e:
            print(f"\nTraining step error:")
            print(f"Input shape: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"Device: {input_ids.device if input_ids is not None else 'None'}")
            print(f"Error: {str(e)}")
            raise

    def validation_step(self, batch, batch_idx):
        try:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask", None)
            
            # Ensure tensors are contiguous and on the correct device
            inputs = input_ids[..., :-1].contiguous()
            labels = input_ids[..., 1:].contiguous()
            
            if attention_mask is not None:
                attention_mask = attention_mask[..., :-1].contiguous()
            
            # Forward pass
            logits = self(inputs, attention_mask)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.model.vocab_size),
                labels.view(-1),
                ignore_index=self.config.model.pad_token_id if self.config.model.pad_token_id is not None else -100,
                reduction='mean'
            )
            
            # Detach loss for logging
            loss_value = loss.detach().float()
            
            # Log metrics
            self.log('val_loss', loss_value, prog_bar=True, on_epoch=True, sync_dist=True)
            
            return loss
            
        except Exception as e:
            print(f"\nValidation step error:")
            print(f"Input shape: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"Device: {input_ids.device if input_ids is not None else 'None'}")
            print(f"Error: {str(e)}")
            raise
    
    def configure_optimizers(self):
        # Create optimizer with explicit type conversion
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.config.scheduler.learning_rate),
            weight_decay=float(self.config.optimizer.weight_decay),
            betas=(float(self.config.optimizer.adam_beta1), 
                   float(self.config.optimizer.adam_beta2)),
            eps=float(self.config.optimizer.adam_eps),
        )
        
        # Create scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=float(self.config.scheduler.max_lr),
            total_steps=int(self.config.training.max_steps),
            pct_start=float(self.config.scheduler.pct_start),
            anneal_strategy=self.config.scheduler.anneal_strategy,
            cycle_momentum=bool(self.config.scheduler.cycle_momentum),
            div_factor=float(self.config.scheduler.div_factor),
            final_div_factor=float(self.config.scheduler.final_div_factor),
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def train_dataloader(self):
        dataset = TextDataset(self.config, split="train")
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.loading.num_workers,
            pin_memory=self.config.data.loading.pin_memory,
            persistent_workers=True,
            prefetch_factor=self.config.data.loading.prefetch_factor,
            drop_last=True  # Drop incomplete batches
        )
    
    def val_dataloader(self):
        dataset = TextDataset(self.config, split="validation")
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.loading.num_workers,
            pin_memory=self.config.data.loading.pin_memory,
            persistent_workers=True,
            prefetch_factor=self.config.data.loading.prefetch_factor
        ) 