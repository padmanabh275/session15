import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    type: str = "custom"
    name: str = "smollm2_transformer"
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M"
    vocab_size: int = 49152
    hidden_size: int = 576
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    num_hidden_layers: int = 30
    intermediate_size: int = 1536
    hidden_act: str = "gelu"
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    max_length: int = 512

    def __post_init__(self):
        # Ensure numeric values are proper types
        self.vocab_size = int(self.vocab_size)
        self.hidden_size = int(self.hidden_size)
        self.num_attention_heads = int(self.num_attention_heads)
        self.num_key_value_heads = int(self.num_key_value_heads)
        self.num_hidden_layers = int(self.num_hidden_layers)
        self.intermediate_size = int(self.intermediate_size)
        self.max_position_embeddings = int(self.max_position_embeddings)
        self.initializer_range = float(self.initializer_range)
        self.rms_norm_eps = float(self.rms_norm_eps)
        self.max_length = int(self.max_length)

@dataclass
class OptimizerConfig:
    type: str = "adamW"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    torch_adam_is_fused: bool = True
    clip_grad: float = 1.0
    accumulate_grad_in_fp32: bool = True

    def __post_init__(self):
        # Ensure numeric values are proper floats
        self.weight_decay = float(self.weight_decay)
        self.adam_beta1 = float(self.adam_beta1)
        self.adam_beta2 = float(self.adam_beta2)
        self.adam_eps = float(self.adam_eps)
        self.clip_grad = float(self.clip_grad)

@dataclass
class SchedulerConfig:
    type: str = "one_cycle"
    learning_rate: float = 0.003
    warmup_steps: int = 100
    max_lr: float = 0.003
    pct_start: float = 0.02
    anneal_strategy: str = "cos"
    cycle_momentum: bool = False
    div_factor: float = 25.0
    final_div_factor: float = 1000.0

@dataclass
class TrainingConfig:
    output_dir: str = "./results"
    batch_size: int = 2
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    sequence_length: int = 512
    learning_rate: float = 0.003
    max_steps: int = 5050
    first_phase_steps: int = 5000
    second_phase_steps: int = 50
    sample_frequency: int = 500
    second_phase_sample_frequency: int = 10
    logging_dir: str = "./logs"
    logging_steps: int = 1
    save_steps: int = 500
    checkpoint_dir: str = "checkpoints"
    sample_prompt: str = "Explain what machine learning is:"
    max_generate_length: int = 100

@dataclass
class HardwareConfig:
    precision: str = "16-mixed"
    accelerator: str = "gpu"
    devices: int = 1
    strategy: str = "auto"
    gradient_clip: float = 1.0

@dataclass
class DatasetConfig:
    name: str
    path: str
    subset: str
    weight: float
    split_ratio: float = 1.0  # Default to using full dataset

@dataclass
class DataLoadingConfig:
    num_workers: int = 2
    batch_size: int = 32
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

@dataclass
class DataConfig:
    datasets: List[DatasetConfig] = field(default_factory=list)
    loading: DataLoadingConfig = field(default_factory=DataLoadingConfig)

class SmolLM2Config:
    def __init__(self, config_path: str = None):
        self.model = ModelConfig()
        self.optimizer = OptimizerConfig()
        self.scheduler = SchedulerConfig()
        self.training = TrainingConfig()
        self.hardware = HardwareConfig()
        self.data = DataConfig()
        
        # Default dataset configuration
        self.data.datasets = [
            DatasetConfig(
                name="wikitext",
                path="wikitext",
                subset="wikitext-2-raw-v1",
                weight=1.0
            )
        ]
        
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Update configurations from yaml
        if 'model' in config_dict:
            for k, v in config_dict['model'].items():
                setattr(self.model, k, v)
                
        if 'optimizer' in config_dict:
            for k, v in config_dict['optimizer'].items():
                setattr(self.optimizer, k, v)
                
        if 'scheduler' in config_dict:
            for k, v in config_dict['scheduler'].items():
                setattr(self.scheduler, k, v)
                
        if 'training' in config_dict:
            for k, v in config_dict['training'].items():
                setattr(self.training, k, v)
                
        if 'hardware' in config_dict:
            for k, v in config_dict['hardware'].items():
                setattr(self.hardware, k, v)
                
        if 'data' in config_dict:
            for k, v in config_dict['data'].items():
                if k == 'datasets':
                    for dataset in v:
                        self.data.datasets.append(DatasetConfig(**dataset))
                elif k == 'loading':
                    for k, v in config_dict['data']['loading'].items():
                        setattr(self.data.loading, k, v)
    
    def save_to_yaml(self, config_path: str):
        config_dict = {
            'model': self.model.__dict__,
            'optimizer': self.optimizer.__dict__,
            'scheduler': self.scheduler.__dict__,
            'training': self.training.__dict__,
            'hardware': self.hardware.__dict__,
            'data': self.data.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def __repr__(self):
        return f"SmolLM2Config(\n" \
               f"  model={self.model}\n" \
               f"  optimizer={self.optimizer}\n" \
               f"  scheduler={self.scheduler}\n" \
               f"  training={self.training}\n" \
               f"  hardware={self.hardware}\n" \
               f"  data={self.data}\n" \
               f")" 