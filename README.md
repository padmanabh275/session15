# DeepSeek-Inspired Language Model Implementation

A PyTorch implementation of a language model inspired by DeepSeek architecture, featuring Multi-Query Attention with Lossless Attention (MLHA) and Mixture of Experts (MoE) with load balancing.

## Architecture Features

### Multi-Query Attention with Lossless Attention (MLHA)
- 8 attention heads with 2 key-value heads
- Efficient attention computation with reduced memory footprint
- Rotary positional embeddings
- Pre-norm architecture with RMSNorm

### Mixture of Experts (MoE)
- 4 expert networks with load balancing
- Expert capacity factor of 2
- MoE layers at positions [2, 4, 6]
- Loss-less load balancing for optimal expert utilization

### Model Configuration
- Hidden size: 512
- Intermediate size: 2048
- Number of layers: 8
- Vocabulary size: 50,257 (GPT-2 tokenizer)
- Maximum sequence length: 512
- SiLU activation function

## Installation

1. Clone the repository:
bash
git clone (https://github.com/padmanabh275/session15)
cd deepseek-implementation

2. Create and activate conda
bash
conda env create -f environment.yml
conda activate torch_env

## Training

1. Configure training parameters in `config.yaml`:
- Model architecture settings
- Training hyperparameters
- Hardware configuration
- Dataset settings

2. Start training:
bash
python train.py

### Training Logs can be found in ([metrices.csv]) along with the ([hparams.yaml]
### Training Features
- Mixed precision training (16-bit)
- Gradient accumulation
- One-cycle learning rate scheduler
- Regular checkpointing
- CSV logging of metrics and generations
- Generation samples during training

## Model Inference

1. Using the API:
bash
Start the model service
docker-compose up model-service
Start the client interface
docker-compose up client-service

2. Access the web interface at `http://localhost:7860`

## Project Structure
├── model.py # Core model implementation (MLHA, MoE) <br/>
├── moe.py # Mixture of Experts implementation <br/>
├── config.py # Configuration management <br/>
├── train_script.py # Training loop and monitoring <br/>
├── api.py # REST API for inference <br/>
├── client_app.py # Gradio interface <br/>
└── docker/ # Containerization <br/>

## Training Data
- Uses WikiText-103 dataset
- Configurable dataset size (default: 1% for faster iteration)
- Dynamic batching with efficient data loading

## Hardware Requirements
- CUDA-capable GPU (12GB+ VRAM recommended)
- 16GB+ RAM
- Python 3.10+
- PyTorch 2.0+

## Monitoring and Logging
- Training metrics logged to CSV
- Generation samples at configurable intervals
- Hardware utilization tracking
- Checkpoint management

## Docker Support
- Separate containers for model and client
- GPU support with NVIDIA Container Toolkit
- Health monitoring
- Easy deployment

## References
- DeepSeek architecture
- Mixture of Experts papers
- Multi-Query Attention research

## License
MIT

## Contributing
Contributions welcome! Please check the issues page.

