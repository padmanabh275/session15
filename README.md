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

### Training Logs 
- [Training Metrics (metrics.csv)](https://github.com/padmanabh275/session15/blob/master/metrics.csv)
- [Hyperparameters (hparams.yaml)](https://github.com/padmanabh275/session15/blob/master/hparams.yaml))


=== Analysis for version_2 ===

Training Statistics:
Total steps completed: 799
Initial loss: 11.0467
Final loss: 0.0310
Best loss: 0.0000
Average loss: 1.0651

Loss plot saved as: training_loss_version_2.png

Loss at key steps:
Step     0: 11.0467
Step   199: 0.8805
Step   399: 1.3597
Step   599: 1.2415
Step   798: 0.0310

=== Analysis for version_3 ===

Training Statistics:
Total steps completed: 10000
Initial loss: 11.1667
Final loss: 0.6163
Best loss: 0.0000
Average loss: 0.8022

Loss plot saved as: training_loss_version_3.png

Loss at key steps:
Step     0: 11.1667
Step  2500: 1.8252
Step  5000: 0.4584
Step  7500: 0.9362
Step  9999: 0.6163


### Best Outputs

Analyzing version_2:
Found 39 generation samples

Latest generations:

Step 1:
Prompt: Write a story about a space explorer:
Generated: Write a story about a space explorer: and the 18 century . The first time of the video , and the game and the first game , but the first time in the end of the first can not found on the game , and the season , the United States , and a " .

--------------------------------------------------------------------------------

Step 0:
Prompt: Explain quantum computing:
Generated: Explain quantum computing: is in the film .

--------------------------------------------------------------------------------

Step 1:
Prompt: Write a story about a space explorer:
Generated: Write a story about a space explorer: and the 18 century . The first time of the video , and the game and the first game , but the first time in the end of the first can not found on the game , and the season , the United States , and a " .

--------------------------------------------------------------------------------

Analyzing version_3:
Found 500 generation samples

Latest generations:

Step 2:
Prompt: Describe the process of photosynthesis:
Generated: Describe the process of photosynthesis:en 's wife of the world . They had been given an example and in the " in the team , she was " . " . In his wife , and the " the second on the band 's original game . He was the show 's , and the game 's death , and "

--------------------------------------------------------------------------------

Step 3:
Prompt: What is the theory of relativity?
Generated: What is the theory of relativity? , which is a single in the most of the original , and an area of the player or an attempt to be a man , and are a single . The series has a single , the same year , and the game 's only by the game 's .

--------------------------------------------------------------------------------

Step 4:
Prompt: How does machine learning work?
Generated: How does machine learning work? that " , "

--------------------------------------------------------------------------------
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

