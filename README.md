# Pokemon GAN and Diffusion Models Project

This project implements and compares Generative Adversarial Networks (GANs) and Diffusion Models (DDPM/DDIM) for generating Pokemon images.

## Features

- **GAN Training**: Train a GAN model with Generator and Discriminator networks
- **Diffusion Training**: Train DDPM and DDIM diffusion models
- **Model Comparison**: Compare models using FID, CLIP Score, and other metrics
- **DVC Pipeline**: Reproducible ML pipeline with Data Version Control
- **Docker Support**: Containerized training environment with GPU support
- **Metrics Tracking**: Comprehensive metrics tracking with Weights & Biases

## Project Structure

```
pgm-pokemon-project/
├── src/                          # Source code
│   ├── train_gan.py             # GAN training script
│   ├── train_diff_models.py     # Diffusion training script
│   ├── gan_models.py            # GAN model definitions
│   ├── diff_models.py           # Diffusion model definitions
│   ├── dataset.py               # Dataset loading and preprocessing
│   ├── unified_metrics.py       # Metrics calculation
│   └── visuals.py               # Visualization utilities
├── data/                         # Dataset
│   └── images/                  # Pokemon images
├── models/                       # Trained models
├── output/                       # Generated images and outputs
├── metrics/                      # Metrics and evaluation results
├── checkpoints_diffusion/        # Diffusion model checkpoints
├── params.yaml                   # Training parameters
├── dvc.yaml                      # DVC pipeline definition
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── Makefile                      # Build automation
└── run_pipeline.sh              # Pipeline runner script
```

## Quick Start

### Option 1: Using Docker (Recommended)

1. **Build and run with Docker Compose:**
   ```bash
   # Build the Docker image
   make docker-build
   
   # Run training container
   make docker-run
   
   # Or run Jupyter for development
   make docker-jupyter
   ```

2. **Execute inside container:**
   ```bash
   # Enter the running container
   docker exec -it pokemon-pgm-training bash
   
   # Run the full pipeline
   make repro
   ```

### Option 2: Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup DVC:**
   ```bash
   dvc init --no-scm
   ```

3. **Configure Weights & Biases (optional):**
   ```bash
   export WANDB_API_KEY=your_api_key
   ```

## DVC Pipeline Usage

The project uses DVC for reproducible ML pipelines. Here are the main commands:

### Pipeline Commands

```bash
# Show available commands
make help

# Prepare dataset
make prepare

# Train GAN model only
make train-gan

# Train Diffusion model only  
make train-diff

# Train both models
make train-all

# Evaluate and compare models
make evaluate

# Run entire pipeline
make repro

# Check pipeline status
make status

# View metrics
make metrics

# Clean outputs
make clean
```

### Manual DVC Commands

```bash
# Show pipeline DAG
dvc dag

# Check what changed
dvc status

# Run specific stage
dvc repro train_gan

# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff
```

## Pipeline Stages

1. **prepare_data**: Downloads and prepares the Pokemon dataset from Kaggle
2. **train_gan**: Trains the GAN model (Generator + Discriminator)
3. **train_diffusion**: Trains DDPM and DDIM diffusion models
4. **evaluate_models**: Evaluates and compares all trained models

## Configuration

Edit `params.yaml` to modify training parameters:

```yaml
gan:
  batch_size: 128
  num_epochs: 500
  lr: 0.0002
  # ... other GAN parameters

diffusion:
  training:
    batch_size: 8
    num_epochs: 100
    learning_rate: 1e-4
    # ... other diffusion parameters
```

## Outputs

- **Models**: Saved in `models/` and `checkpoints_diffusion/`
- **Generated Images**: Saved in `output/` and `output_diffusion/`
- **Metrics**: JSON files in `metrics/` directory
- **Visualizations**: Comparison plots and sample grids

## Metrics Tracked

- **FID (Fréchet Inception Distance)**: Measures quality of generated images
- **CLIP Score**: Semantic similarity between generated images and text prompts
- **Originality Score**: Measures diversity of generated samples
- **Training Losses**: Generator and Discriminator losses for GAN

## GPU Support

The project includes GPU support for faster training:

- Docker images use NVIDIA PyTorch base with CUDA
- Automatic GPU detection in training scripts
- NVIDIA Docker runtime required for containerized GPU training

## Development

### Code Formatting
```bash
make format    # Format code with black
make lint      # Run linting checks
```

### Custom Experiments
```bash
# Modify params.yaml for your experiment
# Run with custom config
python src/train_gan.py --config custom_params.yaml
```

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch size in `params.yaml`
2. **Docker GPU Issues**: Ensure NVIDIA Docker runtime is installed
3. **Dataset Download Issues**: Check Kaggle API credentials
4. **DVC Errors**: Run `dvc repro -f` to force reproduction

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Docker & Docker Compose (for containerized training)
- Kaggle API credentials (for dataset download)

## License

This project is for educational purposes. Please respect the original Pokemon dataset licensing terms.
