# Use NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace/pgm-pokemon-project

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install DVC
RUN pip install dvc[all]

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data models output metrics checkpoints_diffusion output_diffusion

# Set environment variables
ENV PYTHONPATH=/workspace/pgm-pokemon-project/src:$PYTHONPATH
ENV WANDB_DIR=/workspace/pgm-pokemon-project/wandb
ENV TORCH_HOME=/workspace/pgm-pokemon-project/.torch

# Expose port for Jupyter notebook (optional)
EXPOSE 8888

# Default command
CMD ["bash"]
