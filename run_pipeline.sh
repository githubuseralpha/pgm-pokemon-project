#!/bin/bash

# DVC Training Pipeline Runner
# This script provides easy commands to run the Pokemon GAN/Diffusion training pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  prepare     - Download and prepare the Pokemon dataset"
    echo "  train-gan   - Train the GAN model"
    echo "  train-diff  - Train the Diffusion model"
    echo "  train-all   - Train both GAN and Diffusion models"
    echo "  evaluate    - Evaluate and compare models"
    echo "  repro       - Reproduce the entire pipeline"
    echo "  status      - Show DVC pipeline status"
    echo "  dag         - Show DVC pipeline DAG"
    echo "  metrics     - Show current metrics"
    echo "  clean       - Clean all outputs and cache"
    echo ""
}

function log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

case "${1:-}" in
    prepare)
        log_info "Preparing Pokemon dataset..."
        dvc repro prepare_data
        ;;
    train-gan)
        log_info "Training GAN model..."
        dvc repro train_gan
        ;;
    train-diff)
        log_info "Training Diffusion model..."
        dvc repro train_diffusion
        ;;
    train-all)
        log_info "Training both GAN and Diffusion models..."
        dvc repro train_gan train_diffusion
        ;;
    evaluate)
        log_info "Evaluating models..."
        dvc repro evaluate_models
        ;;
    repro)
        log_info "Reproducing entire pipeline..."
        dvc repro
        ;;
    status)
        log_info "DVC Pipeline Status:"
        dvc status
        ;;
    dag)
        log_info "DVC Pipeline DAG:"
        dvc dag
        ;;
    metrics)
        log_info "Current Metrics:"
        dvc metrics show
        ;;
    clean)
        log_warn "Cleaning all outputs and cache..."
        dvc cache dir
        rm -rf .dvc/cache
        rm -rf models/ output/ metrics/ checkpoints_diffusion/ output_diffusion/
        log_info "Clean completed"
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
