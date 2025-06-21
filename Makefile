# Makefile for Pokemon GAN/Diffusion Project

.PHONY: help setup build run train-gan train-diff train-all evaluate clean docker-build docker-run

# Default target
help:
	@echo "Pokemon GAN/Diffusion Training Pipeline"
	@echo "======================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup        - Install dependencies and setup environment"
	@echo "  build        - Build Docker image"
	@echo "  run          - Run training container"
	@echo "  prepare      - Download and prepare dataset"
	@echo "  train-gan    - Train GAN model"
	@echo "  train-diff   - Train Diffusion model"
	@echo "  train-all    - Train both models"
	@echo "  evaluate     - Evaluate models"
	@echo "  repro        - Reproduce entire pipeline"
	@echo "  status       - Show DVC status"
	@echo "  metrics      - Show metrics"
	@echo "  clean        - Clean outputs"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run in Docker"
	@echo ""

# Setup environment
setup:
	pip install -r requirements.txt
	dvc init --no-scm -f

# Docker commands
docker-build:
	docker build -t pokemon-training .

docker-run:
	docker-compose up -d pokemon-training

docker-jupyter:
	docker-compose up -d jupyter

# DVC pipeline commands
prepare:
	./run_pipeline.sh prepare

train-gan:
	./run_pipeline.sh train-gan

train-diff:
	./run_pipeline.sh train-diff

train-all:
	./run_pipeline.sh train-all

evaluate:
	./run_pipeline.sh evaluate

repro:
	./run_pipeline.sh repro

status:
	./run_pipeline.sh status

metrics:
	./run_pipeline.sh metrics

clean:
	./run_pipeline.sh clean

# Development commands
install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

lint:
	flake8 src/
	black src/ --check

format:
	black src/
