#!/usr/bin/env python3
"""
Model Evaluation Script for Pokemon GAN vs Diffusion Models
This script loads trained models and compares their performance.
"""

import argparse
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import yaml

from dataset import PokemonDatasetLoader
from gan_models import Generator
from diff_models import LargeConvDenoiserNetwork, GaussianDiffusion, DeterministicGaussianDiffusion
from unified_metrics import calculate_FID, calculate_clip_score, calculate_fid_diffusion, originality_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare GAN vs Diffusion models")
    parser.add_argument("--gan-model", type=str, required=True,
                        help="Path to trained GAN generator model")
    parser.add_argument("--diffusion-model", type=str, required=True,
                        help="Path to trained diffusion model")
    parser.add_argument("--config", type=str, default="params.yaml",
                        help="Path to configuration file")
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Number of samples to generate for evaluation")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for evaluation")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_gan_model(model_path, config, device):
    """Load GAN generator model"""
    gan_config = config.get("gan", {})
    nz = gan_config.get("nz", 100)
    ngf = gan_config.get("ngf", 128)
    nc = gan_config.get("nc", 3)
    
    model = Generator(ngpu=1, nz=nz, ngf=ngf, nc=nc)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, nz


def load_diffusion_model(model_path, config, device):
    """Load diffusion model"""
    diff_config = config.get("diffusion", {})
    model_config = diff_config.get("model", {})
    
    model = LargeConvDenoiserNetwork(
        in_channels=model_config.get("in_channels", 3),
        out_channels=model_config.get("out_channels", 3),
        channels=model_config.get("channels", [64, 128, 256, 512, 1024]),
        layers_per_block=model_config.get("layers_per_block", 2),
        downblock=model_config.get("downblock", 'ResnetDownsampleBlock2D'),
        upblock=model_config.get("upblock", 'ResnetUpsampleBlock2D'),
        add_attention=model_config.get("add_attention", True),
        attention_head_dim=model_config.get("attention_head_dim", 64)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    training_config = diff_config.get("training", {})
    num_timesteps = training_config.get("num_timesteps", 1000)
    
    ddpm = GaussianDiffusion(num_timesteps=num_timesteps)
    ddim = DeterministicGaussianDiffusion(num_timesteps=num_timesteps)
    
    return model, ddpm, ddim


def generate_gan_samples(generator, nz, num_samples, batch_size, device):
    """Generate samples from GAN"""
    samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in range(num_batches):
            current_batch_size = min(batch_size, num_samples - len(samples))
            noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
            batch_samples = generator(noise).cpu()
            samples.append(batch_samples)
    
    return torch.cat(samples, dim=0)[:num_samples]


def generate_diffusion_samples(model, sampler, num_samples, batch_size, image_size, device):
    """Generate samples from diffusion model"""
    samples = []
    
    with torch.no_grad():
        noise = torch.randn(num_samples, 3, image_size, image_size, device=device)
        sample = sampler.p_sample_loop(
            model, noise=noise, return_trajectory=False,
            clip=True, quiet=True, device=device
        )
    
    return torch.from_numpy(sample).to(device)[:num_samples]


def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset - use test split for final evaluation
    loader = PokemonDatasetLoader(target_folder="data/images", image_size=128)
    test_dataset = loader.get_dataset("test")
    
    # Load models
    print("Loading GAN model...")
    gan_generator, nz = load_gan_model(args.gan_model, config, device)
    
    print("Loading Diffusion model...")
    diff_model, ddpm, ddim = load_diffusion_model(args.diffusion_model, config, device)
    
    # Generate samples
    print(f"Generating {args.sample_size} samples from each model...")
    
    gan_fid_samples = generate_gan_samples(
        gan_generator, nz, 250, args.batch_size, device
    )

    gan_samples = generate_gan_samples(
        gan_generator, nz, args.sample_size, args.batch_size, device
    )
    
    ddpm_samples = generate_diffusion_samples(
        diff_model, ddpm, args.sample_size, args.batch_size, 128, device
    )
    
    ddim_samples = generate_diffusion_samples(
        diff_model, ddim, args.sample_size, args.batch_size, 128, device
    )
    
    # Calculate metrics
    print("Calculating metrics...")
    
    # FID scores
    gan_fid = calculate_FID(gan_fid_samples, test_dataset, device, nz)
    ddpm_fid = calculate_fid_diffusion(diff_model, ddpm, test_dataset, timesteps=30, device=device, 
                                       fid_sample_size=args.sample_size, batch_size=args.batch_size)
    ddim_fid = calculate_fid_diffusion(diff_model, ddim, test_dataset, timesteps=30, device=device,
                                       fid_sample_size=args.sample_size, batch_size=args.batch_size)
    
    # CLIP scores
    text_prompt = "Image of a Pokemon character"
    gan_clip = calculate_clip_score(gan_samples, text_prompt, device)
    ddpm_clip = calculate_clip_score(ddpm_samples, text_prompt, device)
    ddim_clip = calculate_clip_score(ddim_samples, text_prompt, device)
    
    # Originality scores
    gan_originality = originality_score(gan_generator, test_dataset, device, nz)
    
    # Compile results
    results = {
        "gan": {
            "fid": float(gan_fid),
            "clip_score": float(gan_clip),
            "originality": float(gan_originality)
        },
        "ddpm": {
            "fid": float(ddpm_fid),
            "clip_score": float(ddpm_clip)
        },
        "ddim": {
            "fid": float(ddim_fid),
            "clip_score": float(ddim_clip)
        },
        "comparison": {
            "best_fid": "gan" if gan_fid < min(ddpm_fid, ddim_fid) else ("ddpm" if ddpm_fid < ddim_fid else "ddim"),
            "best_clip": "gan" if gan_clip > max(ddpm_clip, ddim_clip) else ("ddpm" if ddpm_clip > ddim_clip else "ddim")
        }
    }
    
    # Save results
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/comparison_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(f"GAN - FID: {gan_fid:.4f}, CLIP: {gan_clip:.4f}, Originality: {gan_originality:.4f}")
    print(f"DDPM - FID: {ddpm_fid:.4f}, CLIP: {ddpm_clip:.4f}")
    print(f"DDIM - FID: {ddim_fid:.4f}, CLIP: {ddim_clip:.4f}")
    print("\nBest Models:")
    print(f"FID (lower is better): {results['comparison']['best_fid']}")
    print(f"CLIP Score (higher is better): {results['comparison']['best_clip']}")
    print("="*50)


if __name__ == "__main__":
    main()
