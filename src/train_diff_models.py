import argparse
import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import PokemonDatasetLoader
from diff_models import LargeConvDenoiserNetwork, GaussianDiffusion, DeterministicGaussianDiffusion
from visuals import show_images, plot_image_trajectories

def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM and DDIM models on Pokemon images")
    parser.add_argument("--dataroot", type=str, default="Data",
                        help="Path to the dataset directory containing images")
    parser.add_argument("--config", type=str, default="configs/default_diffusion.yaml",
                        help="Path to the YAML configuration file")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config.get("seed", 42)
    print(f"Random Seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(project=config.get("wandb_project", "pokemon-diffusion"),
               name=config.get("wandb_run_name", "ddpm-training"),
               config=config)

    loader = PokemonDatasetLoader(target_folder=args.dataroot,
                                  image_size=config.get("image_size", 128))
    if not os.path.isdir(args.dataroot) or not os.listdir(args.dataroot):
        loader.download_and_prepare()
    dataset = loader.get_dataset()
    dataloader = DataLoader(dataset,
                            batch_size=config.get("batch_size", 8),
                            shuffle=True,
                            num_workers=config.get("num_workers", 6),
                            pin_memory=True)

    # Model and Diffusion
    model = LargeConvDenoiserNetwork(
        in_channels=3,
        out_channels=3,
        channels=config.get("channels", [64, 128, 256, 512, 1024]),
        layers_per_block=config.get("layers_per_block", 2),
        downblock=config.get("downblock", 'ResnetDownsampleBlock2D'),
        upblock=config.get("upblock", 'ResnetUpsampleBlock2D'),
        add_attention=config.get("add_attention", True),
        attention_head_dim=config.get("attention_head_dim", 64)
    ).to(device)
    diffusion = GaussianDiffusion(num_timesteps=config.get("num_timesteps", 1000))
    ddim = DeterministicGaussianDiffusion(num_timesteps=config.get("num_timesteps", 1000))

    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))

    # Checkpoint dir
    save_dir = config.get("checkpoint_dir", "checkpoints_diffusion")
    os.makedirs(save_dir, exist_ok=True)

    # Fixed noise
    sample_batch = config.get("sample_batch_size", 48)
    fixed_noise = torch.randn(sample_batch, 3,
                              config.get("image_size", 32),
                              config.get("image_size", 32),
                              device=device)

    # Training Loop
    loss_history = []
    best_loss = float('inf')
    for epoch in range(1, config.get("epochs", 1) + 1):
        model.train()
        losses = []
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = diffusion.train_losses(model, batch, DEVICE=device)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        loss_history.append(avg_loss)

        # Logging
        print(f"Epoch {epoch}/{config.get('epochs')} - Loss: {avg_loss:.6f}")
        wandb.log({"epoch": epoch, "train_loss": avg_loss})

        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        if epoch % config.get("save_interval", 10) == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pt"))

    # Sampling & Trajectories
    model.eval()
    with torch.no_grad():
        ddpm_samples, ddpm_traj = diffusion.p_sample_loop(
            model, noise=fixed_noise, return_trajectory=True,
            clip=True, quiet=False, device=device)
        ddim_samples, ddim_traj = ddim.p_sample_loop(
            model, noise=fixed_noise, return_trajectory=True,
            clip=True, quiet=False, device=device)

    # Create output directory
    out_dir = config.get("output_dir", "output_diffusion")
    os.makedirs(out_dir, exist_ok=True)

    fig1 = plot_image_trajectories(traj_ddim=ddim_traj, traj_ddpm=ddpm_traj)
    fig1.savefig(os.path.join(out_dir, "trajectories.png"))
    plt.close(fig1)

    fig2 = show_images(generated_ddim=ddim_samples, generated_ddpm=ddpm_samples)
    fig2.savefig(os.path.join(out_dir, "images.png"))
    plt.close(fig2)

    wandb.finish()


if __name__ == "__main__":
    main()
