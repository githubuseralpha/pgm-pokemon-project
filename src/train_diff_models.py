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

from unified_metrics import calculate_clip_score, calculate_fid_diffusion
from dataset import PokemonDatasetLoader
from diff_models import LargeConvDenoiserNetwork, GaussianDiffusion, DeterministicGaussianDiffusion
from visuals import show_images, plot_image_trajectories

def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM and DDIM models on Pokemon images")
    parser.add_argument("--dataroot", type=str, default="Data",
                        help="Path to the dataset directory containing images")
    parser.add_argument("--config", type=str, default="params.yaml",
                        help="Path to the YAML configuration file")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    args = parse_args()
    config = load_config(args.config)["diffusion"]

    seed = config.get("seed", 42)
    print(f"Random Seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    # Training Loop
    loss_history = []
    best_loss = float('inf')
    for epoch in range(1, config.get("epochs", 50) + 1):
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

        print(f"Epoch {epoch}/{config.get('epochs')} - Loss: {avg_loss:.6f}")
        wandb.log({"epoch": epoch, "train_loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        if epoch % config.get("save_interval", 10) == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pt"))

    sample_batch = config.get("sample_batch_size", 48)
    fixed_noise = torch.randn(sample_batch, 3,
                              config.get("image_size", 128),
                              config.get("image_size", 128),
                              device=device)
    
    model.eval()
    with torch.no_grad():
        ddpm_samples, ddpm_traj = diffusion.p_sample_loop(
            model, noise=fixed_noise, return_trajectory=True,
            clip=True, quiet=False, device=device)
        ddim_samples, ddim_traj = ddim.p_sample_loop(
            model, noise=fixed_noise, return_trajectory=True,
            clip=True, quiet=False, device=device)
        
    fid_ddpm = calculate_fid_diffusion(
        model=model,
        sampler=diffusion,
        image_ds=dataset,
        timesteps=30,
        device=device,
        fid_sample_size=1000,
        batch_size=8
    )

    fid_ddim = calculate_fid_diffusion(
        model=model,
        sampler=ddim,
        image_ds=dataset,
        timesteps=30,
        device=device,
        fid_sample_size=1000,
        batch_size=8
    )

    print(f"FID DDPM: {fid_ddpm:.4f}")
    print(f"FID DDIM: {fid_ddim:.4f}")

    wandb.log({
        "fid_ddpm": fid_ddpm,
        "fid_ddim": fid_ddim
    })
        
    text_prompt = "Image of a Pokemon character"

    clip_score_ddpm = calculate_clip_score(ddpm_samples, text_prompt, device)
    clip_score_ddim = calculate_clip_score(ddim_samples, text_prompt, device)

    print(f"ClipScore DDPM: {clip_score_ddpm:.4f}")
    print(f"ClipScore DDIM: {clip_score_ddim:.4f}")

    wandb.log({
        "clip_score_ddpm": clip_score_ddpm,
        "clip_score_ddim": clip_score_ddim,
    })
    
    out_dir = config.get("output_dir", "output_diffusion")
    os.makedirs(out_dir, exist_ok=True)

    generated_dir = os.path.join(out_dir, "generated_images")
    os.makedirs(generated_dir, exist_ok=True)

    fig1 = plot_image_trajectories(traj_ddim=ddim_traj, traj_ddpm=ddpm_traj)
    fig1.savefig(os.path.join(out_dir, "trajectories.png"))
    plt.close(fig1)

    fig2 = show_images(generated_ddim=ddim_samples, generated_ddpm=ddpm_samples)
    fig2.savefig(os.path.join(out_dir, "images.png"))
    plt.close(fig2)

    for i, (ddpm_img, ddim_img) in enumerate(zip(ddpm_samples, ddim_samples)):
        if isinstance(ddpm_img, np.ndarray):
            ddpm_img = torch.from_numpy(ddpm_img)
        if isinstance(ddim_img, np.ndarray):
            ddim_img = torch.from_numpy(ddim_img)

        vutils.save_image(ddpm_img, os.path.join(generated_dir, f"ddpm_sample_{i}.png"),
                        normalize=True, value_range=(-1, 1))
        vutils.save_image(ddim_img, os.path.join(generated_dir, f"ddim_sample_{i}.png"),
                        normalize=True, value_range=(-1, 1))

    wandb.finish()


if __name__ == "__main__":
    main()
