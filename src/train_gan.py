import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb
import yaml
import json
import numpy as np

from dataset import PokemonDatasetLoader
from gan_models import Generator, Discriminator, weights_init
from unified_metrics import calculate_FID, originality_score, calculate_clip_score

seed = 42

print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

wandb.init(
    project="pgm-pokemon-project",
    name="gan-training",
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN on Pokemon images")
    parser.add_argument("--dataroot", type=str, default="/workspace/pgm-pokemon-project/Data",
                        help="Path to the dataset root directory")
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to the configuration file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def evaluate_model(netG, fixed_noise, dataset, device):
    netG.eval()
    batch_size = 32
    batches = [fixed_noise[i:i + batch_size] for i in range(0, len(fixed_noise), batch_size)]
    fake_images = []
    for batch in batches:
        with torch.no_grad():
            fake_batch = netG(batch.to(device)).detach().cpu()
            fake_images.append(fake_batch)
    fake_images = torch.cat(fake_images, dim=0)

    print(f"Generated {fake_images.shape[0]} fake images.")
    # Calculate FID
    real_images = [sample for sample in dataset]
    real_images = torch.stack(real_images).to(device)
    fid_score = calculate_FID(real_images, fake_images, device)
    originality_score_ = originality_score(real_images, fake_images[:200], device)
    print(f"FID Score: {fid_score}, Originality Score: {originality_score_}")
    wandb.log({"FID Score": fid_score, "Originality Score": originality_score_})
    
    return fid_score, originality_score_


def train_epoch(    
    dataloader,
    netG,
    netD,
    criterion,
    optimizerG,
    optimizerD,
    device,
    nz,
    real_label=1.0,
    fake_label=0.0,
    num_epochs=100,
    iters=0,
    fixed_noise=None,
    G_losses=None,
    D_losses=None,
    img_list=None,
    epoch=0,
):
    loss_D_sum = 0.0
    loss_G_sum = 0.0
    D_x_sum = 0.0
    D_G_z1_sum = 0.0
    D_G_z2_sum = 0.0
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()

        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)

        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        loss_D_sum += errD.item()
        loss_G_sum += errG.item()
        D_x_sum += D_x
        D_G_z1_sum += D_G_z1
        D_G_z2_sum += D_G_z2

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        iters += 1

    return loss_D_sum / len(dataloader), loss_G_sum / len(dataloader), D_x_sum / len(dataloader), \
           D_G_z1_sum / len(dataloader), D_G_z2_sum / len(dataloader), iters


def train(
    dataroot = "/workspace/pgm-pokemon-project/Data",
    workers = 2,
    batch_size = 128,
    image_size = 128,
    nc = 3,
    nz = 100,
    ngf = 64,
    ndf = 64,
    num_epochs = 300,
    lr = 0.0002,
    beta1 = 0.5,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()

    netG = Generator(ngpu, nz, ngf, nc).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)

    netD = Discriminator(ngpu, nc, ndf).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    transforms_ = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Use PokemonDatasetLoader with train/val splits
    dataset_loader = PokemonDatasetLoader(target_folder=dataroot, image_size=image_size)
    
    # Check if train split exists, if not download and prepare with splits
    train_path = os.path.join(dataroot, "train")
    if not os.path.exists(train_path) or not os.listdir(train_path):
        print("Train dataset not found, downloading and preparing with splits...")
        dataset_loader.download_and_prepare()
    
    # Get train and validation datasets
    train_dataset = dataset_loader.get_dataset("train")
    val_dataset = dataset_loader.get_dataset("val")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False
    )

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        loss_D, loss_G, D_x, D_G_z1, D_G_z2, iters = train_epoch(
            train_dataloader,
            netG,
            netD,
            criterion,
            optimizerG,
            optimizerD,
            device,
            nz,
            real_label,
            fake_label,
            num_epochs,
            iters,
            fixed_noise,
            G_losses,
            D_losses,
            img_list,
            epoch
        )
        print(f"[{epoch}/{num_epochs}] Loss_D: {loss_D:.4f} Loss_G: {loss_G:.4f} "
              f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")
        wandb.log({
            "epoch": epoch,
            "loss_D": loss_D,
            "loss_G": loss_G,
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2
        })

        # save generated images
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        if not os.path.exists("output"):
            os.makedirs("output")
        vutils.save_image(fake, f"output/fake_samples_epoch_{epoch}.png", normalize=True)
        
        # Evaluate on validation set every 10 epochs
        if (epoch + 1) % 10 == 0:
            noise_val = torch.randn(len(val_dataset), nz, 1, 1, device=device)
            val_fid, val_originality = evaluate_model(netG, noise_val, val_dataset, device)
            print(f"Validation - FID: {val_fid:.4f}, Originality: {val_originality:.4f}")
            wandb.log({
                "val_fid": val_fid,
                "val_originality": val_originality
            })

    
    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save(netG.state_dict(), "models/gan_generator.pth")
    torch.save(netD.state_dict(), "models/gan_discriminator.pth")
    
    # Calculate final metrics on validation set
    noise_val = torch.randn(len(val_dataset), nz, 1, 1, device=device)
    final_fid, final_originality = evaluate_model(netG, noise_val, val_dataset, device)
    
    # Calculate CLIP score with Pokemon-related text prompt
    print("Calculating CLIP score...")
    with torch.no_grad():
        final_fake_images = netG(fixed_noise)
    final_clip_score = calculate_clip_score(final_fake_images, "Image of a Pokemon character", device)
    print(f"Final CLIP Score: {final_clip_score:.4f}")
    
    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics = {
        "final_fid": f"{final_fid:.4f}",
        "final_originality": f"{final_originality:.4f}",
        "final_clip_score": f"{final_clip_score:.4f}",
        "final_loss_G": f"{G_losses[-1]:.4f}" if G_losses else 0,
        "final_loss_D": f"{D_losses[-1]:.4f}" if D_losses else 0,
        "num_epochs": num_epochs
    }
    
    with open("metrics/gan_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    wandb.log({
        "final_fid": final_fid,
        "final_originality": final_originality,
        "final_clip_score": final_clip_score
    })
    
    print("Training complete.")
    print(f"Final FID: {final_fid:.4f}")
    print(f"Final Originality: {final_originality:.4f}")
    print(f"Final CLIP Score: {final_clip_score:.4f}")


def main():
    args = parse_args()
    config = load_config(args.config)["gan"]

    wandb.config.update(config)

    train(
        dataroot=args.dataroot,
        workers=config.get("workers", 2),
        batch_size=config.get("batch_size", 128),
        image_size=config.get("image_size", 128),
        nc=config.get("nc", 3),
        nz=config.get("nz", 100),
        ngf=config.get("ngf", 64),
        ndf=config.get("ndf", 64),
        num_epochs=config.get("num_epochs", 300),
        lr=config.get("lr", 0.0002),
        beta1=config.get("beta1", 0.5)
    )

if __name__ == "__main__":
    main()
    wandb.finish()
