import math
import torch
import timm
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm

import clip
import torchvision.transforms as T
from diff_models import InceptionV3


def embed_images(images, model, device):
    """
    Function to transform images to latent space using a model

    :images: images to transform
    :model: model for transforming images
    :device: device for computations (CPU or GPU)
    """ 
    print(f"Embedding images on device: {device}")
    model.eval()
    batch_size = 32
    images = [i for i in images]
    images = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    latents = []
    for batch in images:
        batch = torch.stack(batch, dim=0)
        batch = batch.to(device)
        with torch.no_grad():
            latent_batch = model(batch).cpu().numpy()
            latents.append(latent_batch)
    latents = np.concatenate(latents, axis=0)
    print(f"Latent shape: {latents.shape}")
    return latents


def fit_n_dimensional_gaussian(latents):
    """Fit n-dimensional Gaussian to latent representations"""
    mu = np.mean(latents, axis=0)
    sigma = np.cov(latents, rowvar=False)
    return mu, sigma


def wasserstein_distance_sqrtm(mu1, sigma1, mu2, sigma2):
    """
    Alternative implementation using scipy's sqrtm (used in diffusion models)
    """
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean))


def calculate_FID(real_images, fake_images, device, use_timm=True):
    """
    Calculate FID score for GAN-style evaluation
    
    :real_images: tensor of real images
    :fake_images: tensor of generated images
    :device: computation device
    :use_timm: whether to use timm's inception model (default for GANs)
    """
    inception_net = InceptionV3(normalize_input=False).to(device)

    real_images = embed_images(real_images[:1000], inception_net, device)
    fake_images = embed_images(fake_images[:1000], inception_net, device)

    print("Created embeddings for real and fake images")

    mu_real, sigma_real = fit_n_dimensional_gaussian(real_images)
    mu_fake, sigma_fake = fit_n_dimensional_gaussian(fake_images)

    fid = wasserstein_distance_sqrtm(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid.item() if isinstance(fid, np.ndarray) else fid


def calculate_fid_diffusion(model, sampler, image_ds, timesteps, device, fid_sample_size=1000, batch_size=250):
    """
    Calculate FID score for diffusion models
    
    :model: diffusion model
    :sampler: sampling algorithm (DDPM/DDIM)
    :image_ds: real image dataset
    :timesteps: number of timesteps for sampling
    :device: computation device
    :fid_sample_size: number of samples for FID calculation
    :batch_size: batch size for processing
    """
    iterations = math.ceil(fid_sample_size / batch_size)

    inception_net = InceptionV3(normalize_input=False).to(device)

    @torch.no_grad()
    def embed_real_data(ds):
        real_latents = []
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        for x in tqdm(dl, desc="Embedding real data", total=iterations):
            if len(real_latents) * batch_size >= fid_sample_size:
                break
            x = x.to(device)
            z = inception_net(x)
            real_latents.append(z)
        real_latents = torch.cat(real_latents, dim=0)[:fid_sample_size].cpu().numpy()
        return real_latents

    @torch.no_grad()
    def embed_generated_data(num_timesteps):
        fake_latents = []
        for _ in tqdm(range(iterations), desc=f"Generating ({sampler.__class__.__name__})", total=iterations):
            if len(fake_latents) * batch_size >= fid_sample_size:
                break
            samples = sampler.p_sample_loop(
                model,
                noise=torch.randn(batch_size, 3, 128, 128, device=device),
                num_inference_steps=num_timesteps,
                return_trajectory=False,
                clip=True,
                quiet=True,
                device=device
            )
            samples = torch.from_numpy(samples).to(device)
            z = inception_net(samples)
            fake_latents.append(z)
        fake_latents = torch.cat(fake_latents, dim=0)[:fid_sample_size].cpu().numpy()
        return fake_latents

    real_latents = embed_real_data(image_ds)
    real_mu, real_sigma = fit_n_dimensional_gaussian(real_latents)

    fake_latents = embed_generated_data(timesteps)
    fake_mu, fake_sigma = fit_n_dimensional_gaussian(fake_latents)
    
    fid = wasserstein_distance_sqrtm(real_mu, real_sigma, fake_mu, fake_sigma)
    return fid


def originality_score(real_images, fake_images, device):
    """
    Calculate originality score comparing generated images to real images
    
    :real_images: tensor of real images
    :fake_images: tensor of generated images (subset for efficiency)
    :device: computation device
    """
    model = timm.create_model('inception_v3', pretrained=True)
    model.to(device)
    real_latents = embed_images(real_images, model, device)  # size 1500 x 1000
    fake_latents = embed_images(fake_images, model, device)  # size 200 x 1000

    real_latents = real_latents / np.linalg.norm(real_latents, axis=1, keepdims=True)
    fake_latents = fake_latents / np.linalg.norm(fake_latents, axis=1, keepdims=True)

    # Calculate cosine similarity and convert to cosine distance
    cosine_similarities = np.dot(fake_latents, real_latents.T)
    cosine_distances = 1 - cosine_similarities
        
    min_distances = np.min(cosine_distances, axis=1)
    return np.mean(min_distances) / 2  # Normalize to [0, 1] range


def calculate_clip_score(images_tensor, text_prompt, device):
    """
    Calculate CLIP score between images and text prompt
    
    :images_tensor: tensor of images to evaluate
    :text_prompt: text description to compare against
    :device: computation device
    """
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    # Process text
    text = clip.tokenize([text_prompt]).to(device)

    # Normalization pipeline
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # convert from [-1,1] to [0,1]
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    if isinstance(images_tensor, np.ndarray):
        images_tensor = torch.from_numpy(images_tensor).to(device)

    images_tensor = images_tensor.to(device)
    imgs = (images_tensor + 1) / 2
    imgs = torch.clamp(imgs, 0, 1)

    # Process images in batches
    processed_imgs = []
    for img in imgs:
        processed_imgs.append(transform(img).unsqueeze(0))
    processed_imgs = torch.cat(processed_imgs, dim=0).to(device)

    # Get embeddings
    with torch.no_grad():
        image_features = model.encode_image(processed_imgs)
        text_features = model.encode_text(text)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Calculate cosine similarity
        similarity = (image_features @ text_features.T).squeeze(1)

    clip_score = similarity.mean().item()
    return clip_score
