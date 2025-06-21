import math
import torch
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
from diff_models import InceptionV3
import numpy as np
import torch
from tqdm import tqdm
import math
from scipy.linalg import sqrtm

import clip
from PIL import Image
import torchvision.transforms as T

def calculate_clip_score(images_tensor, text_prompt, device):
    
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    # Przetworzenie tekstu
    text = clip.tokenize([text_prompt]).to(device)

    # Normalizacja
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # zamiana z [-1,1] do [0,1]
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    if isinstance(images_tensor, np.ndarray):
        images_tensor = torch.from_numpy(images_tensor).to(device)

    images_tensor = images_tensor.to(device)
    imgs = (images_tensor + 1) / 2
    imgs = torch.clamp(imgs, 0, 1)

    # Przetwarzanie po batchu
    processed_imgs = []
    for img in imgs:
        processed_imgs.append(transform(img).unsqueeze(0))
    processed_imgs = torch.cat(processed_imgs, dim=0).to(device)

    # embeddingi
    with torch.no_grad():
        image_features = model.encode_image(processed_imgs)
        text_features = model.encode_text(text)

        # Normalizacja
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine similarity
        similarity = (image_features @ text_features.T).squeeze(1)

    clip_score = similarity.mean().item()
    return clip_score

def calculate_fid(model, sampler, image_ds, timesteps, device, fid_sample_size=1000, batch_size=250):
    device = device
    
    iterations = math.ceil(fid_sample_size / batch_size)

    inception_net = InceptionV3(normalize_input=False).to(device)

    @torch.no_grad()
    def embed_real_data(ds):
        real_latents = []
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        for x in tqdm(dl, desc="Osadzanie (real)", total=iterations):
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
        for _ in tqdm(range(iterations), desc=f"Generowanie ({sampler.__class__.__name__})", total=iterations):
            if len(fake_latents) * batch_size >= fid_sample_size:
                break
            samples = sampler.p_sample_loop(
                model,
                noise=torch.randn(batch_size, 3, 32, 32, device=device),
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

    def fit_n_dimensional_gaussian(latents):
        mu = np.mean(latents, axis=0)
        sigma = np.cov(latents, rowvar=False)
        return mu, sigma

    def wasserstein_distance(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean))

    real_latents = embed_real_data(image_ds)
    real_mu, real_sigma = fit_n_dimensional_gaussian(real_latents)

    fake_latents = embed_generated_data(timesteps)
    fake_mu, fake_sigma = fit_n_dimensional_gaussian(fake_latents)
    fid = wasserstein_distance(real_mu, real_sigma, fake_mu, fake_sigma)

    return fid
