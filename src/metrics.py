import torch
import timm
import numpy as np

def embed_images(images, model, device):
    """
    Funkcja przekształcająca obrazy do przestrzeni latentnej przy użyciu modelu

    :images: obrazy do przekształcenia
    :model: model do przekształcania obrazów
    :device: urządzenie, na którym wykonywane są obliczenia (CPU lub GPU)
    """ 
    print(device)
    model.eval()
    batch_size = 32
    images = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    latents = []
    for batch in images:
        batch = batch.to(device)
        with torch.no_grad():
            latent_batch = model(batch).cpu().numpy()
            latents.append(latent_batch)
    latents = np.concatenate(latents, axis=0)
    return latents


def fit_n_dimensional_gaussian(latents):
    mu, sigma = None, None
    mu = np.mean(latents, axis=0)
    sigma = np.cov(latents, rowvar=False)
    return mu, sigma


def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    """
    Funkcja wyznaczająca odległość Wassersteina pomiędzy dwoma rozkładami Gaussa o parametrach mu i sigma

    :mu1: średnia pierwszego rozkładu
    :sigma1: macierz kowariancji pierwszego rozkładu
    :mu2: średnia drugiego rozkładu
    :sigma2: macierz kowariancji drugiego rozkładu
    """
    diff = mu1 - mu2
    euclidean_sq = np.dot(diff, diff)

    trace_cov1 = np.trace(sigma1)
    trace_cov2 = np.trace(sigma2)

    M = sigma1 @ sigma2

    eigvals = np.linalg.eigvals(M)
    real_eigvals = np.real(eigvals)
    nonneg_eigvals = np.maximum(real_eigvals, 0)

    trace_sqrt = np.sum(np.sqrt(nonneg_eigvals))

    result = euclidean_sq + trace_cov1 + trace_cov2 - 2 * trace_sqrt
    return result


def calculate_FID(real_images, fake_images, device):
    model = timm.create_model('inception_v3', pretrained=True)
    model.to(device)
    real_images = embed_images(real_images, model, device)
    fake_images = embed_images(fake_images, model, device)

    print("created embeddings for real and fake images")

    mu_real, sigma_real = fit_n_dimensional_gaussian(real_images)
    mu_fake, sigma_fake = fit_n_dimensional_gaussian(fake_images)

    fid = wasserstein_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid.item() if isinstance(fid, np.ndarray) else fid


def originality_score(real_images, fake_images, device):
    model = timm.create_model('inception_v3', pretrained=True)
    model.to(device)
    real_latents = embed_images(real_images, model, device) # size 1500 x 1000
    fake_latents = embed_images(fake_images, model, device) # size 200 x 1000

    real_latents = real_latents / np.linalg.norm(real_latents, axis=1, keepdims=True)
    fake_latents = fake_latents / np.linalg.norm(fake_latents, axis=1, keepdims=True)

    # Calculate cosine similarity and convert to cosine distance
    cosine_similarities = np.dot(fake_latents, real_latents.T)
    cosine_distances = 1 - cosine_similarities
        
    min_distances = np.min(cosine_distances, axis=1)
    return np.mean(min_distances) / 2 # Normalize to [0, 1] range
