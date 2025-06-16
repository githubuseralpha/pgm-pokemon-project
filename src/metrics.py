from pytorch_fid import fid_score

def calculate_FID(real_images, fake_images, device):
    
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)
    
    # Calculate FID score
    fid_value = fid_score.calculate_fid_given_paths(
        (real_images, fake_images), 
        batch_size=50, 
        device=device, 
        dims=2048
    )
    
    return fid_value