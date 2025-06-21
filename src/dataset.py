import kagglehub
import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class PokemonDatasetLoader:
    def __init__(self, target_folder="Data", image_size=128):
        self.dataset_name = "hlrhegemony/pokemon-image-dataset"
        self.target_folder = target_folder
        self.image_size = image_size
        self.dataset_path = None
        
    def download_and_prepare(self):
        self.dataset_path = kagglehub.dataset_download(self.dataset_name)
        
        self._all_images_to_folder(self.dataset_path, self.target_folder)
        
    def _all_images_to_folder(self, source, target):
        os.makedirs(target, exist_ok=True)
        counter = 0
        for root, _, files in os.walk(source):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in [".jpg", ".jpeg", ".png"]:
                    continue
                
                src_path = os.path.join(root, file)
                parent_folder = os.path.basename(os.path.dirname(src_path))
                base_filename = os.path.splitext(file)[0]
                new_filename = f"{parent_folder}_{base_filename}{ext}"
                dst_path = os.path.join(target, new_filename)
                shutil.copy(src_path, dst_path)
                counter += 1
        print(f"Copied {counter} files into '{target}' folder")
        
    def get_dataset(self):
        return Pokemon_Dataset(self.target_folder, self.image_size)

class Pokemon_Dataset(Dataset):
    def __init__(self, folder_path, image_size):
        super().__init__()
        self.image_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Pokemon dataset")
    parser.add_argument("--prepare", action="store_true", help="Download and prepare dataset")
    parser.add_argument("--target-folder", type=str, default="data/images", help="Target folder for images")
    parser.add_argument("--image-size", type=int, default=128, help="Image size")
    args = parser.parse_args()
    
    if args.prepare:
        loader = PokemonDatasetLoader(target_folder=args.target_folder, image_size=args.image_size)
        loader.download_and_prepare()
        
        dataset = loader.get_dataset()
        print(f"Dataset prepared with {len(dataset)} images.")
    else:
        print("Use --prepare flag to download and prepare the dataset")