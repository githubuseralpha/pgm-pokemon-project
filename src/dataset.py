import kagglehub
import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random

class PokemonDatasetLoader:
    def __init__(self, target_folder="Data", image_size=128, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        self.dataset_name = "hlrhegemony/pokemon-image-dataset"
        self.target_folder = target_folder
        self.image_size = image_size
        self.dataset_path = None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Ensure ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Train, val, test ratios must sum to 1.0"
        
    def download_and_prepare(self):
        self.dataset_path = kagglehub.dataset_download(self.dataset_name)
        
        self._all_images_to_folder(self.dataset_path, self.target_folder)
        self._create_train_val_test_splits()
        
    def _create_train_val_test_splits(self):
        """Create train/val/test splits based on fixed seed for reproducibility"""
        # Set seed for reproducible splits
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Get all image files
        all_images = [f for f in os.listdir(self.target_folder) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        # Shuffle the list with fixed seed
        random.shuffle(all_images)
        
        # Calculate split indices
        total_images = len(all_images)
        train_end = int(total_images * self.train_ratio)
        val_end = train_end + int(total_images * self.val_ratio)
        
        # Split the data
        train_images = all_images[:train_end]
        val_images = all_images[train_end:val_end]
        test_images = all_images[val_end:]
        
        # Create subdirectories
        train_dir = os.path.join(self.target_folder, "train")
        val_dir = os.path.join(self.target_folder, "val")
        test_dir = os.path.join(self.target_folder, "test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Move images to respective folders
        for img in train_images:
            src = os.path.join(self.target_folder, img)
            dst = os.path.join(train_dir, img)
            if os.path.exists(src):
                shutil.move(src, dst)
                
        for img in val_images:
            src = os.path.join(self.target_folder, img)
            dst = os.path.join(val_dir, img)
            if os.path.exists(src):
                shutil.move(src, dst)
                
        for img in test_images:
            src = os.path.join(self.target_folder, img)
            dst = os.path.join(test_dir, img)
            if os.path.exists(src):
                shutil.move(src, dst)
        
        print(f"Created splits:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Val: {len(val_images)} images")
        print(f"  Test: {len(test_images)} images")
        
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
        
    def get_dataset(self, split="train"):
        """Get dataset for specific split (train/val/test)"""
        split_folder = os.path.join(self.target_folder, split)
        if not os.path.exists(split_folder):
            raise ValueError(f"Split '{split}' not found. Available splits should be in {self.target_folder}")
        return Pokemon_Dataset(split_folder, self.image_size)
    
    def get_all_datasets(self):
        """Get all three datasets (train, val, test)"""
        return {
            'train': self.get_dataset('train'),
            'val': self.get_dataset('val'), 
            'test': self.get_dataset('test')
        }

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
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of test data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    args = parser.parse_args()
    
    if args.prepare:
        loader = PokemonDatasetLoader(
            target_folder=args.target_folder, 
            image_size=args.image_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        loader.download_and_prepare()
        
        datasets = loader.get_all_datasets()
        print(f"Dataset prepared with splits:")
        for split, dataset in datasets.items():
            print(f"  {split}: {len(dataset)} images")
    else:
        print("Use --prepare flag to download and prepare the dataset")