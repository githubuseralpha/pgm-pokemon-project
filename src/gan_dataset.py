import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from typing import Optional, Callable


class PokemonDataset(Dataset):    
    def __init__(self, root: str, subfolders: list[str], 
                 extensions: Optional[list[str]] = None,
                 transform: Optional[Callable] = None):
        self.root = root
        self.subfolders = subfolders
        self.transform = transform
        
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.extensions = [ext.lower() for ext in extensions]
        
        self.images = []
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(root, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, filename)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(filename)
                        if ext.lower() in self.extensions:
                            try:
                                image = Image.open(file_path).convert('RGB')
                                self.images.append(image)
                            except Exception as e:
                                print(f"Warning: Could not load image {file_path}: {e}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def get_class_names(self):
        return self.subfolders
    
    def get_file_path(self, idx):
        return self.file_paths[idx]


def train_val_test_split(root: str, train_ratio: float = 0.8, 
                        val_ratio: float = 0.1, 
                        test_ratio: float = 0.1,
                        seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)

    subfolders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    np.random.shuffle(subfolders)
    n_total = len(subfolders)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_folders = subfolders[:n_train]
    val_folders = subfolders[n_train:n_train + n_val]
    test_folders = subfolders[n_train + n_val:]
    return train_folders, val_folders, test_folders


if __name__ == "__main__":
    root = '/workspace/pgm-pokemon-project/data/images'
    train_folders, val_folders, test_folders = train_val_test_split(root)
    
    train_dataset = PokemonDataset(root, train_folders)
    val_dataset = PokemonDataset(root, val_folders)
    test_dataset = PokemonDataset(root, test_folders)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
