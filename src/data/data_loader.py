import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List
from pathlib import Path
from PIL import Image

class DualSatelliteDataset(Dataset):
    """Dataset for dual low-resolution satellite images"""
    
    def __init__(self, data_dir: str, transform=None, target_size=(256, 256)):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        self.image_pairs = self._load_image_pairs()
    
    def _load_image_pairs(self) -> List[Tuple[str, str]]:
        """Load pairs of low-resolution images"""
        lr1_dir = self.data_dir / "low_res_1"
        lr2_dir = self.data_dir / "low_res_2"
        
        pairs = []
        # Look for various image formats
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            for img1_path in lr1_dir.glob(ext):
                img2_path = lr2_dir / img1_path.name
                if img2_path.exists():
                    pairs.append((str(img1_path), str(img2_path)))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img1_path, img2_path = self.image_pairs[idx]
        
        # Load images using PIL
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Resize images
        img1 = img1.resize(self.target_size)
        img2 = img2.resize(self.target_size)
        
        # Convert to numpy arrays and normalize
        img1 = np.array(img1).astype(np.float32) / 255.0
        img2 = np.array(img2).astype(np.float32) / 255.0
        
        # Convert to tensors
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1)
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1)
        
        if self.transform:
            img1_tensor = self.transform(img1_tensor)
            img2_tensor = self.transform(img2_tensor)
        
        return img1_tensor, img2_tensor

def create_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    dataset = DualSatelliteDataset(
        data_dir=config['data']['data_dir'],
        target_size=tuple(config['data']['input_size'])
    )
    
    # Check if dataset is empty
    if len(dataset) == 0:
        raise ValueError(
            f"No image pairs found in {config['data']['data_dir']}. "
            f"Please ensure you have matching images in both low_res_1 and low_res_2 directories."
        )
    
    print(f"Found {len(dataset)} image pairs in dataset")
    
    # Split dataset
    train_size = max(1, int(config['data']['train_split'] * len(dataset)))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config['data']['batch_size'], len(train_dataset)),
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(config['data']['batch_size'], len(val_dataset)),
            shuffle=False,
            num_workers=config['data']['num_workers']
        )
    else:
        val_loader = train_loader  # Use training data for validation if no val data
    
    return train_loader, val_loader
