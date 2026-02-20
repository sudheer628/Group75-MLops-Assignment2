"""
PyTorch Dataset class for Cats vs Dogs classification.
Handles image loading, transforms, and augmentation.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Image settings
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# Class mapping
CLASS_TO_IDX = {"cats": 0, "dogs": 1}
IDX_TO_CLASS = {0: "cat", 1: "dog"}


def get_train_transforms() -> transforms.Compose:
    """Get transforms for training data with augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Get transforms for validation/test data (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class CatsDogsDataset(Dataset):
    """
    PyTorch Dataset for Cats vs Dogs classification.
    
    Args:
        root_dir: Path to the split directory (e.g., data/processed/train)
        transform: Optional transform to apply to images
        max_samples: If set, limit dataset to this many samples
    """
    
    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all image paths and labels
        self.samples = []
        
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
                    self.samples.append((img_path, class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        # Limit samples if requested
        if max_samples and len(self.samples) > max_samples:
            import random
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> dict:
        """Get count of samples per class."""
        counts = {name: 0 for name in CLASS_TO_IDX.keys()}
        for _, label in self.samples:
            class_name = [k for k, v in CLASS_TO_IDX.items() if v == label][0]
            counts[class_name] += 1
        return counts


def create_dataloaders(
    data_dir: str | Path = "data/processed",
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: int = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, val, and test dataloaders.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        max_samples: If set, limit each dataset to this many samples
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    train_dataset = CatsDogsDataset(
        data_dir / "train",
        transform=get_train_transforms(),
        max_samples=max_samples,
    )
    val_dataset = CatsDogsDataset(
        data_dir / "val",
        transform=get_val_transforms(),
        max_samples=max_samples // 4 if max_samples else None,
    )
    test_dataset = CatsDogsDataset(
        data_dir / "test",
        transform=get_val_transforms(),
        max_samples=max_samples // 4 if max_samples else None,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
