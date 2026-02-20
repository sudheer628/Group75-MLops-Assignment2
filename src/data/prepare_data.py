"""
Data preparation module for Cats vs Dogs classification.
Handles dataset download, organization, and train/val/test splitting.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple

import kagglehub


# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def download_dataset() -> Path:
    """
    Download the Cats vs Dogs dataset from Kaggle.
    
    Returns:
        Path to downloaded dataset
    """
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
    print(f"Dataset downloaded to: {path}")
    return Path(path)


def organize_raw_data(source_path: Path) -> None:
    """
    Copy downloaded data to our raw data directory.
    
    Args:
        source_path: Path where kagglehub downloaded the data
    """
    print("Organizing raw data...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find the actual data folders (cats and dogs)
    for item in source_path.rglob("*"):
        if item.is_dir() and item.name.lower() in ["cats", "dogs", "cat", "dog"]:
            dest = RAW_DIR / item.name.lower()
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"  Copied {item.name} -> {dest}")


def get_image_files(directory: Path) -> list:
    """
    Get all image files from a directory.
    
    Args:
        directory: Path to search for images
        
    Returns:
        List of image file paths
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(files)


def create_splits(
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[dict, dict, dict]:
    """
    Split the raw data into train/val/test sets.
    
    Args:
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_files, val_files, test_files) dicts with 'cat' and 'dog' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.seed(seed)
    
    splits = {"train": {}, "val": {}, "test": {}}
    
    for class_name in ["cats", "dogs"]:
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            # Try singular form
            class_dir = RAW_DIR / class_name[:-1]
        
        if not class_dir.exists():
            raise FileNotFoundError(f"Could not find directory for {class_name}")
        
        files = get_image_files(class_dir)
        random.shuffle(files)
        
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits["train"][class_name] = files[:n_train]
        splits["val"][class_name] = files[n_train:n_train + n_val]
        splits["test"][class_name] = files[n_train + n_val:]
        
        print(f"{class_name}: train={len(splits['train'][class_name])}, "
              f"val={len(splits['val'][class_name])}, test={len(splits['test'][class_name])}")
    
    return splits["train"], splits["val"], splits["test"]


def copy_split_files(splits: dict, split_name: str) -> None:
    """
    Copy files for a split to the processed directory.
    
    Args:
        splits: Dict with class names as keys and file lists as values
        split_name: Name of the split (train/val/test)
    """
    split_dir = PROCESSED_DIR / split_name
    
    for class_name, files in splits.items():
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for src_file in files:
            dst_file = class_dir / src_file.name
            shutil.copy2(src_file, dst_file)


def prepare_dataset(
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    skip_download: bool = False
) -> dict:
    """
    Main function to prepare the entire dataset.
    
    Args:
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        test_ratio: Fraction for testing
        seed: Random seed
        skip_download: If True, skip download and use existing raw data
        
    Returns:
        Dict with dataset statistics
    """
    print("=" * 50)
    print("PREPARING CATS VS DOGS DATASET")
    print("=" * 50)
    
    # Step 1: Download if needed
    if not skip_download:
        source_path = download_dataset()
        organize_raw_data(source_path)
    else:
        print("Skipping download, using existing raw data...")
    
    # Step 2: Create splits
    print("\nCreating train/val/test splits...")
    train_files, val_files, test_files = create_splits(
        train_ratio, val_ratio, test_ratio, seed
    )
    
    # Step 3: Copy files to processed directory
    print("\nCopying files to processed directory...")
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    
    copy_split_files(train_files, "train")
    copy_split_files(val_files, "val")
    copy_split_files(test_files, "test")
    
    # Step 4: Calculate statistics
    stats = {
        "train": {k: len(v) for k, v in train_files.items()},
        "val": {k: len(v) for k, v in val_files.items()},
        "test": {k: len(v) for k, v in test_files.items()},
    }
    stats["total"] = {
        "train": sum(stats["train"].values()),
        "val": sum(stats["val"].values()),
        "test": sum(stats["test"].values()),
    }
    
    print("\nDataset preparation complete!")
    print(f"  Train: {stats['total']['train']} images")
    print(f"  Val: {stats['total']['val']} images")
    print(f"  Test: {stats['total']['test']} images")
    
    return stats


if __name__ == "__main__":
    stats = prepare_dataset()
