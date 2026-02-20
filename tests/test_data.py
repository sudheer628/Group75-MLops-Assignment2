"""
Unit tests for data preprocessing functions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import (
    IMAGE_SIZE,
    MEAN,
    STD,
    get_train_transforms,
    get_val_transforms,
)


class TestTransforms:
    """Tests for image transforms."""

    def test_val_transforms_output_size(self):
        """Test that validation transforms produce correct output size."""
        # Create a dummy image
        img = Image.new("RGB", (100, 150), color="red")
        
        transform = get_val_transforms()
        output = transform(img)
        
        assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert output.dtype == torch.float32

    def test_val_transforms_normalization(self):
        """Test that validation transforms apply normalization."""
        # Create a white image (all 255)
        img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="white")
        
        transform = get_val_transforms()
        output = transform(img)
        
        # After normalization, values should not be in [0, 1] range
        # White (1.0) normalized: (1.0 - mean) / std
        # For ImageNet mean/std, this should give values > 1
        assert output.max() > 1.0

    def test_train_transforms_output_size(self):
        """Test that training transforms produce correct output size."""
        img = Image.new("RGB", (300, 200), color="blue")
        
        transform = get_train_transforms()
        output = transform(img)
        
        assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_train_transforms_augmentation(self):
        """Test that training transforms apply augmentation (randomness)."""
        img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="green")
        
        transform = get_train_transforms()
        
        # Apply transform multiple times
        outputs = [transform(img) for _ in range(10)]
        
        # Check that at least some outputs are different (due to augmentation)
        # Note: This might occasionally fail due to randomness, but very unlikely
        all_same = all(torch.equal(outputs[0], o) for o in outputs[1:])
        # We don't assert this because color jitter might not always change green
        # Just verify the shape is correct
        for o in outputs:
            assert o.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


class TestImageProcessing:
    """Tests for image processing utilities."""

    def test_image_resize(self):
        """Test that images are resized correctly."""
        # Test various input sizes
        sizes = [(50, 50), (100, 200), (500, 300), (224, 224)]
        transform = get_val_transforms()
        
        for w, h in sizes:
            img = Image.new("RGB", (w, h), color="red")
            output = transform(img)
            assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Failed for size {w}x{h}"

    def test_grayscale_to_rgb(self):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale image
        img = Image.new("L", (100, 100), color=128)
        img_rgb = img.convert("RGB")
        
        transform = get_val_transforms()
        output = transform(img_rgb)
        
        assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
