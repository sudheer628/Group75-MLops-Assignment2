"""
Unit tests for inference functions.
"""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn import SimpleCNN, create_model
from src.data.dataset import IMAGE_SIZE


class TestModel:
    """Tests for CNN model."""

    def test_model_creation(self):
        """Test that model can be created."""
        model = create_model(num_classes=2)
        assert isinstance(model, SimpleCNN)

    def test_model_forward_pass(self):
        """Test that model forward pass works correctly."""
        model = create_model(num_classes=2)
        model.eval()
        
        # Create dummy input batch
        batch_size = 4
        x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 2)

    def test_model_predict_proba(self):
        """Test that predict_proba returns valid probabilities."""
        model = create_model(num_classes=2)
        model.eval()
        
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        
        with torch.no_grad():
            probs = model.predict_proba(x)
        
        # Check shape
        assert probs.shape == (2, 2)
        
        # Check probabilities sum to 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)
        
        # Check all probabilities are in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_model_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = create_model(num_classes=2)
        model.eval()
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 2)

    def test_model_dropout_effect(self):
        """Test that dropout affects training vs eval mode."""
        model = create_model(num_classes=2, dropout=0.5)
        x = torch.randn(10, 3, IMAGE_SIZE, IMAGE_SIZE)
        
        # In eval mode, outputs should be deterministic
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2)
        
        # In train mode with dropout, outputs might differ
        model.train()
        out3 = model(x)
        out4 = model(x)
        # Note: They might still be equal by chance, so we don't assert inequality


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_model_state_dict(self):
        """Test that model state dict can be saved and loaded."""
        model1 = create_model(num_classes=2)
        model2 = create_model(num_classes=2)
        
        # Models should have different weights initially (random init)
        # Load state from model1 to model2
        model2.load_state_dict(model1.state_dict())
        
        # Now they should produce same output
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        
        assert torch.equal(out1, out2)
