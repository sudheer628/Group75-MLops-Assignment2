"""
CNN model for Cats vs Dogs classification.
Efficient architecture optimized for CPU training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleCNN(nn.Module):
    """
    Efficient CNN for binary image classification.
    Optimized for faster training on CPU while maintaining good accuracy.
    
    Input: 224x224 RGB images
    Output: 2 class logits (cat, dog)
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        
        # Conv block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112
        
        # Conv block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56
        
        # Conv block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28
        
        # Conv block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 28 -> 14
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv blocks with ReLU and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class TransferLearningCNN(nn.Module):
    """
    Transfer-learning wrapper over torchvision backbones.

    Supported architectures:
    - mobilenet_v3_small
    - efficientnet_b0
    """

    def __init__(
        self,
        architecture: str = "mobilenet_v3_small",
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.architecture = architecture

        if architecture == "mobilenet_v3_small":
            weights = (
                models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            backbone = models.mobilenet_v3_small(weights=weights)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        elif architecture == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                "Choose from: simple_cnn, mobilenet_v3_small, efficientnet_b0"
            )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.architecture in {"mobilenet_v3_small", "efficientnet_b0"}:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(
    num_classes: int = 2,
    dropout: float = 0.5,
    architecture: str = "simple_cnn",
    pretrained: bool = True,
) -> nn.Module:
    """Create and return a classification model."""
    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    return TransferLearningCNN(
        architecture=architecture,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )
