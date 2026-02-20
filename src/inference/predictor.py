"""
Inference module for Cats vs Dogs classifier.
Handles model loading and prediction.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import get_val_transforms, IDX_TO_CLASS
from src.models.cnn import create_model


class CatsDogsPredictor:
    """
    Predictor class for Cats vs Dogs classification.
    Loads model once and provides prediction interface.
    """
    
    def __init__(
        self,
        model_path: str = "models/best_model.pt",
        class_mapping_path: str = "models/class_mapping.json",
    ):
        self.model_path = Path(model_path)
        self.class_mapping_path = Path(class_mapping_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_val_transforms()
        self.class_mapping = IDX_TO_CLASS.copy()
        self._loaded = False
    
    def load(self) -> None:
        """Load model and class mapping."""
        if self._loaded:
            return
        
        print(f"Loading model from {self.model_path}...")
        
        # Load class mapping if exists
        if self.class_mapping_path.exists():
            with open(self.class_mapping_path) as f:
                self.class_mapping = {int(k): v for k, v in json.load(f).items()}
        
        # Create and load model
        self.model = create_model(num_classes=2)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._loaded = True
        print("Model loaded successfully")
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def predict(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Dict:
        """
        Predict class for a single image.
        
        Args:
            image: Path to image or PIL Image
            
        Returns:
            Dict with prediction results
        """
        if not self._loaded:
            self.load()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a path or PIL Image")
        
        # Transform and predict
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
        
        return {
            "prediction": self.class_mapping[predicted_idx],
            "confidence": confidence,
            "probabilities": {
                self.class_mapping[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            },
        }
    
    def predict_batch(
        self,
        images: list,
    ) -> list:
        """Predict classes for multiple images."""
        return [self.predict(img) for img in images]


# Global predictor instance (for API use)
_predictor = None


def get_predictor() -> CatsDogsPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = CatsDogsPredictor()
    return _predictor
