"""
Pydantic schemas for request/response models.
"""

from pydantic import BaseModel
from typing import Dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = None
