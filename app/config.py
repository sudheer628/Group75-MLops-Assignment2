"""
Application configuration.
"""

import os


class Settings:
    """Application settings loaded from environment variables."""
    
    APP_NAME: str = "Cats vs Dogs Classifier"
    APP_VERSION: str = "1.0.0"
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_model.pt")
    CLASS_MAPPING_PATH: str = os.getenv("CLASS_MAPPING_PATH", "models/class_mapping.json")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
