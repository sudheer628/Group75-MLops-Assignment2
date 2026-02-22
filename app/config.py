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
    MODEL_CONFIG_PATH: str = os.getenv("MODEL_CONFIG_PATH", "models/model_config.json")
    MODEL_ARCH: str = os.getenv("MODEL_ARCH", "simple_cnn")
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "224"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
