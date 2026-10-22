"""
MalwareGuard AI — Application Configuration
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "MalwareGuard AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model paths
    model_dir: Path = BASE_DIR / "data" / "models"
    rf_model_path: Path = model_dir / "random_forest.joblib"
    xgb_model_path: Path = model_dir / "xgboost.joblib"
    scaler_path: Path = model_dir / "scaler.joblib"
    feature_names_path: Path = model_dir / "feature_names.joblib"

    # Classification
    default_classifier: str = Field(default="random_forest", pattern="^(random_forest|xgboost)$")
    confidence_threshold: float = 0.5

    # Upload
    max_file_size_mb: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
