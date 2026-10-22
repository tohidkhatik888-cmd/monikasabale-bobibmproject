"""
MalwareGuard AI — ML Classifier Service

Wraps both a Random Forest and an XGBoost classifier.
Handles loading pre-trained models from disk and running inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.services.feature_extractor import FEATURE_NAMES, NUM_FEATURES

logger = logging.getLogger(__name__)

# Label mapping
LABEL_MAP = {0: "benign", 1: "malware"}
LABEL_MAP_INV = {"benign": 0, "malware": 1}

# Malware family sub-labels (multi-class, optional future use)
FAMILY_MAP = {
    0: "benign",
    1: "trojan",
    2: "ransomware",
    3: "spyware",
    4: "adware",
    5: "worm",
    6: "rootkit",
    7: "backdoor",
}


class ClassifierService:
    """
    Loads trained models from ``data/models/`` and exposes a unified
    ``predict`` interface.
    """

    def __init__(self) -> None:
        self._rf = None
        self._xgb = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: list[str] = FEATURE_NAMES
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all available models from disk (called at startup)."""
        model_dir: Path = settings.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        if settings.rf_model_path.exists():
            self._rf = joblib.load(settings.rf_model_path)
            logger.info("Random Forest model loaded from %s", settings.rf_model_path)
        else:
            logger.warning("Random Forest model not found at %s", settings.rf_model_path)

        if settings.xgb_model_path.exists():
            self._xgb = joblib.load(settings.xgb_model_path)
            logger.info("XGBoost model loaded from %s", settings.xgb_model_path)
        else:
            logger.warning("XGBoost model not found at %s", settings.xgb_model_path)

        if settings.scaler_path.exists():
            self._scaler = joblib.load(settings.scaler_path)
            logger.info("Scaler loaded from %s", settings.scaler_path)

        if settings.feature_names_path.exists():
            self._feature_names = joblib.load(settings.feature_names_path)

        self._loaded = True

    @property
    def is_ready(self) -> bool:
        return self._loaded and (self._rf is not None or self._xgb is not None)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        features: np.ndarray,
        classifier: str = "random_forest",
    ) -> dict:
        """
        Run inference on a pre-extracted feature vector.

        Parameters
        ----------
        features : np.ndarray, shape (NUM_FEATURES,)
        classifier : "random_forest" | "xgboost"

        Returns
        -------
        dict with keys: label, label_index, confidence, probabilities
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        model = self._rf if classifier == "random_forest" else self._xgb
        if model is None:
            raise RuntimeError(f"Classifier '{classifier}' is not available.")

        x = features.reshape(1, -1)
        if self._scaler is not None:
            x = self._scaler.transform(x)

        proba = model.predict_proba(x)[0]
        label_idx = int(np.argmax(proba))
        label = LABEL_MAP.get(label_idx, "unknown")
        confidence = float(proba[label_idx])

        return {
            "label": label,
            "label_index": label_idx,
            "confidence": confidence,
            "probabilities": {
                LABEL_MAP.get(i, str(i)): float(p)
                for i, p in enumerate(proba)
            },
        }

    def predict_batch(
        self,
        features_matrix: np.ndarray,
        classifier: str = "random_forest",
    ) -> list[dict]:
        return [
            self.predict(row, classifier=classifier)
            for row in features_matrix
        ]


# Module-level singleton — imported by FastAPI lifespan
classifier_service = ClassifierService()
