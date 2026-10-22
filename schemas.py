"""
MalwareGuard AI — Pydantic Request/Response Schemas
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class ClassifyResponse(BaseModel):
    filename: str
    file_size_bytes: int
    label: str = Field(..., description="'benign' or 'malware'")
    label_index: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    classifier_used: str
    is_malware: bool

    model_config = {"json_schema_extra": {
        "example": {
            "filename": "sample.exe",
            "file_size_bytes": 204800,
            "label": "malware",
            "label_index": 1,
            "confidence": 0.97,
            "probabilities": {"benign": 0.03, "malware": 0.97},
            "classifier_used": "random_forest",
            "is_malware": True,
        }
    }}


class BatchClassifyItem(BaseModel):
    filename: str
    label: str
    confidence: float
    is_malware: bool
    error: Optional[str] = None


class BatchClassifyResponse(BaseModel):
    results: list[BatchClassifyItem]
    total: int
    malware_count: int
    benign_count: int
    error_count: int


# ---------------------------------------------------------------------------
# Feature extraction (debug endpoint)
# ---------------------------------------------------------------------------

class FeaturesResponse(BaseModel):
    filename: str
    file_size_bytes: int
    features: dict[str, float]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]
