"""
MalwareGuard AI — REST API Router

Endpoints
---------
POST /classify          Upload a single PE file for classification
POST /classify/batch    Upload multiple PE files at once
POST /features          Return raw extracted feature vector (debug)
GET  /health            Liveness / readiness check
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.models.schemas import (
    BatchClassifyResponse,
    BatchClassifyItem,
    ClassifyResponse,
    FeaturesResponse,
    HealthResponse,
)
from app.services.classifier import classifier_service, LABEL_MAP
from app.services.feature_extractor import extract_features, FEATURE_NAMES

router = APIRouter()

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _save_upload(upload: UploadFile) -> Path:
    """Write an uploaded file to a temp path and return the path."""
    content = await upload.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb} MB.",
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pe")
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return Path(tmp.name), len(content)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """Liveness and readiness check."""
    return HealthResponse(
        status="ok" if classifier_service.is_ready else "degraded",
        version=settings.app_version,
        models_loaded={
            "random_forest": classifier_service._rf is not None,
            "xgboost": classifier_service._xgb is not None,
        },
    )


@router.post("/classify", response_model=ClassifyResponse, tags=["classification"])
async def classify_file(
    file: UploadFile = File(..., description="PE executable file to classify"),
    classifier: str = Form(default="random_forest", description="'random_forest' or 'xgboost'"),
):
    """
    Upload a single PE file and receive a malware/benign prediction.
    """
    if classifier not in ("random_forest", "xgboost"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="classifier must be 'random_forest' or 'xgboost'",
        )
    if not classifier_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded. Run the training script first.",
        )

    tmp_path, file_size = await _save_upload(file)
    try:
        features = extract_features(tmp_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    result = classifier_service.predict(features, classifier=classifier)
    return ClassifyResponse(
        filename=file.filename or "unknown",
        file_size_bytes=file_size,
        classifier_used=classifier,
        is_malware=result["label"] == "malware",
        **result,
    )


@router.post("/classify/batch", response_model=BatchClassifyResponse, tags=["classification"])
async def classify_batch(
    files: List[UploadFile] = File(..., description="List of PE files"),
    classifier: str = Form(default="random_forest"),
):
    """
    Upload multiple PE files and classify each one.
    """
    if not classifier_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded.",
        )

    results: list[BatchClassifyItem] = []
    malware_count = benign_count = error_count = 0

    for upload in files:
        tmp_path = None
        try:
            tmp_path, _ = await _save_upload(upload)
            features = extract_features(tmp_path)
            pred = classifier_service.predict(features, classifier=classifier)
            is_malware = pred["label"] == "malware"
            results.append(BatchClassifyItem(
                filename=upload.filename or "unknown",
                label=pred["label"],
                confidence=pred["confidence"],
                is_malware=is_malware,
            ))
            if is_malware:
                malware_count += 1
            else:
                benign_count += 1
        except Exception as exc:
            results.append(BatchClassifyItem(
                filename=upload.filename or "unknown",
                label="error",
                confidence=0.0,
                is_malware=False,
                error=str(exc),
            ))
            error_count += 1
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return BatchClassifyResponse(
        results=results,
        total=len(results),
        malware_count=malware_count,
        benign_count=benign_count,
        error_count=error_count,
    )


@router.post("/features", response_model=FeaturesResponse, tags=["debug"])
async def extract_features_endpoint(
    file: UploadFile = File(..., description="PE file for feature inspection"),
):
    """
    Return the raw feature vector extracted from a PE file. Useful for
    debugging and model analysis.
    """
    tmp_path, file_size = await _save_upload(file)
    try:
        features = extract_features(tmp_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return FeaturesResponse(
        filename=file.filename or "unknown",
        file_size_bytes=file_size,
        features={name: float(val) for name, val in zip(FEATURE_NAMES, features)},
    )
