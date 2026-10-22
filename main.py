"""
MalwareGuard AI — FastAPI Application Factory
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.api.routes import router
from app.services.classifier import classifier_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup; clean up on shutdown."""
    logger.info("Starting MalwareGuard AI v{}", settings.app_version)
    classifier_service.load()
    if classifier_service.is_ready:
        logger.info("Classifier service is ready.")
    else:
        logger.warning(
            "No trained models found. Run `python scripts/train.py` to train."
        )
    yield
    logger.info("Shutting down MalwareGuard AI.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "REST API for static PE malware classification using Random Forest "
            "and XGBoost trained on PE header features, entropy, and import/export tables."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
