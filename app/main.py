"""
Application entry point — initialises DB, warms the ML model, mounts routes.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine
from app import models
from app.routes import router
from app.predictor import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    models.Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified.")
    load_model()   # warm the model cache so first request is instant
    logger.info("ML model loaded and ready.")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down.")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time transaction fraud scoring powered by a Random Forest model "
        "with SHAP-based AI explanations. Every decision is logged for compliance audit."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Fraud Detection API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
