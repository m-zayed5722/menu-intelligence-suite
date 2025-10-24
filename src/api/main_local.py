"""
Simplified API main for local demo (no database).
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import get_settings
from src.api.deps_local import check_health, get_items_db
from src.api.schemas import HealthResponse

# Import routers
from src.api.routers import search_local as search
from src.api.routers import tagging, dedup, recommend, ingest, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup."""
    print("🚀 Starting Menu Intelligence Suite (Local Demo)")
    health = check_health()
    print(f"📊 Status: {health}")
    yield
    print("👋 Shutting down")


app = FastAPI(
    title="Menu Intelligence Suite",
    description="Multilingual semantic search for food delivery",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(tagging.router, prefix="/api", tags=["tagging"])
app.include_router(dedup.router, prefix="/api", tags=["deduplication"])
app.include_router(recommend.router, prefix="/api", tags=["recommendations"])
app.include_router(ingest.router, prefix="/api", tags=["ingest"])
app.include_router(metrics.router, prefix="/api", tags=["metrics"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Menu Intelligence Suite API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    health_status = check_health()
    return HealthResponse(
        status=health_status["status"],
        database_connected=True,  # In-memory
        redis_connected=True,  # Mock
        vector_store_ready=health_status["vector_count"] > 0,
    )
