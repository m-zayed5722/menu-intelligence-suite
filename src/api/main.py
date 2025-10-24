"""Main FastAPI application."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import deps
from src.api.config import get_settings
from src.api.logging_conf import setup_logging, get_logger
from src.api.routers import dedup, ingest, metrics, recommend, search, tagging

settings = get_settings()
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Menu Intelligence Suite API")
    
    # Warm up model
    from src.core.embeddings import get_model
    model = get_model()
    logger.info(f"Model loaded: {settings.model_name}")
    
    # Check dependencies
    db_ok = deps.check_db_health()
    redis_ok = deps.check_redis_health()
    logger.info(f"Database: {'OK' if db_ok else 'FAIL'}, Redis: {'OK' if redis_ok else 'FAIL'}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="Menu Intelligence Suite",
    description="Multilingual semantic search and intelligence for food delivery",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(tagging.router, prefix="/tag", tags=["Tagging"])
app.include_router(dedup.router, prefix="/dedup", tags=["Deduplication"])
app.include_router(recommend.router, prefix="/recommend", tags=["Recommendations"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])


@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint."""
    return {
        "service": "Menu Intelligence Suite",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    from src.api.schemas import HealthResponse
    
    db_ok = deps.check_db_health()
    redis_ok = deps.check_redis_health()
    
    try:
        from src.core.embeddings import get_model
        get_model()
        model_ok = True
    except Exception:
        model_ok = False
    
    vector_store = deps.get_vector_store()
    try:
        count = vector_store.count()
    except Exception:
        count = 0
    
    status = "ok" if (db_ok and model_ok) else "degraded"
    
    return HealthResponse(
        status=status,
        db_connected=db_ok,
        redis_connected=redis_ok,
        model_loaded=model_ok,
        vector_store_count=count,
    )
