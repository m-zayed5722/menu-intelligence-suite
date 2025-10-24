"""Configuration management for MIS API."""
import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "mis"
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_url: str = "postgresql://postgres:postgres@localhost:5432/mis"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: str = "redis://localhost:6379/0"
    
    # Model
    model_name: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    
    # Vector Store
    vector_backend: str = "pgvector"  # or "faiss"
    ann_lists: int = 100
    ef_search: int = 50
    
    # Hybrid Search
    hybrid_alpha: float = 0.4
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    # Workers
    worker_concurrency: int = 2
    
    # App (Streamlit)
    app_host: str = "0.0.0.0"
    app_port: int = 8501
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
