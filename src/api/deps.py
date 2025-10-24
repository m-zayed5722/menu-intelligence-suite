"""Dependency injection for FastAPI."""
import os
from functools import lru_cache

import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.api.config import get_settings
from src.core.embeddings import get_model
from src.core.sparse import BM25Retriever
from src.core.tagging import LabelTagger
from src.core.vector_store import PgVectorStore, FAISSVectorStore

settings = get_settings()

# Database engine
engine = create_engine(settings.db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@lru_cache
def get_redis_client():
    """Get Redis client."""
    return redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def get_vector_store():
    """Get vector store instance."""
    if settings.vector_backend == "pgvector":
        return PgVectorStore(dimension=settings.embedding_dim)
    elif settings.vector_backend == "faiss":
        return FAISSVectorStore(dimension=settings.embedding_dim)
    else:
        raise ValueError(f"Unknown vector backend: {settings.vector_backend}")


@lru_cache
def get_bm25_retriever():
    """Get BM25 retriever (lazy loaded)."""
    return BM25Retriever()


@lru_cache
def get_label_tagger():
    """Get label tagger."""
    labels_path = os.path.join("src", "data", "seed", "labels.json")
    if os.path.exists(labels_path):
        return LabelTagger(labels_path)
    else:
        # Create default labels
        tagger = LabelTagger()
        tagger.set_labels("cuisine", [
            "Lebanese", "Saudi", "Turkish", "Indian", "Italian",
            "Seafood", "Dessert", "Fast Food", "Asian", "Mexican"
        ])
        tagger.set_labels("diet", [
            "vegan", "vegetarian", "gluten free", "spicy",
            "halal", "keto", "healthy"
        ])
        return tagger


def build_sparse_index(db: Session):
    """Build BM25 index from database."""
    retriever = get_bm25_retriever()
    
    # Get all items with normalized text
    query = text("""
        SELECT item_id, title_norm || ' ' || COALESCE(desc_norm, '')
        FROM items
        WHERE title_norm IS NOT NULL
        ORDER BY item_id
    """)
    
    results = db.execute(query).fetchall()
    
    if not results:
        return
    
    ids = [row[0] for row in results]
    corpus = [row[1] for row in results]
    
    retriever.fit(corpus, ids)
    
    return len(ids)


def check_db_health() -> bool:
    """Check database connectivity."""
    try:
        with Session(engine) as session:
            session.execute(text("SELECT 1"))
            return True
    except Exception:
        return False


def check_redis_health() -> bool:
    """Check Redis connectivity."""
    try:
        client = get_redis_client()
        client.ping()
        return True
    except Exception:
        return False
