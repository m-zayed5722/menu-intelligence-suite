"""Background jobs for MIS."""
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.core.embeddings import encode_texts
from src.core.normalize import normalize_text

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/mis")
engine = create_engine(DB_URL)


def embed_items_job(item_ids: list[int]):
    """Background job to generate embeddings for items."""
    with Session(engine) as db:
        for item_id in item_ids:
            # Fetch item
            query = text("""
                SELECT title_en, title_ar, description
                FROM items
                WHERE item_id = :item_id
            """)
            row = db.execute(query, {"item_id": item_id}).fetchone()
            
            if not row:
                continue
            
            # Combine text
            text_parts = [t for t in row if t]
            text_combined = " ".join(text_parts)
            
            # Normalize
            text_norm = normalize_text(text_combined)
            
            # Generate embedding
            embedding = encode_texts([text_norm], normalize=True)[0]
            embedding_list = embedding.tolist()
            
            # Update item
            update_query = text("""
                UPDATE items
                SET embedding = :embedding, updated_at = NOW()
                WHERE item_id = :item_id
            """)
            db.execute(update_query, {
                "item_id": item_id,
                "embedding": embedding_list,
            })
        
        db.commit()
    
    return len(item_ids)


def rebuild_sparse_index_job():
    """Background job to rebuild BM25 index."""
    from src.api.deps import build_sparse_index
    
    with Session(engine) as db:
        count = build_sparse_index(db)
    
    return count
