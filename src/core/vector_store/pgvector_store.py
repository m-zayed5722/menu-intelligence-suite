"""pgvector-based vector store implementation."""
import os
from typing import Any

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.core.vector_store.base import VectorStore

# Get DB connection from environment
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/mis")


class PgVectorStore(VectorStore):
    """Vector store using PostgreSQL with pgvector extension."""
    
    def __init__(self, table_name: str = "items", dimension: int = 384):
        self.table_name = table_name
        self.dimension = dimension
        self.engine = create_engine(DB_URL)
    
    def add(self, ids: list[Any], vectors: np.ndarray, metadata: list[dict] | None = None):
        """Add vectors to database (updates existing items)."""
        if len(ids) != len(vectors):
            raise ValueError("IDs and vectors must have same length")
        
        with Session(self.engine) as session:
            for i, (item_id, vec) in enumerate(zip(ids, vectors)):
                # Convert numpy array to list for pgvector
                vec_list = vec.tolist()
                
                # Update item with embedding
                query = text(f"""
                    UPDATE {self.table_name}
                    SET embedding = :embedding, updated_at = NOW()
                    WHERE item_id = :item_id
                """)
                session.execute(query, {"item_id": item_id, "embedding": vec_list})
            
            session.commit()
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
    ) -> list[tuple[Any, float]]:
        """
        Search using cosine distance.
        
        Note: pgvector uses <=> for cosine distance (0 = identical, 2 = opposite)
        We convert to similarity score (1 - distance/2) for consistency.
        """
        vec_list = query_vector.tolist()
        
        with Session(self.engine) as session:
            # Set ef_search if provided (for IVFFlat index tuning)
            if ef_search:
                session.execute(text(f"SET ivfflat.probes = {min(ef_search, 100)}"))
            
            # Query with cosine distance
            query = text(f"""
                SELECT item_id, embedding <=> :query_vec AS distance
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :query_vec
                LIMIT :k
            """)
            
            results = session.execute(
                query,
                {"query_vec": vec_list, "k": k}
            ).fetchall()
        
        # Convert distance to similarity (higher is better)
        # Cosine distance in pgvector: 0 = same, 2 = opposite
        # Convert to similarity: 1 - (distance / 2)
        return [(row[0], 1.0 - (row[1] / 2.0)) for row in results]
    
    def delete(self, ids: list[Any]):
        """Delete embeddings (set to NULL)."""
        with Session(self.engine) as session:
            query = text(f"""
                UPDATE {self.table_name}
                SET embedding = NULL
                WHERE item_id = ANY(:ids)
            """)
            session.execute(query, {"ids": ids})
            session.commit()
    
    def clear(self):
        """Clear all embeddings."""
        with Session(self.engine) as session:
            query = text(f"UPDATE {self.table_name} SET embedding = NULL")
            session.execute(query)
            session.commit()
    
    def count(self) -> int:
        """Count items with embeddings."""
        with Session(self.engine) as session:
            query = text(f"""
                SELECT COUNT(*) FROM {self.table_name}
                WHERE embedding IS NOT NULL
            """)
            result = session.execute(query).scalar()
            return result or 0
    
    def get_all_embeddings(self) -> tuple[list[int], np.ndarray]:
        """Get all embeddings for building sparse index."""
        with Session(self.engine) as session:
            query = text(f"""
                SELECT item_id, embedding
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
                ORDER BY item_id
            """)
            results = session.execute(query).fetchall()
        
        if not results:
            return [], np.array([])
        
        ids = [row[0] for row in results]
        # Convert pgvector arrays to numpy
        embeddings = np.array([row[1] for row in results], dtype=np.float32)
        
        return ids, embeddings
