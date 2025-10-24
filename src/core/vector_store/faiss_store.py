"""FAISS-based vector store implementation."""
from typing import Any

import faiss
import numpy as np

from src.core.vector_store.base import VectorStore


class FAISSVectorStore(VectorStore):
    """Vector store using FAISS for ANN search."""
    
    def __init__(self, dimension: int = 384, index_type: str = "IVFFlat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index: faiss.Index | None = None
        self.id_map: list[Any] = []  # Maps FAISS index -> item_id
        self.metadata: list[dict] = []
        self._is_trained = False
    
    def _create_index(self, n_vectors: int):
        """Create FAISS index based on type."""
        if self.index_type == "IVFFlat":
            # IVFFlat with 100 clusters
            quantizer = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine for normalized)
            nlist = min(100, max(1, n_vectors // 39))  # Heuristic: sqrt(N)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "Flat":
            # Exact search
            self.index = faiss.IndexFlatIP(self.dimension)
            self._is_trained = True  # Flat index doesn't need training
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add(self, ids: list[Any], vectors: np.ndarray, metadata: list[dict] | None = None):
        """Add vectors to FAISS index."""
        if len(ids) != len(vectors):
            raise ValueError("IDs and vectors must have same length")
        
        # Ensure vectors are normalized for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-9)
        
        # Create index if needed
        if self.index is None:
            self._create_index(len(vectors))
        
        # Train index if needed (IVFFlat requires training)
        if not self._is_trained and isinstance(self.index, faiss.IndexIVFFlat):
            if len(vectors) >= 39:  # Minimum for training
                self.index.train(vectors)
                self._is_trained = True
            else:
                # Fall back to flat index for small datasets
                self.index = faiss.IndexFlatIP(self.dimension)
                self._is_trained = True
        
        # Add vectors
        self.index.add(vectors)
        self.id_map.extend(ids)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(ids))
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
    ) -> list[tuple[Any, float]]:
        """Search for nearest neighbors."""
        if self.index is None or len(self.id_map) == 0:
            return []
        
        # Normalize query vector
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # Set search parameters
        if ef_search and isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = min(ef_search, self.index.nlist)
        
        # Search
        k = min(k, len(self.id_map))
        distances, indices = self.index.search(query_vector, k)
        
        # Map back to IDs (distances are already similarity scores for IndexFlatIP)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.id_map):
                results.append((self.id_map[idx], float(dist)))
        
        return results
    
    def delete(self, ids: list[Any]):
        """Delete not supported in FAISS - would require rebuild."""
        raise NotImplementedError("FAISS delete requires index rebuild")
    
    def clear(self):
        """Clear all vectors."""
        self.index = None
        self.id_map = []
        self.metadata = []
        self._is_trained = False
    
    def count(self) -> int:
        """Get number of vectors."""
        return len(self.id_map)
    
    def save(self, path: str):
        """Save index to disk."""
        if self.index:
            faiss.write_index(self.index, path)
    
    def load(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(path)
        self._is_trained = True
