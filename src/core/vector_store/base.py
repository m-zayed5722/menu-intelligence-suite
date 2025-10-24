"""Base interface for vector stores."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VectorStore(ABC):
    """Abstract base class for vector storage and retrieval."""
    
    @abstractmethod
    def add(self, ids: list[Any], vectors: np.ndarray, metadata: list[dict] | None = None):
        """
        Add vectors to the store.
        
        Args:
            ids: List of document IDs
            vectors: Array of vectors (N x D)
            metadata: Optional metadata for each vector
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
    ) -> list[tuple[Any, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_vector: Query vector (D,)
            k: Number of results
            ef_search: Search parameter (for HNSW-based indexes)
        
        Returns:
            List of (id, distance/similarity) tuples
        """
        pass
    
    @abstractmethod
    def delete(self, ids: list[Any]):
        """Delete vectors by ID."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all vectors."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of vectors."""
        pass
