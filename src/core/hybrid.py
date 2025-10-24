"""Hybrid retrieval combining sparse and dense methods."""
import numpy as np

from src.core.utils import min_max_normalize


def combine_scores(
    sparse_scores: list[tuple[any, float]],
    dense_scores: list[tuple[any, float]],
    alpha: float = 0.4,
) -> list[tuple[any, float]]:
    """
    Combine sparse and dense scores using weighted sum.
    
    Args:
        sparse_scores: List of (id, score) from sparse retrieval
        dense_scores: List of (id, score) from dense retrieval
        alpha: Weight for sparse (0-1); dense gets (1-alpha)
    
    Returns:
        Combined and sorted list of (id, score)
    """
    # Normalize both score lists
    sparse_dict = {id_: score for id_, score in sparse_scores}
    dense_dict = {id_: score for id_, score in dense_scores}
    
    # Get all unique IDs
    all_ids = set(sparse_dict.keys()) | set(dense_dict.keys())
    
    # Normalize scores
    if sparse_scores:
        sparse_vals = list(sparse_dict.values())
        sparse_normalized = min_max_normalize(sparse_vals)
        sparse_dict = dict(zip(sparse_dict.keys(), sparse_normalized))
    
    if dense_scores:
        dense_vals = list(dense_dict.values())
        dense_normalized = min_max_normalize(dense_vals)
        dense_dict = dict(zip(dense_dict.keys(), dense_normalized))
    
    # Combine scores
    combined = {}
    for id_ in all_ids:
        s = sparse_dict.get(id_, 0.0)
        d = dense_dict.get(id_, 0.0)
        combined[id_] = alpha * s + (1 - alpha) * d
    
    # Sort by combined score
    result = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    
    return result


def hybrid_search(
    query: str,
    sparse_retriever,
    dense_retriever,
    k: int = 10,
    alpha: float = 0.4,
    sparse_k: int = 100,
    dense_k: int = 100,
) -> list[tuple[any, float]]:
    """
    Perform hybrid search.
    
    Args:
        query: Search query
        sparse_retriever: Sparse retrieval object with search() method
        dense_retriever: Dense retrieval object with search() method
        k: Final number of results
        alpha: Sparse weight
        sparse_k: Number of sparse candidates
        dense_k: Number of dense candidates
    
    Returns:
        Top-k results after hybrid scoring
    """
    # Get candidates from both retrievers
    sparse_results = sparse_retriever.search(query, k=sparse_k)
    dense_results = dense_retriever.search(query, k=dense_k)
    
    # Combine scores
    combined = combine_scores(sparse_results, dense_results, alpha=alpha)
    
    # Return top-k
    return combined[:k]
