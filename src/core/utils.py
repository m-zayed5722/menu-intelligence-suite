"""Utility functions for MIS."""
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable


@contextmanager
def timer(label: str = "Operation"):
    """Context manager to time operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"{label}: {elapsed_ms:.2f}ms")


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure and return execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
    return wrapper


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] using min-max scaling."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score - min_score < 1e-9:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


def merge_scores(
    ids_scores_list: list[list[tuple[Any, float]]],
    weights: list[float],
) -> list[tuple[Any, float]]:
    """
    Merge multiple ranked lists with weighted scores.
    
    Args:
        ids_scores_list: List of [(id, score), ...] lists
        weights: Weight for each list (must sum to 1.0)
    
    Returns:
        Merged and sorted [(id, score), ...] list
    """
    assert len(ids_scores_list) == len(weights), "Mismatched lengths"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    # Collect all scores by ID
    combined = {}
    
    for idx, ids_scores in enumerate(ids_scores_list):
        # Normalize scores for this list
        if not ids_scores:
            continue
        
        scores = [s for _, s in ids_scores]
        normalized = min_max_normalize(scores)
        
        for (item_id, _), norm_score in zip(ids_scores, normalized):
            if item_id not in combined:
                combined[item_id] = 0.0
            combined[item_id] += weights[idx] * norm_score
    
    # Sort by combined score
    result = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return result
