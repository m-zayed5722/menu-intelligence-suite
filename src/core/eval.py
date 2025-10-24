"""Offline evaluation metrics for search and tagging."""
from typing import Any

import numpy as np


def recall_at_k(
    predictions: list[list[Any]],
    ground_truth: list[list[Any]],
    k: int = 10,
) -> float:
    """
    Compute Recall@k for ranking.
    
    Args:
        predictions: List of ranked prediction lists
        ground_truth: List of relevant item sets
        k: Cutoff rank
    
    Returns:
        Average recall@k across queries
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    recalls = []
    for pred, true in zip(predictions, ground_truth):
        true_set = set(true)
        if not true_set:
            continue
        
        # Get top-k predictions
        pred_k = pred[:k]
        pred_set = set(pred_k)
        
        # Recall = |relevant ∩ retrieved| / |relevant|
        recall = len(pred_set & true_set) / len(true_set)
        recalls.append(recall)
    
    return float(np.mean(recalls)) if recalls else 0.0


def mean_reciprocal_rank(
    predictions: list[list[Any]],
    ground_truth: list[list[Any]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        predictions: List of ranked prediction lists
        ground_truth: List of relevant item sets
    
    Returns:
        MRR score
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    reciprocal_ranks = []
    for pred, true in zip(predictions, ground_truth):
        true_set = set(true)
        if not true_set:
            continue
        
        # Find first relevant item
        for rank, item in enumerate(pred, start=1):
            if item in true_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No relevant item found
            reciprocal_ranks.append(0.0)
    
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def precision_at_k(
    predictions: list[list[Any]],
    ground_truth: list[list[Any]],
    k: int = 10,
) -> float:
    """
    Compute Precision@k.
    
    Args:
        predictions: List of ranked prediction lists
        ground_truth: List of relevant item sets
        k: Cutoff rank
    
    Returns:
        Average precision@k
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    precisions = []
    for pred, true in zip(predictions, ground_truth):
        true_set = set(true)
        if not true_set:
            continue
        
        # Get top-k
        pred_k = pred[:k]
        pred_set = set(pred_k)
        
        # Precision = |relevant ∩ retrieved| / |retrieved|
        precision = len(pred_set & true_set) / k if k > 0 else 0.0
        precisions.append(precision)
    
    return float(np.mean(precisions)) if precisions else 0.0


def ndcg_at_k(
    predictions: list[list[Any]],
    ground_truth: list[list[Any]],
    k: int = 10,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k).
    
    Args:
        predictions: List of ranked prediction lists
        ground_truth: List of relevant item sets
        k: Cutoff rank
    
    Returns:
        Average NDCG@k
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    ndcgs = []
    for pred, true in zip(predictions, ground_truth):
        true_set = set(true)
        if not true_set:
            continue
        
        # DCG
        dcg = 0.0
        for i, item in enumerate(pred[:k], start=1):
            if item in true_set:
                dcg += 1.0 / np.log2(i + 1)
        
        # IDCG (perfect ranking)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(true_set), k) + 1))
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def evaluate_search(
    predictions: list[list[Any]],
    ground_truth: list[list[Any]],
    ks: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Comprehensive search evaluation.
    
    Args:
        predictions: List of ranked prediction lists
        ground_truth: List of relevant item sets
        ks: List of k values to evaluate
    
    Returns:
        Dict of metrics
    """
    metrics = {"mrr": mean_reciprocal_rank(predictions, ground_truth)}
    
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(predictions, ground_truth, k)
        metrics[f"precision@{k}"] = precision_at_k(predictions, ground_truth, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(predictions, ground_truth, k)
    
    return metrics


def per_query_metrics(
    predictions: list[list[Any]],
    ground_truth: list[list[Any]],
    query_ids: list[str] | None = None,
    k: int = 10,
) -> list[dict]:
    """
    Compute per-query metrics for analysis.
    
    Args:
        predictions: List of ranked prediction lists
        ground_truth: List of relevant item sets
        query_ids: Optional query identifiers
        k: Cutoff rank
    
    Returns:
        List of per-query metric dicts
    """
    if query_ids is None:
        query_ids = [f"q{i}" for i in range(len(predictions))]
    
    results = []
    for qid, pred, true in zip(query_ids, predictions, ground_truth):
        true_set = set(true)
        pred_k = pred[:k]
        pred_set = set(pred_k)
        
        # Find first hit
        first_hit_rank = None
        for rank, item in enumerate(pred, start=1):
            if item in true_set:
                first_hit_rank = rank
                break
        
        results.append({
            "query_id": qid,
            "num_relevant": len(true_set),
            "hit": 1 if (pred_set & true_set) else 0,
            "first_hit_rank": first_hit_rank,
            "recall": len(pred_set & true_set) / len(true_set) if true_set else 0.0,
            "precision": len(pred_set & true_set) / k if k > 0 else 0.0,
        })
    
    return results
