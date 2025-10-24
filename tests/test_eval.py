"""Tests for evaluation metrics."""
import pytest

from src.core.eval import recall_at_k, mean_reciprocal_rank, precision_at_k, ndcg_at_k


def test_recall_at_k():
    """Test Recall@k metric."""
    predictions = [
        [1, 2, 3, 4, 5],
        [10, 11, 12],
    ]
    
    ground_truth = [
        [1, 3, 6],  # 2/3 in top-5
        [11, 13],   # 1/2 in top-3
    ]
    
    recall = recall_at_k(predictions, ground_truth, k=5)
    
    # Expected: (2/3 + 1/2) / 2 = 0.583
    assert 0.58 <= recall <= 0.59


def test_mrr():
    """Test Mean Reciprocal Rank."""
    predictions = [
        [1, 2, 3],  # First relevant at rank 1
        [10, 11, 12],  # First relevant at rank 2
        [20, 21, 22],  # No relevant items
    ]
    
    ground_truth = [
        [1, 5],
        [11, 15],
        [100],
    ]
    
    mrr = mean_reciprocal_rank(predictions, ground_truth)
    
    # Expected: (1/1 + 1/2 + 0) / 3 = 0.5
    assert abs(mrr - 0.5) < 0.01


def test_precision_at_k():
    """Test Precision@k metric."""
    predictions = [
        [1, 2, 3, 4, 5],  # 2 relevant in top-5
    ]
    
    ground_truth = [
        [1, 3, 10],
    ]
    
    precision = precision_at_k(predictions, ground_truth, k=5)
    
    # Expected: 2/5 = 0.4
    assert precision == 0.4


def test_ndcg_at_k():
    """Test NDCG@k metric."""
    predictions = [
        [1, 2, 3, 4, 5],
    ]
    
    ground_truth = [
        [1, 3, 5],
    ]
    
    ndcg = ndcg_at_k(predictions, ground_truth, k=5)
    
    # Should be between 0 and 1
    assert 0 <= ndcg <= 1
    
    # Perfect ranking
    predictions_perfect = [
        [1, 3, 5, 2, 4],
    ]
    
    ndcg_perfect = ndcg_at_k(predictions_perfect, ground_truth, k=5)
    assert ndcg_perfect > ndcg  # Perfect ranking should score higher


def test_empty_predictions():
    """Test metrics with empty predictions."""
    predictions = [[]]
    ground_truth = [[1, 2, 3]]
    
    recall = recall_at_k(predictions, ground_truth, k=5)
    assert recall == 0.0
    
    precision = precision_at_k(predictions, ground_truth, k=5)
    assert precision == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
