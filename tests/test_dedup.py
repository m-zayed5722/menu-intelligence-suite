"""Tests for deduplication functionality."""
import numpy as np
import pytest

from src.core.dedup import deduplicate_items, UnionFind, evaluate_dedup_pairs


def test_union_find():
    """Test union-find clustering."""
    uf = UnionFind()
    
    # Add items
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(4, 5)
    
    # Get clusters
    clusters = uf.get_clusters()
    
    # Should have 2 clusters
    assert len(clusters) == 2
    
    # Find cluster with item 1
    cluster_1 = None
    for items in clusters.values():
        if 1 in items:
            cluster_1 = set(items)
            break
    
    assert cluster_1 == {1, 2, 3}


def test_deduplication():
    """Test item deduplication."""
    # Create synthetic embeddings
    # Items 0,1 are similar; 2,3 are similar; 4 is unique
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],  # Very similar to 0
        [0.0, 1.0, 0.0],
        [0.0, 0.98, 0.02],  # Very similar to 2
        [0.0, 0.0, 1.0],     # Unique
    ], dtype=np.float32)
    
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    item_ids = [10, 11, 20, 21, 30]
    
    # Run dedup with high threshold
    clusters = deduplicate_items(
        item_ids=item_ids,
        embeddings=embeddings,
        sim_threshold=0.95,
    )
    
    # Should find 2 clusters
    assert len(clusters) == 2


def test_dedup_with_blocking():
    """Test deduplication with city blocking."""
    embeddings = np.random.randn(10, 5).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    item_ids = list(range(10))
    blocks = ["City1"] * 5 + ["City2"] * 5  # Two cities
    
    # Dedup with blocking
    clusters = deduplicate_items(
        item_ids=item_ids,
        embeddings=embeddings,
        sim_threshold=0.8,
        blocks=blocks,
    )
    
    # Verify that clusters don't cross city boundaries
    for items in clusters.values():
        cities = set([blocks[item_ids.index(i)] for i in items])
        assert len(cities) == 1  # All items in cluster from same city


def test_dedup_evaluation():
    """Test deduplication evaluation."""
    predicted = [(1, 2), (2, 3), (4, 5)]
    ground_truth = [(1, 2), (1, 3), (6, 7)]
    
    metrics = evaluate_dedup_pairs(predicted, ground_truth)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    
    # (1,2) is correct, (2,3) and (4,5) are false positives
    # (1,3) and (6,7) are false negatives
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
