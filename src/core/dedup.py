"""Item deduplication using embedding-based similarity and union-find clustering."""
from collections import defaultdict
from typing import Any

import numpy as np

from src.core.embeddings import batch_cosine_similarity


class UnionFind:
    """Union-Find data structure for clustering."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        """Find root with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def get_clusters(self):
        """Get all clusters as dict of {cluster_id: [item_ids]}."""
        clusters = defaultdict(list)
        for item in list(self.parent.keys()):
            root = self.find(item)
            clusters[root].append(item)
        return dict(clusters)


def deduplicate_items(
    item_ids: list[Any],
    embeddings: np.ndarray,
    sim_threshold: float = 0.82,
    blocks: list[Any] | None = None,
) -> dict[int, list[Any]]:
    """
    Find duplicate items using cosine similarity and union-find.
    
    Args:
        item_ids: List of item IDs
        embeddings: Item embeddings (N x D)
        sim_threshold: Similarity threshold for duplicates
        blocks: Optional blocking keys (e.g., city) to reduce comparisons
    
    Returns:
        Dictionary of {cluster_id: [item_ids]}
    """
    n = len(item_ids)
    uf = UnionFind()
    
    # Initialize all items
    for item_id in item_ids:
        uf.find(item_id)
    
    # If blocks provided, only compare within blocks
    if blocks:
        block_groups = defaultdict(list)
        for i, (item_id, block) in enumerate(zip(item_ids, blocks)):
            block_groups[block].append((i, item_id))
        
        # Process each block
        for block_items in block_groups.values():
            if len(block_items) < 2:
                continue
            
            indices = [i for i, _ in block_items]
            ids = [id_ for _, id_ in block_items]
            block_embeddings = embeddings[indices]
            
            # Compute pairwise similarities
            sim_matrix = batch_cosine_similarity(block_embeddings, block_embeddings)
            
            # Find pairs above threshold
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if sim_matrix[i, j] >= sim_threshold:
                        uf.union(ids[i], ids[j])
    else:
        # No blocking - compare all pairs (expensive for large datasets)
        sim_matrix = batch_cosine_similarity(embeddings, embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= sim_threshold:
                    uf.union(item_ids[i], item_ids[j])
    
    # Get clusters
    clusters = uf.get_clusters()
    
    # Filter to only clusters with multiple items
    dedup_clusters = {
        cluster_id: items
        for cluster_id, items in clusters.items()
        if len(items) > 1
    }
    
    return dedup_clusters


def compute_dedup_pairs(clusters: dict[int, list[Any]]) -> list[tuple[Any, Any]]:
    """
    Convert clusters to pairwise duplicate relationships.
    
    Args:
        clusters: Dictionary of {cluster_id: [item_ids]}
    
    Returns:
        List of (item_a, item_b) duplicate pairs
    """
    pairs = []
    for items in clusters.values():
        # Generate all pairs within cluster
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pairs.append((items[i], items[j]))
    
    return pairs


def evaluate_dedup_pairs(
    predicted_pairs: list[tuple[Any, Any]],
    true_pairs: list[tuple[Any, Any]],
) -> dict[str, float]:
    """
    Evaluate deduplication performance.
    
    Args:
        predicted_pairs: Predicted duplicate pairs
        true_pairs: Ground truth duplicate pairs
    
    Returns:
        Dict with precision, recall, f1
    """
    pred_set = {tuple(sorted(pair)) for pair in predicted_pairs}
    true_set = {tuple(sorted(pair)) for pair in true_pairs}
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
