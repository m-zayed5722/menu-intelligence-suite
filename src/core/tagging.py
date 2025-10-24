"""Auto-tagging for cuisine and diet labels using label centroids."""
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.core.embeddings import encode_texts, batch_cosine_similarity


class LabelTagger:
    """Assign labels based on nearest centroid in embedding space."""
    
    def __init__(self, labels_path: str | None = None):
        """
        Initialize tagger with label definitions.
        
        Args:
            labels_path: Path to JSON file with label groups
        """
        self.label_groups = {}
        self.label_embeddings = {}
        
        if labels_path and os.path.exists(labels_path):
            self.load_labels(labels_path)
    
    def load_labels(self, labels_path: str):
        """Load label definitions from JSON."""
        with open(labels_path) as f:
            self.label_groups = json.load(f)
        
        # Compute embeddings for each label group
        for group_name, labels in self.label_groups.items():
            # Encode label strings
            embeddings = encode_texts(labels, normalize=True)
            self.label_embeddings[group_name] = {
                "labels": labels,
                "embeddings": embeddings,
            }
    
    def set_labels(self, group_name: str, labels: list[str]):
        """Manually set labels for a group."""
        self.label_groups[group_name] = labels
        embeddings = encode_texts(labels, normalize=True)
        self.label_embeddings[group_name] = {
            "labels": labels,
            "embeddings": embeddings,
        }
    
    def assign_labels(
        self,
        text: str,
        group_name: str,
        top_n: int = 1,
        threshold: float = 0.35,
    ) -> list[tuple[str, float]]:
        """
        Assign labels from a group to text.
        
        Args:
            text: Text to tag
            group_name: Label group (e.g., "cuisine", "diet")
            top_n: Number of labels to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (label, score) tuples
        """
        if group_name not in self.label_embeddings:
            return []
        
        # Encode text
        text_embedding = encode_texts([text], normalize=True)[0]
        
        # Get label embeddings
        label_data = self.label_embeddings[group_name]
        labels = label_data["labels"]
        label_embs = label_data["embeddings"]
        
        # Compute similarities
        similarities = batch_cosine_similarity(
            text_embedding.reshape(1, -1),
            label_embs
        )[0]
        
        # Get top-n above threshold
        results = []
        for label, score in zip(labels, similarities):
            if score >= threshold:
                results.append((label, float(score)))
        
        # Sort by score and return top-n
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def assign_all_groups(
        self,
        text: str,
        top_n: int = 1,
        threshold: float = 0.35,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Assign labels from all groups.
        
        Args:
            text: Text to tag
            top_n: Number of labels per group
            threshold: Minimum similarity threshold
        
        Returns:
            Dict of {group_name: [(label, score), ...]}
        """
        results = {}
        for group_name in self.label_groups:
            results[group_name] = self.assign_labels(
                text, group_name, top_n=top_n, threshold=threshold
            )
        return results


def evaluate_tagging(
    predictions: list[list[str]],
    ground_truth: list[list[str]],
) -> dict[str, float]:
    """
    Evaluate multi-label classification performance.
    
    Args:
        predictions: List of predicted label sets
        ground_truth: List of true label sets
    
    Returns:
        Dict with precision, recall, f1 (macro-averaged)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    precisions = []
    recalls = []
    f1s = []
    
    for pred, true in zip(predictions, ground_truth):
        pred_set = set(pred)
        true_set = set(true)
        
        if len(pred_set) == 0 and len(true_set) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
        
        tp = len(pred_set & true_set)
        
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = tp / len(true_set) if len(true_set) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }
