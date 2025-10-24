"""Simple recommendation baseline using content-based filtering."""
from collections import defaultdict
from typing import Any

import numpy as np

from src.core.embeddings import batch_cosine_similarity


class ContentBasedRecommender:
    """Content-based recommender using item embeddings."""
    
    def __init__(self):
        self.item_ids: list[Any] = []
        self.item_embeddings: np.ndarray | None = None
        self.user_profiles: dict[str, np.ndarray] = {}
        self.item_popularity: dict[Any, int] = defaultdict(int)
    
    def set_items(self, item_ids: list[Any], embeddings: np.ndarray):
        """Set item catalog."""
        self.item_ids = item_ids
        self.item_embeddings = embeddings
    
    def update_user_profile(
        self,
        user_id: str,
        interacted_item_ids: list[Any],
        popularity_prior: float = 0.1,
    ):
        """
        Build user profile from interaction history.
        
        Args:
            user_id: User identifier
            interacted_item_ids: List of items user interacted with
            popularity_prior: Weight for popularity regularization
        """
        if self.item_embeddings is None:
            return
        
        # Find embeddings for interacted items
        interacted_embeddings = []
        for item_id in interacted_item_ids:
            if item_id in self.item_ids:
                idx = self.item_ids.index(item_id)
                interacted_embeddings.append(self.item_embeddings[idx])
                self.item_popularity[item_id] += 1
        
        if not interacted_embeddings:
            return
        
        # User profile = mean of interacted item embeddings
        user_profile = np.mean(interacted_embeddings, axis=0)
        
        # Normalize
        user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-9)
        
        self.user_profiles[user_id] = user_profile
    
    def recommend_for_user(
        self,
        user_id: str,
        k: int = 10,
        exclude_items: list[Any] | None = None,
        popularity_boost: float = 0.1,
    ) -> list[tuple[Any, float]]:
        """
        Recommend items for a user.
        
        Args:
            user_id: User identifier
            k: Number of recommendations
            exclude_items: Items to exclude (e.g., already interacted)
            popularity_boost: Boost popular items
        
        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_profiles or self.item_embeddings is None:
            # Fall back to popularity
            return self._recommend_popular(k, exclude_items)
        
        user_profile = self.user_profiles[user_id]
        
        # Compute similarities
        similarities = batch_cosine_similarity(
            user_profile.reshape(1, -1),
            self.item_embeddings
        )[0]
        
        # Apply popularity boost
        if popularity_boost > 0:
            max_pop = max(self.item_popularity.values()) if self.item_popularity else 1
            for i, item_id in enumerate(self.item_ids):
                pop_score = self.item_popularity.get(item_id, 0) / max_pop
                similarities[i] += popularity_boost * pop_score
        
        # Sort and filter
        results = []
        exclude_set = set(exclude_items or [])
        
        for item_id, score in zip(self.item_ids, similarities):
            if item_id not in exclude_set:
                results.append((item_id, float(score)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def recommend_similar_items(
        self,
        item_id: Any,
        k: int = 10,
        exclude_self: bool = True,
    ) -> list[tuple[Any, float]]:
        """
        Recommend items similar to a given item.
        
        Args:
            item_id: Reference item
            k: Number of recommendations
            exclude_self: Exclude the reference item
        
        Returns:
            List of (item_id, score) tuples
        """
        if self.item_embeddings is None or item_id not in self.item_ids:
            return []
        
        # Get item embedding
        idx = self.item_ids.index(item_id)
        item_embedding = self.item_embeddings[idx]
        
        # Compute similarities
        similarities = batch_cosine_similarity(
            item_embedding.reshape(1, -1),
            self.item_embeddings
        )[0]
        
        # Sort
        results = []
        for i, (other_id, score) in enumerate(zip(self.item_ids, similarities)):
            if exclude_self and other_id == item_id:
                continue
            results.append((other_id, float(score)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _recommend_popular(
        self,
        k: int,
        exclude_items: list[Any] | None = None,
    ) -> list[tuple[Any, float]]:
        """Fallback to popularity-based recommendations."""
        exclude_set = set(exclude_items or [])
        
        # Sort by popularity
        popular = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = []
        for item_id, count in popular:
            if item_id not in exclude_set:
                results.append((item_id, float(count)))
                if len(results) >= k:
                    break
        
        # If not enough popular items, add random items
        if len(results) < k and self.item_ids:
            for item_id in self.item_ids:
                if item_id not in exclude_set and item_id not in [x[0] for x in results]:
                    results.append((item_id, 0.0))
                    if len(results) >= k:
                        break
        
        return results
