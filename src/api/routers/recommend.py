"""Recommendation router - content-based recommendations."""
import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import deps
from src.api.logging_conf import get_logger
from src.api.schemas import RecommendItem, RecommendRequest, RecommendResponse
from src.core.recommend import ContentBasedRecommender

router = APIRouter()
logger = get_logger(__name__)

# Global recommender instance (in production, use Redis/cache)
_recommender = None


def get_recommender(db: Session) -> ContentBasedRecommender:
    """Get or initialize recommender."""
    global _recommender
    
    if _recommender is None:
        _recommender = ContentBasedRecommender()
        
        # Load all items with embeddings
        query = text("""
            SELECT item_id, embedding
            FROM items
            WHERE embedding IS NOT NULL
            ORDER BY item_id
        """)
        rows = db.execute(query).fetchall()
        
        if rows:
            item_ids = [row[0] for row in rows]
            embeddings = np.array([row[1] for row in rows], dtype=np.float32)
            _recommender.set_items(item_ids, embeddings)
    
    return _recommender


@router.post("", response_model=RecommendResponse)
def recommend(
    request: RecommendRequest,
    db: Session = Depends(deps.get_db),
):
    """
    Get recommendations.
    
    Provide either:
    - user_id: Get personalized recommendations based on interaction history
    - item_id: Get similar items (content-based)
    - neither: Get popular items
    """
    recommender = get_recommender(db)
    
    if request.item_id:
        # Item-based recommendations
        results = recommender.recommend_similar_items(
            item_id=request.item_id,
            k=request.k,
            exclude_self=True,
        )
        mode = "item_based"
    
    elif request.user_id:
        # Load user interaction history
        query = text("""
            SELECT item_id
            FROM user_interactions
            WHERE user_id = :user_id
            ORDER BY timestamp DESC
            LIMIT 50
        """)
        rows = db.execute(query, {"user_id": request.user_id}).fetchall()
        interacted_ids = [row[0] for row in rows]
        
        if interacted_ids:
            # Update user profile
            recommender.update_user_profile(request.user_id, interacted_ids)
            
            # Get recommendations
            results = recommender.recommend_for_user(
                user_id=request.user_id,
                k=request.k,
                exclude_items=interacted_ids,
                popularity_boost=0.1,
            )
            mode = "user_based"
        else:
            # No history - fall back to popular
            results = recommender._recommend_popular(k=request.k)
            mode = "popular"
    
    else:
        # No user or item - return popular
        results = recommender._recommend_popular(k=request.k)
        mode = "popular"
    
    # Format response
    items = [
        RecommendItem(item_id=item_id, score=score)
        for item_id, score in results
    ]
    
    return RecommendResponse(items=items, mode=mode)
