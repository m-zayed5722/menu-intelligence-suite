"""Tagging router - auto-tagging for cuisine and diet labels."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import deps
from src.api.logging_conf import get_logger
from src.api.schemas import TagRequest, TagResponse, LabelScore

router = APIRouter()
logger = get_logger(__name__)


@router.post("", response_model=TagResponse)
def tag_item(
    request: TagRequest,
    db: Session = Depends(deps.get_db),
):
    """
    Auto-tag an item with cuisine and diet labels.
    
    Provide either item_id or text. If item_id is provided, uses item's metadata.
    """
    # Get text to tag
    if request.item_id:
        # Fetch item from database
        query = text("""
            SELECT title_en, title_ar, description
            FROM items
            WHERE item_id = :item_id
        """)
        row = db.execute(query, {"item_id": request.item_id}).fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Combine available text
        text_parts = [t for t in row if t]
        text = " ".join(text_parts)
    elif request.text:
        text = request.text
    else:
        raise HTTPException(status_code=400, detail="Provide either item_id or text")
    
    # Get tagger
    tagger = deps.get_label_tagger()
    
    # Assign labels for all groups
    results = tagger.assign_all_groups(
        text,
        top_n=request.top_n,
        threshold=request.threshold,
    )
    
    # Format response
    cuisine_labels = [
        LabelScore(label=label, score=score)
        for label, score in results.get("cuisine", [])
    ]
    
    diet_labels = [
        LabelScore(label=label, score=score)
        for label, score in results.get("diet", [])
    ]
    
    return TagResponse(
        cuisine=cuisine_labels,
        diet=diet_labels,
        item_id=request.item_id,
    )
