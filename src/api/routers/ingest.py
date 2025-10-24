"""Ingest router - load menu items into the system."""
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import deps
from src.api.logging_conf import get_logger
from src.api.schemas import IngestRequest, IngestResponse
from src.core.embeddings import encode_texts
from src.core.normalize import normalize_text

router = APIRouter()
logger = get_logger(__name__)


@router.post("", response_model=IngestResponse)
def ingest_items(
    request: IngestRequest,
    db: Session = Depends(deps.get_db),
):
    """
    Ingest menu items into the system.
    
    Creates normalized text fields and embeddings for search.
    """
    if not request.items:
        return IngestResponse(inserted=0, updated=0, message="No items provided")
    
    inserted = 0
    updated = 0
    
    for item in request.items:
        # Normalize text
        title_parts = []
        if item.title_en:
            title_parts.append(item.title_en)
        if item.title_ar:
            title_parts.append(item.title_ar)
        
        title_combined = " ".join(title_parts)
        title_norm = normalize_text(title_combined)
        desc_norm = normalize_text(item.description or "")
        
        # Generate embedding
        text_for_embedding = f"{title_combined} {item.description or ''}"
        embedding = encode_texts([text_for_embedding], normalize=True)[0]
        embedding_list = embedding.tolist()
        
        # Check if item exists
        check_query = text("""
            SELECT item_id FROM items
            WHERE outlet_id = :outlet_id AND title_en = :title_en
        """)
        existing = db.execute(check_query, {
            "outlet_id": item.outlet_id,
            "title_en": item.title_en,
        }).fetchone()
        
        if existing:
            # Update
            update_query = text("""
                UPDATE items SET
                    outlet_name = :outlet_name,
                    city = :city,
                    lat = :lat,
                    lon = :lon,
                    title_ar = :title_ar,
                    description = :description,
                    price = :price,
                    cuisine_tags = :cuisine_tags,
                    diet_tags = :diet_tags,
                    title_norm = :title_norm,
                    desc_norm = :desc_norm,
                    embedding = :embedding,
                    updated_at = NOW()
                WHERE item_id = :item_id
            """)
            db.execute(update_query, {
                "item_id": existing[0],
                "outlet_name": item.outlet_name,
                "city": item.city,
                "lat": item.lat,
                "lon": item.lon,
                "title_ar": item.title_ar,
                "description": item.description,
                "price": item.price,
                "cuisine_tags": item.cuisine_tags,
                "diet_tags": item.diet_tags,
                "title_norm": title_norm,
                "desc_norm": desc_norm,
                "embedding": embedding_list,
            })
            updated += 1
        else:
            # Insert
            insert_query = text("""
                INSERT INTO items (
                    outlet_id, outlet_name, city, lat, lon,
                    title_en, title_ar, description, price,
                    cuisine_tags, diet_tags,
                    title_norm, desc_norm, embedding
                ) VALUES (
                    :outlet_id, :outlet_name, :city, :lat, :lon,
                    :title_en, :title_ar, :description, :price,
                    :cuisine_tags, :diet_tags,
                    :title_norm, :desc_norm, :embedding
                )
            """)
            db.execute(insert_query, {
                "outlet_id": item.outlet_id,
                "outlet_name": item.outlet_name,
                "city": item.city,
                "lat": item.lat,
                "lon": item.lon,
                "title_en": item.title_en,
                "title_ar": item.title_ar,
                "description": item.description,
                "price": item.price,
                "cuisine_tags": item.cuisine_tags,
                "diet_tags": item.diet_tags,
                "title_norm": title_norm,
                "desc_norm": desc_norm,
                "embedding": embedding_list,
            })
            inserted += 1
    
    db.commit()
    
    # Rebuild sparse index
    logger.info("Rebuilding BM25 index...")
    count = deps.build_sparse_index(db)
    logger.info(f"BM25 index rebuilt with {count} items")
    
    return IngestResponse(
        inserted=inserted,
        updated=updated,
        message=f"Processed {len(request.items)} items",
    )
