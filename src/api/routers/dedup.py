"""Deduplication router - find duplicate menu items."""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import deps
from src.api.logging_conf import get_logger
from src.api.schemas import DedupCluster, DedupClusterRequest, DedupClusterResponse, DedupStats
from src.core.dedup import deduplicate_items

router = APIRouter()
logger = get_logger(__name__)


@router.post("/cluster", response_model=DedupClusterResponse)
def cluster_duplicates(
    request: DedupClusterRequest,
    db: Session = Depends(deps.get_db),
):
    """
    Find and cluster duplicate items.
    
    Optionally filter by city to reduce comparisons.
    Uses cosine similarity on item embeddings.
    """
    # Build query
    where_clause = ""
    params = {"sim_thresh": request.sim_threshold}
    
    if request.city:
        where_clause = "WHERE city = :city"
        params["city"] = request.city
    
    # Fetch items with embeddings
    query = text(f"""
        SELECT item_id, embedding, city
        FROM items
        {where_clause}
        AND embedding IS NOT NULL
        ORDER BY item_id
    """)
    
    rows = db.execute(query, params).fetchall()
    
    if len(rows) < 2:
        return DedupClusterResponse(
            clusters=[],
            stats=DedupStats(
                total_items=len(rows),
                num_clusters=0,
                num_duplicates=0,
                pairs_compared=0,
            ),
        )
    
    # Extract data
    item_ids = [row[0] for row in rows]
    embeddings = np.array([row[1] for row in rows], dtype=np.float32)
    cities = [row[2] for row in rows] if request.city is None else None
    
    # Run deduplication
    clusters = deduplicate_items(
        item_ids=item_ids,
        embeddings=embeddings,
        sim_threshold=request.sim_threshold,
        blocks=cities,
    )
    
    # Format response
    cluster_list = []
    for cluster_id, items in clusters.items():
        cluster_list.append(DedupCluster(
            cluster_id=cluster_id,
            item_ids=sorted(items),
        ))
    
    # Compute stats
    num_duplicates = sum(len(items) for items in clusters.values())
    pairs_compared = len(item_ids) * (len(item_ids) - 1) // 2
    
    # Save clusters to database
    if clusters:
        # Clear existing clusters for these items
        db.execute(
            text("DELETE FROM dedup_clusters WHERE item_id = ANY(:ids)"),
            {"ids": item_ids}
        )
        
        # Insert new clusters
        for cluster_id, items in clusters.items():
            for item_id in items:
                db.execute(
                    text("INSERT INTO dedup_clusters (item_id, cluster_id) VALUES (:item_id, :cluster_id)"),
                    {"item_id": item_id, "cluster_id": cluster_id}
                )
        
        db.commit()
    
    return DedupClusterResponse(
        clusters=cluster_list,
        stats=DedupStats(
            total_items=len(item_ids),
            num_clusters=len(clusters),
            num_duplicates=num_duplicates,
            pairs_compared=pairs_compared,
        ),
    )
