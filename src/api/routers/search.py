"""Search router - hybrid retrieval endpoint."""
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import deps
from src.api.logging_conf import get_logger
from src.api.schemas import SearchRequest, SearchResponse, SearchResultItem, SearchTimings
from src.core.embeddings import encode_texts
from src.core.hybrid import combine_scores
from src.core.normalize import normalize_text

router = APIRouter()
logger = get_logger(__name__)


@router.post("", response_model=SearchResponse)
def search(
    request: SearchRequest,
    db: Session = Depends(deps.get_db),
):
    """
    Search for menu items using sparse, dense, or hybrid retrieval.
    
    - **sparse**: BM25-based keyword search
    - **dense**: Semantic search using embeddings
    - **hybrid**: Weighted combination (alpha * sparse + (1-alpha) * dense)
    """
    start_time = time.perf_counter()
    timings = {}
    
    # Normalize query
    query_normalized = normalize_text(request.query, remove_diacritics=request.normalize_arabic)
    
    sparse_results = []
    dense_results = []
    
    # Sparse retrieval
    if request.mode in ["sparse", "hybrid"]:
        t0 = time.perf_counter()
        try:
            bm25 = deps.get_bm25_retriever()
            sparse_results = bm25.search(query_normalized, k=100)
        except Exception as e:
            logger.warning(f"Sparse search failed: {e}")
            if request.mode == "sparse":
                raise HTTPException(status_code=500, detail="Sparse search failed")
        timings["sparse_ms"] = (time.perf_counter() - t0) * 1000
    
    # Dense retrieval
    if request.mode in ["dense", "hybrid"]:
        # Encode query
        t0 = time.perf_counter()
        query_vector = encode_texts([query_normalized], normalize=True)[0]
        timings["encode_ms"] = (time.perf_counter() - t0) * 1000
        
        # ANN search
        t0 = time.perf_counter()
        vector_store = deps.get_vector_store()
        dense_results = vector_store.search(
            query_vector,
            k=request.k if request.mode == "dense" else 100,
            ef_search=request.ef_search,
        )
        timings["dense_ms"] = (time.perf_counter() - t0) * 1000
    
    # Combine results
    if request.mode == "hybrid":
        t0 = time.perf_counter()
        combined = combine_scores(sparse_results, dense_results, alpha=request.alpha)
        final_results = combined[:request.k]
        timings["hybrid_ms"] = (time.perf_counter() - t0) * 1000
    elif request.mode == "sparse":
        final_results = sparse_results[:request.k]
    else:  # dense
        final_results = dense_results
    
    # Get item details from database
    if final_results:
        item_ids = [item_id for item_id, _ in final_results]
        score_map = {item_id: score for item_id, score in final_results}
        
        placeholders = ','.join([':id' + str(i) for i in range(len(item_ids))])
        query_sql = text(f"""
            SELECT item_id, title_en, title_ar, outlet_name, city, price
            FROM items
            WHERE item_id IN ({placeholders})
        """)
        
        params = {f'id{i}': item_id for i, item_id in enumerate(item_ids)}
        rows = db.execute(query_sql, params).fetchall()
        
        # Build response maintaining order
        id_to_row = {row[0]: row for row in rows}
        results = []
        for item_id, _ in final_results:
            if item_id in id_to_row:
                row = id_to_row[item_id]
                results.append(SearchResultItem(
                    item_id=row[0],
                    score=score_map[item_id],
                    title_en=row[1],
                    title_ar=row[2],
                    outlet_name=row[3],
                    city=row[4],
                    price=float(row[5]) if row[5] else None,
                ))
    else:
        results = []
    
    # Total time
    total_ms = (time.perf_counter() - start_time) * 1000
    
    return SearchResponse(
        results=results,
        timings=SearchTimings(
            encode_ms=timings.get("encode_ms", 0.0),
            sparse_ms=timings.get("sparse_ms"),
            dense_ms=timings.get("dense_ms"),
            hybrid_ms=timings.get("hybrid_ms"),
            total_ms=total_ms,
        ),
        query=request.query,
        mode=request.mode,
    )
