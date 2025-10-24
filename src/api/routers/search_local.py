"""Search router - hybrid retrieval endpoint (local demo version)."""
import time

from fastapi import APIRouter, HTTPException

from src.api import deps_local as deps
from src.api.schemas import SearchRequest, SearchResponse, SearchResultItem, SearchTimings
from src.core.embeddings import encode_texts
from src.core.hybrid import combine_scores
from src.core.normalize import normalize_text

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
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
            if bm25:
                sparse_results = bm25.search(query_normalized, k=100)
        except Exception as e:
            print(f"Sparse search failed: {e}")
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
    
    # Get item details from in-memory store
    items_db = deps.get_items_db()
    results = []
    
    for item_id, score in final_results:
        if item_id in items_db:
            item = items_db[item_id]
            results.append(SearchResultItem(
                item_id=item_id,
                score=score,
                title_en=item.get("title_en"),
                title_ar=item.get("title_ar"),
                outlet_name=item.get("outlet_name"),
                city=item.get("city"),
                price=item.get("price"),
            ))
    
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
