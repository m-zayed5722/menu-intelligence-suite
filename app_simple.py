"""
Simplified API main for local demo (minimal dependencies).
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from src.api.config import get_settings
from src.api.deps_local import (
    check_health, 
    get_items_db, 
    get_vector_store,
    get_bm25_retriever,
    get_label_tagger,
)
from src.api.schemas import (
    HealthResponse, 
    SearchRequest, 
    SearchResponse, 
    SearchResultItem, 
    SearchTimings,
    TagRequest,
    TagResponse,
    LabelScore,
)
from src.core.embeddings import encode_texts
from src.core.hybrid import combine_scores
from src.core.normalize import normalize_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup."""
    print("Starting Menu Intelligence Suite (Local Demo)")
    health = check_health()
    print(f"Status: {health}")
    yield
    print("Shutting down")


app = FastAPI(
    title="Menu Intelligence Suite",
    description="Multilingual semantic search for food delivery",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Menu Intelligence Suite API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    health_status = check_health()
    vector_store = get_vector_store()
    return HealthResponse(
        status=health_status["status"],
        db_connected=True,  # In-memory
        redis_connected=True,  # Mock
        model_loaded=True,  # Loaded on first use
        vector_store_count=vector_store.count(),
    )


@app.post("/api/search", response_model=SearchResponse)
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
            bm25 = get_bm25_retriever()
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
        vector_store = get_vector_store()
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
    items_db = get_items_db()
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


@app.post("/api/tag", response_model=TagResponse)
def tag(request: TagRequest):
    """Auto-tag items with cuisine/diet labels."""
    items_db = get_items_db()
    
    # Get text to tag
    if request.item_id:
        if request.item_id not in items_db:
            raise HTTPException(status_code=404, detail="Item not found")
        item = items_db[request.item_id]
        text = f"{item['title_en']} {item.get('description', '')}"
    else:
        text = request.text
    
    if not text:
        raise HTTPException(status_code=400, detail="No text to tag")
    
    # Get tagger
    tagger = get_label_tagger()
    
    # Assign labels
    cuisine_labels = tagger.assign_labels(
        text, 
        group="cuisine",
        top_n=request.top_n,
        threshold=request.threshold,
    )
    
    diet_labels = tagger.assign_labels(
        text,
        group="diet",
        top_n=request.top_n,
        threshold=request.threshold,
    )
    
    # Format results
    cuisine_results = [
        LabelScore(label=label, score=score)
        for label, score in cuisine_labels
    ]
    
    diet_results = [
        LabelScore(label=label, score=score)
        for label, score in diet_labels
    ]
    
    return TagResponse(
        item_id=request.item_id,
        cuisine=cuisine_results,
        diet=diet_results,
    )
