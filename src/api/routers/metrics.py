"""Metrics router - offline evaluation of search quality."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import deps
from src.api.logging_conf import get_logger
from src.api.schemas import QueryMetric, SearchMetricsRequest, SearchMetricsResponse
from src.core.embeddings import encode_texts
from src.core.eval import per_query_metrics, recall_at_k, mean_reciprocal_rank, precision_at_k
from src.core.hybrid import combine_scores
from src.core.normalize import normalize_text

router = APIRouter()
logger = get_logger(__name__)


@router.get("/search", response_model=SearchMetricsResponse)
def evaluate_search(
    k: int = Query(default=5, ge=1, le=20),
    mode: str = Query(default="hybrid", pattern="^(sparse|dense|hybrid)$"),
    alpha: float = Query(default=0.4, ge=0.0, le=1.0),
    ef_search: int | None = Query(default=None, ge=1, le=200),
    db: Session = Depends(deps.get_db),
):
    """
    Evaluate search performance on labeled queries.
    
    Returns Recall@k, MRR, and per-query metrics.
    """
    # Load query labels
    query = text("""
        SELECT qid, query, relevant_ids
        FROM query_labels
        ORDER BY qid
    """)
    rows = db.execute(query).fetchall()
    
    if not rows:
        return SearchMetricsResponse(
            recall_at_k=0.0,
            mrr=0.0,
            precision_at_k=0.0,
            per_query=[],
        )
    
    # Run search for each query
    predictions = []
    ground_truth = []
    queries = []
    
    bm25 = deps.get_bm25_retriever()
    vector_store = deps.get_vector_store()
    
    for row in rows:
        qid, query_text, relevant_ids = row
        
        # Normalize query
        query_norm = normalize_text(query_text)
        
        sparse_results = []
        dense_results = []
        
        # Sparse
        if mode in ["sparse", "hybrid"]:
            try:
                sparse_results = bm25.search(query_norm, k=100)
            except Exception as e:
                logger.warning(f"Sparse search failed for query {qid}: {e}")
        
        # Dense
        if mode in ["dense", "hybrid"]:
            query_vector = encode_texts([query_norm], normalize=True)[0]
            dense_results = vector_store.search(
                query_vector,
                k=100,
                ef_search=ef_search,
            )
        
        # Combine
        if mode == "hybrid":
            combined = combine_scores(sparse_results, dense_results, alpha=alpha)
            final_results = combined[:k * 2]  # Get more for evaluation
        elif mode == "sparse":
            final_results = sparse_results[:k * 2]
        else:
            final_results = dense_results[:k * 2]
        
        # Extract IDs
        pred_ids = [item_id for item_id, _ in final_results]
        
        predictions.append(pred_ids)
        ground_truth.append(list(relevant_ids))
        queries.append(query_text)
    
    # Compute metrics
    recall = recall_at_k(predictions, ground_truth, k=k)
    mrr = mean_reciprocal_rank(predictions, ground_truth)
    precision = precision_at_k(predictions, ground_truth, k=k)
    
    # Per-query metrics
    per_query = per_query_metrics(predictions, ground_truth, queries, k=k)
    
    # Format response
    query_metrics = []
    for qm in per_query:
        query_metrics.append(QueryMetric(
            query=qm["query_id"],
            hit=qm["hit"],
            first_rank=qm["first_hit_rank"],
            recall=qm["recall"],
        ))
    
    return SearchMetricsResponse(
        recall_at_k=recall,
        mrr=mrr,
        precision_at_k=precision,
        per_query=query_metrics,
    )
