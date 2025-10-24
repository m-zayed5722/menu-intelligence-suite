"""Pydantic schemas for API requests and responses."""
from typing import Any

from pydantic import BaseModel, Field


# Ingest schemas
class ItemInput(BaseModel):
    """Input schema for menu item."""
    outlet_id: int
    outlet_name: str
    city: str
    lat: float | None = None
    lon: float | None = None
    title_en: str | None = None
    title_ar: str | None = None
    description: str | None = None
    price: float | None = None
    cuisine_tags: list[str] | None = None
    diet_tags: list[str] | None = None


class IngestRequest(BaseModel):
    """Request schema for ingesting items."""
    items: list[ItemInput]


class IngestResponse(BaseModel):
    """Response schema for ingest."""
    inserted: int
    updated: int
    message: str = "Items ingested successfully"


# Search schemas
class SearchRequest(BaseModel):
    """Request schema for search."""
    query: str
    k: int = Field(default=10, ge=1, le=100)
    mode: str = Field(default="hybrid", pattern="^(sparse|dense|hybrid)$")
    alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    ef_search: int | None = Field(default=None, ge=1, le=200)
    normalize_arabic: bool = True


class SearchResultItem(BaseModel):
    """Single search result."""
    item_id: int
    score: float
    title_en: str | None
    title_ar: str | None
    outlet_name: str | None
    city: str | None
    price: float | None


class SearchTimings(BaseModel):
    """Timing breakdown for search."""
    encode_ms: float
    sparse_ms: float | None = None
    dense_ms: float | None = None
    hybrid_ms: float | None = None
    total_ms: float


class SearchResponse(BaseModel):
    """Response schema for search."""
    results: list[SearchResultItem]
    timings: SearchTimings
    query: str
    mode: str


# Tagging schemas
class TagRequest(BaseModel):
    """Request schema for tagging."""
    item_id: int | None = None
    text: str | None = None
    top_n: int = Field(default=1, ge=1, le=5)
    threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class LabelScore(BaseModel):
    """Label with confidence score."""
    label: str
    score: float


class TagResponse(BaseModel):
    """Response schema for tagging."""
    cuisine: list[LabelScore]
    diet: list[LabelScore]
    item_id: int | None = None


# Dedup schemas
class DedupClusterRequest(BaseModel):
    """Request schema for dedup clustering."""
    city: str | None = None
    sim_threshold: float = Field(default=0.82, ge=0.5, le=1.0)


class DedupCluster(BaseModel):
    """Single dedup cluster."""
    cluster_id: int
    item_ids: list[int]


class DedupStats(BaseModel):
    """Dedup statistics."""
    total_items: int
    num_clusters: int
    num_duplicates: int
    pairs_compared: int


class DedupClusterResponse(BaseModel):
    """Response schema for dedup clustering."""
    clusters: list[DedupCluster]
    stats: DedupStats


# Recommendation schemas
class RecommendRequest(BaseModel):
    """Request schema for recommendations."""
    user_id: str | None = None
    item_id: int | None = None
    k: int = Field(default=10, ge=1, le=50)


class RecommendItem(BaseModel):
    """Recommended item."""
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    """Response schema for recommendations."""
    items: list[RecommendItem]
    mode: str  # "user_based", "item_based", or "popular"


# Metrics schemas
class SearchMetricsRequest(BaseModel):
    """Request schema for search metrics."""
    k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(sparse|dense|hybrid)$")
    alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    ef_search: int | None = None


class QueryMetric(BaseModel):
    """Per-query metric."""
    query: str
    hit: int
    first_rank: int | None
    recall: float


class SearchMetricsResponse(BaseModel):
    """Response schema for search metrics."""
    recall_at_k: float
    mrr: float
    precision_at_k: float
    per_query: list[QueryMetric]


# Health schema
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    db_connected: bool
    redis_connected: bool
    model_loaded: bool
    vector_store_count: int
