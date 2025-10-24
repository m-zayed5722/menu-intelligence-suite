-- Menu Intelligence Suite - Initial Schema
-- Requires PostgreSQL 15+ with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Items table (menu items from outlets)
CREATE TABLE IF NOT EXISTS items (
    item_id BIGSERIAL PRIMARY KEY,
    outlet_id BIGINT,
    outlet_name TEXT,
    city TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    title_en TEXT,
    title_ar TEXT,
    description TEXT,
    price NUMERIC(10, 2),
    cuisine_tags TEXT[],
    diet_tags TEXT[],
    title_norm TEXT,
    desc_norm TEXT,
    embedding vector(384),  -- Adjust dimension based on model
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Query labels for offline evaluation
CREATE TABLE IF NOT EXISTS query_labels (
    qid BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    relevant_ids BIGINT[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Deduplication pairs (ground truth)
CREATE TABLE IF NOT EXISTS dedup_pairs (
    a BIGINT,
    b BIGINT,
    is_duplicate BOOLEAN NOT NULL DEFAULT TRUE,
    score DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (a, b),
    CHECK (a < b)  -- Ensure canonical ordering
);

-- Deduplication clusters (computed)
CREATE TABLE IF NOT EXISTS dedup_clusters (
    item_id BIGINT PRIMARY KEY REFERENCES items(item_id),
    cluster_id BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User interactions (for recommendations)
CREATE TABLE IF NOT EXISTS user_interactions (
    user_id TEXT NOT NULL,
    item_id BIGINT NOT NULL REFERENCES items(item_id),
    interaction_type TEXT,  -- view, click, order
    timestamp TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, item_id, timestamp)
);

-- Metrics log (for observability)
CREATE TABLE IF NOT EXISTS metrics_log (
    log_id BIGSERIAL PRIMARY KEY,
    service TEXT NOT NULL,
    operation TEXT NOT NULL,
    duration_ms DOUBLE PRECISION,
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Create indexes (basic ones; more in 002_indexes.sql)
CREATE INDEX IF NOT EXISTS idx_items_outlet ON items(outlet_id);
CREATE INDEX IF NOT EXISTS idx_items_city ON items(city);
CREATE INDEX IF NOT EXISTS idx_query_labels_query ON query_labels(query);

COMMENT ON TABLE items IS 'Menu items with multilingual metadata and embeddings';
COMMENT ON TABLE query_labels IS 'Ground truth for search evaluation';
COMMENT ON TABLE dedup_pairs IS 'Known duplicate pairs for evaluation';
COMMENT ON TABLE dedup_clusters IS 'Computed duplicate clusters';
COMMENT ON TABLE user_interactions IS 'User interaction history for recommendations';
COMMENT ON TABLE metrics_log IS 'Performance and operation metrics';
