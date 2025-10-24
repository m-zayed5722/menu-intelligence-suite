-- Additional indexes and optimizations for MIS

-- Vector similarity search index (IVFFlat)
-- Adjust 'lists' parameter based on dataset size (rule of thumb: rows/1000)
CREATE INDEX IF NOT EXISTS idx_items_embedding_ivfflat 
ON items 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Alternative: HNSW index (better quality, more memory)
-- Uncomment if needed:
-- CREATE INDEX IF NOT EXISTS idx_items_embedding_hnsw 
-- ON items 
-- USING hnsw (embedding vector_cosine_ops) 
-- WITH (m = 16, ef_construction = 64);

-- Text search indexes
CREATE INDEX IF NOT EXISTS idx_items_title_norm ON items(title_norm);
CREATE INDEX IF NOT EXISTS idx_items_desc_norm ON items(desc_norm);

-- GIN indexes for array fields
CREATE INDEX IF NOT EXISTS idx_items_cuisine_tags ON items USING GIN(cuisine_tags);
CREATE INDEX IF NOT EXISTS idx_items_diet_tags ON items USING GIN(diet_tags);

-- Spatial index for location-based queries
CREATE INDEX IF NOT EXISTS idx_items_location ON items(lat, lon);

-- Dedup cluster lookup
CREATE INDEX IF NOT EXISTS idx_dedup_clusters_cluster_id ON dedup_clusters(cluster_id);

-- User interactions indexes
CREATE INDEX IF NOT EXISTS idx_user_interactions_user ON user_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_item ON user_interactions(item_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_time ON user_interactions(timestamp DESC);

-- Metrics indexes
CREATE INDEX IF NOT EXISTS idx_metrics_log_operation ON metrics_log(operation);
CREATE INDEX IF NOT EXISTS idx_metrics_log_timestamp ON metrics_log(timestamp DESC);

-- Partial index for items with embeddings
CREATE INDEX IF NOT EXISTS idx_items_with_embedding 
ON items(item_id) 
WHERE embedding IS NOT NULL;

-- Update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER items_updated_at
BEFORE UPDATE ON items
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- Vacuum and analyze for statistics
ANALYZE items;
ANALYZE query_labels;
ANALYZE dedup_pairs;
ANALYZE dedup_clusters;
