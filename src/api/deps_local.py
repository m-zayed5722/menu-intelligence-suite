"""
Simplified dependencies for local demo (no DB, no Redis).
"""
import pickle
from pathlib import Path
from functools import lru_cache
from typing import Optional

from src.core.embeddings import get_model
from src.core.sparse import BM25Retriever
from src.core.vector_store.faiss_store import FAISSVectorStore
from src.core.tagging import LabelTagger
from src.api.config import get_settings

# In-memory data store
_items_db = {}
_query_labels = []

# Cache directory
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_items_db():
    """Get in-memory items database."""
    return _items_db


def set_items_db(items):
    """Set in-memory items database."""
    global _items_db
    _items_db = items


def get_query_labels():
    """Get query labels for evaluation."""
    return _query_labels


def set_query_labels(labels):
    """Set query labels for evaluation."""
    global _query_labels
    _query_labels = labels


@lru_cache()
def get_vector_store() -> FAISSVectorStore:
    """Get cached FAISS vector store."""
    settings = get_settings()
    store = FAISSVectorStore(
        dimension=settings.embedding_dim,
        index_type="IVFFlat",
    )
    
    # Try to load from disk
    index_path = CACHE_DIR / "faiss_index.idx"
    meta_path = CACHE_DIR / "faiss_metadata.pkl"
    if index_path.exists():
        try:
            store.load(str(index_path))
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    data = pickle.load(f)
                    store.id_map = data["id_map"]
                    store.metadata = data.get("metadata", [])
            print(f"[OK] Loaded FAISS index with {store.count()} vectors")
        except Exception as e:
            print(f"Warning: Could not load index: {e}")
    
    return store


@lru_cache()
def get_bm25_retriever() -> Optional[BM25Retriever]:
    """Get cached BM25 retriever."""
    retriever_path = CACHE_DIR / "bm25_retriever.pkl"
    if retriever_path.exists():
        try:
            with open(retriever_path, "rb") as f:
                retriever = pickle.load(f)
            print(f"[OK] Loaded BM25 index with {len(retriever.corpus)} documents")
            return retriever
        except Exception as e:
            print(f"Warning: Could not load BM25: {e}")
    
    # Build from items
    items = get_items_db()
    if items:
        corpus = []
        ids = []
        for item_id, item in items.items():
            text = f"{item.get('title_norm', '')} {item.get('desc_norm', '')}"
            corpus.append(text)
            ids.append(item_id)
        
        retriever = BM25Retriever()
        retriever.fit(corpus, ids)
        
        # Save for next time
        with open(retriever_path, "wb") as f:
            pickle.dump(retriever, f)
        
        print(f"[OK] Built BM25 index with {len(corpus)} documents")
        return retriever
    
    return None


def build_sparse_index(items: dict):
    """Build BM25 index from items."""
    corpus = []
    ids = []
    for item_id, item in items.items():
        text = f"{item.get('title_norm', '')} {item.get('desc_norm', '')}"
        corpus.append(text)
        ids.append(item_id)
    
    retriever = BM25Retriever()
    retriever.fit(corpus, ids)
    
    # Save to disk
    retriever_path = CACHE_DIR / "bm25_retriever.pkl"
    with open(retriever_path, "wb") as f:
        pickle.dump(retriever, f)
    
    # Clear cache to reload
    get_bm25_retriever.cache_clear()
    
    return retriever


@lru_cache()
def get_label_tagger() -> LabelTagger:
    """Get cached label tagger."""
    model = get_model()
    tagger = LabelTagger(model)
    
    # Load default labels
    labels_path = Path("src/data/labels.json")
    if labels_path.exists():
        tagger.load_labels(str(labels_path))
        print(f"[OK] Loaded label tagger")
    
    return tagger


def save_vector_store():
    """Save FAISS index to disk."""
    store = get_vector_store()
    index_path = CACHE_DIR / "faiss_index.idx"
    meta_path = CACHE_DIR / "faiss_metadata.pkl"
    
    store.save(str(index_path))
    
    # Save metadata separately
    with open(meta_path, "wb") as f:
        pickle.dump({
            "id_map": store.id_map,
            "metadata": store.metadata,
        }, f)
    
    print(f"[OK] Saved FAISS index with {store.count()} vectors")


def check_health() -> dict:
    """Check system health."""
    return {
        "status": "healthy",
        "items_count": len(get_items_db()),
        "vector_count": get_vector_store().count(),
        "bm25_ready": get_bm25_retriever() is not None,
    }
