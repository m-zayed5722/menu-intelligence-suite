"""Tests for search functionality."""
import pytest

from src.core.normalize import normalize_text
from src.core.embeddings import encode_texts, cosine_similarity
from src.core.sparse import BM25Retriever
from src.core.hybrid import combine_scores


def test_normalize_arabic():
    """Test Arabic normalization."""
    # Test diacritics removal
    text_with_diac = "مَرْحَبًا"
    normalized = normalize_text(text_with_diac)
    assert "َ" not in normalized  # Fatha removed
    assert "ْ" not in normalized  # Sukun removed
    
    # Test digit conversion
    text_with_ar_digits = "٠١٢٣٤٥"
    normalized = normalize_text(text_with_ar_digits)
    assert "012345" in normalized
    
    # Test Alef normalization
    text_with_alef = "أإآ"
    normalized = normalize_text(text_with_alef)
    assert normalized == "ااا"


def test_embeddings():
    """Test embedding generation."""
    texts = ["chicken shawarma", "شاورما دجاج", "pizza"]
    embeddings = encode_texts(texts, normalize=True)
    
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] > 0
    
    # Test similarity
    sim = cosine_similarity(embeddings[0], embeddings[1])
    assert 0 <= sim <= 1


def test_sparse_retrieval():
    """Test BM25 sparse retrieval."""
    corpus = [
        "chicken shawarma wrap",
        "beef kebab plate",
        "chicken tikka masala",
        "pizza margherita",
    ]
    ids = [1, 2, 3, 4]
    
    retriever = BM25Retriever()
    retriever.fit(corpus, ids)
    
    # Search for chicken
    results = retriever.search("chicken", k=2)
    
    assert len(results) <= 2
    assert results[0][0] in [1, 3]  # Should return chicken items


def test_hybrid_scoring():
    """Test hybrid score combination."""
    sparse_results = [(1, 10.0), (2, 5.0), (3, 2.0)]
    dense_results = [(2, 0.9), (1, 0.7), (4, 0.5)]
    
    # Pure sparse (alpha=1.0)
    combined = combine_scores(sparse_results, dense_results, alpha=1.0)
    assert combined[0][0] == 1  # Highest sparse score
    
    # Pure dense (alpha=0.0)
    combined = combine_scores(sparse_results, dense_results, alpha=0.0)
    assert combined[0][0] == 2  # Highest dense score
    
    # Hybrid (alpha=0.5)
    combined = combine_scores(sparse_results, dense_results, alpha=0.5)
    assert len(combined) == 4


def test_arabic_vs_english_search():
    """Test that dense search improves Arabic queries."""
    # This is a placeholder - would need actual data
    # In practice, test that Arabic queries have higher recall with dense/hybrid
    pass


@pytest.mark.benchmark
def test_search_latency():
    """Benchmark search latency."""
    # Placeholder for performance test
    # Would measure p95 latency for sparse/dense/hybrid
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
