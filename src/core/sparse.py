"""Sparse retrieval using BM25."""
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.normalize import normalize_text


class BM25Retriever:
    """BM25-based sparse retrieval."""
    
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.corpus_ids: list[Any] = []
        self.tokenized_corpus: list[list[str]] = []
    
    def fit(self, corpus: list[str], ids: list[Any]):
        """
        Fit BM25 on a corpus.
        
        Args:
            corpus: List of documents (normalized)
            ids: List of document IDs
        """
        self.corpus_ids = ids
        
        # Tokenize corpus (simple whitespace)
        self.tokenized_corpus = [doc.split() for doc in corpus]
        
        # Fit BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query: str, k: int = 10) -> list[tuple[Any, float]]:
        """
        Search for top-k documents.
        
        Args:
            query: Query string (normalized)
            k: Number of results
        
        Returns:
            List of (id, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 not fitted. Call fit() first.")
        
        # Tokenize query
        tokenized_query = query.split()
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:k]
        
        # Return (id, score) pairs
        results = [(self.corpus_ids[i], float(scores[i])) for i in top_k_idx]
        
        return results


class TfidfRetriever:
    """TF-IDF based sparse retrieval (alternative to BM25)."""
    
    def __init__(self, ngram_range=(1, 2), max_features=10000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=False,  # Already normalized
        )
        self.corpus_ids: list[Any] = []
        self.tfidf_matrix = None
    
    def fit(self, corpus: list[str], ids: list[Any]):
        """Fit TF-IDF on corpus."""
        self.corpus_ids = ids
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
    
    def search(self, query: str, k: int = 10) -> list[tuple[Any, float]]:
        """Search using TF-IDF cosine similarity."""
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF not fitted. Call fit() first.")
        
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        
        # Get top-k
        top_k_idx = np.argsort(scores)[::-1][:k]
        
        results = [(self.corpus_ids[i], float(scores[i])) for i in top_k_idx]
        
        return results
