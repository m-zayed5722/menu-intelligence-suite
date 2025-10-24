"""Multilingual embedding utilities using sentence-transformers."""
import functools
import os
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.normalize import normalize_batch

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


@functools.lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Load and cache the sentence transformer model.
    
    Returns:
        Loaded SentenceTransformer model
    """
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    return model


def encode_texts(
    texts: list[str],
    normalize: bool = True,
    normalize_embeddings: bool = True,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode texts to embeddings.
    
    Args:
        texts: List of texts to encode
        normalize: Whether to normalize text before encoding
        normalize_embeddings: Whether to L2-normalize embeddings
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    
    Returns:
        Array of embeddings (N x D)
    """
    if not texts:
        return np.array([]).reshape(0, EMBEDDING_DIM)
    
    # Normalize text if requested
    if normalize:
        texts = normalize_batch(texts)
    
    # Get model and encode
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    
    return np.asarray(embeddings, dtype="float32")


def encode_single(text: str, normalize: bool = True) -> np.ndarray:
    """Encode a single text to embedding vector."""
    return encode_texts([text], normalize=normalize)[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: Vector 1
        b: Vector 2
    
    Returns:
        Cosine similarity [-1, 1]
    """
    if a.ndim == 1 and b.ndim == 1:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    raise ValueError("Inputs must be 1D vectors")


def batch_cosine_similarity(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities.
    
    Args:
        queries: Query embeddings (N x D)
        docs: Document embeddings (M x D)
    
    Returns:
        Similarity matrix (N x M)
    """
    # Normalize if not already
    queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    docs_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9)
    
    return queries_norm @ docs_norm.T
