#!/usr/bin/env python3
"""
Polymath V2 Embeddings - BGE-M3 Only

CRITICAL: All embeddings MUST use BGE-M3 (1024 dimensions).
Using any other model will create incompatible vectors.

This module provides a singleton embedder to avoid reloading the model.
"""

from typing import List, Union, Optional
import numpy as np
from .config import config

# Singleton instance
_embedder = None


class BGEEmbedder:
    """
    BGE-M3 Embedder with singleton pattern.

    BGE-M3 advantages over MPNet:
    - 1024 dimensions (vs 384) = better semantic capture
    - Multi-lingual support
    - Better for scientific/technical text
    - Dense + sparse + ColBERT representations (we use dense)
    """

    def __init__(self, use_fp16: bool = True):
        """
        Initialize BGE-M3 embedder.

        Args:
            use_fp16: Use FP16 for faster inference (recommended for GPU)
        """
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding not installed. Run: pip install FlagEmbedding"
            )

        self.model_name = config.EMBEDDING_MODEL
        self.dim = config.EMBEDDING_DIM

        print(f"Loading {self.model_name}...")
        self.model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16)
        print(f"BGE-M3 loaded ({self.dim} dimensions)")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings (n_texts, 1024)
        """
        if isinstance(texts, str):
            texts = [texts]

        # BGE-M3 returns dict with 'dense_vecs', 'sparse', 'colbert'
        # We only use dense embeddings
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )

        # Handle both dict output (full model) and array output (dense only)
        if isinstance(output, dict):
            embeddings = output['dense_vecs']
        else:
            embeddings = output

        return np.array(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query.
        Same as encode() but semantically clearer for search operations.
        """
        return self.encode(query)[0]

    def similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query vector (1024,)
            doc_embeddings: Document vectors (n_docs, 1024)

        Returns:
            Similarity scores (n_docs,)
        """
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Cosine similarity
        return np.dot(doc_norms, query_norm)


def get_embedder() -> BGEEmbedder:
    """
    Get singleton BGE-M3 embedder instance.

    Usage:
        from lib.embeddings import get_embedder
        embedder = get_embedder()
        vectors = embedder.encode(["text1", "text2"])
    """
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder


def embed_texts(texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
    """
    Convenience function to embed texts without managing embedder instance.

    Usage:
        from lib.embeddings import embed_texts
        vectors = embed_texts(["text1", "text2"])
    """
    embedder = get_embedder()
    return embedder.encode(texts, batch_size=batch_size)


# Validation at import time
def _validate_model():
    """Ensure config specifies BGE-M3."""
    if "bge-m3" not in config.EMBEDDING_MODEL.lower():
        raise ValueError(
            f"EMBEDDING_MODEL must be BGE-M3, got: {config.EMBEDDING_MODEL}\n"
            "Using other models will create incompatible vectors!"
        )

_validate_model()
