"""
Polymath V2 Library

Core modules for consistent knowledge ingestion:
- config: Single source of truth for configuration
- chunking: Structure-aware markdown chunking
- embeddings: BGE-M3 embeddings (1024 dim only)
- metadata: pdf2doi + CrossRef/arXiv lookup
"""

from .config import config
from .chunking import chunk_markdown_by_headers, Chunk
from .embeddings import get_embedder, embed_texts
from .metadata import get_paper_metadata, PaperMetadata

__all__ = [
    'config',
    'chunk_markdown_by_headers',
    'Chunk',
    'get_embedder',
    'embed_texts',
    'get_paper_metadata',
    'PaperMetadata',
]
