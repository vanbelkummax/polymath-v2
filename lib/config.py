#!/usr/bin/env python3
"""
Polymath V2 Configuration - Single Source of Truth

All scripts MUST import from here. No direct .env reading elsewhere.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Load .env from repo root
_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Config:
    """Polymath V2 Configuration."""

    # Embedding Model - MUST be BGE-M3 (1024 dim)
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024

    # Database Paths (Linux ext4 only - NEVER /mnt/)
    POSTGRES_URI: str = os.getenv("DATABASE_URL", "postgresql://polymath:polymath@localhost/polymath")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "polymathic2026")

    # ChromaDB (deprecated - migrating to pgvector)
    CHROMADB_PATH: str = "/home/user/polymath-repo/chromadb"

    # Zotero
    ZOTERO_API_KEY: Optional[str] = os.getenv("ZOTERO_API_KEY")
    ZOTERO_USER_ID: Optional[str] = os.getenv("ZOTERO_USER_ID")

    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

    # Paths
    PDF_ARCHIVE: Path = Path("/scratch/polymath_archive/pdfs")
    INGEST_STAGING: Path = Path("/home/user/work/polymax/ingest_staging")
    LOG_DIR: Path = Path("/home/user/work/polymax/logs")

    # Chunking Parameters (V2 - Structure-Aware)
    CHUNK_MAX_SIZE: int = 4000  # chars - only used as fallback for huge sections
    CHUNK_MIN_SIZE: int = 50   # chars - skip empty sections

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of errors."""
        errors = []

        if not self.ZOTERO_API_KEY:
            errors.append("ZOTERO_API_KEY not set")
        if not self.ZOTERO_USER_ID:
            errors.append("ZOTERO_USER_ID not set")
        if not self.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY not set (needed for concept extraction)")

        # Check paths
        if not self.PDF_ARCHIVE.exists():
            errors.append(f"PDF_ARCHIVE not found: {self.PDF_ARCHIVE}")

        return errors


# Global config instance
config = Config()
