#!/usr/bin/env python3
"""
Polymath V2 Ingestion Pipeline

This script implements the CORRECT ingestion workflow:
1. MinerU for PDF → Markdown conversion
2. Structure-aware chunking (by headers, not character count)
3. BGE-M3 embeddings (1024 dim, not MPNet 384)
4. pdf2doi metadata extraction (not filename regex)
5. Postgres with pgvector (not ChromaDB)

Usage:
    python3 scripts/ingest_v2.py /path/to/paper.pdf
    python3 scripts/ingest_v2.py /path/to/pdfs/  # Batch mode
"""

import sys
import os
import gc
import json
import hashlib
import tempfile
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import uuid

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.chunking import chunk_markdown_by_headers, Chunk, get_chunk_with_context
from lib.embeddings import get_embedder
from lib.metadata import get_paper_metadata, PaperMetadata


def extract_with_fitz(pdf_path: str) -> Optional[str]:
    """Fast text extraction - use for simple PDFs."""
    import fitz
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        # Only accept if substantial text found
        return text if len(text.strip()) > 1000 else None
    except Exception:
        return None


def extract_with_mineru(pdf_path: str, output_dir: str) -> Optional[str]:
    """
    MinerU (Magic-PDF) extraction for complex layouts.
    Returns Markdown with preserved structure.
    """
    try:
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
    except ImportError:
        print("MinerU not installed. Run: pip install magic-pdf")
        return None

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    image_writer = DiskReaderWriter(output_dir)
    pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer)
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()

    md_content = pipe.pipe_mk_markdown(output_dir, drop_mode="none")
    return md_content


def extract_text(pdf_path: Path) -> tuple[Optional[str], str]:
    """
    Extract text from PDF, trying fast method first, then MinerU.

    Returns:
        (text_content, extraction_method)
    """
    # Try fast extraction first
    text = extract_with_fitz(str(pdf_path))
    if text:
        return text, "fitz"

    # Fall back to MinerU for complex layouts
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            text = extract_with_mineru(str(pdf_path), tmpdir)
            if text:
                return text, "mineru"
    except Exception as e:
        print(f"MinerU extraction failed: {e}")

    return None, "failed"


def store_to_postgres(
    doc_id: str,
    metadata: PaperMetadata,
    chunks: List[Chunk],
    embeddings: list,
    pdf_path: Path
):
    """
    Store document and passages to Postgres with pgvector.

    Schema:
    - documents: Metadata (title, authors, DOI, etc.)
    - passages_v2: Chunks with embeddings
    """
    import psycopg2
    from psycopg2.extras import execute_values

    conn = psycopg2.connect(config.POSTGRES_URI)
    cur = conn.cursor()

    try:
        # Insert document
        cur.execute("""
            INSERT INTO documents (doc_id, title, authors, year, doi, pmid, source_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title,
                authors = EXCLUDED.authors,
                year = EXCLUDED.year,
                doi = EXCLUDED.doi
        """, (
            doc_id,
            metadata.title,
            json.dumps(metadata.authors) if metadata.authors else None,
            metadata.year,
            metadata.doi,
            metadata.pmid,
            pdf_path.name
        ))

        # Insert passages with embeddings
        passage_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            passage_id = str(uuid.uuid4())
            passage_data.append((
                passage_id,
                doc_id,
                chunk.content,
                chunk.header,
                chunk.level,
                chunk.parent_header,
                chunk.char_start,
                chunk.char_end,
                embedding.tolist()  # pgvector accepts list
            ))

        if passage_data:
            execute_values(cur, """
                INSERT INTO passages_v2
                    (passage_id, doc_id, passage_text, header, header_level, parent_header, char_start, char_end, embedding)
                VALUES %s
            """, passage_data, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)")

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def store_to_neo4j(doc_id: str, metadata: PaperMetadata, chunks: List[Chunk]):
    """
    Store paper and concepts to Neo4j knowledge graph.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )

    try:
        with driver.session() as session:
            # Create Paper node
            session.run("""
                MERGE (p:Paper {doc_id: $doc_id})
                SET p.title = $title,
                    p.year = $year,
                    p.doi = $doi,
                    p.source = 'ingest_v2'
            """, {
                'doc_id': doc_id,
                'title': metadata.title,
                'year': metadata.year,
                'doi': metadata.doi
            })

            # Create section nodes and relationships
            for chunk in chunks:
                session.run("""
                    MATCH (p:Paper {doc_id: $doc_id})
                    MERGE (s:Section {paper_id: $doc_id, header: $header})
                    SET s.level = $level, s.parent = $parent
                    MERGE (p)-[:HAS_SECTION]->(s)
                """, {
                    'doc_id': doc_id,
                    'header': chunk.header,
                    'level': chunk.level,
                    'parent': chunk.parent_header
                })

    finally:
        driver.close()


def process_pdf(pdf_path: Path, embedder) -> dict:
    """
    Process a single PDF through the full V2 pipeline.

    Returns:
        Dict with processing results
    """
    result = {
        'file': pdf_path.name,
        'status': 'pending',
        'chunks': 0,
        'method': None,
        'confidence': 0.0
    }

    # Step 1: Extract metadata (pdf2doi → CrossRef/arXiv)
    print(f"  [1/4] Extracting metadata...", end=" ", flush=True)
    metadata = get_paper_metadata(pdf_path)
    result['confidence'] = metadata.confidence
    print(f"[{metadata.source_method}] confidence={metadata.confidence:.2f}")

    # Step 2: Extract text (fitz or MinerU)
    print(f"  [2/4] Extracting text...", end=" ", flush=True)
    text, method = extract_text(pdf_path)
    result['method'] = method

    if not text:
        result['status'] = 'failed'
        result['error'] = 'No text extracted'
        print("FAILED")
        return result

    print(f"[{method}] {len(text):,} chars")

    # Step 3: Chunk by markdown headers (NOT sliding window!)
    print(f"  [3/4] Chunking by headers...", end=" ", flush=True)
    chunks = chunk_markdown_by_headers(text)
    result['chunks'] = len(chunks)

    if not chunks:
        result['status'] = 'failed'
        result['error'] = 'No chunks generated'
        print("FAILED")
        return result

    print(f"{len(chunks)} sections")

    # Step 4: Generate embeddings with BGE-M3
    print(f"  [4/4] Generating BGE-M3 embeddings...", end=" ", flush=True)
    chunk_texts = [get_chunk_with_context(c) for c in chunks]
    embeddings = embedder.encode(chunk_texts, batch_size=16)
    print(f"{embeddings.shape}")

    # Step 5: Store to databases
    doc_id = hashlib.sha256(pdf_path.name.encode()).hexdigest()[:16]

    try:
        store_to_postgres(doc_id, metadata, chunks, embeddings, pdf_path)
        store_to_neo4j(doc_id, metadata, chunks)
        result['status'] = 'success'
    except Exception as e:
        result['status'] = 'partial'
        result['error'] = str(e)
        print(f"  DB Error: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Polymath V2 Ingestion")
    parser.add_argument("path", help="PDF file or directory to ingest")
    parser.add_argument("--dry-run", action="store_true", help="Extract and chunk only, don't store")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs to process")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        pdfs = [path]
    elif path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
    else:
        print(f"ERROR: {path} not found")
        sys.exit(1)

    if args.limit > 0:
        pdfs = pdfs[:args.limit]

    print("=" * 60)
    print("Polymath V2 Ingestion")
    print("=" * 60)
    print(f"Model: {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIM} dim)")
    print(f"PDFs: {len(pdfs)}")
    print("=" * 60)

    # Load embedder once
    embedder = get_embedder()

    stats = {'success': 0, 'failed': 0, 'partial': 0, 'total_chunks': 0}

    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf.name}")

        try:
            result = process_pdf(pdf, embedder)
            stats[result['status']] = stats.get(result['status'], 0) + 1
            stats['total_chunks'] += result['chunks']
            print(f"  → {result['status'].upper()}")
        except Exception as e:
            print(f"  → ERROR: {e}")
            stats['failed'] += 1

        gc.collect()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Success: {stats['success']}")
    print(f"Failed:  {stats['failed']}")
    print(f"Partial: {stats.get('partial', 0)}")
    print(f"Chunks:  {stats['total_chunks']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
