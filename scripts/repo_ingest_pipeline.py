#!/usr/bin/env python3
"""
Automated Repository Ingestion Pipeline

This script provides end-to-end automation for:
1. Detecting GitHub repos mentioned in papers (during PDF ingestion)
2. Queuing repos for ingestion
3. Cloning and ingesting repos
4. Linking repos back to papers

Usage:
    # Queue repos from existing papers
    python scripts/repo_ingest_pipeline.py --queue-from-papers

    # Process queue (clone and ingest)
    python scripts/repo_ingest_pipeline.py --process-queue --limit 5

    # Show queue status
    python scripts/repo_ingest_pipeline.py --status

    # Queue a specific repo
    python scripts/repo_ingest_pipeline.py --queue https://github.com/owner/repo --doc-id UUID

Integration with paper ingestion:
    from scripts.repo_ingest_pipeline import queue_repos_from_paper
    queue_repos_from_paper(conn, doc_id, passage_text)
"""

import os
import re
import sys
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

REPO_CLONE_DIR = Path("/scratch/polymath_repos")  # Where to clone repos
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("PAT_2")
MAX_REPO_SIZE_MB = 500  # Skip repos larger than this

# =============================================================================
# Schema
# =============================================================================

SCHEMA_SQL = """
-- Queue for repos pending ingestion
CREATE TABLE IF NOT EXISTS repo_ingest_queue (
    id SERIAL PRIMARY KEY,
    repo_url TEXT NOT NULL UNIQUE,
    repo_owner TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    source_doc_id UUID REFERENCES documents(doc_id) ON DELETE SET NULL,
    status TEXT DEFAULT 'pending',  -- pending, cloning, ingesting, completed, failed, skipped
    priority INT DEFAULT 0,  -- Higher = more important
    clone_path TEXT,
    error_message TEXT,
    file_count INT,
    chunk_count INT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_repo_queue_status ON repo_ingest_queue(status);
CREATE INDEX IF NOT EXISTS idx_repo_queue_priority ON repo_ingest_queue(priority DESC, created_at);

-- Track which papers triggered which repo ingestions
CREATE TABLE IF NOT EXISTS paper_repo_triggers (
    id SERIAL PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    queue_id INT NOT NULL REFERENCES repo_ingest_queue(id) ON DELETE CASCADE,
    context TEXT,  -- Where in the paper the repo was mentioned
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(doc_id, queue_id)
);
"""


def ensure_schema(conn):
    """Create queue tables if they don't exist."""
    cur = conn.cursor()
    try:
        cur.execute(SCHEMA_SQL)
        conn.commit()
        logger.info("Schema ensured")
    except Exception as e:
        logger.error(f"Schema creation failed: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()


# =============================================================================
# GitHub URL Extraction (reused from link_paper_repos.py)
# =============================================================================

GITHUB_PATTERNS = [
    r'https?://(?:www\.)?github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)(?:/[^\s\)\]]*)?',
    r'(?<![a-zA-Z0-9])github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)(?:/[^\s\)\]]*)?',
]

# Blocklist for repos we don't want to ingest
REPO_BLOCKLIST = {
    'numpy/numpy', 'tensorflow/tensorflow', 'pytorch/pytorch',
    'pandas-dev/pandas', 'scikit-learn/scikit-learn', 'matplotlib/matplotlib',
    'python/cpython', 'torvalds/linux', 'facebook/react',
}


def extract_github_repos(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract GitHub repos from text.
    Returns: List of (url, owner, repo_name)
    """
    results = []
    seen = set()

    for pattern in GITHUB_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            owner = match.group(1)
            repo = match.group(2).rstrip('.').rstrip(',')

            # Skip common false positives
            if owner.lower() in ['issues', 'pull', 'blob', 'tree', 'wiki']:
                continue
            if repo.lower() in ['issues', 'pull', 'blob', 'tree']:
                continue

            repo_key = f"{owner}/{repo}".lower()

            # Skip blocklisted repos (common libraries)
            if repo_key in REPO_BLOCKLIST:
                continue

            if repo_key not in seen:
                seen.add(repo_key)
                url = f"https://github.com/{owner}/{repo}"
                results.append((url, owner, repo))

    return results


# =============================================================================
# Queue Management
# =============================================================================

def queue_repo(
    conn,
    repo_url: str,
    owner: str,
    repo_name: str,
    source_doc_id: Optional[str] = None,
    context: Optional[str] = None,
    priority: int = 0
) -> Optional[int]:
    """
    Add a repo to the ingestion queue.
    Returns queue_id if added, None if already exists.
    """
    cur = conn.cursor()

    try:
        # Insert or get existing
        cur.execute("""
            INSERT INTO repo_ingest_queue (repo_url, repo_owner, repo_name, priority)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (repo_url) DO UPDATE SET
                priority = GREATEST(repo_ingest_queue.priority, EXCLUDED.priority)
            RETURNING id, (xmax = 0) as is_new
        """, (repo_url, owner, repo_name, priority))

        row = cur.fetchone()
        queue_id = row[0]
        is_new = row[1]

        # Track which paper triggered this
        if source_doc_id:
            cur.execute("""
                INSERT INTO paper_repo_triggers (doc_id, queue_id, context)
                VALUES (%s, %s, %s)
                ON CONFLICT (doc_id, queue_id) DO NOTHING
            """, (source_doc_id, queue_id, context[:500] if context else None))

            # Also add to paper_repos if not exists
            cur.execute("""
                INSERT INTO paper_repos (doc_id, repo_url, repo_owner, repo_name, detection_method, confidence)
                VALUES (%s, %s, %s, %s, 'auto_queue', 0.8)
                ON CONFLICT (doc_id, repo_url) DO NOTHING
            """, (source_doc_id, repo_url, owner, repo_name))

        conn.commit()

        if is_new:
            logger.info(f"Queued new repo: {owner}/{repo_name}")
        return queue_id

    except Exception as e:
        logger.error(f"Failed to queue repo {repo_url}: {e}")
        conn.rollback()
        return None
    finally:
        cur.close()


def queue_repos_from_paper(conn, doc_id: str, text: str) -> int:
    """
    Extract and queue all repos mentioned in a paper's text.
    Call this during paper ingestion.
    Returns number of repos queued.
    """
    repos = extract_github_repos(text)
    queued = 0

    for url, owner, repo_name in repos:
        # Get context around the URL
        idx = text.lower().find(f"github.com/{owner}/{repo_name}".lower())
        context = text[max(0, idx-100):idx+100] if idx >= 0 else None

        if queue_repo(conn, url, owner, repo_name, doc_id, context):
            queued += 1

    return queued


def queue_from_existing_papers(conn) -> int:
    """Queue repos from all papers that have GitHub mentions."""
    cur = conn.cursor()

    # Find papers with GitHub URLs not yet processed
    cur.execute("""
        SELECT DISTINCT p.doc_id, string_agg(p.passage_text, ' ') as full_text
        FROM passages p
        WHERE p.passage_text ILIKE '%github.com/%'
          AND NOT EXISTS (
              SELECT 1 FROM paper_repo_triggers t WHERE t.doc_id = p.doc_id
          )
        GROUP BY p.doc_id
    """)

    total_queued = 0
    for doc_id, text in cur.fetchall():
        queued = queue_repos_from_paper(conn, str(doc_id), text)
        total_queued += queued
        if queued > 0:
            logger.info(f"Paper {doc_id}: queued {queued} repos")

    cur.close()
    return total_queued


def get_queue_status(conn) -> Dict:
    """Get queue statistics."""
    cur = conn.cursor()

    cur.execute("""
        SELECT status, COUNT(*)
        FROM repo_ingest_queue
        GROUP BY status
    """)
    status_counts = dict(cur.fetchall())

    cur.execute("""
        SELECT COUNT(*) FROM repo_ingest_queue
    """)
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT repo_owner || '/' || repo_name, status, created_at
        FROM repo_ingest_queue
        WHERE status = 'pending'
        ORDER BY priority DESC, created_at
        LIMIT 10
    """)
    next_in_queue = cur.fetchall()

    cur.close()

    return {
        'total': total,
        'by_status': status_counts,
        'next_up': next_in_queue
    }


# =============================================================================
# Repo Cloning and Ingestion
# =============================================================================

def clone_repo(repo_url: str, owner: str, repo_name: str) -> Optional[Path]:
    """
    Clone a GitHub repo.
    Returns path to cloned repo or None if failed.
    """
    REPO_CLONE_DIR.mkdir(parents=True, exist_ok=True)
    clone_path = REPO_CLONE_DIR / f"{owner}__{repo_name}"

    # Skip if already cloned
    if clone_path.exists():
        logger.info(f"Repo already cloned: {clone_path}")
        return clone_path

    # Build clone URL with token if available
    if GITHUB_TOKEN:
        clone_url = f"https://{GITHUB_TOKEN}@github.com/{owner}/{repo_name}.git"
    else:
        clone_url = repo_url

    try:
        # Shallow clone to save space
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )

        if result.returncode != 0:
            logger.error(f"Clone failed: {result.stderr}")
            return None

        logger.info(f"Cloned {owner}/{repo_name} to {clone_path}")
        return clone_path

    except subprocess.TimeoutExpired:
        logger.error(f"Clone timed out for {owner}/{repo_name}")
        if clone_path.exists():
            shutil.rmtree(clone_path)
        return None
    except Exception as e:
        logger.error(f"Clone error: {e}")
        return None


def get_repo_size(clone_path: Path) -> int:
    """Get repo size in MB."""
    total = 0
    for f in clone_path.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total // (1024 * 1024)


def ingest_repo(clone_path: Path, repo_name: str, conn) -> Tuple[int, int]:
    """
    Ingest a cloned repo into the database.
    Returns (file_count, chunk_count).
    """
    # Import here to avoid circular dependency
    sys.path.insert(0, str(Path("/home/user/polymath-repo")))

    try:
        from lib.unified_ingest import TransactionalIngestor
    except ImportError:
        logger.error("Could not import TransactionalIngestor")
        # Fallback to simpler ingestion
        return ingest_repo_simple(clone_path, repo_name, conn)

    ingestor = TransactionalIngestor(use_ocr=False)

    # Find code files
    code_extensions = ['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h']
    files = []
    for ext in code_extensions:
        files.extend(clone_path.glob(f"**/*{ext}"))

    # Filter out test files, etc.
    skip_patterns = ['__pycache__', 'node_modules', '.git', 'test_', '_test.py', 'venv']
    files = [f for f in files if not any(p in str(f) for p in skip_patterns)]

    file_count = len(files)
    chunk_count = 0

    for f in files[:500]:  # Limit to 500 files per repo
        try:
            result = ingestor.ingest_code_file(str(f), repo_name=repo_name)
            if result and hasattr(result, 'chunk_count'):
                chunk_count += result.chunk_count
        except Exception as e:
            logger.warning(f"Failed to ingest {f}: {e}")

    return file_count, chunk_count


def ingest_repo_simple(clone_path: Path, repo_name: str, conn) -> Tuple[int, int]:
    """Simple fallback ingestion using direct database writes."""
    import hashlib
    import uuid

    cur = conn.cursor()

    code_extensions = ['.py', '.js', '.ts', '.java', '.go', '.rs']
    skip_patterns = ['__pycache__', 'node_modules', '.git', 'test_', 'venv']

    files = []
    for ext in code_extensions:
        files.extend(clone_path.glob(f"**/*{ext}"))

    files = [f for f in files if not any(p in str(f) for p in skip_patterns)]

    file_count = 0
    chunk_count = 0

    for f in files[:500]:
        try:
            content = f.read_text(errors='ignore')
            if not content.strip():
                continue

            rel_path = str(f.relative_to(clone_path))
            file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            # Insert file
            cur.execute("""
                INSERT INTO code_files (repo_name, file_path, language, file_hash, loc)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (repo_name, file_path, head_commit_sha) DO UPDATE
                    SET file_hash = EXCLUDED.file_hash
                RETURNING file_id
            """, (repo_name, rel_path, f.suffix[1:], file_hash, len(content.splitlines())))

            file_id = cur.fetchone()[0]
            file_count += 1

            # Simple chunking by function/class
            chunks = simple_chunk_code(content, f.suffix)

            for chunk_type, name, chunk_text, start_line, end_line in chunks:
                chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
                cur.execute("""
                    INSERT INTO code_chunks (file_id, chunk_type, name, start_line, end_line, content, chunk_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (file_id, chunk_type, name, start_line, end_line, chunk_text, chunk_hash))
                chunk_count += 1

            conn.commit()

        except Exception as e:
            logger.warning(f"Failed to ingest {f}: {e}")
            conn.rollback()

    cur.close()
    return file_count, chunk_count


def simple_chunk_code(content: str, extension: str) -> List[Tuple[str, str, str, int, int]]:
    """Simple code chunking - extract functions and classes."""
    chunks = []
    lines = content.splitlines()

    if extension in ['.py']:
        # Python: look for def and class
        current_chunk = []
        current_type = None
        current_name = None
        start_line = 0

        for i, line in enumerate(lines):
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                if current_chunk and current_name:
                    chunks.append((current_type, current_name, '\n'.join(current_chunk), start_line, i-1))
                current_type = 'function'
                match = re.match(r'(?:async\s+)?def\s+(\w+)', line.strip())
                current_name = match.group(1) if match else 'unknown'
                current_chunk = [line]
                start_line = i
            elif line.strip().startswith('class '):
                if current_chunk and current_name:
                    chunks.append((current_type, current_name, '\n'.join(current_chunk), start_line, i-1))
                current_type = 'class'
                match = re.match(r'class\s+(\w+)', line.strip())
                current_name = match.group(1) if match else 'unknown'
                current_chunk = [line]
                start_line = i
            elif current_chunk:
                current_chunk.append(line)

        if current_chunk and current_name:
            chunks.append((current_type, current_name, '\n'.join(current_chunk), start_line, len(lines)-1))

    # If no chunks found, chunk the whole file
    if not chunks and len(content) > 100:
        chunks.append(('file', 'module', content[:10000], 0, len(lines)-1))

    return chunks


def process_queue_item(conn, queue_id: int) -> bool:
    """Process a single item from the queue."""
    cur = conn.cursor()

    # Get queue item
    cur.execute("""
        SELECT repo_url, repo_owner, repo_name
        FROM repo_ingest_queue
        WHERE id = %s AND status = 'pending'
        FOR UPDATE
    """, (queue_id,))

    row = cur.fetchone()
    if not row:
        cur.close()
        return False

    repo_url, owner, repo_name = row

    # Update status to cloning
    cur.execute("""
        UPDATE repo_ingest_queue
        SET status = 'cloning', started_at = NOW()
        WHERE id = %s
    """, (queue_id,))
    conn.commit()

    try:
        # Clone
        clone_path = clone_repo(repo_url, owner, repo_name)
        if not clone_path:
            cur.execute("""
                UPDATE repo_ingest_queue
                SET status = 'failed', error_message = 'Clone failed'
                WHERE id = %s
            """, (queue_id,))
            conn.commit()
            cur.close()
            return False

        # Check size
        size_mb = get_repo_size(clone_path)
        if size_mb > MAX_REPO_SIZE_MB:
            cur.execute("""
                UPDATE repo_ingest_queue
                SET status = 'skipped', error_message = %s
                WHERE id = %s
            """, (f'Repo too large: {size_mb}MB', queue_id))
            conn.commit()
            cur.close()
            return False

        # Update status to ingesting
        cur.execute("""
            UPDATE repo_ingest_queue
            SET status = 'ingesting', clone_path = %s
            WHERE id = %s
        """, (str(clone_path), queue_id))
        conn.commit()

        # Ingest
        file_count, chunk_count = ingest_repo(clone_path, repo_name, conn)

        # Update to completed
        cur.execute("""
            UPDATE repo_ingest_queue
            SET status = 'completed', completed_at = NOW(),
                file_count = %s, chunk_count = %s
            WHERE id = %s
        """, (file_count, chunk_count, queue_id))
        conn.commit()

        logger.info(f"Completed {owner}/{repo_name}: {file_count} files, {chunk_count} chunks")
        cur.close()
        return True

    except Exception as e:
        logger.error(f"Error processing {owner}/{repo_name}: {e}")
        cur.execute("""
            UPDATE repo_ingest_queue
            SET status = 'failed', error_message = %s
            WHERE id = %s
        """, (str(e)[:500], queue_id))
        conn.commit()
        cur.close()
        return False


def process_queue(conn, limit: int = 10) -> int:
    """Process pending items in the queue."""
    cur = conn.cursor()

    cur.execute("""
        SELECT id FROM repo_ingest_queue
        WHERE status = 'pending'
        ORDER BY priority DESC, created_at
        LIMIT %s
    """, (limit,))

    queue_ids = [row[0] for row in cur.fetchall()]
    cur.close()

    processed = 0
    for qid in queue_ids:
        if process_queue_item(conn, qid):
            processed += 1

    return processed


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Automated repo ingestion pipeline')
    parser.add_argument('--queue-from-papers', action='store_true',
                       help='Queue repos from existing papers')
    parser.add_argument('--process-queue', action='store_true',
                       help='Process pending queue items')
    parser.add_argument('--limit', type=int, default=10,
                       help='Max items to process (default: 10)')
    parser.add_argument('--status', action='store_true',
                       help='Show queue status')
    parser.add_argument('--queue', type=str,
                       help='Queue a specific repo URL')
    parser.add_argument('--doc-id', type=str,
                       help='Associate queued repo with a document')

    args = parser.parse_args()

    conn = psycopg2.connect(config.POSTGRES_URI)

    try:
        ensure_schema(conn)

        if args.queue_from_papers:
            logger.info("Queuing repos from existing papers...")
            queued = queue_from_existing_papers(conn)
            logger.info(f"Queued {queued} repos from papers")

        if args.queue:
            # Parse repo URL
            match = re.match(r'https?://github\.com/([^/]+)/([^/]+)', args.queue)
            if not match:
                print(f"Invalid GitHub URL: {args.queue}")
                return 1
            owner, repo_name = match.groups()
            repo_name = repo_name.rstrip('.git')

            queue_id = queue_repo(conn, args.queue, owner, repo_name, args.doc_id)
            if queue_id:
                print(f"Queued {owner}/{repo_name} (id: {queue_id})")

        if args.process_queue:
            logger.info(f"Processing up to {args.limit} queue items...")
            processed = process_queue(conn, args.limit)
            logger.info(f"Processed {processed} repos")

        if args.status:
            status = get_queue_status(conn)
            print("\n=== Repo Ingestion Queue Status ===")
            print(f"Total: {status['total']}")
            print("\nBy status:")
            for s, count in status['by_status'].items():
                print(f"  {s}: {count}")
            print("\nNext in queue:")
            for repo, st, created in status['next_up']:
                print(f"  {repo} ({st}) - queued {created}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
