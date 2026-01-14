#!/usr/bin/env python3
"""
Paper-Repository Linker

Automatically detects GitHub repositories mentioned in papers and creates
linkages for integrated code+paper search.

Detection strategies:
1. Pattern matching for GitHub URLs in passage text
2. "Code availability" section parsing
3. Footnote/reference extraction (e.g., "Code available at...")

Usage:
    python scripts/link_paper_repos.py --scan        # Scan and extract repos
    python scripts/link_paper_repos.py --stats       # Show linking statistics
    python scripts/link_paper_repos.py --verify      # Verify repo URLs are valid
"""

import re
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RepoLink:
    """A detected paper-repository link."""
    doc_id: str
    repo_url: str
    repo_owner: str
    repo_name: str
    detection_method: str  # 'url_pattern', 'code_availability', 'footnote'
    confidence: float
    context: str  # Surrounding text where URL was found


# =============================================================================
# Schema Setup
# =============================================================================

SCHEMA_SQL = """
-- Paper-Repository linkage table
CREATE TABLE IF NOT EXISTS paper_repos (
    id SERIAL PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    repo_url TEXT NOT NULL,
    repo_owner TEXT,
    repo_name TEXT,
    detection_method TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    context TEXT,
    verified BOOLEAN DEFAULT FALSE,
    verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(doc_id, repo_url)
);

-- Index for fast lookup
CREATE INDEX IF NOT EXISTS idx_paper_repos_doc ON paper_repos(doc_id);
CREATE INDEX IF NOT EXISTS idx_paper_repos_repo ON paper_repos(repo_owner, repo_name);

-- View to easily query paper-repo relationships
CREATE OR REPLACE VIEW v_paper_code AS
SELECT
    d.doc_id,
    d.title,
    d.doi,
    d.year,
    pr.repo_url,
    pr.repo_owner,
    pr.repo_name,
    pr.confidence,
    pr.verified,
    COALESCE(cc.chunk_count, 0) as chunk_count
FROM documents d
JOIN paper_repos pr ON d.doc_id = pr.doc_id
LEFT JOIN LATERAL (
    SELECT COUNT(*) as chunk_count
    FROM code_files cf
    JOIN code_chunks ch ON cf.file_id = ch.file_id
    WHERE cf.repo_name = pr.repo_name
) cc ON true;
"""


def ensure_schema(conn):
    """Create the paper_repos table if it doesn't exist."""
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
# GitHub URL Extraction
# =============================================================================

# Patterns for detecting GitHub URLs
GITHUB_PATTERNS = [
    # Standard GitHub URLs
    r'https?://(?:www\.)?github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)(?:/[^\s\)\]]*)?',
    # GitHub URLs without protocol
    r'(?<![a-zA-Z0-9])github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)(?:/[^\s\)\]]*)?',
]

# Patterns indicating official code release
CODE_AVAILABILITY_PATTERNS = [
    r'code\s+(?:is\s+)?available\s+at\s+([^\s]+github[^\s\)]+)',
    r'(?:our\s+)?code\s+(?:can\s+be\s+)?(?:found|accessed)\s+at\s+([^\s]+github[^\s\)]+)',
    r'implementation\s+(?:is\s+)?available\s+at\s+([^\s]+github[^\s\)]+)',
    r'source\s+code[:\s]+([^\s]+github[^\s\)]+)',
    r'github[:\s]+([^\s]+github\.com[^\s\)]+)',
]


def extract_github_urls(text: str) -> List[Tuple[str, str, str, str]]:
    """
    Extract GitHub URLs from text.

    Returns:
        List of (full_url, owner, repo_name, detection_method)
    """
    results = []
    seen_repos = set()

    # Method 1: Direct URL pattern matching
    for pattern in GITHUB_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            owner = match.group(1)
            repo = match.group(2).rstrip('.').rstrip(',')  # Clean trailing punctuation

            # Skip common false positives
            if owner.lower() in ['issues', 'pull', 'blob', 'tree', 'wiki']:
                continue
            if repo.lower() in ['issues', 'pull', 'blob', 'tree']:
                continue

            repo_key = f"{owner}/{repo}".lower()
            if repo_key not in seen_repos:
                seen_repos.add(repo_key)
                full_url = f"https://github.com/{owner}/{repo}"
                results.append((full_url, owner, repo, 'url_pattern'))

    # Method 2: Code availability statements (higher confidence)
    for pattern in CODE_AVAILABILITY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            url_text = match.group(1) if match.lastindex else match.group(0)

            # Extract owner/repo from the matched URL
            for url_pattern in GITHUB_PATTERNS:
                url_match = re.search(url_pattern, url_text, re.IGNORECASE)
                if url_match:
                    owner = url_match.group(1)
                    repo = url_match.group(2).rstrip('.').rstrip(',')
                    repo_key = f"{owner}/{repo}".lower()

                    if repo_key not in seen_repos:
                        seen_repos.add(repo_key)
                        full_url = f"https://github.com/{owner}/{repo}"
                        results.append((full_url, owner, repo, 'code_availability'))
                    break

    return results


def get_context(text: str, url: str, context_chars: int = 200) -> str:
    """Extract surrounding context for a URL."""
    idx = text.lower().find(url.lower().replace('https://', '').replace('http://', ''))
    if idx == -1:
        # Try finding just the repo name
        parts = url.split('/')
        if len(parts) >= 2:
            idx = text.lower().find(parts[-1].lower())

    if idx == -1:
        return ""

    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(url) + context_chars)
    return text[start:end].strip()


# =============================================================================
# Database Operations
# =============================================================================

def scan_passages_for_repos(conn, batch_size: int = 1000) -> List[RepoLink]:
    """Scan all passages for GitHub URLs."""
    cur = conn.cursor()
    links = []

    # Get total count
    cur.execute("SELECT COUNT(DISTINCT doc_id) FROM passages WHERE passage_text ILIKE '%github%'")
    total_docs = cur.fetchone()[0]
    logger.info(f"Scanning {total_docs} documents with GitHub mentions...")

    # Process in batches
    cur.execute("""
        SELECT doc_id, string_agg(passage_text, ' ') as full_text
        FROM passages
        WHERE passage_text ILIKE '%github%'
        GROUP BY doc_id
    """)

    processed = 0
    for doc_id, full_text in cur:
        extracted = extract_github_urls(full_text)

        for url, owner, repo, method in extracted:
            context = get_context(full_text, url)

            # Assign confidence based on detection method
            confidence = 0.9 if method == 'code_availability' else 0.7

            links.append(RepoLink(
                doc_id=str(doc_id),
                repo_url=url,
                repo_owner=owner,
                repo_name=repo,
                detection_method=method,
                confidence=confidence,
                context=context[:500]  # Truncate context
            ))

        processed += 1
        if processed % 100 == 0:
            logger.info(f"Processed {processed}/{total_docs} documents, found {len(links)} links")

    cur.close()
    logger.info(f"Scan complete: {len(links)} repo links found in {total_docs} documents")
    return links


def save_links(conn, links: List[RepoLink]) -> int:
    """Save extracted links to database."""
    if not links:
        return 0

    cur = conn.cursor()

    # Prepare data for bulk insert
    data = [
        (
            link.doc_id,
            link.repo_url,
            link.repo_owner,
            link.repo_name,
            link.detection_method,
            link.confidence,
            link.context
        )
        for link in links
    ]

    try:
        execute_values(cur, """
            INSERT INTO paper_repos
                (doc_id, repo_url, repo_owner, repo_name, detection_method, confidence, context)
            VALUES %s
            ON CONFLICT (doc_id, repo_url) DO UPDATE SET
                confidence = GREATEST(paper_repos.confidence, EXCLUDED.confidence),
                detection_method = CASE
                    WHEN EXCLUDED.detection_method = 'code_availability' THEN 'code_availability'
                    ELSE paper_repos.detection_method
                END
        """, data)
        conn.commit()
        logger.info(f"Saved {len(links)} links to database")
        return len(links)
    except Exception as e:
        logger.error(f"Failed to save links: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()


def get_stats(conn) -> Dict:
    """Get linking statistics."""
    cur = conn.cursor()

    stats = {}

    # Total links
    cur.execute("SELECT COUNT(*) FROM paper_repos")
    stats['total_links'] = cur.fetchone()[0]

    # Unique papers with repos
    cur.execute("SELECT COUNT(DISTINCT doc_id) FROM paper_repos")
    stats['papers_with_repos'] = cur.fetchone()[0]

    # Unique repos
    cur.execute("SELECT COUNT(DISTINCT repo_url) FROM paper_repos")
    stats['unique_repos'] = cur.fetchone()[0]

    # By detection method
    cur.execute("""
        SELECT detection_method, COUNT(*)
        FROM paper_repos
        GROUP BY detection_method
    """)
    stats['by_method'] = dict(cur.fetchall())

    # Papers with indexed code
    cur.execute("""
        SELECT COUNT(DISTINCT pr.doc_id)
        FROM paper_repos pr
        JOIN code_files cf ON pr.repo_name = cf.repo_name
    """)
    stats['papers_with_indexed_code'] = cur.fetchone()[0]

    # Top repos by paper count
    cur.execute("""
        SELECT repo_owner || '/' || repo_name as repo, COUNT(DISTINCT doc_id) as papers
        FROM paper_repos
        GROUP BY repo_owner, repo_name
        ORDER BY papers DESC
        LIMIT 10
    """)
    stats['top_repos'] = cur.fetchall()

    cur.close()
    return stats


# =============================================================================
# Repo Search Integration
# =============================================================================

def get_repos_for_paper(conn, doc_id: str) -> List[Dict]:
    """Get all repos linked to a paper."""
    cur = conn.cursor()
    cur.execute("""
        SELECT repo_url, repo_owner, repo_name, confidence, verified
        FROM paper_repos
        WHERE doc_id = %s
        ORDER BY confidence DESC
    """, (doc_id,))

    repos = []
    for row in cur.fetchall():
        repos.append({
            'url': row[0],
            'owner': row[1],
            'name': row[2],
            'confidence': row[3],
            'verified': row[4]
        })

    cur.close()
    return repos


def get_papers_for_repo(conn, repo_name: str) -> List[Dict]:
    """Get all papers linked to a repo."""
    cur = conn.cursor()
    cur.execute("""
        SELECT d.doc_id, d.title, d.year, d.doi, pr.confidence
        FROM paper_repos pr
        JOIN documents d ON pr.doc_id = d.doc_id
        WHERE pr.repo_name = %s OR pr.repo_url ILIKE %s
        ORDER BY pr.confidence DESC, d.year DESC
    """, (repo_name, f'%/{repo_name}%'))

    papers = []
    for row in cur.fetchall():
        papers.append({
            'doc_id': str(row[0]),
            'title': row[1],
            'year': row[2],
            'doi': row[3],
            'confidence': row[4]
        })

    cur.close()
    return papers


def search_paper_with_code(conn, doc_id: str, query: str) -> Dict:
    """
    Search both paper passages and linked repo code for a query.

    Returns combined results from paper and associated code.
    """
    cur = conn.cursor()

    results = {
        'paper_passages': [],
        'code_chunks': [],
        'repos': []
    }

    # Get paper passages
    cur.execute("""
        SELECT passage_id, passage_text, section,
               ts_rank(to_tsvector('english', passage_text),
                       plainto_tsquery('english', %s)) as score
        FROM passages
        WHERE doc_id = %s
          AND to_tsvector('english', passage_text) @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT 10
    """, (query, doc_id, query))

    for row in cur.fetchall():
        results['paper_passages'].append({
            'passage_id': str(row[0]),
            'text': row[1][:500],
            'section': row[2],
            'score': float(row[3])
        })

    # Get linked repos
    repos = get_repos_for_paper(conn, doc_id)
    results['repos'] = repos

    # Search code in linked repos
    if repos:
        repo_names = [r['name'] for r in repos]
        cur.execute("""
            SELECT ch.chunk_id, cf.repo_name, cf.file_path, ch.content as chunk_text, ch.chunk_type,
                   ts_rank(ch.search_vector, plainto_tsquery('english', %s)) as score
            FROM code_chunks ch
            JOIN code_files cf ON ch.file_id = cf.file_id
            WHERE cf.repo_name = ANY(%s)
              AND ch.search_vector @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT 10
        """, (query, repo_names, query))

        for row in cur.fetchall():
            results['code_chunks'].append({
                'chunk_id': str(row[0]),
                'repo': row[1],
                'file': row[2],
                'code': row[3][:500],
                'type': row[4],
                'score': float(row[5])
            })

    cur.close()
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Link papers to GitHub repositories')
    parser.add_argument('--scan', action='store_true', help='Scan passages and extract repo links')
    parser.add_argument('--stats', action='store_true', help='Show linking statistics')
    parser.add_argument('--verify', action='store_true', help='Verify repo URLs are accessible')
    parser.add_argument('--search', type=str, help='Search paper+code for a doc_id')
    parser.add_argument('--query', type=str, default='', help='Query for --search')

    args = parser.parse_args()

    conn = psycopg2.connect(config.POSTGRES_URI)

    try:
        ensure_schema(conn)

        if args.scan:
            logger.info("Scanning passages for GitHub repositories...")
            links = scan_passages_for_repos(conn)
            if links:
                save_links(conn, links)
                logger.info(f"Extraction complete: {len(links)} links saved")
            else:
                logger.info("No new links found")

        if args.stats:
            stats = get_stats(conn)
            print("\n=== Paper-Repository Linking Statistics ===")
            print(f"Total links: {stats['total_links']}")
            print(f"Papers with repos: {stats['papers_with_repos']}")
            print(f"Unique repos: {stats['unique_repos']}")
            print(f"Papers with indexed code: {stats['papers_with_indexed_code']}")
            print(f"\nBy detection method:")
            for method, count in stats.get('by_method', {}).items():
                print(f"  {method}: {count}")
            print(f"\nTop repos by paper count:")
            for repo, count in stats.get('top_repos', []):
                print(f"  {repo}: {count} papers")

        if args.search:
            if not args.query:
                print("Please provide --query with --search")
                return

            results = search_paper_with_code(conn, args.search, args.query)
            print(f"\n=== Search Results for '{args.query}' ===")
            print(f"\nPaper passages ({len(results['paper_passages'])}):")
            for p in results['paper_passages'][:3]:
                print(f"  [{p['section']}] {p['text'][:200]}...")

            print(f"\nLinked repos ({len(results['repos'])}):")
            for r in results['repos']:
                print(f"  {r['url']} (confidence: {r['confidence']:.2f})")

            print(f"\nCode chunks ({len(results['code_chunks'])}):")
            for c in results['code_chunks'][:3]:
                print(f"  [{c['repo']}] {c['file']}")
                print(f"    {c['code'][:150]}...")

        if args.verify:
            logger.info("Verification not yet implemented")
            # TODO: Use requests to check if repo URLs are accessible

    finally:
        conn.close()


if __name__ == "__main__":
    main()
