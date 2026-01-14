#!/usr/bin/env python3
"""
Load Zotero CSV exports into Postgres for metadata enrichment.

This creates a lookup table that can be queried during PDF ingestion
to enrich extracted metadata with Zotero's cleaner data.

Usage:
    python scripts/load_zotero_metadata.py /path/to/export.csv
    python scripts/load_zotero_metadata.py --all  # Load from default staging
"""

import sys
import csv
import hashlib
import argparse
from pathlib import Path
from typing import Optional, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config


def normalize_doi(doi: str) -> Optional[str]:
    """Normalize DOI for consistent matching."""
    if not doi:
        return None
    doi = doi.strip().lower()
    # Remove common prefixes
    for prefix in ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi.org/']:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi if doi else None


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching."""
    import re
    if not title:
        return ""
    # Lowercase, remove punctuation, collapse whitespace
    title = title.lower()
    title = re.sub(r'[^\w\s]', ' ', title)
    title = ' '.join(title.split())
    return title


def title_hash(title: str) -> str:
    """Create hash of normalized title for fast lookup."""
    normalized = normalize_title(title)
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def windows_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL path."""
    if not win_path:
        return ""
    # Handle multiple attachments (semicolon separated)
    first_path = win_path.split(';')[0].strip()
    # Convert C:\... to /mnt/c/...
    wsl_path = first_path.replace('C:\\', '/mnt/c/').replace('\\', '/')
    return wsl_path


def parse_zotero_csv(filepath: Path) -> List[Dict]:
    """Parse Zotero CSV export into list of records."""
    records = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract key fields
            record = {
                'zotero_key': row.get('Key', ''),
                'item_type': row.get('Item Type', ''),
                'title': row.get('Title', '').strip(),
                'title_normalized': normalize_title(row.get('Title', '')),
                'title_hash': title_hash(row.get('Title', '')),
                'authors': row.get('Author', '').strip(),
                'year': None,
                'doi': normalize_doi(row.get('DOI', '')),
                'url': row.get('Url', '').strip(),
                'abstract': row.get('Abstract Note', '').strip(),
                'publication': row.get('Publication Title', '').strip(),
                'issn': row.get('ISSN', '').strip(),
                'pages': row.get('Pages', '').strip(),
                'volume': row.get('Volume', '').strip(),
                'issue': row.get('Issue', '').strip(),
                'tags_manual': row.get('Manual Tags', '').strip(),
                'tags_auto': row.get('Automatic Tags', '').strip(),
                'pdf_path_windows': row.get('File Attachments', '').strip(),
                'pdf_path_wsl': windows_to_wsl_path(row.get('File Attachments', '')),
                'extra': row.get('Extra', '').strip(),
            }

            # Parse year
            year_str = row.get('Publication Year', '').strip()
            if year_str and year_str.isdigit():
                record['year'] = int(year_str)

            # Skip empty records
            if record['title'] or record['doi']:
                records.append(record)

    return records


def create_metadata_table(conn):
    """Create the Zotero metadata lookup table."""
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS zotero_metadata (
            id SERIAL PRIMARY KEY,
            zotero_key TEXT UNIQUE,
            item_type TEXT,

            -- Core identifiers
            title TEXT,
            title_normalized TEXT,
            title_hash TEXT,
            doi TEXT,

            -- Metadata
            authors TEXT,
            year INTEGER,
            abstract TEXT,
            publication TEXT,
            issn TEXT,
            pages TEXT,
            volume TEXT,
            issue TEXT,
            url TEXT,

            -- Tags
            tags_manual TEXT,
            tags_auto TEXT,

            -- File paths
            pdf_path_windows TEXT,
            pdf_path_wsl TEXT,

            -- Extra info
            extra TEXT,

            -- Tracking
            source_csv TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # Create partial unique index for non-null DOIs only
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_zotero_doi_unique
        ON zotero_metadata(doi)
        WHERE doi IS NOT NULL
    """)

    # Create indexes for fast lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_zotero_doi ON zotero_metadata(doi)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_zotero_title_hash ON zotero_metadata(title_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_zotero_year ON zotero_metadata(year)")

    conn.commit()
    cur.close()
    print("✓ Created zotero_metadata table")


def load_records(conn, records: List[Dict], source_csv: str):
    """Load records into the metadata table."""
    cur = conn.cursor()

    inserted = 0
    updated = 0
    skipped = 0

    for record in records:
        record['source_csv'] = source_csv

        try:
            # Try insert, update on zotero_key conflict
            cur.execute("""
                INSERT INTO zotero_metadata (
                    zotero_key, item_type, title, title_normalized, title_hash,
                    doi, authors, year, abstract, publication, issn, pages,
                    volume, issue, url, tags_manual, tags_auto,
                    pdf_path_windows, pdf_path_wsl, extra, source_csv
                ) VALUES (
                    %(zotero_key)s, %(item_type)s, %(title)s, %(title_normalized)s, %(title_hash)s,
                    %(doi)s, %(authors)s, %(year)s, %(abstract)s, %(publication)s, %(issn)s, %(pages)s,
                    %(volume)s, %(issue)s, %(url)s, %(tags_manual)s, %(tags_auto)s,
                    %(pdf_path_windows)s, %(pdf_path_wsl)s, %(extra)s, %(source_csv)s
                )
                ON CONFLICT (zotero_key) DO UPDATE SET
                    title = EXCLUDED.title,
                    doi = COALESCE(EXCLUDED.doi, zotero_metadata.doi),
                    abstract = CASE
                        WHEN LENGTH(EXCLUDED.abstract) > LENGTH(COALESCE(zotero_metadata.abstract, ''))
                        THEN EXCLUDED.abstract
                        ELSE zotero_metadata.abstract
                    END,
                    source_csv = EXCLUDED.source_csv
            """, record)
            conn.commit()
            inserted += 1

        except Exception as e:
            conn.rollback()
            # DOI conflict = duplicate paper with different zotero_key, just skip
            skipped += 1

    cur.close()

    return inserted, updated, skipped


def generate_manifest(conn) -> Dict:
    """Generate a manifest of what's ready for ingestion."""
    cur = conn.cursor()

    # Get stats
    cur.execute("SELECT COUNT(*) FROM zotero_metadata")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM zotero_metadata WHERE doi IS NOT NULL")
    with_doi = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM zotero_metadata WHERE LENGTH(abstract) > 50")
    with_abstract = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM zotero_metadata WHERE pdf_path_wsl != ''")
    with_pdf = cur.fetchone()[0]

    # Check PDF accessibility
    cur.execute("SELECT pdf_path_wsl FROM zotero_metadata WHERE pdf_path_wsl != '' LIMIT 100")
    sample_paths = [r[0] for r in cur.fetchall()]

    import os
    accessible = sum(1 for p in sample_paths if os.path.exists(p))

    cur.close()

    return {
        'total_records': total,
        'with_doi': with_doi,
        'with_abstract': with_abstract,
        'with_pdf_path': with_pdf,
        'sample_pdf_accessible': f"{accessible}/{len(sample_paths)}",
    }


def main():
    parser = argparse.ArgumentParser(description="Load Zotero metadata into Postgres")
    parser.add_argument("csv_file", nargs='?', help="Path to Zotero CSV export")
    parser.add_argument("--all", action="store_true", help="Load all CSVs from staging")
    parser.add_argument("--manifest", action="store_true", help="Just show manifest, don't load")
    args = parser.parse_args()

    import psycopg2
    conn = psycopg2.connect(config.POSTGRES_URI)

    # Create table
    create_metadata_table(conn)

    if args.manifest:
        manifest = generate_manifest(conn)
        print("\n=== INGESTION MANIFEST ===")
        for k, v in manifest.items():
            print(f"  {k}: {v}")
        conn.close()
        return

    # Determine which files to load
    csv_files = []
    if args.all:
        staging = Path("/home/user/work/polymax/metadata_staging")
        csv_files = list(staging.glob("*.csv"))
    elif args.csv_file:
        csv_files = [Path(args.csv_file)]
    else:
        parser.print_help()
        return

    # Load each CSV
    for csv_file in csv_files:
        print(f"\n=== Loading {csv_file.name} ===")
        records = parse_zotero_csv(csv_file)
        print(f"  Parsed {len(records)} records")

        inserted, updated, skipped = load_records(conn, records, csv_file.name)
        print(f"  Inserted: {inserted}, Skipped (duplicates): {skipped}")

    # Show manifest
    manifest = generate_manifest(conn)
    print("\n=== FINAL MANIFEST ===")
    for k, v in manifest.items():
        print(f"  {k}: {v}")

    conn.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
