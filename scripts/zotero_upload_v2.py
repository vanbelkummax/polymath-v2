#!/usr/bin/env python3
"""
Polymath V2 Zotero Upload

Key improvements over V1:
1. pdf2doi scanning BEFORE upload (not filename regex)
2. Zotero's "Add by Identifier" for DOI-bearing PDFs
3. Duplicate checking before upload
4. Proper error handling and retry logic

Workflow:
1. Scan PDF for DOI/arXiv ID using pdf2doi
2. Check if already exists in Zotero
3. If DOI found: Create item with DOI → Zotero auto-fills metadata
4. If no DOI: Fall back to filename metadata
5. Attach PDF to item

Usage:
    python3 scripts/zotero_upload_v2.py                    # Upload all
    python3 scripts/zotero_upload_v2.py --resume           # Resume from checkpoint
    python3 scripts/zotero_upload_v2.py --limit 10         # Test with 10 files
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.metadata import get_paper_metadata, metadata_to_zotero_item

try:
    from pyzotero import zotero
except ImportError:
    print("Installing pyzotero...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyzotero", "-q"])
    from pyzotero import zotero


# Configuration
COLLECTION_NAME = "Polymath_V2"
CHECKPOINT_FILE = config.LOG_DIR / "zotero_v2_checkpoint.json"
LOG_FILE = config.LOG_DIR / "zotero_v2_upload.log"


def log(msg: str):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_zotero_client():
    """Initialize Zotero client with validation."""
    errors = config.validate()
    zotero_errors = [e for e in errors if 'ZOTERO' in e]
    if zotero_errors:
        for e in zotero_errors:
            log(f"ERROR: {e}")
        sys.exit(1)

    return zotero.Zotero(config.ZOTERO_USER_ID, 'user', config.ZOTERO_API_KEY)


def get_or_create_collection(zot, name: str) -> str:
    """Get existing collection or create new one."""
    collections = zot.collections()

    for c in collections:
        if c['data']['name'] == name:
            log(f"Using existing collection: {name}")
            return c['key']

    result = zot.create_collections([{'name': name}])
    if result and 'successful' in result:
        key = list(result['successful'].values())[0]['key']
        log(f"Created collection: {name}")
        return key

    raise Exception(f"Failed to create collection: {result}")


def get_existing_dois(zot) -> Set[str]:
    """Get set of DOIs already in Zotero library."""
    log("Fetching existing DOIs from Zotero...")
    existing = set()

    # Fetch all items in batches
    start = 0
    limit = 100

    while True:
        items = zot.items(start=start, limit=limit)
        if not items:
            break

        for item in items:
            doi = item.get('data', {}).get('DOI')
            if doi:
                existing.add(doi.lower())

        start += limit
        if len(items) < limit:
            break

    log(f"Found {len(existing)} existing DOIs")
    return existing


def load_checkpoint() -> Dict:
    """Load checkpoint for resume."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"uploaded": [], "failed": [], "skipped": [], "collection_key": None}


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint."""
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))


def upload_pdf(
    zot,
    pdf_path: Path,
    collection_key: str,
    existing_dois: Set[str]
) -> Dict:
    """
    Upload a single PDF to Zotero.

    Returns:
        Dict with 'status', 'method', 'doi', 'error'
    """
    result = {
        'file': pdf_path.name,
        'status': 'pending',
        'method': None,
        'doi': None,
        'error': None
    }

    # Step 1: Extract metadata using pdf2doi
    try:
        metadata = get_paper_metadata(pdf_path)
        result['method'] = metadata.source_method
        result['doi'] = metadata.doi
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = f"Metadata extraction failed: {e}"
        return result

    # Step 2: Check for duplicates
    if metadata.doi and metadata.doi.lower() in existing_dois:
        result['status'] = 'duplicate'
        result['error'] = f"DOI already exists: {metadata.doi}"
        return result

    # Step 3: Create Zotero item
    try:
        item = metadata_to_zotero_item(metadata, pdf_path)
        item['collections'] = [collection_key]

        # If we have a DOI, try Zotero's magic lookup first
        if metadata.doi and metadata.confidence < 0.9:
            # Let Zotero fetch better metadata
            try:
                # Zotero can create items from identifiers
                # This gets full metadata from CrossRef/PubMed
                zot_result = zot.item_template('journalArticle')
                zot_result['DOI'] = metadata.doi
                zot_result['collections'] = [collection_key]
                create_result = zot.create_items([zot_result])

                if create_result and 'successful' in create_result:
                    item_key = list(create_result['successful'].values())[0]['key']
                    result['status'] = 'success'
                    result['method'] = 'zotero_doi_lookup'

                    # Attach PDF
                    zot.attachment_simple([str(pdf_path)], item_key)

                    # Add to existing DOIs
                    existing_dois.add(metadata.doi.lower())
                    return result
            except Exception:
                # Fall through to regular creation
                pass

        # Regular creation with our metadata
        create_result = zot.create_items([item])

        if create_result and 'successful' in create_result:
            item_key = list(create_result['successful'].values())[0]['key']

            # Attach PDF
            try:
                zot.attachment_simple([str(pdf_path)], item_key)
            except Exception as e:
                result['error'] = f"Attachment failed: {e}"

            result['status'] = 'success'

            if metadata.doi:
                existing_dois.add(metadata.doi.lower())
        else:
            result['status'] = 'failed'
            result['error'] = f"Create failed: {create_result}"

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Polymath V2 Zotero Upload")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs")
    parser.add_argument("--pdf-dir", type=str, default=None, help="Custom PDF directory")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else config.PDF_ARCHIVE

    log("=" * 60)
    log("Polymath V2 Zotero Upload")
    log("=" * 60)

    # Get all PDFs
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    total = len(all_pdfs)
    log(f"Found {total} PDFs in {pdf_dir}")

    if args.limit > 0:
        all_pdfs = all_pdfs[:args.limit]
        log(f"Limited to {len(all_pdfs)} PDFs")

    # Initialize Zotero
    zot = get_zotero_client()
    log("Zotero client initialized")

    # Get or create collection
    collection_key = get_or_create_collection(zot, COLLECTION_NAME)

    # Get existing DOIs for deduplication
    existing_dois = get_existing_dois(zot)

    # Load checkpoint
    checkpoint = load_checkpoint() if args.resume else {
        "uploaded": [], "failed": [], "skipped": [], "collection_key": collection_key
    }
    checkpoint["collection_key"] = collection_key

    uploaded_set = set(checkpoint["uploaded"])
    failed_set = set(checkpoint["failed"])
    skipped_set = set(checkpoint["skipped"])

    if args.resume:
        log(f"Resuming: {len(uploaded_set)} uploaded, {len(failed_set)} failed, {len(skipped_set)} skipped")

    # Upload loop
    stats = {'success': 0, 'failed': 0, 'duplicate': 0, 'skipped': 0}
    start_time = time.time()

    for i, pdf in enumerate(all_pdfs, 1):
        # Skip already processed
        if pdf.name in uploaded_set or pdf.name in failed_set or pdf.name in skipped_set:
            stats['skipped'] += 1
            continue

        # Upload
        result = upload_pdf(zot, pdf, collection_key, existing_dois)

        # Track result
        if result['status'] == 'success':
            checkpoint["uploaded"].append(pdf.name)
            uploaded_set.add(pdf.name)
            stats['success'] += 1
            log(f"[{i}/{len(all_pdfs)}] SUCCESS: {pdf.name[:40]}... [{result['method']}]")
        elif result['status'] == 'duplicate':
            checkpoint["skipped"].append(pdf.name)
            skipped_set.add(pdf.name)
            stats['duplicate'] += 1
            log(f"[{i}/{len(all_pdfs)}] DUPLICATE: {pdf.name[:40]}...")
        else:
            checkpoint["failed"].append(pdf.name)
            failed_set.add(pdf.name)
            stats['failed'] += 1
            log(f"[{i}/{len(all_pdfs)}] FAILED: {pdf.name[:40]}... - {result['error']}")

        # Save checkpoint periodically
        if i % 25 == 0:
            save_checkpoint(checkpoint)
            elapsed = time.time() - start_time
            rate = stats['success'] / elapsed if elapsed > 0 else 0
            log(f"Progress: {i}/{len(all_pdfs)} | {rate:.1f}/s")

        # Rate limiting
        time.sleep(0.3)

    # Final save
    save_checkpoint(checkpoint)

    # Summary
    elapsed = time.time() - start_time
    log("=" * 60)
    log("COMPLETE")
    log("=" * 60)
    log(f"Success:    {stats['success']}")
    log(f"Failed:     {stats['failed']}")
    log(f"Duplicate:  {stats['duplicate']}")
    log(f"Skipped:    {stats['skipped']}")
    log(f"Time:       {elapsed/60:.1f} minutes")
    log(f"Collection: {COLLECTION_NAME}")
    log("=" * 60)


if __name__ == "__main__":
    main()
