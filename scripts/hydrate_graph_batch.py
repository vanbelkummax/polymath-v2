#!/usr/bin/env python3
"""
Batch Graph Hydration Pipeline using Gemini Batch API

This script implements a three-phase batch processing workflow for entity extraction:
1. PREPARE: Query unprocessed passages → Write JSONL → Upload to GCS
2. SUBMIT: Create batch job via Gemini Batch API
3. INGEST: Download results → Parse → Store in Neo4j

Benefits over sequential processing:
- 50% lower cost (batch pricing)
- Massive parallelism (no rate limits)
- Google manages retries
- Process 50K+ passages per batch

Usage:
    # Phase 1: Prepare batch (creates JSONL and uploads to GCS)
    python scripts/hydrate_graph_batch.py prepare --limit 50000

    # Phase 2: Submit batch job
    python scripts/hydrate_graph_batch.py submit --input gs://bucket/batch_input.jsonl

    # Phase 3: Check job status
    python scripts/hydrate_graph_batch.py status --job-id <job_name>

    # Phase 4: Ingest completed results
    python scripts/hydrate_graph_batch.py ingest --job-id <job_name>

    # All-in-one (prepare + submit, then poll until done + ingest)
    python scripts/hydrate_graph_batch.py run --limit 50000

Prerequisites:
    - GCS bucket configured (GCS_BUCKET_NAME in .env)
    - Google Cloud credentials (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)
    - Gemini API access with batch capabilities
"""

import os
import sys
import json
import time
import argparse
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.entity_extraction import (
    prepare_batch_file,
    parse_batch_file,
    store_extraction_to_neo4j,
    ExtractionResult
)

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'hydrate_batch.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BatchJobState:
    """Track batch job state for resume capability."""
    job_id: Optional[str] = None
    input_gcs_uri: Optional[str] = None
    output_gcs_uri: Optional[str] = None
    status: str = "pending"  # pending, submitted, processing, completed, failed
    passages_submitted: int = 0
    passages_processed: int = 0
    entities_extracted: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'BatchJobState':
        if not path.exists():
            return cls()
        with open(path, 'r') as f:
            return cls(**json.load(f))


# =============================================================================
# GCS Operations
# =============================================================================

def upload_to_gcs(local_path: str, destination_blob_name: str) -> str:
    """
    Upload a local file to Google Cloud Storage.

    Args:
        local_path: Path to local file
        destination_blob_name: Name for the blob in GCS

    Returns:
        GCS URI (gs://bucket/blob)
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
        raise

    storage_client = storage.Client()
    bucket = storage_client.bucket(config.GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    logger.info(f"Uploading {local_path} to gs://{config.GCS_BUCKET_NAME}/{destination_blob_name}")
    blob.upload_from_filename(local_path)

    return f"gs://{config.GCS_BUCKET_NAME}/{destination_blob_name}"


def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    """
    Download a file from Google Cloud Storage.

    Args:
        gcs_uri: GCS URI (gs://bucket/blob)
        local_path: Local path to save file

    Returns:
        Local path
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
        raise

    # Parse GCS URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    logger.info(f"Downloading {gcs_uri} to {local_path}")
    blob.download_to_filename(local_path)

    return local_path


def list_gcs_blobs(prefix: str) -> List[str]:
    """List blobs in GCS bucket with given prefix."""
    try:
        from google.cloud import storage
    except ImportError:
        return []

    storage_client = storage.Client()
    bucket = storage_client.bucket(config.GCS_BUCKET_NAME)

    blobs = bucket.list_blobs(prefix=prefix)
    return [f"gs://{config.GCS_BUCKET_NAME}/{blob.name}" for blob in blobs]


# =============================================================================
# Gemini Batch API Operations
# =============================================================================

def create_batch_job(input_gcs_uri: str, display_name: str = None) -> Tuple[str, str]:
    """
    Submit a batch job to Gemini.

    Args:
        input_gcs_uri: GCS URI of the input JSONL file
        display_name: Optional job display name

    Returns:
        Tuple of (job_id, output_gcs_uri)
    """
    try:
        from google import genai
        from google.genai.types import CreateBatchJobConfig
    except ImportError:
        logger.error("google-genai not installed. Run: pip install google-genai")
        raise

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    if not display_name:
        display_name = f"polymath_hydration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_uri = f"gs://{config.GCS_BUCKET_NAME}/results/{display_name}/"

    logger.info(f"Creating batch job: {display_name}")
    logger.info(f"  Input: {input_gcs_uri}")
    logger.info(f"  Output: {output_uri}")

    # Create the batch job
    batch_job = client.batches.create(
        model=config.BATCH_MODEL,
        src=input_gcs_uri,
        config=CreateBatchJobConfig(
            dest=output_uri,
            display_name=display_name
        )
    )

    logger.info(f"Job created: {batch_job.name}")
    logger.info(f"State: {batch_job.state}")

    return batch_job.name, output_uri


def get_batch_job_status(job_id: str) -> Dict:
    """
    Get the status of a batch job.

    Args:
        job_id: The job name/ID

    Returns:
        Dict with job status info
    """
    try:
        from google import genai
    except ImportError:
        logger.error("google-genai not installed")
        raise

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    job = client.batches.get(name=job_id)

    return {
        'name': job.name,
        'state': str(job.state),
        'display_name': getattr(job, 'display_name', ''),
        'create_time': str(getattr(job, 'create_time', '')),
        'update_time': str(getattr(job, 'update_time', '')),
        'dest': getattr(job, 'dest', ''),
    }


def wait_for_batch_job(
    job_id: str,
    initial_interval: int = 30,
    max_interval: int = 300,
    max_wait: int = 86400
) -> Dict:
    """
    Poll until batch job completes using exponential backoff.

    Reduces API chatter by starting with shorter intervals and
    increasing them over time (batch jobs typically take 10+ minutes).

    Args:
        job_id: The job name/ID
        initial_interval: Starting seconds between checks (default: 30s)
        max_interval: Maximum seconds between checks (default: 5min)
        max_wait: Maximum seconds to wait total (default: 24h)

    Returns:
        Final job status
    """
    start_time = time.time()
    current_interval = initial_interval
    poll_count = 0

    while time.time() - start_time < max_wait:
        poll_count += 1
        elapsed = time.time() - start_time

        status = get_batch_job_status(job_id)
        state = status.get('state', '')

        logger.info(f"[Poll #{poll_count}, {elapsed/60:.1f}m elapsed] Job {job_id}: {state}")

        if state in ['JOB_STATE_SUCCEEDED', 'SUCCEEDED', 'STATE_SUCCEEDED']:
            logger.info("Batch job completed successfully!")
            return status
        elif state in ['JOB_STATE_FAILED', 'FAILED', 'STATE_FAILED']:
            logger.error("Batch job failed!")
            return status
        elif state in ['JOB_STATE_CANCELLED', 'CANCELLED']:
            logger.warning("Batch job was cancelled")
            return status

        # Exponential backoff with jitter
        logger.debug(f"Sleeping for {current_interval}s before next poll")
        time.sleep(current_interval)

        # Increase interval with exponential backoff (1.5x each time, capped)
        current_interval = min(int(current_interval * 1.5), max_interval)

    logger.warning(f"Timed out waiting for job after {max_wait}s ({poll_count} polls)")
    return get_batch_job_status(job_id)


# =============================================================================
# Database Operations
# =============================================================================

def get_unprocessed_passages(pg_conn, limit: int = 0) -> List[Dict]:
    """
    Get passages that haven't been processed for entity extraction.

    Checks the passage_concepts table to find unprocessed passages.
    """
    cur = pg_conn.cursor()

    query = """
        SELECT p.passage_id, p.doc_id, p.passage_text
        FROM passages p
        LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id
        WHERE pc.id IS NULL
          AND LENGTH(p.passage_text) > 100
        ORDER BY p.doc_id
    """

    if limit > 0:
        query += f" LIMIT {limit}"

    cur.execute(query)

    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]

    cur.close()
    return results


def get_passage_texts(pg_conn, passage_ids: List[str]) -> Dict[str, str]:
    """
    Fetch passage texts for a list of passage IDs.

    Returns dict mapping passage_id -> passage_text
    """
    if not passage_ids:
        return {}

    cur = pg_conn.cursor()
    cur.execute("""
        SELECT passage_id::text, passage_text
        FROM passages
        WHERE passage_id = ANY(%s::uuid[])
    """, (passage_ids,))

    result = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    return result


def record_extraction(pg_conn, extraction: ExtractionResult):
    """Record extraction in passage_concepts table for tracking."""
    cur = pg_conn.cursor()

    try:
        for entity in extraction.entities:
            cur.execute("""
                INSERT INTO passage_concepts
                    (passage_id, concept_name, concept_type, confidence, extractor, extractor_version)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (passage_id, concept_name) DO UPDATE SET
                    confidence = GREATEST(passage_concepts.confidence, EXCLUDED.confidence),
                    extractor_version = EXCLUDED.extractor_version
            """, (
                extraction.passage_id,
                entity.name,
                entity.entity_type,
                entity.confidence,
                'gemini-batch',
                extraction.extractor_version
            ))

        pg_conn.commit()

    except Exception as e:
        logger.warning(f"Failed to record extraction for {extraction.passage_id}: {e}")
        pg_conn.rollback()
    finally:
        cur.close()


# =============================================================================
# Pipeline Phases
# =============================================================================

def phase_prepare(pg_conn, limit: int, state: BatchJobState) -> str:
    """
    Phase 1: Prepare batch input file and upload to GCS.

    Returns:
        GCS URI of uploaded batch file
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: PREPARE")
    logger.info("=" * 60)

    # Get unprocessed passages
    logger.info(f"Fetching up to {limit} unprocessed passages...")
    passages = get_unprocessed_passages(pg_conn, limit=limit)
    logger.info(f"Found {len(passages)} passages to process")

    if not passages:
        logger.info("No passages to process!")
        return ""

    # Create JSONL file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_path = Path(tempfile.gettempdir()) / f"batch_input_{timestamp}.jsonl"

    logger.info(f"Writing batch file to {local_path}...")
    count = prepare_batch_file(passages, local_path)
    logger.info(f"Wrote {count} requests")

    state.passages_submitted = count

    # Upload to GCS
    gcs_blob = f"inputs/batch_input_{timestamp}.jsonl"
    gcs_uri = upload_to_gcs(str(local_path), gcs_blob)

    state.input_gcs_uri = gcs_uri
    logger.info(f"Uploaded to {gcs_uri}")

    # Clean up local file
    local_path.unlink()

    return gcs_uri


def phase_submit(input_gcs_uri: str, state: BatchJobState) -> str:
    """
    Phase 2: Submit batch job to Gemini.

    Returns:
        Job ID
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: SUBMIT")
    logger.info("=" * 60)

    job_id, output_uri = create_batch_job(input_gcs_uri)

    state.job_id = job_id
    state.output_gcs_uri = output_uri
    state.status = "submitted"
    state.started_at = datetime.now().isoformat()

    return job_id


def phase_status(job_id: str) -> Dict:
    """
    Check batch job status.
    """
    logger.info("=" * 60)
    logger.info("PHASE: STATUS CHECK")
    logger.info("=" * 60)

    status = get_batch_job_status(job_id)

    logger.info(f"Job: {status.get('name')}")
    logger.info(f"State: {status.get('state')}")
    logger.info(f"Display Name: {status.get('display_name')}")
    logger.info(f"Created: {status.get('create_time')}")
    logger.info(f"Updated: {status.get('update_time')}")
    logger.info(f"Output: {status.get('dest')}")

    return status


def phase_ingest(
    pg_conn,
    neo4j_driver,
    job_id: str,
    state: BatchJobState
) -> int:
    """
    Phase 3: Download results and ingest into Neo4j.

    Returns:
        Number of entities extracted
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: INGEST")
    logger.info("=" * 60)

    # Get job status to find output location
    status = get_batch_job_status(job_id)
    output_uri = status.get('dest', state.output_gcs_uri)

    if not output_uri:
        logger.error("No output URI found for job")
        return 0

    # List output files
    output_blobs = list_gcs_blobs(output_uri.replace(f"gs://{config.GCS_BUCKET_NAME}/", ""))

    if not output_blobs:
        logger.error(f"No output files found at {output_uri}")
        return 0

    logger.info(f"Found {len(output_blobs)} output files")

    # Download and process each output file
    total_entities = 0
    total_passages = 0

    for blob_uri in output_blobs:
        if not blob_uri.endswith('.jsonl'):
            continue

        # Download result file
        local_result = Path(tempfile.gettempdir()) / f"batch_result_{total_passages}.jsonl"
        download_from_gcs(blob_uri, str(local_result))

        # Get passage IDs from result file to fetch original texts
        passage_ids = []
        with open(local_result, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    passage_ids.append(data.get('custom_id', ''))

        # Fetch original texts from Postgres
        logger.info(f"Fetching {len(passage_ids)} passage texts from database...")
        text_lookup = get_passage_texts(pg_conn, passage_ids)

        # Parse results
        logger.info("Parsing batch results...")
        results = parse_batch_file(local_result, text_lookup)

        logger.info(f"Parsed {len(results)} extractions")

        # Store in Neo4j
        for extraction in results:
            if extraction.entities:
                # Get doc_id for this passage
                cur = pg_conn.cursor()
                cur.execute(
                    "SELECT doc_id::text FROM passages WHERE passage_id = %s::uuid",
                    (extraction.passage_id,)
                )
                row = cur.fetchone()
                cur.close()

                if row:
                    doc_id = row[0]
                    try:
                        store_extraction_to_neo4j(neo4j_driver, extraction, doc_id)
                        record_extraction(pg_conn, extraction)
                        total_entities += len(extraction.entities)
                    except Exception as e:
                        logger.warning(f"Failed to store extraction: {e}")

            total_passages += 1

        # Clean up local file
        local_result.unlink()

    state.passages_processed = total_passages
    state.entities_extracted = total_entities
    state.status = "completed"
    state.completed_at = datetime.now().isoformat()

    logger.info(f"Ingestion complete: {total_passages} passages, {total_entities} entities")

    return total_entities


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Graph Hydration using Gemini Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare and submit a batch job
    python hydrate_graph_batch.py prepare --limit 10000
    python hydrate_graph_batch.py submit --input gs://bucket/inputs/batch.jsonl

    # Check job status
    python hydrate_graph_batch.py status --job-id batches/123456

    # Ingest results
    python hydrate_graph_batch.py ingest --job-id batches/123456

    # Full pipeline (prepare + submit + wait + ingest)
    python hydrate_graph_batch.py run --limit 10000
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare batch input file')
    prepare_parser.add_argument('--limit', type=int, default=50000,
                               help='Maximum passages to process (default: 50000)')

    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit batch job')
    submit_parser.add_argument('--input', required=True,
                              help='GCS URI of input JSONL file')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument('--job-id', required=True,
                              help='Batch job ID')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest completed batch results')
    ingest_parser.add_argument('--job-id', required=True,
                              help='Batch job ID')

    # Run command (full pipeline)
    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    run_parser.add_argument('--limit', type=int, default=50000,
                           help='Maximum passages to process')
    run_parser.add_argument('--poll-interval', type=int, default=30,
                           help='Initial seconds between status checks, increases with backoff (default: 30)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load state
    state_file = log_dir / "batch_job_state.json"
    state = BatchJobState.load(state_file)

    # Connect to databases
    import psycopg2
    from neo4j import GraphDatabase

    pg_conn = None
    neo4j_driver = None

    try:
        logger.info("Connecting to databases...")
        pg_conn = psycopg2.connect(config.POSTGRES_URI)
        neo4j_driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        logger.info("  Connected to Postgres and Neo4j")

        if args.command == 'prepare':
            gcs_uri = phase_prepare(pg_conn, args.limit, state)
            if gcs_uri:
                state.save(state_file)
                logger.info(f"\nNext step: python hydrate_graph_batch.py submit --input {gcs_uri}")

        elif args.command == 'submit':
            job_id = phase_submit(args.input, state)
            state.save(state_file)
            logger.info(f"\nJob submitted: {job_id}")
            logger.info(f"Check status: python hydrate_graph_batch.py status --job-id {job_id}")

        elif args.command == 'status':
            phase_status(args.job_id)

        elif args.command == 'ingest':
            entities = phase_ingest(pg_conn, neo4j_driver, args.job_id, state)
            state.save(state_file)
            logger.info(f"\nExtracted {entities} entities")

        elif args.command == 'run':
            # Full pipeline
            logger.info("Running full batch pipeline...")

            # Phase 1: Prepare
            gcs_uri = phase_prepare(pg_conn, args.limit, state)
            if not gcs_uri:
                return
            state.save(state_file)

            # Phase 2: Submit
            job_id = phase_submit(gcs_uri, state)
            state.save(state_file)

            # Wait for completion (exponential backoff reduces API chatter)
            logger.info(f"\nWaiting for job to complete (starting interval: {args.poll_interval}s, with backoff)...")
            final_status = wait_for_batch_job(job_id, initial_interval=args.poll_interval)

            if final_status.get('state') not in ['JOB_STATE_SUCCEEDED', 'SUCCEEDED', 'STATE_SUCCEEDED']:
                logger.error("Job did not complete successfully")
                return

            # Phase 3: Ingest
            entities = phase_ingest(pg_conn, neo4j_driver, job_id, state)
            state.save(state_file)

            logger.info("\n" + "=" * 60)
            logger.info("BATCH PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Passages submitted: {state.passages_submitted}")
            logger.info(f"Passages processed: {state.passages_processed}")
            logger.info(f"Entities extracted: {state.entities_extracted}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

    finally:
        if pg_conn:
            pg_conn.close()
        if neo4j_driver:
            neo4j_driver.close()


if __name__ == "__main__":
    main()
