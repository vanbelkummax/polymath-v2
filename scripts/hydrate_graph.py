#!/usr/bin/env python3
"""
Graph Hydration Script - Bridge Ingestion to Entity Extraction

This script closes the "Integration Gap" between:
- Skeleton (Text + Vectors in Postgres) ← Created by ingest_v2.py
- Muscle (Entity Extraction to Neo4j) ← Created by this script

Architecture (based on paper corpus findings):

1. Aragog-style JIT Processing:
   - Don't process all passages at once
   - Batch by document for context coherence
   - Adapt batch size based on LLM rate limits

2. HaluMem-style Validation:
   - Validate extractions before storing
   - Track extraction quality metrics
   - Skip passages that fail validation

3. Domain-Specific Schema (GraphRAG paper):
   - Use SPATIAL_TX_ENTITY_TYPES from config
   - Typed nodes for better graph queries

Usage:
    # Hydrate all unprocessed passages
    python3 scripts/hydrate_graph.py

    # Hydrate specific document
    python3 scripts/hydrate_graph.py --doc-id abc123

    # Dry run (no writes)
    python3 scripts/hydrate_graph.py --dry-run --limit 100

    # Resume from checkpoint
    python3 scripts/hydrate_graph.py --resume
"""

import sys
import os
import json
import time
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import logging

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.entity_extraction import (
    extract_entities_sync,
    extract_entities_pattern,
    store_extraction_to_neo4j,
    ExtractionResult,
    Entity,
    Relation
)
from lib.hallucination_detector import HallucinationDetector, ValidationResult

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'hydrate_graph.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class HydrationStats:
    """Track hydration progress and quality metrics."""
    passages_processed: int = 0
    passages_skipped: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    validation_failures: int = 0
    llm_calls: int = 0
    pattern_only_fallbacks: int = 0
    start_time: datetime = None
    last_checkpoint: str = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['start_time'] = self.start_time.isoformat() if self.start_time else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'HydrationStats':
        if d.get('start_time'):
            d['start_time'] = datetime.fromisoformat(d['start_time'])
        return cls(**d)


class GraphHydrator:
    """
    Hydrates Neo4j knowledge graph from ingested passages.

    Design principles:
    - Idempotent: Can be run multiple times safely
    - Resumable: Checkpoints progress for long-running jobs
    - Validated: HaluMem-style validation before storage
    - Batched: Process by document for context coherence
    """

    def __init__(
        self,
        pg_conn,
        neo4j_driver,
        batch_size: int = 50,
        use_llm: bool = True,
        dry_run: bool = False
    ):
        self.pg_conn = pg_conn
        self.neo4j_driver = neo4j_driver
        self.batch_size = batch_size
        self.use_llm = use_llm
        self.dry_run = dry_run

        self.detector = HallucinationDetector()
        self.stats = HydrationStats(start_time=datetime.now())
        self.checkpoint_file = Path(__file__).parent.parent / "logs" / "hydrate_checkpoint.json"

    def get_unprocessed_passages(
        self,
        doc_id: Optional[str] = None,
        limit: int = 0
    ) -> List[Dict]:
        """
        Get passages that haven't been processed for entity extraction.

        Uses a LEFT JOIN to find passages without corresponding Neo4j entries.
        """
        cur = self.pg_conn.cursor()

        # Base query: passages without entity extractions
        query = """
            SELECT p.passage_id, p.doc_id, p.passage_text, p.header, d.title
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            LEFT JOIN passage_extractions pe ON p.passage_id = pe.passage_id
            WHERE pe.passage_id IS NULL
              AND LENGTH(p.passage_text) > 100
        """

        params = []

        if doc_id:
            query += " AND p.doc_id = %s"
            params.append(doc_id)

        query += " ORDER BY d.doc_id, p.char_start"  # Group by document

        if limit > 0:
            query += f" LIMIT {limit}"

        cur.execute(query, params)

        columns = [desc[0] for desc in cur.description]
        results = [dict(zip(columns, row)) for row in cur.fetchall()]

        cur.close()
        return results

    def get_unprocessed_passages_simple(self, limit: int = 0) -> List[Dict]:
        """
        Simpler query that doesn't require passage_extractions table.
        Uses Neo4j to check what's already processed.
        """
        cur = self.pg_conn.cursor()

        # Get all passages with sufficient text
        query = """
            SELECT p.passage_id, p.doc_id, p.passage_text, p.header
            FROM passages p
            WHERE LENGTH(p.passage_text) > 100
            ORDER BY p.doc_id
        """

        if limit > 0:
            query += f" LIMIT {limit}"

        cur.execute(query)

        columns = [desc[0] for desc in cur.description]
        results = [dict(zip(columns, row)) for row in cur.fetchall()]

        cur.close()

        # Filter out already processed (check Neo4j)
        processed_ids = self._get_processed_passage_ids()
        return [r for r in results if r['passage_id'] not in processed_ids]

    def _get_processed_passage_ids(self) -> set:
        """Get passage IDs that already have extractions in Neo4j."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH ()-[r:MENTIONED_IN]->()
                RETURN DISTINCT r.passage_id AS passage_id
            """)
            return {r['passage_id'] for r in result if r['passage_id']}

    def process_passage(self, passage: Dict) -> Optional[ExtractionResult]:
        """
        Extract entities from a single passage with validation.

        Returns None if extraction fails validation.
        """
        passage_id = passage['passage_id']
        text = passage['passage_text']

        # Step 1: Extract entities
        try:
            if self.use_llm:
                result = extract_entities_sync(text, passage_id)
                self.stats.llm_calls += 1
            else:
                # Pattern-only mode (faster, lower recall)
                entities = extract_entities_pattern(text, passage_id)
                result = ExtractionResult(
                    passage_id=passage_id,
                    entities=entities,
                    relations=[],
                    extractor_version="v2.0-pattern-only"
                )
                self.stats.pattern_only_fallbacks += 1

        except Exception as e:
            logger.warning(f"Extraction failed for {passage_id}: {e}")
            # Fallback to pattern extraction
            entities = extract_entities_pattern(text, passage_id)
            result = ExtractionResult(
                passage_id=passage_id,
                entities=entities,
                relations=[],
                extractor_version="v2.0-pattern-fallback"
            )
            self.stats.pattern_only_fallbacks += 1

        # Step 2: Validate extractions (HaluMem-style)
        validated_entities = []
        for entity in result.entities:
            validation = self.detector.validate_extraction(
                entity.name,
                entity.entity_type,
                text,
                entity.source_text
            )

            if validation.is_valid:
                validated_entities.append(entity)
            else:
                logger.debug(
                    f"Entity failed validation: {entity.name} ({entity.entity_type})"
                    f" - Issues: {validation.issues}"
                )
                self.stats.validation_failures += 1

        # Update result with validated entities only
        result.entities = validated_entities

        return result

    def store_extraction(self, extraction: ExtractionResult, doc_id: str):
        """Store validated extraction to Neo4j."""
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would store {len(extraction.entities)} entities, "
                f"{len(extraction.relations)} relations for passage {extraction.passage_id}"
            )
            return

        try:
            store_extraction_to_neo4j(self.neo4j_driver, extraction, doc_id)
            self.stats.entities_extracted += len(extraction.entities)
            self.stats.relations_extracted += len(extraction.relations)
        except Exception as e:
            logger.error(f"Failed to store extraction: {e}")
            raise

    def _record_extraction(
        self,
        passage_id: str,
        extractor_version: str,
        entity_count: int,
        relation_count: int
    ):
        """
        Record extraction in Postgres tracking table.

        This enables fast SQL-only filtering via get_unprocessed_passages()
        instead of the slower Neo4j-based check.
        """
        cur = self.pg_conn.cursor()
        try:
            cur.execute("""
                INSERT INTO passage_extractions
                    (passage_id, extractor_version, entity_count, relation_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (passage_id) DO UPDATE SET
                    extractor_version = EXCLUDED.extractor_version,
                    entity_count = EXCLUDED.entity_count,
                    relation_count = EXCLUDED.relation_count,
                    extracted_at = NOW()
            """, (passage_id, extractor_version, entity_count, relation_count))
            self.pg_conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record extraction tracking: {e}")
            self.pg_conn.rollback()
        finally:
            cur.close()

    def save_checkpoint(self, last_passage_id: str):
        """Save progress checkpoint for resume capability."""
        self.stats.last_checkpoint = last_passage_id
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)

    def load_checkpoint(self) -> Optional[str]:
        """Load last checkpoint for resume."""
        if not self.checkpoint_file.exists():
            return None

        with open(self.checkpoint_file, 'r') as f:
            data = json.load(f)
            self.stats = HydrationStats.from_dict(data)
            return self.stats.last_checkpoint

    def hydrate(
        self,
        doc_id: Optional[str] = None,
        limit: int = 0,
        resume: bool = False
    ):
        """
        Main hydration loop.

        Args:
            doc_id: Optional specific document to process
            limit: Maximum passages to process (0 = unlimited)
            resume: Resume from last checkpoint
        """
        logger.info("=" * 60)
        logger.info("GRAPH HYDRATION STARTING")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"LLM Extraction: {'ENABLED' if self.use_llm else 'PATTERN ONLY'}")
        logger.info(f"Batch Size: {self.batch_size}")

        # Resume from checkpoint if requested
        last_checkpoint = None
        if resume:
            last_checkpoint = self.load_checkpoint()
            if last_checkpoint:
                logger.info(f"Resuming from checkpoint: {last_checkpoint}")

        # Get unprocessed passages
        logger.info("Fetching unprocessed passages...")
        passages = self.get_unprocessed_passages_simple(limit=limit)
        logger.info(f"Found {len(passages)} passages to process")

        if not passages:
            logger.info("No passages to process. Graph is up to date!")
            return

        # Skip to checkpoint if resuming
        if last_checkpoint:
            skip_count = 0
            for i, p in enumerate(passages):
                if p['passage_id'] == last_checkpoint:
                    passages = passages[i+1:]
                    skip_count = i + 1
                    break
            logger.info(f"Skipped {skip_count} passages (already processed)")

        # Process in batches
        current_doc = None
        batch_count = 0

        for passage in passages:
            try:
                # Track document changes for logging
                if passage['doc_id'] != current_doc:
                    if current_doc:
                        logger.info(f"Completed document {current_doc}")
                    current_doc = passage['doc_id']
                    logger.info(f"Processing document: {current_doc}")

                # Extract and validate
                extraction = self.process_passage(passage)

                if extraction and extraction.entities:
                    self.store_extraction(extraction, passage['doc_id'])

                    # Track in Postgres for fast SQL-only filtering
                    if not self.dry_run:
                        self._record_extraction(
                            passage['passage_id'],
                            extraction.extractor_version,
                            len(extraction.entities),
                            len(extraction.relations)
                        )

                self.stats.passages_processed += 1

                # Rate limiting for LLM calls
                if self.use_llm:
                    time.sleep(0.1)  # 10 requests/sec max

                # Checkpoint every batch_size passages
                batch_count += 1
                if batch_count >= self.batch_size:
                    self.save_checkpoint(passage['passage_id'])
                    logger.info(
                        f"Checkpoint: {self.stats.passages_processed} processed, "
                        f"{self.stats.entities_extracted} entities, "
                        f"{self.stats.relations_extracted} relations"
                    )
                    batch_count = 0

            except KeyboardInterrupt:
                logger.info("Interrupted! Saving checkpoint...")
                self.save_checkpoint(passage['passage_id'])
                break

            except Exception as e:
                logger.error(f"Error processing passage {passage['passage_id']}: {e}")
                self.stats.passages_skipped += 1
                continue

        # Final checkpoint
        self.save_checkpoint(passages[-1]['passage_id'] if passages else '')

        # Summary
        logger.info("=" * 60)
        logger.info("HYDRATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Passages Processed: {self.stats.passages_processed}")
        logger.info(f"Passages Skipped: {self.stats.passages_skipped}")
        logger.info(f"Entities Extracted: {self.stats.entities_extracted}")
        logger.info(f"Relations Extracted: {self.stats.relations_extracted}")
        logger.info(f"Validation Failures: {self.stats.validation_failures}")
        logger.info(f"LLM Calls: {self.stats.llm_calls}")
        logger.info(f"Pattern-Only Fallbacks: {self.stats.pattern_only_fallbacks}")


def ensure_schema(pg_conn):
    """Ensure required tables exist for tracking extractions."""
    cur = pg_conn.cursor()

    # Create tracking table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS passage_extractions (
            passage_id UUID PRIMARY KEY REFERENCES passages(passage_id),
            extracted_at TIMESTAMP DEFAULT NOW(),
            extractor_version TEXT,
            entity_count INTEGER,
            relation_count INTEGER
        );
    """)

    pg_conn.commit()
    cur.close()


def ensure_neo4j_indexes(driver):
    """Create Neo4j indexes for efficient queries."""
    with driver.session() as session:
        # Fulltext index for concept search
        try:
            session.run("""
                CREATE FULLTEXT INDEX concept_name_fulltext IF NOT EXISTS
                FOR (n:METHOD|DATASET|CELL_TYPE|GENE|TISSUE|ALGORITHM|LOSS_FUNCTION|DATA_STRUCTURE|METRIC|MECHANISM)
                ON EACH [n.name]
            """)
        except Exception as e:
            logger.warning(f"Could not create fulltext index: {e}")

        # Regular indexes for common lookups
        for label in config.SPATIAL_TX_ENTITY_TYPES:
            try:
                session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)")
            except Exception:
                pass

        logger.info("Neo4j indexes verified")


def main():
    parser = argparse.ArgumentParser(description="Hydrate Neo4j graph from ingested passages")
    parser.add_argument("--doc-id", help="Process specific document only")
    parser.add_argument("--limit", type=int, default=0, help="Limit passages to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Checkpoint batch size")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Neo4j")
    parser.add_argument("--no-llm", action="store_true", help="Pattern extraction only (faster)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Connect to databases
    import psycopg2
    from neo4j import GraphDatabase

    logger.info("Connecting to databases...")

    pg_conn = psycopg2.connect(config.POSTGRES_URI)
    logger.info("  Postgres: connected")

    neo4j_driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    logger.info("  Neo4j: connected")

    try:
        # Ensure schema exists
        ensure_schema(pg_conn)
        ensure_neo4j_indexes(neo4j_driver)

        # Run hydration
        hydrator = GraphHydrator(
            pg_conn=pg_conn,
            neo4j_driver=neo4j_driver,
            batch_size=args.batch_size,
            use_llm=not args.no_llm,
            dry_run=args.dry_run
        )

        hydrator.hydrate(
            doc_id=args.doc_id,
            limit=args.limit,
            resume=args.resume
        )

    finally:
        pg_conn.close()
        neo4j_driver.close()
        logger.info("Database connections closed")


if __name__ == "__main__":
    main()
