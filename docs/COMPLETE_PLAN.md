# Polymath V2: Complete Implementation Plan

**Version**: 2.0
**Date**: January 2026
**Author**: Max Van Belkum

A scientifically rigorous RAG system for cross-domain knowledge discovery in spatial transcriptomics research.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Architecture](#2-core-architecture)
3. [Implementation Status](#3-implementation-status)
4. [Paper Corpus Enhancements](#4-paper-corpus-enhancements)
5. [Module Reference](#5-module-reference)
6. [Deployment Guide](#6-deployment-guide)
7. [Validation & Testing](#7-validation--testing)
8. [Roadmap](#8-roadmap)

---

## 1. Executive Summary

### Problem Statement

The original Polymath system suffered from **Implementation Drift**: architecture documents described BGE-M3 embeddings with hierarchical chunking, but actual scripts used MPNet with sliding window chunking. This created:

- **Inconsistent embeddings** (384-dim vs 1024-dim)
- **Lost document structure** (sliding window destroys sections)
- **Poor metadata extraction** (filename regex vs pdf2doi)
- **Sync overhead** (ChromaDB + Postgres duplication)

### Solution

Polymath V2 is a ground-up reimplementation with:

| Component | V1 (Broken) | V2 (Fixed) |
|-----------|-------------|------------|
| Embeddings | MPNet (384-dim) | **BGE-M3 (1024-dim)** |
| Chunking | Sliding window | **Markdown header-aware** |
| Metadata | Filename regex | **pdf2doi → CrossRef/arXiv** |
| Vectors | ChromaDB + Postgres | **pgvector in Postgres only** |
| PDF Extraction | fitz (breaks 2-column) | **MinerU first, fitz fallback** |
| Entity Schema | Generic | **Domain-specific (spatial TX)** |

### Key Enhancements from Paper Corpus

Analysis of 6 papers on agentic memory systems yielded 5 actionable improvements:

| Rank | Enhancement | Source Paper | Status |
|------|-------------|--------------|--------|
| 1 | Domain-specific entity schema | GraphRAG on Technical Documents | ✅ Implemented |
| 2 | JIT memory retrieval | General Agentic Memory (GAM) | ✅ Implemented |
| 3 | Hallucination detection | HaluMem (2026) | ✅ Implemented |
| 4 | RL-based context curation | Memory-as-Action | 📋 Future |
| 5 | Memory-layer architectures | UltraMemV2 | 📋 Future |

---

## 2. Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         POLYMATH V2 ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   MinerU     │    │  pdf2doi     │    │   BGE-M3     │              │
│  │ (PDF→MD)     │    │ (Metadata)   │    │ (Embeddings) │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             ▼                                           │
│                   ┌─────────────────┐                                   │
│                   │  INGESTION      │                                   │
│                   │  PIPELINE       │                                   │
│                   │  (ingest_v2.py) │                                   │
│                   └────────┬────────┘                                   │
│                            │                                            │
│         ┌──────────────────┼──────────────────┐                        │
│         ▼                  ▼                  ▼                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  Postgres   │    │   Neo4j     │    │ Hallucination│                │
│  │  (pgvector) │    │  (Concepts) │    │  Detector    │                │
│  │             │    │             │    │              │                │
│  │ • passages  │    │ • METHOD    │    │ • extraction │                │
│  │ • documents │    │ • DATASET   │    │ • updating   │                │
│  │ • hybrid    │    │ • CELL_TYPE │    │ • QA stage   │                │
│  │   search    │    │ • GENE      │    │              │                │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                │
│         │                  │                  │                        │
│         └──────────────────┼──────────────────┘                        │
│                            ▼                                            │
│                   ┌─────────────────┐                                   │
│                   │  JIT RETRIEVAL  │                                   │
│                   │  (Runtime)      │                                   │
│                   │                 │                                   │
│                   │ • Query classify│                                   │
│                   │ • Concept lookup│                                   │
│                   │ • Deep passage  │                                   │
│                   │ • RT synthesis  │                                   │
│                   └─────────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion** (offline, batch)
   - PDF → MinerU → Structured Markdown
   - Markdown → Header-aware chunking → Passages
   - pdf2doi → CrossRef/arXiv → Metadata
   - Passages → BGE-M3 → Embeddings (1024-dim)
   - Entity extraction → Neo4j (domain-specific schema)
   - All stored to Postgres (single source of truth)

2. **Retrieval** (online, per-query)
   - Query → Classification (factual/comparison/exploratory/synthesis)
   - Lightweight index lookup (Neo4j concepts)
   - Deep passage retrieval (Postgres pgvector)
   - Runtime synthesis (Gemini, query-specific)
   - Hallucination validation (grounding check)

---

## 3. Implementation Status

### Core Components

| File | Purpose | Status |
|------|---------|--------|
| `lib/config.py` | Configuration + domain schema | ✅ Complete |
| `lib/chunking.py` | Header-aware chunking | ✅ Complete |
| `lib/embeddings.py` | BGE-M3 singleton | ✅ Complete |
| `lib/metadata.py` | pdf2doi + API lookups | ✅ Complete |
| `lib/entity_extraction.py` | Domain-specific extraction | ✅ Complete |
| `lib/hallucination_detector.py` | HaluMem-style validation | ✅ Complete |
| `lib/jit_retrieval.py` | GAM-style JIT memory | ✅ Complete |
| `scripts/ingest_v2.py` | Full ingestion pipeline | ✅ Complete |
| `scripts/zotero_upload_v2.py` | Zotero integration | ✅ Complete |
| `config/schema_v2.sql` | Postgres + pgvector schema | ✅ Complete |

### Database Schema

```sql
-- Documents (metadata)
documents_v2 (
    doc_id UUID PRIMARY KEY,
    title TEXT,
    authors JSONB,
    year INTEGER,
    doi TEXT UNIQUE,
    pmid TEXT,
    source_file TEXT,
    metadata_confidence REAL,
    created_at TIMESTAMP
)

-- Passages (with structure preservation)
passages_v2 (
    passage_id UUID PRIMARY KEY,
    doc_id UUID REFERENCES documents_v2,
    passage_text TEXT NOT NULL,
    header TEXT,           -- Section header (Methods, Results, etc.)
    header_level INTEGER,  -- 1, 2, or 3
    parent_header TEXT,    -- Parent section for hierarchy
    char_start INTEGER,
    char_end INTEGER,
    embedding vector(1024), -- BGE-M3 dimensions
    search_vector tsvector  -- For hybrid search
)

-- Indexes for fast retrieval
CREATE INDEX ON passages_v2 USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON passages_v2 USING gin (search_vector);
```

---

## 4. Paper Corpus Enhancements

### 4.1 Domain-Specific Entity Schema (GraphRAG)

**Source**: "GraphRAG on Technical Documents – Impact of Knowledge Graph Schema"
**DOI**: 10.4230/TGDK.3.2.3

**Key Insight**: Domain-expert schemas extract 10% more relevant entities than auto-generated schemas.

**Implementation** (`lib/config.py`):

```python
SPATIAL_TX_ENTITY_TYPES = (
    'METHOD',           # Img2ST, TESLA, Tangram
    'DATASET',          # Visium HD, 10x Xenium
    'CELL_TYPE',        # T-cell, macrophage
    'GENE',             # EGFR, TP53
    'TISSUE',           # colon, liver
    'ALGORITHM',        # optimal transport, GNN
    'LOSS_FUNCTION',    # MSE, Poisson NLL
    'DATA_STRUCTURE',   # point cloud, graph
    'METRIC',           # PCC, SSIM, AUC
    'MECHANISM',        # attention, convolution
)

SPATIAL_TX_RELATION_TYPES = (
    'APPLIES_TO',       # method → dataset
    'PREDICTS',         # method → target
    'OUTPERFORMS',      # method → method
    'REQUIRES',         # method → requirement
    'TRAINED_ON',       # method → dataset
    'OPERATES_ON',      # algorithm → data_structure
    'EXPRESSES',        # cell_type → gene
    'FOUND_IN',         # cell_type → tissue
    'USES_LOSS',        # method → loss_function
    'IMPLEMENTS',       # method → mechanism
)
```

### 4.2 JIT Memory Retrieval (GAM)

**Source**: "General Agentic Memory Via Deep Research" (2025)

**Key Insight**: Static memory (ahead-of-time compression) loses information. JIT compilation retrieves and synthesizes at query time.

**Implementation** (`lib/jit_retrieval.py`):

```python
class JITRetriever:
    """
    Two-tier architecture:
    - Memorizer: Lightweight index (Neo4j) + raw storage (Postgres)
    - Researcher: Query-guided deep retrieval at runtime
    """

    async def retrieve(self, query: str) -> JITResult:
        # 1. Classify query type
        query_type = classify_query(query)  # factual/comparison/exploratory/synthesis

        # 2. Lookup lightweight concept index
        concepts = await lookup_concept_index(query, self.neo4j_driver)

        # 3. Deep passage retrieval (guided by concepts)
        passages = await retrieve_passages_deep(query, self.pg_conn, concept_doc_ids)

        # 4. Runtime synthesis (NOT pre-computed!)
        synthesis = await synthesize_at_runtime(query, passages, concepts)

        return JITResult(passages, concepts, synthesis)
```

### 4.3 Hallucination Detection (HaluMem)

**Source**: "HaluMem: Evaluating Hallucinations in Memory Systems of Agents" (2026)
**DOI**: 10.48550/arxiv.2511.03506

**Key Insight**: Memory systems hallucinate at three stages - extraction, updating, and QA.

**Implementation** (`lib/hallucination_detector.py`):

```python
class HallucinationDetector:
    def validate_extraction(self, entity, entity_type, source_text, source_span):
        """Check entity is grounded in source text."""
        # Exact/fuzzy span matching
        # Entity type plausibility
        # Abbreviation validation

    def validate_update(self, new_entry, existing_entries):
        """Check for contradictions with existing knowledge."""
        # Semantic opposition detection
        # Numeric value comparison
        # Temporal consistency

    def validate_answer(self, answer, passages, question):
        """Check answer is grounded in retrieved passages."""
        # Claim extraction and grounding
        # Fabricated number detection
        # Citation verification
```

### 4.4 Future Enhancements

#### Memory-as-Action (RL-based Context Curation)

**Source**: "Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks" (2025)
**DOI**: 10.48550/arxiv.2510.12635

**Concept**: Train RL policy to decide which passages to include in context, rather than fixed top-k retrieval.

```python
# Future implementation
class ContextCurationPolicy:
    def select_context(self, query, candidates, history):
        """RL-trained policy for dynamic context selection."""
        # State: (query_embedding, candidate_embeddings, history_summary)
        # Action: Include/exclude each candidate
        # Reward: Answer quality (downstream task performance)
```

#### UltraMemV2 Architecture

**Source**: "UltraMemV2: Memory Networks Scaling to 120B Parameters" (2025)
**Author**: ByteDance Seed

**Concept**: Memory-layer architecture for custom embedders handling very long documents.

---

## 5. Module Reference

### lib/config.py

**Purpose**: Single source of truth for all configuration.

```python
from lib.config import config

# Access settings
config.EMBEDDING_MODEL      # "BAAI/bge-m3"
config.EMBEDDING_DIM        # 1024
config.POSTGRES_URI         # from .env
config.SPATIAL_TX_ENTITY_TYPES    # ('METHOD', 'DATASET', ...)
config.ENTITY_CONFIDENCE_THRESHOLD # 0.7
```

### lib/chunking.py

**Purpose**: Structure-aware Markdown chunking.

```python
from lib.chunking import chunk_markdown_by_headers, Chunk

chunks = chunk_markdown_by_headers(markdown_text)
for chunk in chunks:
    print(f"{chunk.header} (level {chunk.level}): {len(chunk.content)} chars")
```

### lib/embeddings.py

**Purpose**: BGE-M3 embeddings (refuses to use other models).

```python
from lib.embeddings import get_embedder

embedder = get_embedder()  # Singleton
embeddings = embedder.encode(["text1", "text2"])  # Shape: (2, 1024)
```

### lib/entity_extraction.py

**Purpose**: Domain-specific entity extraction.

```python
from lib.entity_extraction import extract_entities_sync

result = extract_entities_sync(passage_text, passage_id="abc123")
for entity in result.entities:
    print(f"{entity.entity_type}: {entity.name} (conf={entity.confidence})")
```

### lib/hallucination_detector.py

**Purpose**: Validate extractions, updates, and QA outputs.

```python
from lib.hallucination_detector import HallucinationDetector

detector = HallucinationDetector()

# Validate extraction
result = detector.validate_extraction("EGFR", "GENE", source_text, "EGFR expression")
if not result.is_valid:
    print(f"Issues: {result.issues}")

# Validate answer
result = detector.validate_answer(answer, passages, question)
```

### lib/jit_retrieval.py

**Purpose**: Query-time retrieval and synthesis.

```python
from lib.jit_retrieval import JITRetriever

retriever = JITRetriever(pg_conn, neo4j_driver)
result = retriever.retrieve_sync("What methods predict gene expression from H&E?")

print(f"Passages: {len(result.passages)}")
print(f"Concepts: {len(result.concepts)}")
print(f"Synthesis:\n{result.synthesis}")
```

---

## 6. Deployment Guide

### Prerequisites

```bash
# Python 3.10+
python3 --version

# Postgres with pgvector
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Neo4j 5.x
neo4j status

# MinerU
pip install magic-pdf
```

### Installation

```bash
git clone https://github.com/vanbelkummax/polymath-v2.git
cd polymath-v2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your credentials

# Initialize database schema
psql -U polymath -d polymath -f config/schema_v2.sql
```

### Running Ingestion

```bash
# Single PDF
python3 scripts/ingest_v2.py /path/to/paper.pdf

# Batch mode
python3 scripts/ingest_v2.py /path/to/pdfs/ --limit 100

# Dry run (no database writes)
python3 scripts/ingest_v2.py /path/to/paper.pdf --dry-run
```

---

## 7. Validation & Testing

### Unit Tests

```bash
# Test chunking
python3 -c "from lib.chunking import chunk_markdown_by_headers; ..."

# Test entity extraction
python3 lib/entity_extraction.py

# Test hallucination detection
python3 lib/hallucination_detector.py

# Test JIT retrieval
python3 lib/jit_retrieval.py
```

### Integration Tests

```bash
# Full pipeline test
python3 scripts/ingest_v2.py test_papers/ --dry-run

# Verify database
psql -U polymath -d polymath -c "SELECT COUNT(*) FROM passages_v2;"
```

### Hallucination Validation Metrics

Target thresholds (based on HaluMem paper):

| Stage | Metric | Target |
|-------|--------|--------|
| Extraction | Grounding accuracy | >90% |
| Updating | Contradiction rate | <5% |
| QA | Claim grounding | >85% |

---

## 8. Roadmap

### Phase 1: Core System (✅ Complete)

- [x] BGE-M3 embeddings
- [x] Header-aware chunking
- [x] pdf2doi metadata extraction
- [x] pgvector in Postgres
- [x] MinerU PDF extraction
- [x] Domain-specific entity schema
- [x] JIT retrieval
- [x] Hallucination detection

### Phase 2: Enhancement (Q1 2026)

- [ ] RL-based context curation (Memory-as-Action)
- [ ] Multi-hop reasoning with Neo4j
- [ ] Streaming ingestion pipeline
- [ ] Web UI for search

### Phase 3: Scale (Q2 2026)

- [ ] Distributed embedding computation
- [ ] Incremental index updates
- [ ] Custom memory-layer embedder (UltraMemV2)
- [ ] Production deployment

---

## References

1. **GraphRAG on Technical Documents**
   DOI: 10.4230/TGDK.3.2.3
   Key: Domain-specific schema improves entity extraction by 10%

2. **General Agentic Memory (GAM)**
   Title: "General Agentic Memory Via Deep Research" (2025)
   Key: JIT memory compilation preserves information vs pre-summarization

3. **HaluMem**
   DOI: 10.48550/arxiv.2511.03506
   Key: Validate at extraction, updating, and QA stages

4. **Memory-as-Action**
   DOI: 10.48550/arxiv.2510.12635
   Key: RL-trained context curation outperforms heuristics

5. **UltraMemV2**
   Author: ByteDance Seed (2025)
   Key: Memory-layer architecture for long-context models

6. **Microsoft GraphRAG**
   URL: https://github.com/microsoft/graphrag
   Key: Leiden hierarchical communities for mechanism clustering

7. **LightRAG**
   URL: https://github.com/HKUDS/LightRAG
   Key: Dual-level retrieval (entity vs relation)

8. **Letta (MemGPT)**
   URL: https://github.com/letta-ai/letta
   Key: Virtual memory paging with three-tier architecture
