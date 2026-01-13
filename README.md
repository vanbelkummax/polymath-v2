# Polymath V2: Consistent Implementation

**Author**: Max Van Belkum
**Institution**: Vanderbilt University MD-PhD Program
**Date**: January 2026

---

## The Problem: Implementation Drift

The original Polymath architecture documents describe a sophisticated system:
- BGE-M3 embeddings (1024 dimensions)
- Structure-aware hierarchical passages
- Mechanism-centric knowledge extraction
- Zotero-first metadata workflow

But the actual scripts were building something different:
- MPNet embeddings (384 dimensions) - **INCOMPATIBLE**
- Sliding window chunking (destroys document structure)
- Filename regex for metadata (garbage-in)
- ChromaDB with sync scripts (fragile)

**This repository fixes the implementation to match the architecture.**

---

## Key Changes from V1

| Component | V1 (Broken) | V2 (Fixed) |
|-----------|-------------|------------|
| **Embedding Model** | `all-mpnet-base-v2` (384 dim) | `BAAI/bge-m3` (1024 dim) |
| **Chunking** | `chunk_text(size=1500)` sliding window | `chunk_markdown_by_headers()` structure-aware |
| **Metadata** | Filename regex | pdf2doi → CrossRef/arXiv lookup |
| **Vector Store** | ChromaDB (separate sync) | pgvector in Postgres |
| **Deduplication** | None | DOI-based before upload |

---

## Architecture

```
PDF Input
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. METADATA EXTRACTION (lib/metadata.py)│
│    pdf2doi scan → CrossRef/arXiv lookup │
│    Confidence: 0.95 (DOI) vs 0.3 (regex)│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 2. TEXT EXTRACTION                       │
│    PyMuPDF (fast) → MinerU (complex)    │
│    Output: Structured Markdown          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 3. CHUNKING (lib/chunking.py)           │
│    Split by Markdown headers (# ## ###) │
│    Preserve: Tables, Math, Code blocks  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 4. EMBEDDING (lib/embeddings.py)        │
│    BGE-M3 (1024 dim) - ONLY THIS MODEL  │
│    Includes header context for search   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 5. STORAGE                              │
│    Postgres + pgvector (hybrid search)  │
│    Neo4j (knowledge graph)              │
└─────────────────────────────────────────┘
```

---

## Installation

```bash
# Clone
git clone https://github.com/vanbelkummax/polymath-v2.git
cd polymath-v2

# Dependencies
pip install -r requirements.txt

# Database setup
psql -U polymath -d polymath -f config/schema_v2.sql
```

### Required Dependencies

```
# Core
FlagEmbedding>=1.2.0  # BGE-M3
psycopg2-binary>=2.9
neo4j>=5.0
pyzotero>=1.5

# PDF Processing
PyMuPDF>=1.23
magic-pdf>=0.6  # MinerU
pdf2doi>=1.5

# Optional
tqdm
```

---

## Usage

### Ingest Papers

```bash
# Single paper
python3 scripts/ingest_v2.py /path/to/paper.pdf

# Batch directory
python3 scripts/ingest_v2.py /path/to/pdfs/

# Test run (no database writes)
python3 scripts/ingest_v2.py /path/to/pdfs/ --dry-run --limit 5
```

### Upload to Zotero

```bash
# Full upload with deduplication
python3 scripts/zotero_upload_v2.py

# Resume from checkpoint
python3 scripts/zotero_upload_v2.py --resume

# Test with limit
python3 scripts/zotero_upload_v2.py --limit 10
```

### Hybrid Search (SQL)

```sql
-- Vector + keyword search in one query
SELECT * FROM hybrid_search_v2(
    'spatial transcriptomics prediction',  -- query text
    $embedding_vector,                     -- 1024-dim BGE-M3 vector
    0.3,                                   -- keyword weight
    0.7,                                   -- vector weight
    20                                     -- result limit
);
```

---

## File Structure

```
polymath-v2/
├── lib/
│   ├── config.py       # Single source of truth for configuration
│   ├── chunking.py     # Markdown header-aware chunking
│   ├── embeddings.py   # BGE-M3 singleton embedder
│   └── metadata.py     # pdf2doi + CrossRef/arXiv lookup
├── scripts/
│   ├── ingest_v2.py    # Full ingestion pipeline
│   └── zotero_upload_v2.py
├── config/
│   └── schema_v2.sql   # Postgres + pgvector schema
├── docs/
│   └── IMPLEMENTATION_PLAN.md
└── .env.example
```

---

## Configuration

Create `.env` in the repository root:

```bash
# Database
DATABASE_URL=postgresql://polymath:polymath@localhost/polymath
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=polymathic2026

# Zotero
ZOTERO_API_KEY=your_key_here
ZOTERO_USER_ID=your_user_id

# APIs (for concept extraction)
GEMINI_API_KEY=your_key_here
```

---

## Why These Choices?

### BGE-M3 over MPNet

| Model | Dimensions | Scientific Text | Multi-lingual |
|-------|------------|-----------------|---------------|
| MPNet | 384 | Good | No |
| BGE-M3 | 1024 | **Excellent** | **Yes** |

BGE-M3 captures more semantic nuance, essential for cross-domain knowledge discovery.

### Markdown Header Chunking over Sliding Window

**Sliding window (V1)**:
```
"...the results showed significant improvement. # Meth"
"ods We used a transformer-based approach with..."
```
Destroys section boundaries. Mixes Results into Methods.

**Header chunking (V2)**:
```
{"header": "Results", "content": "The results showed significant improvement..."}
{"header": "Methods", "content": "We used a transformer-based approach with..."}
```
Preserves semantic units. Enables section-specific search.

### pdf2doi over Filename Regex

**Filename regex (V1)**:
- `paper.pdf` → No metadata, title = "paper"
- Confidence: 0.3

**pdf2doi (V2)**:
- Scans PDF binary for DOI
- Looks up CrossRef for full metadata
- Confidence: 0.95

### pgvector over ChromaDB

**ChromaDB (V1)**:
- Separate process
- Sync scripts needed
- Drift risk

**pgvector (V2)**:
- In Postgres (single DB)
- ACID transactions
- Hybrid search in one query

---

## Migration from V1

```bash
# 1. Apply new schema
psql -U polymath -d polymath -f config/schema_v2.sql

# 2. Re-embed existing passages with BGE-M3
# (Required because V1 used 384-dim MPNet)
python3 scripts/migrate_embeddings.py

# 3. Re-upload to Zotero with proper metadata
python3 scripts/zotero_upload_v2.py
```

**Warning**: V1 embeddings (384 dim) are INCOMPATIBLE with V2 (1024 dim). Full re-embedding required.

---

## Contributing

This repository implements the Polymath V2 specification exactly. All changes must:

1. Use BGE-M3 for embeddings (no other models)
2. Use markdown header chunking (no sliding window)
3. Use pdf2doi for metadata (no filename regex)
4. Store vectors in pgvector (no ChromaDB)

---

## License

MIT
