# Polymath V2: Consistent Implementation

**Author**: Max Van Belkum
**Institution**: Vanderbilt University MD-PhD Program
**Date**: January 2026
**Status**: Production Ready

---

## System Statistics (Current)

| Component | Count |
|-----------|-------|
| **Papers (documents)** | 30,650 |
| **Passages** | 748,325 |
| **Code Files** | 90,051 |
| **Code Chunks** | 578,830 |
| **Paper-Repo Links** | 1,597 |
| **Repos Queued for Ingestion** | 1,414 |

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
| **Code Search** | None | Paper-repo linking + code chunks |
| **Graph Integration** | Manual sync | Hydration bridge with JIT retrieval |

---

## Architecture Overview

```
                           POLYMATH V2 ARCHITECTURE
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │   PDF Input ─────────────► Metadata Extraction (pdf2doi + CrossRef)  │
    │        │                                                             │
    │        ▼                                                             │
    │   Text Extraction ──────► Markdown Header Chunking                   │
    │        │                                                             │
    │        ▼                                                             │
    │   BGE-M3 Embedding ─────► Postgres (passages + pgvector)             │
    │        │                                      │                      │
    │        │                                      ▼                      │
    │        │                              Neo4j Knowledge Graph          │
    │        │                              (concepts, methods, domains)   │
    │        │                                      │                      │
    │        ▼                                      ▼                      │
    │   GitHub URL Detection ──► Repo Queue ──► Clone & Ingest             │
    │        │                                      │                      │
    │        ▼                                      ▼                      │
    │   Paper-Repo Links ◄──────────────────── Code Chunks                 │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

                         MCP SERVER (polymath_v2_server.py)
    ┌──────────────────────────────────────────────────────────────────────┐
    │  quick_lookup    │  research_topic  │  search_paper_code  │ ...     │
    └──────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. JIT (Just-In-Time) Memory Retrieval (`lib/jit_retrieval.py`)

Retrieves and synthesizes knowledge at query time by combining:
- **Neo4j concept graph** - finds related concepts via fulltext search
- **Postgres passages** - retrieves source text with embeddings
- **Gemini synthesis** - generates contextual answers

**Test Results:**
```
Query: "optimal transport spatial"
Concepts found: 20 (score range: 2.2 - 8.5)
Passages retrieved: 5
Synthesis confidence: 0.9
```

### 2. Paper-Repository Linking (`scripts/link_paper_repos.py`)

Automatically detects GitHub repositories mentioned in papers:
- Pattern matching for `github.com/owner/repo` URLs
- "Code availability" section parsing
- Context extraction around mentions

**Test Results:**
```
Documents scanned: 924 (with GitHub mentions)
Repo links found: 1,596
Unique repos: 1,421
Papers with repos: 660
Detection methods:
  - url_pattern: 1,596
  - code_availability: (higher confidence when found)
```

### 3. Automated Repo Ingestion (`scripts/repo_ingest_pipeline.py`)

End-to-end automation for repository ingestion:
1. **Queue** - repos from paper mentions or manual addition
2. **Clone** - shallow git clone with optional GitHub token
3. **Ingest** - Python code chunking (functions, classes)
4. **Link** - associate back to source papers

**Test Results:**
```
Repos queued from papers: 1,414
Test processing: 2 repos
  - RobertaColetti/TRIM-IT: 0 files (R/Matlab project)
  - open-mmlab/mmaction2: 500 files, 2,012 chunks (Python)
Status tracking: pending → cloning → ingesting → completed
```

### 4. MCP Server (`mcp/polymath_v2_server.py`)

Model Context Protocol server providing tools:
- `quick_lookup` - fast concept/paper lookup
- `research_topic` - comprehensive topic research
- `search_paper_code` - search both papers and linked code
- `get_paper_repos` - retrieve repos linked to a paper
- `ingest_paper` / `ingest_code` - add new content

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

# Environment
cp .env.example .env
# Edit .env with your credentials
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

# MCP Server
fastmcp>=0.1.0
google-genai>=0.3.0  # For synthesis

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

### Paper-Repository Linking

```bash
# Scan all papers and extract GitHub URLs
python3 scripts/link_paper_repos.py --scan

# Show linking statistics
python3 scripts/link_paper_repos.py --stats

# Search paper + linked code
python3 scripts/link_paper_repos.py --search <doc_id> --query "transformer"
```

### Automated Repo Ingestion

```bash
# Queue repos from existing papers (one-time scan)
python3 scripts/repo_ingest_pipeline.py --queue-from-papers

# Check queue status
python3 scripts/repo_ingest_pipeline.py --status

# Process pending repos (clone & ingest)
python3 scripts/repo_ingest_pipeline.py --process-queue --limit 10

# Queue a specific repo manually
python3 scripts/repo_ingest_pipeline.py --queue https://github.com/owner/repo
```

### Integration During Paper Ingestion

```python
# In your paper ingestion code:
from scripts.repo_ingest_pipeline import queue_repos_from_paper

# After extracting paper text:
conn = psycopg2.connect(config.POSTGRES_URI)
queued = queue_repos_from_paper(conn, doc_id, full_text)
print(f"Queued {queued} repos for ingestion")
```

### MCP Server

```bash
# Start MCP server
python3 mcp/polymath_v2_server.py

# Or add to Claude Code config (~/.mcp.json):
{
  "mcpServers": {
    "polymath-v2": {
      "command": "python3",
      "args": ["/home/user/polymath-v2/mcp/polymath_v2_server.py"]
    }
  }
}
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
│   ├── config.py           # Single source of truth for configuration
│   ├── chunking.py         # Markdown header-aware chunking
│   ├── embeddings.py       # BGE-M3 singleton embedder
│   ├── metadata.py         # pdf2doi + CrossRef/arXiv lookup
│   └── jit_retrieval.py    # Just-in-time memory retrieval system
├── scripts/
│   ├── ingest_v2.py        # Full paper ingestion pipeline
│   ├── link_paper_repos.py # Paper-repository linking
│   ├── repo_ingest_pipeline.py  # Automated repo clone & ingest
│   ├── hydrate_graph.py    # Neo4j graph hydration
│   └── zotero_upload_v2.py # Zotero integration
├── mcp/
│   └── polymath_v2_server.py  # MCP server with search tools
├── config/
│   └── schema_v2.sql       # Postgres + pgvector schema
├── docs/
│   ├── IMPLEMENTATION_PLAN.md
│   ├── COMPLETE_PLAN.md
│   └── ADVANCED_RAG_PATTERNS.md
└── .env.example
```

---

## Database Schema

### Core Tables

```sql
-- Papers
documents (doc_id UUID, title, authors, year, doi, pmid, ...)

-- Passages (with embeddings)
passages (passage_id UUID, doc_id, passage_text, section, page_num, embedding vector(1024))

-- Code
code_files (file_id, repo_name, file_path, language, loc)
code_chunks (chunk_id, file_id, chunk_type, name, content, start_line, end_line)

-- Paper-Repo Links
paper_repos (doc_id, repo_url, repo_owner, repo_name, detection_method, confidence)
repo_ingest_queue (repo_url, status, clone_path, file_count, chunk_count)
paper_repo_triggers (doc_id, queue_id, context)
```

### Neo4j Graph

```cypher
// Node types
(:Passage {passage_id, doc_id})
(:Concept {name, type})  -- METHOD, DOMAIN, PROBLEM, etc.

// Relationships
(p:Passage)-[:MENTIONS]->(c:Concept)
```

---

## Configuration

Create `.env` in the repository root:

```bash
# Database
DATABASE_URL=dbname=polymath user=polymath host=/var/run/postgresql
POSTGRES_URI=dbname=polymath user=polymath host=/var/run/postgresql
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=polymathic2026

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024

# APIs
GEMINI_API_KEY=your_key_here

# Optional: GitHub (for private repos)
GITHUB_TOKEN=your_token_here

# Zotero
ZOTERO_API_KEY=your_key_here
ZOTERO_USER_ID=your_user_id
```

---

## Test Results Summary

### JIT Retrieval System
- **Status**: Working
- **Concepts found**: 20 per query (using fulltext index)
- **Passages retrieved**: 5-10 per query
- **Gemini synthesis**: Confidence 0.9 with gemini-2.0-flash

### Paper-Repository Linking
- **Status**: Working
- **Papers scanned**: 924 (with GitHub mentions)
- **Links created**: 1,597 across 660 papers
- **Detection accuracy**: High (URL pattern matching)

### Automated Repo Ingestion
- **Status**: Working
- **Queue populated**: 1,414 repos from existing papers
- **Test ingestion**: mmaction2 (500 files, 2,012 chunks)
- **Blocklist**: Common libraries excluded (numpy, pytorch, etc.)

### MCP Server
- **Status**: Working
- **Tools tested**: quick_lookup, research_topic, search_paper_code
- **Integration**: Compatible with Claude Code

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

### Automatic Paper-Repo Linking

Papers often mention their code repositories in:
- "Code Availability" sections
- Footnotes and references
- Methods sections

Automatically linking these enables:
- **Search both paper AND code** for a method
- **Trace implementation** from paper description
- **Queue for ingestion** without manual curation

---

## Migration from V1

```bash
# 1. Apply new schema
psql -U polymath -d polymath -f config/schema_v2.sql

# 2. Re-embed existing passages with BGE-M3
# (Required because V1 used 384-dim MPNet)
python3 scripts/migrate_embeddings.py

# 3. Scan papers for GitHub repos
python3 scripts/link_paper_repos.py --scan

# 4. Queue detected repos for ingestion
python3 scripts/repo_ingest_pipeline.py --queue-from-papers

# 5. Process queue (run in background)
nohup python3 scripts/repo_ingest_pipeline.py --process-queue --limit 100 &
```

**Warning**: V1 embeddings (384 dim) are INCOMPATIBLE with V2 (1024 dim). Full re-embedding required.

---

## Known Limitations

1. **Code chunking**: Currently only supports Python, JS, TS, Java, Go, Rust. R/Matlab projects show 0 chunks.
2. **Private repos**: Require GITHUB_TOKEN in environment
3. **Large repos**: Skipped if >500MB (configurable)
4. **Rate limits**: GitHub API rate limits apply during cloning

---

## Contributing

This repository implements the Polymath V2 specification exactly. All changes must:

1. Use BGE-M3 for embeddings (no other models)
2. Use markdown header chunking (no sliding window)
3. Use pdf2doi for metadata (no filename regex)
4. Store vectors in pgvector (no ChromaDB)
5. Auto-link paper repos when detected

---

## License

MIT
