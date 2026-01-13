# Polymath V2 Implementation Plan

**Author**: Max Van Belkum
**Date**: January 2026

---

## Executive Summary

This document captures the senior engineer optimization plan for fixing the "Implementation Drift" between Polymath's architecture documentation and actual code implementation.

**Core Problem**: Architecture says BGE-M3 + hierarchical chunking + Zotero-first metadata, but code uses MPNet + sliding window + filename regex.

**Result**: Running current scripts would fill the V2 database with V1-quality, incompatible data.

---

## Critical Fixes Implemented

### 1. Embedding Model Consistency

**Before (V1)**:
```python
# scripts/ingest_mineru.py:133
embed_model = SentenceTransformer('all-mpnet-base-v2')  # 384 dim
```

**After (V2)**:
```python
# lib/embeddings.py
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # 1024 dim
```

**Why it matters**: 384-dim and 1024-dim vectors cannot be compared. Mixing them breaks similarity search entirely.

### 2. Structure-Aware Chunking

**Before (V1)**:
```python
# scripts/ingest_mineru.py:63-72
def chunk_text(text, size=1500, overlap=200, max_chunks=40):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i+size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks
```

This destroys MinerU's structured markdown output.

**After (V2)**:
```python
# lib/chunking.py
def chunk_markdown_by_headers(md_text):
    """Split by headers (# ## ###) to preserve semantic context."""
    header_pattern = r'^(#{1,3})\s+(.+?)$'
    headers = list(re.finditer(header_pattern, md_text, re.MULTILINE))
    # ... preserves section boundaries
```

**Why it matters**: Scientific papers have structure (Abstract, Methods, Results). Sliding window destroys this, making section-specific search impossible.

### 3. Metadata Extraction Priority

**Before (V1)**:
```python
# scripts/zotero_full_upload.py:94-153
def extract_metadata(pdf_path):
    name = pdf_path.stem
    # Parse year from filename regex
    if match := re.search(r'(19|20)\d{2}', name):
        year = match.group(0)
    # ...garbage-in for files named "paper.pdf"
```

**After (V2)**:
```python
# lib/metadata.py
def get_paper_metadata(pdf_path):
    # 1. pdf2doi: Scan PDF binary for DOI (high accuracy)
    id_result = extract_identifier_from_pdf(pdf_path)

    if id_result['identifier']:
        # 2. CrossRef/arXiv lookup (authoritative metadata)
        if id_result['identifier_type'] == 'doi':
            return lookup_crossref(id_result['identifier'])

    # 3. Filename regex (last resort, low confidence)
    return extract_metadata_from_filename(pdf_path)
```

**Why it matters**: DOI-based lookup returns complete, accurate metadata (authors, venue, abstract). Filename parsing fails for generically-named PDFs.

### 4. Vector Storage Consolidation

**Before (V1)**:
- Postgres: Metadata only
- ChromaDB: Vectors in separate process
- Sync scripts: `consistency_check.py`, `sync_chroma.py`

**After (V2)**:
- Postgres + pgvector: Everything in one database
- Hybrid search in single SQL query
- No sync needed - ACID transactions

**Why it matters**: Multiple databases drift apart. Sync scripts are fragile. pgvector eliminates this class of bugs entirely.

---

## File Inventory

| File | Purpose | V1 Issue Fixed |
|------|---------|----------------|
| `lib/config.py` | Single source of truth | Multiple `.env` files |
| `lib/embeddings.py` | BGE-M3 singleton | 15+ scripts using MPNet |
| `lib/chunking.py` | Markdown header split | Sliding window everywhere |
| `lib/metadata.py` | pdf2doi + CrossRef | Filename regex |
| `scripts/ingest_v2.py` | Full pipeline | Scattered ingestion scripts |
| `scripts/zotero_upload_v2.py` | Zotero with dedup | No duplicate checking |
| `config/schema_v2.sql` | pgvector schema | ChromaDB dependency |

---

## Immediate Checklist

1. **Do NOT run any V1 ingestion scripts**
   - They will create incompatible 384-dim vectors
   - They will destroy document structure
   - They will create garbage metadata

2. **Apply V2 schema first**
   ```bash
   psql -U polymath -d polymath -f config/schema_v2.sql
   ```

3. **Test on small batch**
   ```bash
   python3 scripts/ingest_v2.py /path/to/10_pdfs/ --limit 10
   ```

4. **Verify embeddings are 1024 dimensions**
   ```sql
   SELECT passage_id, vector_dims(embedding) as dims
   FROM passages_v2 LIMIT 5;
   -- Should all show 1024
   ```

5. **Scale up only after verification**

---

## Three-Tool Agentic Design

The V2 architecture enables precise, context-aware tools for Claude:

### Tool 1: `search_library`

**Purpose**: High-level survey ("What papers discuss X?")

**Implementation**:
```sql
SELECT * FROM hybrid_search_v2(
    $query_text,
    $query_embedding,
    0.3,  -- keyword weight
    0.7,  -- vector weight
    20    -- limit
);
```

**Output**: Paper titles, DOIs, summaries (not full text)

### Tool 2: `read_section`

**Purpose**: Deep reading ("Read the Methods section of Paper X")

**Implementation**:
```sql
SELECT passage_text, header
FROM passages_v2
WHERE doc_id = $doc_id
  AND header ILIKE $section_name
ORDER BY char_start;
```

**Why possible**: V2 preserves section boundaries. V1 cannot do this.

### Tool 3: `find_mechanism_overlap`

**Purpose**: Cross-domain discovery ("Where else is this math used?")

**Implementation**:
1. Encode mechanism description with BGE-M3
2. Search Methods sections across all papers
3. Filter for different domains using Neo4j

**Why this is the polymathic tool**: Same mechanism in different domains = transfer opportunity.

---

## Risk Mitigation

### Risk: Mixed V1/V2 Data

**Symptom**: Similarity search returns garbage (comparing 384-dim to 1024-dim vectors)

**Prevention**: Never run V1 scripts after applying V2 schema

**Detection**:
```sql
-- Check for dimension mismatches
SELECT vector_dims(embedding), COUNT(*)
FROM passages_v2
GROUP BY vector_dims(embedding);
-- Should only show 1024
```

### Risk: MinerU Failures

**Symptom**: Some PDFs fail to extract

**Mitigation**: Fall back to PyMuPDF (less structure, but still works)

### Risk: pdf2doi Rate Limits

**Symptom**: CrossRef returns 429

**Mitigation**: Add exponential backoff, cache results

---

## Success Criteria

| Metric | V1 | V2 Target |
|--------|-----|-----------|
| Embedding consistency | Mixed 384/1024 | 100% 1024-dim |
| Metadata confidence | ~0.3 avg | >0.8 avg |
| Section-specific search | Impossible | Works |
| Database sync issues | Weekly | Never |

---

## Conclusion

This implementation fixes the architectural drift. Every script now uses:
- BGE-M3 (1024 dimensions)
- Markdown header chunking
- pdf2doi metadata extraction
- pgvector in Postgres

The result is a consistent, searchable knowledge base that matches the architecture documentation.
