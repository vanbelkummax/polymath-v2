-- Polymath V2 Schema - Postgres with pgvector
--
-- Key changes from V1:
-- 1. Uses pgvector for embeddings (eliminates ChromaDB)
-- 2. Structure-aware passages with header hierarchy
-- 3. Better metadata tracking
--
-- Run: psql -U polymath -d polymath -f schema_v2.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table (upgraded from V1)
CREATE TABLE IF NOT EXISTS documents_v2 (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core metadata
    title TEXT NOT NULL,
    authors JSONB,  -- Array of author names
    year INTEGER,

    -- Identifiers (at least one should be populated)
    doi TEXT UNIQUE,
    arxiv_id TEXT UNIQUE,
    pmid TEXT UNIQUE,
    pmcid TEXT,

    -- Source tracking
    source_file TEXT,
    source_method TEXT,  -- 'pdf2doi', 'crossref', 'zotero', 'filename'
    metadata_confidence REAL DEFAULT 0.0,  -- 0.0-1.0

    -- Zotero integration
    zotero_key TEXT,
    zotero_synced_at TIMESTAMP,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Deduplication
    title_hash TEXT GENERATED ALWAYS AS (encode(sha256(lower(title)::bytea), 'hex')) STORED
);

-- Passages table with embeddings (replaces ChromaDB)
CREATE TABLE IF NOT EXISTS passages_v2 (
    passage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents_v2(doc_id) ON DELETE CASCADE,

    -- Content
    passage_text TEXT NOT NULL,

    -- Structure preservation (from markdown headers)
    header TEXT,           -- Section header (e.g., "Methods", "Results")
    header_level INTEGER,  -- 1, 2, or 3
    parent_header TEXT,    -- Parent section for hierarchy

    -- Position tracking
    char_start INTEGER,
    char_end INTEGER,
    page_num INTEGER,

    -- BGE-M3 embedding (1024 dimensions)
    embedding vector(1024),

    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', passage_text)) STORED,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Code chunks table with embeddings
CREATE TABLE IF NOT EXISTS code_chunks_v2 (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Repository info
    repo_name TEXT NOT NULL,
    file_path TEXT NOT NULL,

    -- Content
    chunk_text TEXT NOT NULL,
    chunk_type TEXT,  -- 'function', 'class', 'method', 'module'
    language TEXT,

    -- AST info (from tree-sitter)
    function_name TEXT,
    class_name TEXT,
    docstring TEXT,

    -- Summary for better semantic search
    summary TEXT,  -- LLM-generated description

    -- Embedding
    embedding vector(1024),

    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Concept extraction (V2 - with extractor tracking)
CREATE TABLE IF NOT EXISTS passage_concepts_v2 (
    id SERIAL PRIMARY KEY,
    passage_id UUID REFERENCES passages_v2(passage_id) ON DELETE CASCADE,

    concept_name TEXT NOT NULL,
    concept_type TEXT,  -- 'METHOD', 'PROBLEM', 'DOMAIN', 'MECHANISM'
    confidence REAL DEFAULT 1.0,

    -- Extraction tracking
    extractor TEXT,  -- 'gemini', 'gpt4', 'regex'
    extractor_version TEXT,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(passage_id, concept_name)
);

-- Indexes for performance

-- Vector similarity search (HNSW for fast approximate search)
CREATE INDEX IF NOT EXISTS idx_passages_v2_embedding
    ON passages_v2 USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_code_chunks_v2_embedding
    ON code_chunks_v2 USING hnsw (embedding vector_cosine_ops);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_passages_v2_search
    ON passages_v2 USING gin (search_vector);

CREATE INDEX IF NOT EXISTS idx_code_chunks_v2_search
    ON code_chunks_v2 USING gin (search_vector);

-- Lookups
CREATE INDEX IF NOT EXISTS idx_documents_v2_doi ON documents_v2(doi);
CREATE INDEX IF NOT EXISTS idx_documents_v2_year ON documents_v2(year);
CREATE INDEX IF NOT EXISTS idx_documents_v2_title_hash ON documents_v2(title_hash);
CREATE INDEX IF NOT EXISTS idx_passages_v2_doc ON passages_v2(doc_id);
CREATE INDEX IF NOT EXISTS idx_passages_v2_header ON passages_v2(header);
CREATE INDEX IF NOT EXISTS idx_code_chunks_v2_repo ON code_chunks_v2(repo_name);

-- Hybrid search function (Vector + Keyword in one query)
CREATE OR REPLACE FUNCTION hybrid_search_v2(
    query_text TEXT,
    query_embedding vector(1024),
    keyword_weight REAL DEFAULT 0.3,
    vector_weight REAL DEFAULT 0.7,
    result_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    passage_id UUID,
    doc_id UUID,
    passage_text TEXT,
    header TEXT,
    title TEXT,
    year INTEGER,
    doi TEXT,
    keyword_score REAL,
    vector_score REAL,
    combined_score REAL
) AS $$
BEGIN
    RETURN QUERY
    WITH keyword_results AS (
        SELECT
            p.passage_id,
            ts_rank(p.search_vector, plainto_tsquery('english', query_text)) as k_score
        FROM passages_v2 p
        WHERE p.search_vector @@ plainto_tsquery('english', query_text)
    ),
    vector_results AS (
        SELECT
            p.passage_id,
            1 - (p.embedding <=> query_embedding) as v_score
        FROM passages_v2 p
        ORDER BY p.embedding <=> query_embedding
        LIMIT result_limit * 2
    ),
    combined AS (
        SELECT
            COALESCE(k.passage_id, v.passage_id) as passage_id,
            COALESCE(k.k_score, 0) as keyword_score,
            COALESCE(v.v_score, 0) as vector_score,
            (COALESCE(k.k_score, 0) * keyword_weight +
             COALESCE(v.v_score, 0) * vector_weight) as combined_score
        FROM keyword_results k
        FULL OUTER JOIN vector_results v ON k.passage_id = v.passage_id
    )
    SELECT
        c.passage_id,
        p.doc_id,
        p.passage_text,
        p.header,
        d.title,
        d.year,
        d.doi,
        c.keyword_score,
        c.vector_score,
        c.combined_score
    FROM combined c
    JOIN passages_v2 p ON c.passage_id = p.passage_id
    JOIN documents_v2 d ON p.doc_id = d.doc_id
    ORDER BY c.combined_score DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Migration helper: Track what's been migrated
CREATE TABLE IF NOT EXISTS v2_migrations (
    migration_name TEXT PRIMARY KEY,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    items_processed INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
);

-- Comments
COMMENT ON TABLE documents_v2 IS 'V2 documents with improved metadata tracking';
COMMENT ON TABLE passages_v2 IS 'V2 passages with structure preservation and pgvector embeddings';
COMMENT ON TABLE code_chunks_v2 IS 'V2 code chunks with AST info and summaries';
COMMENT ON FUNCTION hybrid_search_v2 IS 'Combined keyword + vector search in single query';
