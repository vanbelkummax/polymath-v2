-- Polymath V2 Schema - Postgres with pgvector
--
-- This schema is compatible with the existing production tables (documents, passages)
-- but adds V2 enhancements: structure-aware passages, BGE-M3 embeddings, pgvector.
--
-- Key changes from V1:
-- 1. Uses pgvector for embeddings (eliminates ChromaDB)
-- 2. Structure-aware passages with header hierarchy
-- 3. Better metadata tracking
--
-- Run: psql -U polymath -d polymath -f schema_v2.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- CORE TABLES (V2 compatible with existing production data)
-- ============================================================================

-- Documents table (compatible with existing 'documents' table)
CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core metadata
    title TEXT NOT NULL,
    authors JSONB,  -- Array of author names
    year INTEGER,

    -- Identifiers (at least one should be populated)
    doi TEXT,
    arxiv_id TEXT,
    pmid TEXT,
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
    title_hash TEXT
);

-- Add unique constraints if they don't exist (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'documents_doi_key') THEN
        ALTER TABLE documents ADD CONSTRAINT documents_doi_key UNIQUE (doi);
    END IF;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

-- ============================================================================
-- V1 → V2 MIGRATION: Add columns to existing tables
-- These are safe to run multiple times (IF NOT EXISTS / exception handling)
-- ============================================================================

-- Add V2 columns to existing documents table
DO $$
BEGIN
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS arxiv_id TEXT;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS pmcid TEXT;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_method TEXT;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata_confidence REAL DEFAULT 0.0;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS zotero_key TEXT;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS zotero_synced_at TIMESTAMP;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

-- Passages table with embeddings (compatible with existing 'passages' table)
CREATE TABLE IF NOT EXISTS passages (
    passage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,

    -- Content
    passage_text TEXT NOT NULL,

    -- Structure preservation (from markdown headers) - V2 enhancement
    header TEXT,           -- Section header (e.g., "Methods", "Results")
    header_level INTEGER,  -- 1, 2, or 3
    parent_header TEXT,    -- Parent section for hierarchy
    section TEXT,          -- Alias for header (backward compat)

    -- Position tracking
    char_start INTEGER,
    char_end INTEGER,
    page_num INTEGER,

    -- BGE-M3 embedding (1024 dimensions) - V2 enhancement
    embedding vector(1024),

    -- Full-text search
    search_vector tsvector,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Add V2 columns to existing passages table (migration)
DO $$
BEGIN
    ALTER TABLE passages ADD COLUMN IF NOT EXISTS header TEXT;
    ALTER TABLE passages ADD COLUMN IF NOT EXISTS header_level INTEGER;
    ALTER TABLE passages ADD COLUMN IF NOT EXISTS parent_header TEXT;
    ALTER TABLE passages ADD COLUMN IF NOT EXISTS embedding vector(1024);
    ALTER TABLE passages ADD COLUMN IF NOT EXISTS search_vector tsvector;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

-- Add trigger for search_vector if not exists
CREATE OR REPLACE FUNCTION passages_search_vector_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.passage_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvector_update ON passages;
CREATE TRIGGER tsvector_update BEFORE INSERT OR UPDATE ON passages
    FOR EACH ROW EXECUTE FUNCTION passages_search_vector_trigger();

-- ============================================================================
-- CODE TABLES (compatible with existing code_files, code_chunks)
-- ============================================================================

-- Code files table
CREATE TABLE IF NOT EXISTS code_files (
    file_id SERIAL PRIMARY KEY,
    repo_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    language TEXT,
    file_hash TEXT,
    loc INTEGER,
    head_commit_sha TEXT DEFAULT 'HEAD',
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(repo_name, file_path, head_commit_sha)
);

-- Code chunks table
CREATE TABLE IF NOT EXISTS code_chunks (
    chunk_id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES code_files(file_id) ON DELETE CASCADE,

    -- Content
    content TEXT NOT NULL,
    chunk_type TEXT,  -- 'function', 'class', 'method', 'module'
    name TEXT,        -- function/class name

    -- Position
    start_line INTEGER,
    end_line INTEGER,

    -- AST info
    docstring TEXT,
    chunk_hash TEXT,

    -- Full-text search
    search_vector tsvector,

    -- Embedding (optional, for semantic search)
    embedding vector(1024),

    created_at TIMESTAMP DEFAULT NOW()
);

-- Add trigger for code chunks search_vector
CREATE OR REPLACE FUNCTION code_chunks_search_vector_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, '') || ' ' || COALESCE(NEW.name, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS code_chunks_tsvector_update ON code_chunks;
CREATE TRIGGER code_chunks_tsvector_update BEFORE INSERT OR UPDATE ON code_chunks
    FOR EACH ROW EXECUTE FUNCTION code_chunks_search_vector_trigger();

-- ============================================================================
-- PAPER-REPO LINKING TABLES
-- ============================================================================

-- Paper-Repository linkage table
CREATE TABLE IF NOT EXISTS paper_repos (
    id SERIAL PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    repo_url TEXT NOT NULL,
    repo_owner TEXT,
    repo_name TEXT,
    detection_method TEXT NOT NULL,  -- 'url_pattern', 'code_availability', 'auto_queue'
    confidence FLOAT DEFAULT 1.0,
    context TEXT,
    verified BOOLEAN DEFAULT FALSE,
    verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(doc_id, repo_url)
);

-- Queue for repos pending ingestion
CREATE TABLE IF NOT EXISTS repo_ingest_queue (
    id SERIAL PRIMARY KEY,
    repo_url TEXT NOT NULL UNIQUE,
    repo_owner TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    source_doc_id UUID REFERENCES documents(doc_id) ON DELETE SET NULL,
    status TEXT DEFAULT 'pending',  -- pending, cloning, ingesting, completed, failed, skipped
    priority INT DEFAULT 0,
    clone_path TEXT,
    error_message TEXT,
    file_count INT,
    chunk_count INT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Track which papers triggered which repo ingestions
CREATE TABLE IF NOT EXISTS paper_repo_triggers (
    id SERIAL PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    queue_id INT NOT NULL REFERENCES repo_ingest_queue(id) ON DELETE CASCADE,
    context TEXT,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(doc_id, queue_id)
);

-- ============================================================================
-- CONCEPT EXTRACTION TABLES
-- ============================================================================

-- Concept extraction for passages
CREATE TABLE IF NOT EXISTS passage_concepts (
    id SERIAL PRIMARY KEY,
    passage_id UUID REFERENCES passages(passage_id) ON DELETE CASCADE,

    concept_name TEXT NOT NULL,
    concept_type TEXT,  -- 'METHOD', 'PROBLEM', 'DOMAIN', 'MECHANISM', 'GENE', 'CELL_TYPE'
    confidence REAL DEFAULT 1.0,

    -- Extraction tracking
    extractor TEXT,           -- 'gemini', 'gpt4', 'regex', 'spacy'
    extractor_version TEXT,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(passage_id, concept_name)
);

-- Concept extraction for code chunks
CREATE TABLE IF NOT EXISTS chunk_concepts (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES code_chunks(chunk_id) ON DELETE CASCADE,

    concept_name TEXT NOT NULL,
    concept_type TEXT,
    confidence REAL DEFAULT 1.0,

    extractor TEXT,
    extractor_version TEXT,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(chunk_id, concept_name)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Vector similarity search (HNSW for fast approximate search)
CREATE INDEX IF NOT EXISTS idx_passages_embedding
    ON passages USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_code_chunks_embedding
    ON code_chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_passages_search
    ON passages USING gin (search_vector);

CREATE INDEX IF NOT EXISTS idx_code_chunks_search
    ON code_chunks USING gin (search_vector);

-- Document lookups
CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year);
CREATE INDEX IF NOT EXISTS idx_documents_title_hash ON documents(title_hash);

-- Passage lookups
CREATE INDEX IF NOT EXISTS idx_passages_doc ON passages(doc_id);
CREATE INDEX IF NOT EXISTS idx_passages_header ON passages(header);

-- Code lookups
CREATE INDEX IF NOT EXISTS idx_code_files_repo ON code_files(repo_name);
CREATE INDEX IF NOT EXISTS idx_code_chunks_file ON code_chunks(file_id);

-- Paper-repo lookups
CREATE INDEX IF NOT EXISTS idx_paper_repos_doc ON paper_repos(doc_id);
CREATE INDEX IF NOT EXISTS idx_paper_repos_repo ON paper_repos(repo_owner, repo_name);
CREATE INDEX IF NOT EXISTS idx_repo_queue_status ON repo_ingest_queue(status);
CREATE INDEX IF NOT EXISTS idx_repo_queue_priority ON repo_ingest_queue(priority DESC, created_at);

-- ============================================================================
-- HYBRID SEARCH FUNCTION (Optimized)
-- ============================================================================

-- Combined keyword + vector search using "retrieve then rerank" pattern
-- Avoids expensive FULL OUTER JOIN by:
-- 1. Getting top candidates from vector search (fast HNSW index)
-- 2. Scoring those candidates with keyword search
-- 3. Combining scores for final ranking
CREATE OR REPLACE FUNCTION hybrid_search(
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
    WITH
    -- Step 1: Get vector candidates (limited, uses HNSW index)
    vector_candidates AS (
        SELECT
            p.passage_id,
            1 - (p.embedding <=> query_embedding) as v_score
        FROM passages p
        WHERE p.embedding IS NOT NULL
        ORDER BY p.embedding <=> query_embedding
        LIMIT result_limit * 3  -- Over-fetch to allow for keyword boost
    ),
    -- Step 2: Score candidates with keyword search
    scored AS (
        SELECT
            vc.passage_id,
            COALESCE(ts_rank(p.search_vector, plainto_tsquery('english', query_text)), 0) as k_score,
            vc.v_score,
            (COALESCE(ts_rank(p.search_vector, plainto_tsquery('english', query_text)), 0) * keyword_weight +
             vc.v_score * vector_weight) as combined_score
        FROM vector_candidates vc
        JOIN passages p ON vc.passage_id = p.passage_id
    )
    -- Step 3: Return results with metadata
    SELECT
        s.passage_id,
        p.doc_id,
        p.passage_text,
        p.header,
        d.title,
        d.year,
        d.doi,
        s.k_score as keyword_score,
        s.v_score as vector_score,
        s.combined_score
    FROM scored s
    JOIN passages p ON s.passage_id = p.passage_id
    JOIN documents d ON p.doc_id = d.doc_id
    ORDER BY s.combined_score DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Strict mode: requires matches in BOTH vector and keyword search
CREATE OR REPLACE FUNCTION hybrid_search_strict(
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
    WITH
    -- Get keyword matches
    keyword_matches AS (
        SELECT
            p.passage_id,
            ts_rank(p.search_vector, plainto_tsquery('english', query_text)) as k_score
        FROM passages p
        WHERE p.search_vector @@ plainto_tsquery('english', query_text)
          AND p.embedding IS NOT NULL
        LIMIT result_limit * 5
    ),
    -- Score with vectors (INNER JOIN ensures both match)
    combined AS (
        SELECT
            km.passage_id,
            km.k_score,
            1 - (p.embedding <=> query_embedding) as v_score,
            (km.k_score * keyword_weight +
             (1 - (p.embedding <=> query_embedding)) * vector_weight) as combined_score
        FROM keyword_matches km
        JOIN passages p ON km.passage_id = p.passage_id
    )
    SELECT
        c.passage_id,
        p.doc_id,
        p.passage_text,
        p.header,
        d.title,
        d.year,
        d.doi,
        c.k_score as keyword_score,
        c.v_score as vector_score,
        c.combined_score
    FROM combined c
    JOIN passages p ON c.passage_id = p.passage_id
    JOIN documents d ON p.doc_id = d.doc_id
    ORDER BY c.combined_score DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- MIGRATION TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb_migrations (
    id SERIAL PRIMARY KEY,
    job_name TEXT NOT NULL,
    cursor_position TEXT,
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
    items_processed INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(job_name)
);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE documents IS 'Papers with metadata - compatible with V1, enhanced for V2';
COMMENT ON TABLE passages IS 'Paper passages with structure preservation and BGE-M3 embeddings';
COMMENT ON TABLE code_files IS 'Code repository files';
COMMENT ON TABLE code_chunks IS 'Code chunks (functions, classes) with search vectors';
COMMENT ON TABLE paper_repos IS 'Links between papers and their GitHub repositories';
COMMENT ON TABLE repo_ingest_queue IS 'Queue for automatic repository ingestion';
COMMENT ON FUNCTION hybrid_search IS 'Combined keyword + vector search in single query';
