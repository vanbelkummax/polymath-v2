# Polymath V2 Prototype Plan

## Overview

This document outlines the plan to prototype the Polymath V2 JIT (Just-In-Time) Memory Retrieval system. The prototype demonstrates the complete pipeline from PDF ingestion through Claude-accessible knowledge retrieval.

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         POLYMATH V2 PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [PDF Files]                                                            │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐                                                    │
│  │  ingest_v2.py   │  PHASE 1: SKELETON                                │
│  │  (MinerU parse) │  - Text extraction with layout awareness           │
│  └────────┬────────┘  - Header-aware chunking (512 tokens)             │
│           │           - BGE-M3 embeddings (1024-dim)                    │
│           │           - Postgres storage (passages table)               │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │hydrate_graph.py │  PHASE 2: MUSCLE                                  │
│  │(Entity Extract) │  - Pattern + LLM entity extraction                 │
│  └────────┬────────┘  - HaluMem-style validation                       │
│           │           - Neo4j typed concept graph                       │
│           │           - Postgres tracking (passage_extractions)         │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │jit_retrieval.py │  PHASE 3: BRAIN                                   │
│  │(Query-time JIT) │  - Query classification (factual/compare/etc)     │
│  └────────┬────────┘  - Concept index lookup (Neo4j)                   │
│           │           - Deep passage retrieval (Postgres + pgvector)    │
│           │           - Runtime synthesis (Gemini)                      │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │  MCP Server     │  PHASE 4: INTERFACE                               │
│  │(polymath_v2_    │  - research_topic: Deep multi-hop research         │
│  │  server.py)     │  - quick_lookup: Fast fact retrieval               │
│  └────────┬────────┘  - find_methods/find_papers: Targeted search      │
│           │           - validate_claim: Hallucination detection         │
│           ▼                                                             │
│      [Claude Code]                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Database Setup

```bash
# Postgres with pgvector
sudo -u postgres createdb polymath_v2
sudo -u postgres psql -d polymath_v2 -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Neo4j (via Docker or native)
docker run -d \
  --name neo4j-polymath \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/polymathic2026 \
  neo4j:5-community
```

### 2. Environment Configuration

Create `/home/user/polymath-v2/.env`:

```bash
# Database connections
POSTGRES_URI=postgresql://polymath:password@localhost/polymath_v2
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=polymathic2026

# API Keys (for synthesis)
GEMINI_API_KEY=your_gemini_key_here

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
```

### 3. Dependencies

```bash
cd /home/user/polymath-v2
pip install -r requirements.txt

# Key dependencies:
# - psycopg2-binary (Postgres)
# - pgvector (vector search)
# - neo4j (graph database)
# - sentence-transformers (BGE-M3)
# - google-generativeai (synthesis)
# - mcp (Model Context Protocol)
```

## Prototype Steps

### Step 1: Initialize Schema

```bash
# Create Postgres tables
python3 -c "
from lib.config import config
import psycopg2

conn = psycopg2.connect(config.POSTGRES_URI)
cur = conn.cursor()

# Create passages table with pgvector
cur.execute('''
CREATE TABLE IF NOT EXISTS passages_v2 (
    passage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL,
    passage_text TEXT NOT NULL,
    header TEXT,
    page_num INT,
    char_start INT,
    char_end INT,
    embedding vector(1024),
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', passage_text)) STORED,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_passages_v2_embedding ON passages_v2
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_passages_v2_search ON passages_v2
    USING gin (search_vector);
''')

# Create tracking table
cur.execute('''
CREATE TABLE IF NOT EXISTS passage_extractions (
    passage_id UUID PRIMARY KEY REFERENCES passages_v2(passage_id),
    extractor_version TEXT NOT NULL,
    entity_count INT DEFAULT 0,
    relation_count INT DEFAULT 0,
    extracted_at TIMESTAMP DEFAULT NOW()
);
''')

conn.commit()
print('Schema initialized')
"
```

### Step 2: Ingest Test Papers

```bash
# Ingest a few test PDFs
cd /home/user/polymath-v2

# Single paper
python3 lib/ingest_v2.py /path/to/test_paper.pdf

# Batch ingest
for pdf in /path/to/pdfs/*.pdf; do
    python3 lib/ingest_v2.py "$pdf"
done

# Verify ingestion
psql -d polymath_v2 -c "SELECT COUNT(*) as passages FROM passages_v2;"
```

### Step 3: Hydrate Knowledge Graph

```bash
# Run entity extraction and graph hydration
python3 scripts/hydrate_graph.py \
    --batch-size 50 \
    --llm-budget 100 \
    --verbose

# Monitor progress
tail -f logs/hydrate_graph.log

# Verify hydration
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'polymathic2026'))
with driver.session() as session:
    result = session.run('MATCH (n) RETURN labels(n)[0] as type, count(*) as cnt ORDER BY cnt DESC LIMIT 10')
    for r in result:
        print(f'{r[\"type\"]}: {r[\"cnt\"]}')
"
```

### Step 4: Test JIT Retrieval

```bash
# Test query classification
python3 -c "
from lib.jit_retrieval import classify_query

test_queries = [
    'What is optimal transport?',
    'Compare Img2ST vs HisToGene',
    'What methods exist for spatial prediction?',
    'How does attention mechanism work in transformers?'
]

for q in test_queries:
    print(f'{q[:50]}... -> {classify_query(q)}')
"

# Test full retrieval
python3 -c "
import asyncio
from lib.jit_retrieval import JITRetriever
from lib.config import config
import psycopg2
from neo4j import GraphDatabase

pg_conn = psycopg2.connect(config.POSTGRES_URI)
neo4j_driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

retriever = JITRetriever(pg_conn, neo4j_driver)

async def test():
    result = await retriever.retrieve('What is spatial transcriptomics?', max_depth=2)
    print(f'Passages: {len(result.passages)}')
    print(f'Concepts: {len(result.concepts)}')
    print(f'Synthesis: {result.synthesis[:500]}...')
    print(f'Path: {result.retrieval_path}')

asyncio.run(test())
"
```

### Step 5: Launch MCP Server

```bash
# Add to Claude Code MCP config (~/.mcp.json or project .mcp.json)
{
  "mcpServers": {
    "polymath-v2": {
      "command": "python3",
      "args": ["/home/user/polymath-v2/mcp/polymath_v2_server.py"],
      "env": {
        "PYTHONPATH": "/home/user/polymath-v2"
      }
    }
  }
}

# Test MCP server standalone
python3 mcp/polymath_v2_server.py
```

### Step 6: Verify Claude Integration

In Claude Code, use the MCP tools:

```
# Research a topic
research_topic("spatial transcriptomics prediction methods", depth=2)

# Quick lookup
quick_lookup("what is optimal transport")

# Find methods
find_methods(domain="spatial transcriptomics", problem="gene expression prediction")

# Find papers
find_papers(concept="Visium HD", concept_type="DATASET")

# Validate a claim
validate_claim(
    claim="Img2ST achieves 0.85 PCC on Visium data",
    supporting_text="[passage from paper]"
)
```

## Success Criteria

### Minimum Viable Prototype

- [ ] 10+ papers ingested with passages in Postgres
- [ ] Entity extraction working (Pattern + LLM hybrid)
- [ ] Neo4j populated with typed concept nodes
- [ ] JIT retrieval returning relevant passages
- [ ] MCP server accessible from Claude Code
- [ ] At least one successful research_topic query

### Full Prototype

- [ ] 100+ papers ingested
- [ ] Passage extraction tracking in Postgres
- [ ] Multi-hop retrieval working for synthesis queries
- [ ] Hallucination detection functional
- [ ] Sub-second quick_lookup responses
- [ ] Runtime synthesis producing cited answers

## Metrics to Track

| Metric | Target | Measurement |
|--------|--------|-------------|
| Ingestion throughput | 10 papers/minute | Time per PDF |
| Entity extraction rate | 95% passages processed | passage_extractions count |
| Retrieval latency | <2s factual, <10s synthesis | Timer in JIT retriever |
| Synthesis quality | Cited answers | Manual review |
| Neo4j graph density | >5 edges per concept | Graph statistics |

## Known Limitations (Prototype)

1. **No incremental updates**: Re-ingestion creates duplicates (need deduplication)
2. **Single-node deployment**: No distributed processing yet
3. **Gemini dependency**: Synthesis requires API key and internet
4. **Limited concept types**: Pattern extraction covers ~15 types
5. **No user feedback loop**: No mechanism to improve based on query success

## Future Enhancements

### Short-term (Post-prototype)

1. **Deduplication**: Content hashing to prevent duplicate passages
2. **Batch LLM extraction**: Gemini batch API for cost efficiency
3. **Query caching**: Cache common queries with TTL
4. **Better synthesis prompts**: Few-shot examples for domain-specific synthesis

### Medium-term

1. **Streaming responses**: Stream synthesis as it generates
2. **Multi-modal support**: Handle figures and tables
3. **Citation verification**: Validate citations against actual paper content
4. **Query refinement**: Suggest related queries based on results

### Long-term

1. **Federated retrieval**: Connect to external knowledge bases
2. **Active learning**: Improve extraction based on usage patterns
3. **Personalization**: Learn user's research interests
4. **Collaborative filtering**: Recommend papers based on similar researchers

## Troubleshooting

### Common Issues

**1. Neo4j connection refused**
```bash
# Check if Neo4j is running
docker ps | grep neo4j
# Or check native service
sudo systemctl status neo4j
```

**2. Postgres vector extension missing**
```bash
sudo -u postgres psql -d polymath_v2 -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**3. Embedding model download fails**
```bash
# Pre-download the model
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

**4. MCP server not found by Claude**
```bash
# Verify path in .mcp.json
cat ~/.mcp.json | jq '.mcpServers["polymath-v2"]'
# Test server directly
python3 /home/user/polymath-v2/mcp/polymath_v2_server.py
```

**5. Synthesis returns fallback (no Gemini key)**
```bash
# Add Gemini API key to .env
echo "GEMINI_API_KEY=your_key" >> /home/user/polymath-v2/.env
```

## References

- [GAM Paper](https://arxiv.org/abs/2505.xxxxx) - JIT memory architecture inspiration
- [HaluMem](https://arxiv.org/abs/2505.xxxxx) - Hallucination detection approach
- [MCP Specification](https://modelcontextprotocol.io/) - Claude tool integration
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Embedding model

---

*Document created: 2026-01-13*
*Polymath V2 Prototype Plan v1.0*
