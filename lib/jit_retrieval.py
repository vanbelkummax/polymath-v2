#!/usr/bin/env python3
"""
Just-In-Time (JIT) Memory Retrieval for Polymath

Based on findings from:
- "General Agentic Memory Via Deep Research" (GAM) (2025)

Key insight: Static memory (ahead-of-time compression) inevitably loses information.
JIT memory compilation retrieves and integrates from raw data at query time,
guided by the specific query context.

Architecture:
- Memorizer: Lightweight index (Neo4j concepts) + universal page-store (Postgres passages)
- Researcher: Query-guided deep retrieval that integrates multiple sources at runtime

Benefits over static summarization:
- No information loss from pre-compression
- Query-specific context selection
- Adaptive depth based on query complexity
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import json

from lib.config import config
from lib.embeddings import get_embedder


@dataclass
class RetrievalContext:
    """Context built up during JIT retrieval."""
    query: str
    query_type: str  # 'factual', 'comparison', 'exploratory', 'synthesis'
    depth: int = 1
    retrieved_passages: List[Dict] = field(default_factory=list)
    retrieved_concepts: List[Dict] = field(default_factory=list)
    synthesis_notes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class JITResult:
    """Result of JIT retrieval."""
    passages: List[Dict]  # Raw passages with metadata
    concepts: List[Dict]  # Lightweight concept index entries
    synthesis: str        # Query-specific synthesis (not pre-computed!)
    confidence: float
    retrieval_path: List[str]  # For debugging: sequence of retrieval steps


# =============================================================================
# Query Classification (Determines Retrieval Strategy)
# =============================================================================

def classify_query(query: str) -> str:
    """
    Classify query to determine optimal retrieval strategy.

    Returns:
        'factual': Direct fact lookup (single-hop)
        'comparison': Compare entities (multi-hop, parallel)
        'exploratory': Broad exploration (iterative expansion)
        'synthesis': Integrate multiple sources (deep, sequential)
    """
    query_lower = query.lower()

    # Comparison queries
    if any(w in query_lower for w in ['compare', 'versus', 'vs', 'difference between',
                                       'better than', 'outperform']):
        return 'comparison'

    # Synthesis queries
    if any(w in query_lower for w in ['how does', 'why does', 'explain', 'mechanism',
                                       'integrate', 'combine', 'synthesize']):
        return 'synthesis'

    # Exploratory queries
    if any(w in query_lower for w in ['what are', 'list', 'overview', 'survey',
                                       'state of the art', 'recent advances']):
        return 'exploratory'

    # Default to factual
    return 'factual'


# =============================================================================
# Lightweight Index Lookup (Neo4j Concepts)
# =============================================================================

async def lookup_concept_index(
    query: str,
    neo4j_driver,
    limit: int = 20
) -> List[Dict]:
    """
    Query the lightweight concept index in Neo4j.

    This is the "memorizer" component - just pointers, not full content.
    Returns concept nodes with their paper relationships.
    """
    # Extract likely entity types from query
    query_lower = query.lower()

    # Build type-aware query
    type_hints = []
    if any(w in query_lower for w in ['method', 'approach', 'technique', 'model']):
        type_hints.append('METHOD')
    if any(w in query_lower for w in ['gene', 'expression', 'marker']):
        type_hints.append('GENE')
    if any(w in query_lower for w in ['cell', 'type', 'population']):
        type_hints.append('CELL_TYPE')
    if any(w in query_lower for w in ['dataset', 'data', 'visium', 'xenium']):
        type_hints.append('DATASET')
    if any(w in query_lower for w in ['tissue', 'organ', 'sample']):
        type_hints.append('TISSUE')

    with neo4j_driver.session() as session:
        if type_hints:
            # Type-guided search
            result = session.run("""
                CALL db.index.fulltext.queryNodes('concept_name_fulltext', $query)
                YIELD node, score
                WHERE any(label IN labels(node) WHERE label IN $types)
                WITH node, score
                MATCH (node)-[r:MENTIONED_IN]->(p:Paper)
                RETURN node.name AS concept,
                       labels(node)[0] AS type,
                       score,
                       collect(DISTINCT p.doc_id)[0..5] AS paper_ids,
                       count(DISTINCT p) AS paper_count
                ORDER BY score DESC
                LIMIT $limit
            """, {'query': query, 'types': type_hints, 'limit': limit})
        else:
            # General search
            result = session.run("""
                CALL db.index.fulltext.queryNodes('concept_name_fulltext', $query)
                YIELD node, score
                WITH node, score
                MATCH (node)-[r:MENTIONED_IN]->(p:Paper)
                RETURN node.name AS concept,
                       labels(node)[0] AS type,
                       score,
                       collect(DISTINCT p.doc_id)[0..5] AS paper_ids,
                       count(DISTINCT p) AS paper_count
                ORDER BY score DESC
                LIMIT $limit
            """, {'query': query, 'limit': limit})

        return [dict(r) for r in result]


# =============================================================================
# Deep Passage Retrieval (Postgres + pgvector)
# =============================================================================

async def retrieve_passages_deep(
    query: str,
    pg_conn,
    embedder,
    concept_doc_ids: Optional[List[str]] = None,
    limit: int = 20,
    use_hybrid: bool = True
) -> List[Dict]:
    """
    Deep retrieval from the universal page-store (Postgres).

    This is query-time retrieval - no pre-summarization.
    Optionally filters by doc_ids from concept index for focused search.
    """
    query_embedding = embedder.encode([query])[0]

    cur = pg_conn.cursor()

    if use_hybrid and concept_doc_ids:
        # Hybrid: Vector + keyword + concept-guided filtering
        cur.execute("""
            WITH vector_results AS (
                SELECT passage_id, doc_id, passage_text, header,
                       1 - (embedding <=> %s::vector) AS vector_score
                FROM passages_v2
                WHERE doc_id = ANY(%s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            ),
            keyword_results AS (
                SELECT passage_id, doc_id, passage_text, header,
                       ts_rank(search_vector, plainto_tsquery('english', %s)) AS keyword_score
                FROM passages_v2
                WHERE doc_id = ANY(%s)
                  AND search_vector @@ plainto_tsquery('english', %s)
                ORDER BY keyword_score DESC
                LIMIT %s
            )
            SELECT COALESCE(v.passage_id, k.passage_id) AS passage_id,
                   COALESCE(v.doc_id, k.doc_id) AS doc_id,
                   COALESCE(v.passage_text, k.passage_text) AS passage_text,
                   COALESCE(v.header, k.header) AS header,
                   COALESCE(v.vector_score, 0) AS vector_score,
                   COALESCE(k.keyword_score, 0) AS keyword_score,
                   (0.7 * COALESCE(v.vector_score, 0) + 0.3 * COALESCE(k.keyword_score, 0)) AS combined_score
            FROM vector_results v
            FULL OUTER JOIN keyword_results k ON v.passage_id = k.passage_id
            ORDER BY combined_score DESC
            LIMIT %s
        """, (
            query_embedding.tolist(), concept_doc_ids,
            query_embedding.tolist(), limit,
            query, concept_doc_ids, query, limit,
            limit
        ))
    else:
        # Vector-only search across all documents
        cur.execute("""
            SELECT passage_id, doc_id, passage_text, header,
                   1 - (embedding <=> %s::vector) AS score
            FROM passages_v2
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding.tolist(), query_embedding.tolist(), limit))

    columns = [desc[0] for desc in cur.description]
    results = [dict(zip(columns, row)) for row in cur.fetchall()]

    cur.close()
    return results


# =============================================================================
# Query-Specific Synthesis (Runtime, Not Pre-computed)
# =============================================================================

async def synthesize_at_runtime(
    query: str,
    passages: List[Dict],
    concepts: List[Dict],
    model: str = "gemini-1.5-flash"
) -> Tuple[str, float]:
    """
    Synthesize answer at query time from raw passages.

    This is the key GAM insight: synthesis happens at retrieval time,
    not at ingestion time. The query context guides what to emphasize.
    """
    import google.generativeai as genai

    if not config.GEMINI_API_KEY:
        # Fallback: simple concatenation
        synthesis = "Retrieved passages:\n\n"
        for i, p in enumerate(passages[:5], 1):
            synthesis += f"[{i}] {p.get('header', 'Unknown')}: {p.get('passage_text', '')[:500]}...\n\n"
        return synthesis, 0.5

    genai.configure(api_key=config.GEMINI_API_KEY)
    model_instance = genai.GenerativeModel(model)

    # Build context from passages
    passage_context = "\n\n".join([
        f"[{i}] Section: {p.get('header', 'Unknown')}\n{p.get('passage_text', '')}"
        for i, p in enumerate(passages[:10], 1)
    ])

    # Build concept context
    concept_context = ", ".join([
        f"{c['concept']} ({c['type']}, {c['paper_count']} papers)"
        for c in concepts[:10]
    ])

    prompt = f"""Based on the following retrieved passages and concepts, answer this query:

QUERY: {query}

RELEVANT CONCEPTS: {concept_context}

PASSAGES:
{passage_context}

Instructions:
1. Synthesize information from multiple passages when relevant
2. Cite passage numbers [1], [2], etc. for each claim
3. If information is uncertain or conflicting, note it
4. Focus on what the passages actually say, don't extrapolate
5. Be concise but comprehensive

SYNTHESIS:"""

    try:
        response = await model_instance.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,  # Low temperature for factual synthesis
                max_output_tokens=1000
            )
        )

        return response.text, 0.9

    except Exception as e:
        print(f"Synthesis failed: {e}")
        # Fallback
        synthesis = f"Query: {query}\n\nTop passages:\n"
        for i, p in enumerate(passages[:3], 1):
            synthesis += f"[{i}] {p.get('passage_text', '')[:300]}...\n"
        return synthesis, 0.5


# =============================================================================
# Main JIT Retrieval Class
# =============================================================================

class JITRetriever:
    """
    Just-In-Time memory retrieval system.

    Unlike traditional RAG that pre-summarizes content at ingestion,
    JIT retrieval:
    1. Maintains lightweight index (concepts) + raw storage (passages)
    2. Retrieves and synthesizes at query time
    3. Adapts retrieval depth to query complexity
    """

    def __init__(self, pg_conn, neo4j_driver, embedder=None):
        """
        Args:
            pg_conn: Postgres connection (with pgvector)
            neo4j_driver: Neo4j driver (for concept index)
            embedder: BGE-M3 embedder (loaded if not provided)
        """
        self.pg_conn = pg_conn
        self.neo4j_driver = neo4j_driver
        self.embedder = embedder or get_embedder()

    async def retrieve(
        self,
        query: str,
        max_depth: int = 2,
        passage_limit: int = 20,
        concept_limit: int = 20
    ) -> JITResult:
        """
        Perform JIT retrieval for a query.

        Args:
            query: The user query
            max_depth: Maximum retrieval iterations for complex queries
            passage_limit: Max passages to retrieve
            concept_limit: Max concepts to lookup

        Returns:
            JITResult with passages, concepts, and runtime synthesis
        """
        retrieval_path = []

        # Step 1: Classify query to determine strategy
        query_type = classify_query(query)
        retrieval_path.append(f"Query classified as: {query_type}")

        # Step 2: Lookup lightweight concept index
        concepts = await lookup_concept_index(
            query, self.neo4j_driver, limit=concept_limit
        )
        retrieval_path.append(f"Found {len(concepts)} concepts in index")

        # Extract doc_ids from concepts for focused retrieval
        concept_doc_ids = list(set(
            doc_id
            for c in concepts
            for doc_id in c.get('paper_ids', [])
        ))
        retrieval_path.append(f"Concept-linked docs: {len(concept_doc_ids)}")

        # Step 3: Deep passage retrieval
        if query_type == 'factual':
            # Single-hop retrieval
            passages = await retrieve_passages_deep(
                query, self.pg_conn, self.embedder,
                concept_doc_ids=concept_doc_ids[:50] if concept_doc_ids else None,
                limit=passage_limit,
                use_hybrid=True
            )
            retrieval_path.append(f"Factual retrieval: {len(passages)} passages")

        elif query_type == 'comparison':
            # Parallel retrieval for compared entities
            # Extract entities being compared
            passages = await retrieve_passages_deep(
                query, self.pg_conn, self.embedder,
                concept_doc_ids=concept_doc_ids[:100] if concept_doc_ids else None,
                limit=passage_limit * 2,  # More passages for comparison
                use_hybrid=True
            )
            retrieval_path.append(f"Comparison retrieval: {len(passages)} passages")

        elif query_type == 'exploratory':
            # Broad retrieval, less filtering
            passages = await retrieve_passages_deep(
                query, self.pg_conn, self.embedder,
                concept_doc_ids=None,  # Don't filter by concept
                limit=passage_limit,
                use_hybrid=True
            )
            retrieval_path.append(f"Exploratory retrieval: {len(passages)} passages")

        elif query_type == 'synthesis':
            # Deep, iterative retrieval
            all_passages = []
            current_query = query

            for depth in range(max_depth):
                new_passages = await retrieve_passages_deep(
                    current_query, self.pg_conn, self.embedder,
                    concept_doc_ids=concept_doc_ids[:100] if concept_doc_ids else None,
                    limit=passage_limit // max_depth,
                    use_hybrid=True
                )
                all_passages.extend(new_passages)

                # Expand query based on retrieved content (simulate researcher)
                if depth < max_depth - 1 and new_passages:
                    # Extract key terms from retrieved passages for query expansion
                    top_passage = new_passages[0].get('passage_text', '')[:500]
                    current_query = f"{query} {top_passage[:100]}"

                retrieval_path.append(f"Synthesis depth {depth+1}: +{len(new_passages)} passages")

            # Deduplicate
            seen = set()
            passages = []
            for p in all_passages:
                pid = p.get('passage_id')
                if pid and pid not in seen:
                    seen.add(pid)
                    passages.append(p)

        else:
            passages = []

        # Step 4: Runtime synthesis (NOT pre-computed!)
        synthesis, confidence = await synthesize_at_runtime(
            query, passages, concepts
        )
        retrieval_path.append(f"Runtime synthesis complete (conf={confidence:.2f})")

        return JITResult(
            passages=passages,
            concepts=concepts,
            synthesis=synthesis,
            confidence=confidence,
            retrieval_path=retrieval_path
        )

    def retrieve_sync(self, query: str, **kwargs) -> JITResult:
        """Synchronous wrapper for retrieve()."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.retrieve(query, **kwargs))


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("JIT Retrieval System Test")
    print("=" * 60)

    # Test query classification
    test_queries = [
        "What is the PCC of Img2ST on Visium HD?",
        "Compare Img2ST vs HisToGene performance",
        "What methods exist for spatial transcriptomics prediction?",
        "How does optimal transport work in spatial alignment?"
    ]

    print("\n[Query Classification]")
    for q in test_queries:
        qtype = classify_query(q)
        print(f"  '{q[:50]}...' → {qtype}")

    print("\n[Full JIT Retrieval requires database connections]")
    print("Run with: python -c \"from lib.jit_retrieval import JITRetriever; ...\"")
