#!/usr/bin/env python3
"""
Polymath V2 MCP Server

Connects the JIT Retrieval system to Claude via Model Context Protocol.
This is the "hands" that allows Claude to interact with the knowledge base.

Tools provided:
- research_topic: Deep research with multi-hop synthesis
- quick_lookup: Fast fact retrieval
- find_methods: Search for methods/algorithms
- find_papers: Search for papers by concept
- extract_entities: Extract entities from text (for analysis)

Usage:
    # Start server (stdio mode for Claude Code)
    python3 mcp/polymath_v2_server.py

    # Or with uvicorn (HTTP mode)
    uvicorn mcp.polymath_v2_server:app --port 8000

Configuration:
    Reads from lib/config.py which loads .env
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.jit_retrieval import JITRetriever, classify_query
from lib.entity_extraction import extract_entities_sync, extract_entities_pattern
from lib.hallucination_detector import HallucinationDetector

# Try to import FastMCP, fall back to manual implementation if not available
try:
    from mcp.server.fastmcp import FastMCP
    USE_FASTMCP = True
except ImportError:
    USE_FASTMCP = False
    print("FastMCP not available, using manual MCP implementation")

# Database connections (lazy initialization)
_pg_conn = None
_neo4j_driver = None
_retriever = None
_detector = None


def get_pg_conn():
    """Lazy initialization of Postgres connection."""
    global _pg_conn
    if _pg_conn is None:
        import psycopg2
        _pg_conn = psycopg2.connect(config.POSTGRES_URI)
    return _pg_conn


def get_neo4j_driver():
    """Lazy initialization of Neo4j driver."""
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
    return _neo4j_driver


def get_retriever():
    """Lazy initialization of JIT retriever."""
    global _retriever
    if _retriever is None:
        _retriever = JITRetriever(
            pg_conn=get_pg_conn(),
            neo4j_driver=get_neo4j_driver()
        )
    return _retriever


def get_detector():
    """Lazy initialization of hallucination detector."""
    global _detector
    if _detector is None:
        _detector = HallucinationDetector()
    return _detector


# =============================================================================
# MCP Server Definition
# =============================================================================

if USE_FASTMCP:
    mcp = FastMCP("Polymath V2")

    @mcp.tool()
    async def research_topic(
        query: str,
        depth: int = 2,
        max_passages: int = 20
    ) -> str:
        """
        Perform deep research on a topic using the Polymath knowledge base.

        This tool synthesizes information from multiple papers, extracting
        relevant passages and generating a comprehensive answer.

        Use this for:
        - Complex questions requiring synthesis
        - Understanding mechanisms or methods
        - Comparing approaches
        - Literature review

        Args:
            query: The research question or topic
            depth: Search depth (1=fast, 2=thorough, 3=exhaustive)
            max_passages: Maximum passages to retrieve

        Returns:
            Synthesized answer with citations
        """
        retriever = get_retriever()
        result = await retriever.retrieve(
            query,
            max_depth=depth,
            passage_limit=max_passages
        )

        # Format response with citations
        response = f"## Research Results for: {query}\n\n"
        response += f"**Query Type**: {classify_query(query)}\n"
        response += f"**Passages Retrieved**: {len(result.passages)}\n"
        response += f"**Concepts Found**: {len(result.concepts)}\n\n"
        response += "### Synthesis\n\n"
        response += result.synthesis
        response += "\n\n### Key Concepts\n\n"

        for c in result.concepts[:10]:
            response += f"- **{c['concept']}** ({c['type']}): {c['paper_count']} papers\n"

        return response

    @mcp.tool()
    async def quick_lookup(query: str) -> str:
        """
        Fast lookup for specific facts, definitions, or methods.

        Use this for:
        - Definition lookups
        - Quick fact checks
        - Method names or references
        - Simple questions

        Args:
            query: The question or term to look up

        Returns:
            Brief answer with source
        """
        retriever = get_retriever()
        result = await retriever.retrieve(
            query,
            max_depth=1,
            passage_limit=5
        )

        response = f"**Quick Lookup**: {query}\n\n"
        response += result.synthesis

        if result.passages:
            response += "\n\n**Top Source**:\n"
            p = result.passages[0]
            response += f"- Section: {p.get('header', 'Unknown')}\n"
            response += f"- Excerpt: {p.get('passage_text', '')[:300]}..."

        return response

    @mcp.tool()
    async def find_methods(
        domain: str = "spatial transcriptomics",
        problem: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """
        Search for methods and algorithms in the knowledge base.

        Use this for:
        - Finding methods that solve a specific problem
        - Discovering algorithms in a domain
        - Comparing method approaches

        Args:
            domain: The domain to search in
            problem: Optional specific problem the method should solve
            limit: Maximum methods to return

        Returns:
            List of methods with descriptions
        """
        driver = get_neo4j_driver()

        query = f"methods for {problem}" if problem else f"methods in {domain}"

        with driver.session() as session:
            if problem:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('concept_name_fulltext', $query)
                    YIELD node, score
                    WHERE 'METHOD' IN labels(node) OR 'ALGORITHM' IN labels(node)
                    WITH node, score
                    MATCH (node)-[:MENTIONED_IN]->(p:Paper)
                    RETURN node.name AS method,
                           labels(node)[0] AS type,
                           score,
                           count(DISTINCT p) AS paper_count,
                           collect(DISTINCT p.title)[0..3] AS sample_papers
                    ORDER BY paper_count DESC
                    LIMIT $limit
                """, {'query': problem, 'limit': limit})
            else:
                result = session.run("""
                    MATCH (m)
                    WHERE 'METHOD' IN labels(m) OR 'ALGORITHM' IN labels(m)
                    WITH m
                    MATCH (m)-[:MENTIONED_IN]->(p:Paper)
                    RETURN m.name AS method,
                           labels(m)[0] AS type,
                           count(DISTINCT p) AS paper_count,
                           collect(DISTINCT p.title)[0..3] AS sample_papers
                    ORDER BY paper_count DESC
                    LIMIT $limit
                """, {'limit': limit})

            methods = list(result)

        response = f"## Methods Found\n\n"
        response += f"**Domain**: {domain}\n"
        if problem:
            response += f"**Problem**: {problem}\n"
        response += "\n"

        for m in methods:
            response += f"### {m['method']} ({m['type']})\n"
            response += f"- **Papers**: {m['paper_count']}\n"
            if m.get('sample_papers'):
                response += f"- **Example papers**: {', '.join(m['sample_papers'][:2])}\n"
            response += "\n"

        return response

    @mcp.tool()
    async def find_papers(
        concept: str,
        concept_type: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """
        Find papers that mention a specific concept.

        Use this for:
        - Finding papers about a method
        - Literature on a specific gene/cell type
        - Papers using a dataset

        Args:
            concept: The concept to search for
            concept_type: Optional type filter (METHOD, GENE, CELL_TYPE, DATASET, etc.)
            limit: Maximum papers to return

        Returns:
            List of papers with relevance scores
        """
        driver = get_neo4j_driver()

        with driver.session() as session:
            if concept_type:
                result = session.run(f"""
                    MATCH (c:{concept_type} {{name: $concept}})-[:MENTIONED_IN]->(p:Paper)
                    RETURN p.title AS title, p.year AS year, p.doi AS doi,
                           count(*) AS mentions
                    ORDER BY mentions DESC
                    LIMIT $limit
                """, {'concept': concept, 'limit': limit})
            else:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('concept_name_fulltext', $concept)
                    YIELD node, score
                    WITH node, score
                    MATCH (node)-[:MENTIONED_IN]->(p:Paper)
                    RETURN p.title AS title, p.year AS year, p.doi AS doi,
                           sum(score) AS relevance
                    ORDER BY relevance DESC
                    LIMIT $limit
                """, {'concept': concept, 'limit': limit})

            papers = list(result)

        response = f"## Papers mentioning: {concept}\n\n"

        for p in papers:
            response += f"- **{p['title']}**"
            if p.get('year'):
                response += f" ({p['year']})"
            if p.get('doi'):
                response += f" - DOI: {p['doi']}"
            response += "\n"

        return response

    @mcp.tool()
    async def search_paper_code(
        query: str,
        paper_title: Optional[str] = None,
        repo_name: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """
        Search across both paper passages and linked code repositories.

        Use this for:
        - Finding implementation details for a method described in a paper
        - Understanding how a paper's algorithm is actually coded
        - Cross-referencing theoretical descriptions with code

        Args:
            query: Search query
            paper_title: Optional paper title to focus search
            repo_name: Optional repo name to focus search
            limit: Maximum results per category

        Returns:
            Combined results from paper passages and code chunks
        """
        conn = get_pg_conn()
        cur = conn.cursor()

        response = f"## Search Results for: {query}\n\n"

        # If paper_title provided, find the doc_id
        doc_id = None
        if paper_title:
            cur.execute("""
                SELECT doc_id, title FROM documents
                WHERE title ILIKE %s
                LIMIT 1
            """, (f'%{paper_title}%',))
            row = cur.fetchone()
            if row:
                doc_id = str(row[0])
                response += f"**Paper**: {row[1]}\n\n"

        # Search paper passages
        response += "### Paper Passages\n\n"
        if doc_id:
            cur.execute("""
                SELECT passage_text, section,
                       ts_rank(to_tsvector('english', passage_text),
                               plainto_tsquery('english', %s)) as score
                FROM passages
                WHERE doc_id = %s
                  AND to_tsvector('english', passage_text) @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, doc_id, query, limit))
        else:
            cur.execute("""
                SELECT p.passage_text, p.section, d.title,
                       ts_rank(to_tsvector('english', p.passage_text),
                               plainto_tsquery('english', %s)) as score
                FROM passages p
                JOIN documents d ON p.doc_id = d.doc_id
                WHERE to_tsvector('english', p.passage_text) @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, limit))

        passages = cur.fetchall()
        if passages:
            for i, row in enumerate(passages[:5], 1):
                text = row[0][:300] if len(row[0]) > 300 else row[0]
                section = row[1] or "Unknown"
                if len(row) > 3:  # Has title
                    response += f"**[{i}] {row[2][:50]}... ({section})**\n"
                else:
                    response += f"**[{i}] {section}**\n"
                response += f"{text}...\n\n"
        else:
            response += "No matching passages found.\n\n"

        # Get linked repos for the paper
        response += "### Linked Repositories\n\n"
        if doc_id:
            cur.execute("""
                SELECT repo_url, repo_owner, repo_name, confidence
                FROM paper_repos
                WHERE doc_id = %s
                ORDER BY confidence DESC
            """, (doc_id,))
            repos = cur.fetchall()
            if repos:
                for r in repos:
                    response += f"- [{r[1]}/{r[2]}]({r[0]}) (confidence: {r[3]:.2f})\n"
            else:
                response += "No linked repositories found.\n"
        else:
            response += "Provide paper_title to see linked repos.\n"

        # Search code
        response += "\n### Code Matches\n\n"
        if repo_name:
            cur.execute("""
                SELECT cf.repo_name, cf.file_path, ch.name, ch.content,
                       ts_rank(ch.search_vector, plainto_tsquery('english', %s)) as score
                FROM code_chunks ch
                JOIN code_files cf ON ch.file_id = cf.file_id
                WHERE cf.repo_name = %s
                  AND ch.search_vector @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, repo_name, query, limit))
        elif doc_id:
            # Search in repos linked to the paper
            cur.execute("""
                SELECT cf.repo_name, cf.file_path, ch.name, ch.content,
                       ts_rank(ch.search_vector, plainto_tsquery('english', %s)) as score
                FROM code_chunks ch
                JOIN code_files cf ON ch.file_id = cf.file_id
                JOIN paper_repos pr ON cf.repo_name = pr.repo_name
                WHERE pr.doc_id = %s
                  AND ch.search_vector @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, doc_id, query, limit))
        else:
            cur.execute("""
                SELECT cf.repo_name, cf.file_path, ch.name, ch.content,
                       ts_rank(ch.search_vector, plainto_tsquery('english', %s)) as score
                FROM code_chunks ch
                JOIN code_files cf ON ch.file_id = cf.file_id
                WHERE ch.search_vector @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, limit))

        code_results = cur.fetchall()
        if code_results:
            for row in code_results[:5]:
                repo, filepath, name, content, score = row
                response += f"**{repo}** - `{filepath}`\n"
                if name:
                    response += f"Function/Class: `{name}`\n"
                code_preview = content[:200] if len(content) > 200 else content
                response += f"```\n{code_preview}...\n```\n\n"
        else:
            response += "No matching code found.\n"

        cur.close()
        return response

    @mcp.tool()
    async def get_paper_repos(paper_title: str) -> str:
        """
        Get all GitHub repositories linked to a paper.

        Use this for:
        - Finding code implementations for a paper
        - Checking if a paper has open source code
        - Getting repo URLs for papers you're interested in

        Args:
            paper_title: Title or partial title of the paper

        Returns:
            List of linked repositories with metadata
        """
        conn = get_pg_conn()
        cur = conn.cursor()

        cur.execute("""
            SELECT d.title, d.year, d.doi, pr.repo_url, pr.repo_owner, pr.repo_name,
                   pr.confidence, pr.context
            FROM documents d
            JOIN paper_repos pr ON d.doc_id = pr.doc_id
            WHERE d.title ILIKE %s
            ORDER BY pr.confidence DESC
        """, (f'%{paper_title}%',))

        results = cur.fetchall()
        cur.close()

        if not results:
            return f"No papers found matching: {paper_title}"

        response = f"## Repositories for papers matching: {paper_title}\n\n"

        current_title = None
        for row in results:
            title, year, doi, url, owner, name, conf, context = row
            if title != current_title:
                current_title = title
                response += f"### {title}"
                if year:
                    response += f" ({year})"
                response += "\n"
                if doi:
                    response += f"DOI: {doi}\n"
                response += "\n"

            response += f"- **[{owner}/{name}]({url})** (confidence: {conf:.2f})\n"
            if context:
                response += f"  Context: _{context[:100]}..._\n"

        return response

    @mcp.tool()
    async def extract_entities_from_text(text: str) -> str:
        """
        Extract domain-specific entities from a piece of text.

        Use this for:
        - Analyzing a paper abstract
        - Identifying methods mentioned in text
        - Finding genes, cell types, datasets

        Args:
            text: The text to analyze

        Returns:
            Extracted entities with types and confidence
        """
        result = extract_entities_sync(text, "mcp-analysis")

        response = "## Extracted Entities\n\n"
        response += f"**Extractor**: {result.extractor_version}\n\n"

        if result.entities:
            response += "### Entities\n\n"
            by_type = {}
            for e in result.entities:
                by_type.setdefault(e.entity_type, []).append(e)

            for etype, entities in sorted(by_type.items()):
                response += f"**{etype}**:\n"
                for e in entities:
                    response += f"- {e.name} (conf: {e.confidence:.2f})\n"
                response += "\n"

        if result.relations:
            response += "### Relations\n\n"
            for r in result.relations:
                response += f"- {r.source_entity} --[{r.relation_type}]--> {r.target_entity}\n"

        return response

    @mcp.tool()
    async def validate_claim(
        claim: str,
        supporting_text: str
    ) -> str:
        """
        Validate whether a claim is grounded in supporting text.

        Use this for:
        - Checking if an answer is hallucinated
        - Verifying claims against sources
        - Fact-checking generated content

        Args:
            claim: The claim to validate
            supporting_text: The text that should support the claim

        Returns:
            Validation result with confidence and issues
        """
        detector = get_detector()
        result = detector.validate_answer(claim, [supporting_text], "Is this claim grounded?")

        response = "## Claim Validation\n\n"
        response += f"**Claim**: {claim[:200]}...\n\n"
        response += f"**Valid**: {'Yes' if result.is_valid else 'No'}\n"
        response += f"**Confidence**: {result.confidence:.2f}\n\n"

        if result.issues:
            response += "### Issues Found\n\n"
            for issue in result.issues:
                response += f"- {issue}\n"

        if result.suggestions:
            response += "\n### Suggestions\n\n"
            for suggestion in result.suggestions:
                response += f"- {suggestion}\n"

        return response


# =============================================================================
# Manual MCP Implementation (fallback)
# =============================================================================

else:
    import json

    async def handle_request(request: dict) -> dict:
        """Handle MCP request manually."""
        method = request.get('method', '')

        if method == 'tools/list':
            return {
                'tools': [
                    {
                        'name': 'research_topic',
                        'description': 'Deep research on a topic',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'query': {'type': 'string'},
                                'depth': {'type': 'integer', 'default': 2},
                                'max_passages': {'type': 'integer', 'default': 20}
                            },
                            'required': ['query']
                        }
                    },
                    {
                        'name': 'quick_lookup',
                        'description': 'Fast fact lookup',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'query': {'type': 'string'}
                            },
                            'required': ['query']
                        }
                    }
                ]
            }

        elif method == 'tools/call':
            tool_name = request.get('params', {}).get('name')
            args = request.get('params', {}).get('arguments', {})

            if tool_name == 'research_topic':
                retriever = get_retriever()
                result = await retriever.retrieve(
                    args['query'],
                    max_depth=args.get('depth', 2),
                    passage_limit=args.get('max_passages', 20)
                )
                return {'content': [{'type': 'text', 'text': result.synthesis}]}

            elif tool_name == 'quick_lookup':
                retriever = get_retriever()
                result = await retriever.retrieve(
                    args['query'],
                    max_depth=1,
                    passage_limit=5
                )
                return {'content': [{'type': 'text', 'text': result.synthesis}]}

        return {'error': f'Unknown method: {method}'}

    async def run_stdio():
        """Run MCP server in stdio mode."""
        import sys

        while True:
            line = sys.stdin.readline()
            if not line:
                break

            try:
                request = json.loads(line)
                response = await handle_request(request)
                print(json.dumps(response), flush=True)
            except Exception as e:
                print(json.dumps({'error': str(e)}), flush=True)


# =============================================================================
# Main Entry Point
# =============================================================================

def cleanup():
    """Clean up database connections."""
    global _pg_conn, _neo4j_driver
    if _pg_conn:
        _pg_conn.close()
    if _neo4j_driver:
        _neo4j_driver.close()


if __name__ == "__main__":
    import atexit
    atexit.register(cleanup)

    if USE_FASTMCP:
        # Run with FastMCP
        mcp.run()
    else:
        # Run manual stdio implementation
        asyncio.run(run_stdio())
