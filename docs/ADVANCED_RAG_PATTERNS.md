# Advanced RAG Patterns Analysis

**Author**: Max Van Belkum
**Date**: January 2026

Analysis of Microsoft GraphRAG, LightRAG, and MemGPT (Letta) for patterns applicable to Polymath's cross-domain scientific discovery mission.

---

## Executive Summary

| System | Key Innovation | Polymath Application |
|--------|---------------|---------------------|
| **GraphRAG** | Hierarchical Leiden communities | Mechanism-level clustering for cross-domain matching |
| **LightRAG** | Dual-level (entity vs relation) retrieval | Separate indices for "what methods?" vs "what papers mention X?" |
| **MemGPT** | Virtual memory paging with LLM summarization | Long-term agent state for multi-session research |

---

## 1. Microsoft GraphRAG: Hierarchical Community Detection

### Key Mechanisms

**Leiden Algorithm for Semantic Hierarchies**:
- Level-0: Tightly coupled concepts (e.g., "optimal transport" ↔ "Wasserstein distance")
- Level-1+: Meta-themes (e.g., "transport methods" ↔ "distribution matching")
- Resolution parameter controls granularity

**Global vs Local Search**:
- **Local**: Entity-centric, vector similarity → neighbors → chunks
- **Global**: Map-reduce on community summaries for corpus-level queries

**Community Summarization**:
- Each community generates LLM report at each hierarchy level
- 97% token reduction vs full text while maintaining quality

### Actionable for Polymath

```cypher
// Run Leiden separately on METHOD nodes
CALL gds.leiden.write('method-graph', {
    writeProperty: 'mechanismCommunity',
    resolution: 1.0
})

// Then on PROBLEM nodes
CALL gds.leiden.write('problem-graph', {
    writeProperty: 'problemCommunity',
    resolution: 0.8
})

// Cross-domain discovery: Methods in same mechanism community
// that solve problems in DIFFERENT domain communities
MATCH (m:METHOD)-[:SOLVES]->(p:PROBLEM)
WHERE m.mechanismCommunity = $target_mechanism
  AND p.problemCommunity <> $source_domain
RETURN m, p
```

**Implementation Priority**: HIGH
- Solves the "labels without mechanisms" problem
- Community membership becomes proxy for mechanism similarity
- Cross-domain = same mechanism community, different domain community

---

## 2. LightRAG: Dual-Level Retrieval

### Key Mechanisms

**Dual Keyword Extraction**:
- **High-Level (HL)**: Relationship-centric ("How does X relate to Y?")
- **Low-Level (LL)**: Entity-centric ("What properties does X have?")

**Separate Vector Indices**:
```
entities_vdb     → Fast entity search
relationships_vdb → Fast relation search
```

**Conditional Re-summarization**:
- Only regenerate entity summaries when new docs provide contradictory info
- 5-10x cost reduction on incremental ingestion

**Reverse Indices**:
- `entity → [chunk_ids]`: Which chunks mention this entity?
- `relation → [chunk_ids]`: Which chunks describe this relationship?

### Actionable for Polymath

```python
# Dual-index search
async def hybrid_search(query: str):
    # Extract both keyword types
    hl_keywords = extract_high_level(query)  # "methods for", "relationship between"
    ll_keywords = extract_low_level(query)   # "protein X", "gene Y"

    # Parallel retrieval
    method_results = await search_methods_index(hl_keywords)
    entity_results = await search_entities_index(ll_keywords)

    # Round-robin merge with dedup
    return interleave_results(method_results, entity_results)
```

**Implementation Priority**: MEDIUM
- Already have separate passages and concepts tables
- Need to split queries into "what method?" vs "what entity?" paths
- Estimated +30-40% improvement on relationship queries

---

## 3. MemGPT (Letta): Virtual Memory System

### Key Mechanisms

**Three-Tier Memory**:
| Tier | Purpose | Polymath Equivalent |
|------|---------|---------------------|
| **Core Memory** | In-context facts (~8KB) | Current hypothesis + active concepts |
| **Recall Memory** | Message history | Recent search/reasoning traces |
| **Archival Memory** | Long-term vectors | Paper passages + embeddings |

**Virtual Memory Paging**:
1. Detect context overflow (token counting)
2. Find semantic boundary (last assistant message)
3. Compress old messages via LLM summarization
4. Prepend summary to history

**Self-Editing Memory**:
- Agent explicitly calls `core_memory_append()` to remember
- Version tracking enables rollback
- Read-only blocks for immutable facts

### Actionable for Polymath

```python
class ResearchAgent:
    def __init__(self):
        self.core_memory = {
            "current_hypothesis": "",      # Mutable: agent updates this
            "active_concepts": [],         # Mutable: concepts being explored
            "paper_registry": "READ_ONLY"  # Immutable: ingested paper list
        }
        self.archival = VectorStore()      # Long-term paper passages
        self.recall = MessageHistory()     # Recent conversation

    async def on_context_overflow(self):
        """When context exceeds limit, compress old messages."""
        old_messages = self.recall.messages_before_cutoff()

        # LLM-generated summary preserving key discoveries
        summary = await self.llm.summarize(
            old_messages,
            prompt="Extract novel hypotheses and contradictions..."
        )

        # Store summary as structured insight, not just prose
        await self.neo4j.create_insight(summary)

        # Update recall with compressed history
        self.recall.replace_old_with_summary(summary)
```

**Implementation Priority**: LOW (for now)
- Requires building agent framework first
- Valuable for multi-session research continuity
- Key insight: Summaries should extract structured knowledge, not just reduce tokens

---

## 4. Patterns to Incorporate (Priority Order)

### Phase 1: Mechanism Clustering (GraphRAG)

**Goal**: Solve "labels without mechanisms" problem

```sql
-- Add mechanism_community to METHOD nodes
ALTER TABLE passage_concepts_v2
ADD COLUMN mechanism_community INTEGER;

-- Populate via Leiden on Neo4j, then sync back
```

**Validation**: Methods in same community should share data structures/algorithms

### Phase 2: Dual-Index Search (LightRAG)

**Goal**: Separate "what method?" from "what entity?" queries

```python
# In lib/search.py
def route_query(query: str):
    if is_method_query(query):  # "methods for", "approaches to"
        return search_methods_index(query)
    elif is_entity_query(query):  # protein names, gene symbols
        return search_entities_index(query)
    else:
        return hybrid_search(query)  # interleave both
```

### Phase 3: Conditional Summarization (LightRAG)

**Goal**: Reduce LLM cost on incremental ingestion

```python
async def merge_entity(existing, new_mentions):
    if len(new_mentions) < 2:
        # Don't re-summarize for minor additions
        existing.source_ids.extend(new_mentions)
        return existing

    # Only re-summarize if significant new information
    return await llm_summarize(existing.description, new_mentions)
```

### Phase 4: Agent Memory (MemGPT - Future)

**Goal**: Multi-session research continuity

- Track current hypothesis in "core memory"
- Archive discoveries as structured concepts (not prose summaries)
- Enable temporal queries: "What did I believe about X last week?"

---

## 5. Anti-Patterns to Avoid

### From GraphRAG
- **Avoid**: Full re-indexing on every document add
- **Solution**: LightRAG's incremental merge pattern

### From LightRAG
- **Avoid**: Storing all data in separate vector DBs without graph
- **Solution**: Keep Neo4j for relationship traversal, pgvector for similarity

### From MemGPT
- **Avoid**: Summarizing conversations into prose (loses structure)
- **Solution**: Extract structured concepts from summaries into Neo4j

---

## 6. Combined Architecture Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                      POLYMATH V3 (Future)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │ Entity Index│    │ Method Index│    │ Relation Idx│        │
│   │ (pgvector)  │    │ (pgvector)  │    │ (pgvector)  │        │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│          │                  │                  │                │
│          └──────────────────┼──────────────────┘                │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │  Query Router   │                          │
│                    │  (LightRAG)     │                          │
│                    └────────┬────────┘                          │
│                             │                                   │
│          ┌──────────────────┼──────────────────┐                │
│          ▼                  ▼                  ▼                │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │ Local Search│    │Global Search│    │Cross-Domain │        │
│   │ (Entity)    │    │ (Community) │    │ (Mechanism) │        │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│          │                  │                  │                │
│          └──────────────────┼──────────────────┘                │
│                             ▼                                   │
│                    ┌────────────────┐                           │
│                    │    Neo4j       │                           │
│                    │  (Leiden       │                           │
│                    │   Communities) │                           │
│                    └────────────────┘                           │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Agent Layer                          │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│   │  │ Core Mem │  │  Recall  │  │ Archival │              │  │
│   │  │(Hypothes)│  │ (Recent) │  │ (Papers) │              │  │
│   │  └──────────┘  └──────────┘  └──────────┘              │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- Microsoft GraphRAG: https://github.com/microsoft/graphrag
- LightRAG: https://github.com/HKUDS/LightRAG
- Letta (MemGPT): https://github.com/letta-ai/letta
- Zep: https://github.com/getzep/zep (temporal knowledge graphs - not analyzed)
