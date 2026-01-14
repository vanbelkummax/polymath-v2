#!/usr/bin/env python3
"""
Domain-Specific Entity Extraction for Spatial Transcriptomics

Based on findings from:
- "GraphRAG on Technical Documents - Impact of Knowledge Graph Schema" (DOI: 10.4230/TGDK.3.2.3)

Key insight: Domain-expert schemas extract 10% more relevant entities than
generic auto-generated schemas. Simple 5-class schemas outperform complex
auto-schemas on technical documents.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

from lib.config import config


@dataclass
class Entity:
    """Extracted entity with type and provenance."""
    name: str
    entity_type: str  # One of SPATIAL_TX_ENTITY_TYPES
    confidence: float
    source_text: str  # Exact text span from document
    char_start: int
    char_end: int
    aliases: List[str] = field(default_factory=list)


@dataclass
class Relation:
    """Extracted relation between entities."""
    source_entity: str
    relation_type: str  # One of SPATIAL_TX_RELATION_TYPES
    target_entity: str
    confidence: float
    evidence_text: str  # Sentence containing the relation


@dataclass
class ExtractionResult:
    """Complete extraction result for a passage."""
    passage_id: str
    entities: List[Entity]
    relations: List[Relation]
    extractor_version: str = "v2.0-domain-schema"


# =============================================================================
# Pattern-Based Extraction (Fast, High Precision)
# =============================================================================

# Compiled patterns for fast matching
GENE_PATTERN = re.compile(
    r'\b([A-Z][A-Z0-9]{1,5}(?:-[A-Z0-9]{1,3})?)\b'  # EGFR, TP53, BRCA1, HLA-A
)

DATASET_PATTERNS = {
    'Visium': re.compile(r'\b(Visium(?:\s+HD)?)\b', re.IGNORECASE),
    '10x': re.compile(r'\b(10[xX]\s+(?:Xenium|Visium|Genomics))\b'),
    'MERSCOPE': re.compile(r'\b(MERSCOPE|MERFISH)\b', re.IGNORECASE),
    'CosMx': re.compile(r'\b(CosMx|NanoString)\b', re.IGNORECASE),
    'Slide-seq': re.compile(r'\b(Slide-?seq(?:\s+V\d)?)\b', re.IGNORECASE),
    'ST': re.compile(r'\b(spatial\s+transcriptomics?\s+(?:data(?:set)?|platform))\b', re.IGNORECASE),
}

METHOD_PATTERNS = {
    'Img2ST': re.compile(r'\b(Img2ST|Image2ST)\b', re.IGNORECASE),
    'HisToGene': re.compile(r'\b(HisToGene)\b'),
    'TESLA': re.compile(r'\b(TESLA)\b'),
    'Tangram': re.compile(r'\b(Tangram)\b'),
    'SpaGCN': re.compile(r'\b(SpaGCN)\b'),
    'STAGATE': re.compile(r'\b(STAGATE)\b'),
    'STNet': re.compile(r'\b(STNet|ST-Net)\b'),
    'iSCALE': re.compile(r'\b(iSCALE)\b'),
}

ALGORITHM_PATTERNS = {
    'optimal_transport': re.compile(r'\b(optimal\s+transport|Wasserstein|Sinkhorn)\b', re.IGNORECASE),
    'GNN': re.compile(r'\b(graph\s+neural\s+network|GNN|GCN|GAT|GraphSAGE)\b', re.IGNORECASE),
    'transformer': re.compile(r'\b(transformer|attention\s+mechanism|self-attention|ViT)\b', re.IGNORECASE),
    'diffusion': re.compile(r'\b(diffusion\s+model|DDPM|score\s+matching)\b', re.IGNORECASE),
    'VAE': re.compile(r'\b(VAE|variational\s+autoencoder)\b', re.IGNORECASE),
    'CNN': re.compile(r'\b(CNN|convolutional\s+neural\s+network|ResNet|UNet)\b', re.IGNORECASE),
}

LOSS_PATTERNS = {
    'MSE': re.compile(r'\b(MSE|mean\s+squared?\s+error)\b', re.IGNORECASE),
    'Poisson': re.compile(r'\b(Poisson\s+(?:NLL|loss|likelihood))\b', re.IGNORECASE),
    'ZINB': re.compile(r'\b(ZINB|zero-inflated\s+negative\s+binomial)\b', re.IGNORECASE),
    'CrossEntropy': re.compile(r'\b(cross[\s-]?entropy|CE\s+loss)\b', re.IGNORECASE),
    'NLL': re.compile(r'\b(NLL|negative\s+log[\s-]?likelihood)\b', re.IGNORECASE),
}

METRIC_PATTERNS = {
    'PCC': re.compile(r'\b(PCC|Pearson(?:\s+correlation)?)\b', re.IGNORECASE),
    'SSIM': re.compile(r'\b(SSIM|structural\s+similarity)\b', re.IGNORECASE),
    'AUC': re.compile(r'\b(AUC(?:-ROC)?|AUROC)\b', re.IGNORECASE),
    'R2': re.compile(r'\b(R[²2]|R-squared|coefficient\s+of\s+determination)\b', re.IGNORECASE),
    'MAE': re.compile(r'\b(MAE|mean\s+absolute\s+error)\b', re.IGNORECASE),
}

CELL_TYPE_PATTERNS = {
    # Immune cells
    'T-cell': re.compile(r'\b(T[\s-]?cells?|CD[48]\+?\s*T[\s-]?cells?)\b', re.IGNORECASE),
    'B-cell': re.compile(r'\b(B[\s-]?cells?|CD19\+?\s*cells?)\b', re.IGNORECASE),
    'macrophage': re.compile(r'\b(macrophages?|M[12]\s+macrophages?)\b', re.IGNORECASE),
    'fibroblast': re.compile(r'\b(fibroblasts?|CAF|cancer[\s-]?associated\s+fibroblasts?)\b', re.IGNORECASE),
    'epithelial': re.compile(r'\b(epithelial\s+cells?|enterocytes?)\b', re.IGNORECASE),
    'endothelial': re.compile(r'\b(endothelial\s+cells?)\b', re.IGNORECASE),
    'neutrophil': re.compile(r'\b(neutrophils?)\b', re.IGNORECASE),
    'dendritic': re.compile(r'\b(dendritic\s+cells?|DCs?)\b', re.IGNORECASE),
}

TISSUE_PATTERNS = {
    'colon': re.compile(r'\b(colon(?:ic)?|colorectal|CRC)\b', re.IGNORECASE),
    'liver': re.compile(r'\b(liver|hepatic|hepatocellular)\b', re.IGNORECASE),
    'brain': re.compile(r'\b(brain|cerebral|cortex|hippocampus)\b', re.IGNORECASE),
    'lung': re.compile(r'\b(lung|pulmonary)\b', re.IGNORECASE),
    'breast': re.compile(r'\b(breast|mammary)\b', re.IGNORECASE),
    'kidney': re.compile(r'\b(kidney|renal)\b', re.IGNORECASE),
    'skin': re.compile(r'\b(skin|dermal|cutaneous|melanoma)\b', re.IGNORECASE),
}


def extract_entities_pattern(text: str, passage_id: str = "") -> List[Entity]:
    """
    Extract entities using domain-specific patterns.

    This is the fast path - high precision, lower recall.
    Use LLM extraction for comprehensive coverage.
    """
    entities = []

    # Extract datasets
    for name, pattern in DATASET_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(Entity(
                name=name,
                entity_type='DATASET',
                confidence=0.95,
                source_text=match.group(0),
                char_start=match.start(),
                char_end=match.end()
            ))

    # Extract methods
    for name, pattern in METHOD_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(Entity(
                name=name,
                entity_type='METHOD',
                confidence=0.95,
                source_text=match.group(0),
                char_start=match.start(),
                char_end=match.end()
            ))

    # Extract algorithms
    for name, pattern in ALGORITHM_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(Entity(
                name=name,
                entity_type='ALGORITHM',
                confidence=0.9,
                source_text=match.group(0),
                char_start=match.start(),
                char_end=match.end()
            ))

    # Extract loss functions
    for name, pattern in LOSS_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(Entity(
                name=name,
                entity_type='LOSS_FUNCTION',
                confidence=0.9,
                source_text=match.group(0),
                char_start=match.start(),
                char_end=match.end()
            ))

    # Extract metrics
    for name, pattern in METRIC_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(Entity(
                name=name,
                entity_type='METRIC',
                confidence=0.9,
                source_text=match.group(0),
                char_start=match.start(),
                char_end=match.end()
            ))

    # Extract cell types
    for name, pattern in CELL_TYPE_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(Entity(
                name=name,
                entity_type='CELL_TYPE',
                confidence=0.9,
                source_text=match.group(0),
                char_start=match.start(),
                char_end=match.end()
            ))

    # Extract tissues
    for name, pattern in TISSUE_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(Entity(
                name=name,
                entity_type='TISSUE',
                confidence=0.85,
                source_text=match.group(0),
                char_start=match.start(),
                char_end=match.end()
            ))

    # Deduplicate by (name, type) keeping highest confidence
    seen = {}
    for entity in entities:
        key = (entity.name.lower(), entity.entity_type)
        if key not in seen or entity.confidence > seen[key].confidence:
            seen[key] = entity

    return list(seen.values())


# =============================================================================
# LLM-Based Extraction (Comprehensive, uses Gemini)
# =============================================================================

EXTRACTION_PROMPT = """Extract entities and relations from this scientific text using ONLY these types:

ENTITY TYPES:
- METHOD: Specific methods/models (e.g., Img2ST, TESLA, Tangram)
- DATASET: Data sources (e.g., Visium HD, 10x Xenium)
- CELL_TYPE: Cell types (e.g., T-cell, macrophage)
- GENE: Gene names (e.g., EGFR, TP53)
- TISSUE: Tissue/organ types (e.g., colon, liver)
- ALGORITHM: Algorithms/techniques (e.g., optimal transport, GNN)
- LOSS_FUNCTION: Loss functions (e.g., MSE, Poisson NLL)
- DATA_STRUCTURE: Data structures (e.g., point cloud, graph)
- METRIC: Evaluation metrics (e.g., PCC, SSIM, AUC)
- MECHANISM: Mechanisms (e.g., attention, convolution)

RELATION TYPES:
- APPLIES_TO: method → dataset
- PREDICTS: method → target
- OUTPERFORMS: method → method
- REQUIRES: method → requirement
- TRAINED_ON: method → dataset
- OPERATES_ON: algorithm → data_structure
- EXPRESSES: cell_type → gene
- FOUND_IN: cell_type → tissue
- USES_LOSS: method → loss_function
- IMPLEMENTS: method → mechanism

TEXT:
{text}

Return JSON with this exact structure:
{{
  "entities": [
    {{"name": "...", "type": "...", "confidence": 0.0-1.0, "source_span": "exact text"}}
  ],
  "relations": [
    {{"source": "...", "relation": "...", "target": "...", "confidence": 0.0-1.0, "evidence": "sentence"}}
  ]
}}

IMPORTANT:
- Only extract entities that CLEARLY match the defined types
- Include the exact source text span for each entity
- Confidence should reflect how certain you are about the extraction
- Skip generic terms like "method" or "data" unless they refer to specific named entities
"""


async def extract_entities_llm(
    text: str,
    passage_id: str = "",
    model: str = "gemini-1.5-flash"
) -> ExtractionResult:
    """
    Extract entities using LLM with domain-specific schema.

    Uses Gemini for comprehensive extraction with schema guidance.
    Falls back to pattern extraction if LLM fails.
    """
    import google.generativeai as genai

    if not config.GEMINI_API_KEY:
        # Fallback to pattern extraction
        entities = extract_entities_pattern(text, passage_id)
        return ExtractionResult(
            passage_id=passage_id,
            entities=entities,
            relations=[],
            extractor_version="v2.0-pattern-only"
        )

    genai.configure(api_key=config.GEMINI_API_KEY)
    model_instance = genai.GenerativeModel(model)

    try:
        response = await model_instance.generate_content_async(
            EXTRACTION_PROMPT.format(text=text[:8000]),  # Truncate for context window
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1  # Low temperature for consistent extraction
            )
        )

        result = json.loads(response.text)

        entities = []
        for e in result.get('entities', []):
            if e.get('confidence', 0) >= config.ENTITY_CONFIDENCE_THRESHOLD:
                entities.append(Entity(
                    name=e['name'],
                    entity_type=e['type'],
                    confidence=e['confidence'],
                    source_text=e.get('source_span', e['name']),
                    char_start=text.find(e.get('source_span', e['name'])),
                    char_end=text.find(e.get('source_span', e['name'])) + len(e.get('source_span', e['name']))
                ))

        relations = []
        for r in result.get('relations', []):
            if r.get('confidence', 0) >= config.RELATION_CONFIDENCE_THRESHOLD:
                relations.append(Relation(
                    source_entity=r['source'],
                    relation_type=r['relation'],
                    target_entity=r['target'],
                    confidence=r['confidence'],
                    evidence_text=r.get('evidence', '')
                ))

        return ExtractionResult(
            passage_id=passage_id,
            entities=entities,
            relations=relations,
            extractor_version="v2.0-gemini-domain-schema"
        )

    except Exception as e:
        print(f"LLM extraction failed: {e}, falling back to patterns")
        entities = extract_entities_pattern(text, passage_id)
        return ExtractionResult(
            passage_id=passage_id,
            entities=entities,
            relations=[],
            extractor_version="v2.0-pattern-fallback"
        )


# =============================================================================
# Batch API Functions (for hydrate_graph_batch.py)
# =============================================================================

def prepare_batch_request(text: str, passage_id: str) -> Dict[str, Any]:
    """
    Generate a single JSONL request object for the Gemini Batch API.

    The Batch API expects JSONL files where each line is a request object.
    This format matches the Gemini Batch API specification.

    Args:
        text: The passage text to extract entities from
        passage_id: Unique identifier for the passage

    Returns:
        Dict matching Gemini Batch API input format
    """
    prompt = EXTRACTION_PROMPT.format(text=text[:8000])

    return {
        "custom_id": passage_id,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1
            }
        }
    }


def parse_batch_result(
    batch_line: Dict[str, Any],
    original_text: str
) -> Optional[ExtractionResult]:
    """
    Parse a single line from the completed Batch output file.

    Args:
        batch_line: A single parsed JSON line from batch output
        original_text: The original passage text (needed to recalculate char indices)

    Returns:
        ExtractionResult or None if parsing fails
    """
    try:
        passage_id = batch_line.get('custom_id', '')

        # Handle different response formats
        response = batch_line.get('response', {})

        # Check for error
        if 'error' in response:
            print(f"Batch error for {passage_id}: {response['error']}")
            return None

        # Extract the generated content
        body = response.get('body', response)
        candidates = body.get('candidates', [])

        if not candidates:
            print(f"No candidates in response for {passage_id}")
            return None

        # Get the JSON text from the response
        content = candidates[0].get('content', {})
        parts = content.get('parts', [])

        if not parts:
            print(f"No parts in response for {passage_id}")
            return None

        json_text = parts[0].get('text', '')

        # Parse the JSON response
        result = json.loads(json_text)

        # Convert to Entity objects
        entities = []
        for e in result.get('entities', []):
            confidence = float(e.get('confidence', 0))
            if confidence < config.ENTITY_CONFIDENCE_THRESHOLD:
                continue

            source_span = e.get('source_span', e.get('name', ''))

            # Find position in original text
            char_start = original_text.find(source_span) if source_span else -1
            char_end = char_start + len(source_span) if char_start >= 0 else -1

            entities.append(Entity(
                name=e['name'],
                entity_type=e.get('type', 'UNKNOWN'),
                confidence=confidence,
                source_text=source_span,
                char_start=char_start,
                char_end=char_end
            ))

        # Convert to Relation objects
        relations = []
        for r in result.get('relations', []):
            confidence = float(r.get('confidence', 0))
            if confidence < config.RELATION_CONFIDENCE_THRESHOLD:
                continue

            relations.append(Relation(
                source_entity=r.get('source', ''),
                relation_type=r.get('relation', ''),
                target_entity=r.get('target', ''),
                confidence=confidence,
                evidence_text=r.get('evidence', '')
            ))

        return ExtractionResult(
            passage_id=passage_id,
            entities=entities,
            relations=relations,
            extractor_version="v2.0-gemini-batch"
        )

    except json.JSONDecodeError as e:
        print(f"JSON decode error for {batch_line.get('custom_id', 'unknown')}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing batch result: {e}")
        return None


def prepare_batch_file(
    passages: List[Dict[str, Any]],
    output_path: Path
) -> int:
    """
    Write a JSONL file of batch requests for multiple passages.

    Args:
        passages: List of dicts with 'passage_id' and 'passage_text' keys
        output_path: Path to write the JSONL file

    Returns:
        Number of requests written
    """
    count = 0
    with open(output_path, 'w') as f:
        for passage in passages:
            passage_id = str(passage.get('passage_id', ''))
            text = passage.get('passage_text', '')

            if not text or len(text) < 100:
                continue

            request = prepare_batch_request(text, passage_id)
            f.write(json.dumps(request) + '\n')
            count += 1

    return count


def parse_batch_file(
    result_path: Path,
    text_lookup: Dict[str, str]
) -> List[ExtractionResult]:
    """
    Parse a completed batch result JSONL file.

    Args:
        result_path: Path to the batch output JSONL file
        text_lookup: Dict mapping passage_id -> original passage text

    Returns:
        List of ExtractionResult objects
    """
    results = []

    with open(result_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            batch_line = json.loads(line)
            passage_id = batch_line.get('custom_id', '')
            original_text = text_lookup.get(passage_id, '')

            result = parse_batch_result(batch_line, original_text)
            if result:
                results.append(result)

    return results


def extract_entities_sync(text: str, passage_id: str = "") -> ExtractionResult:
    """Synchronous wrapper for entity extraction."""
    import asyncio

    # Try LLM first if available
    if config.GEMINI_API_KEY:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(extract_entities_llm(text, passage_id))

    # Pattern-only fallback
    entities = extract_entities_pattern(text, passage_id)
    return ExtractionResult(
        passage_id=passage_id,
        entities=entities,
        relations=[],
        extractor_version="v2.0-pattern-only"
    )


# =============================================================================
# Neo4j Storage
# =============================================================================

def store_extraction_to_neo4j(driver, extraction: ExtractionResult, doc_id: str):
    """
    Store extracted entities and relations to Neo4j.

    Creates entity nodes with domain-specific labels (METHOD, GENE, etc.)
    and links them to Paper nodes via MENTIONED_IN relationships.

    Note: Paper nodes are created by ingest_v2.py. If the Paper doesn't exist,
    we create a stub node to allow entity storage (it will be enriched later).
    """
    with driver.session() as session:
        # Ensure Paper node exists (MERGE creates if missing)
        session.run("""
            MERGE (p:Paper {doc_id: $doc_id})
            ON CREATE SET p.source = 'hydrate_graph', p.created_at = datetime()
        """, {'doc_id': doc_id})

        # Create entity nodes with their types as labels
        # Note: Using parameterized label requires APOC or separate queries per type
        for entity in extraction.entities:
            # Validate entity type is in our schema
            if entity.entity_type not in config.SPATIAL_TX_ENTITY_TYPES:
                continue

            # Use APOC for dynamic labels if available, otherwise use CASE
            try:
                # Try APOC first (preferred)
                session.run("""
                    CALL apoc.merge.node([$label], {name: $name}, {
                        confidence: $confidence,
                        last_seen: datetime()
                    }, {}) YIELD node
                    WITH node
                    MATCH (p:Paper {doc_id: $doc_id})
                    MERGE (node)-[r:MENTIONED_IN]->(p)
                    SET r.passage_id = $passage_id,
                        r.source_text = $source_text,
                        r.char_start = $char_start,
                        r.char_end = $char_end
                """, {
                    'label': entity.entity_type,
                    'name': entity.name,
                    'confidence': entity.confidence,
                    'doc_id': doc_id,
                    'passage_id': extraction.passage_id,
                    'source_text': entity.source_text,
                    'char_start': entity.char_start,
                    'char_end': entity.char_end
                })
            except Exception:
                # Fallback: Use static query per entity type (less elegant but works)
                _store_entity_static(session, entity, extraction, doc_id)

        # Create relations between entities
        for relation in extraction.relations:
            # Validate relation type
            if relation.relation_type not in config.SPATIAL_TX_RELATION_TYPES:
                continue

            try:
                session.run("""
                    CALL apoc.merge.relationship(
                        (MATCH (a {name: $source_name}) RETURN a),
                        $rel_type,
                        {passage_id: $passage_id},
                        {confidence: $confidence, evidence: $evidence},
                        (MATCH (b {name: $target_name}) RETURN b),
                        {}
                    ) YIELD rel
                    RETURN rel
                """, {
                    'source_name': relation.source_entity,
                    'target_name': relation.target_entity,
                    'rel_type': relation.relation_type,
                    'confidence': relation.confidence,
                    'evidence': relation.evidence_text,
                    'passage_id': extraction.passage_id
                })
            except Exception:
                # Fallback: static query
                _store_relation_static(session, relation, extraction)


def _store_entity_static(session, entity, extraction, doc_id: str):
    """Fallback for storing entities without APOC."""
    # Map entity types to specific queries (verbose but reliable)
    queries = {
        'METHOD': "MERGE (e:METHOD {name: $name})",
        'DATASET': "MERGE (e:DATASET {name: $name})",
        'CELL_TYPE': "MERGE (e:CELL_TYPE {name: $name})",
        'GENE': "MERGE (e:GENE {name: $name})",
        'TISSUE': "MERGE (e:TISSUE {name: $name})",
        'ALGORITHM': "MERGE (e:ALGORITHM {name: $name})",
        'LOSS_FUNCTION': "MERGE (e:LOSS_FUNCTION {name: $name})",
        'DATA_STRUCTURE': "MERGE (e:DATA_STRUCTURE {name: $name})",
        'METRIC': "MERGE (e:METRIC {name: $name})",
        'MECHANISM': "MERGE (e:MECHANISM {name: $name})",
    }

    merge_query = queries.get(entity.entity_type)
    if not merge_query:
        return

    full_query = f"""
        {merge_query}
        SET e.confidence = CASE
            WHEN e.confidence IS NULL OR $confidence > e.confidence
            THEN $confidence ELSE e.confidence END,
            e.last_seen = datetime()
        WITH e
        MATCH (p:Paper {{doc_id: $doc_id}})
        MERGE (e)-[r:MENTIONED_IN]->(p)
        SET r.passage_id = $passage_id,
            r.source_text = $source_text,
            r.char_start = $char_start,
            r.char_end = $char_end
    """

    session.run(full_query, {
        'name': entity.name,
        'confidence': entity.confidence,
        'doc_id': doc_id,
        'passage_id': extraction.passage_id,
        'source_text': entity.source_text,
        'char_start': entity.char_start,
        'char_end': entity.char_end
    })


def _store_relation_static(session, relation, extraction):
    """Fallback for storing relations without APOC."""
    queries = {
        'APPLIES_TO': "MERGE (s)-[r:APPLIES_TO]->(t)",
        'PREDICTS': "MERGE (s)-[r:PREDICTS]->(t)",
        'OUTPERFORMS': "MERGE (s)-[r:OUTPERFORMS]->(t)",
        'REQUIRES': "MERGE (s)-[r:REQUIRES]->(t)",
        'TRAINED_ON': "MERGE (s)-[r:TRAINED_ON]->(t)",
        'OPERATES_ON': "MERGE (s)-[r:OPERATES_ON]->(t)",
        'EXPRESSES': "MERGE (s)-[r:EXPRESSES]->(t)",
        'FOUND_IN': "MERGE (s)-[r:FOUND_IN]->(t)",
        'USES_LOSS': "MERGE (s)-[r:USES_LOSS]->(t)",
        'IMPLEMENTS': "MERGE (s)-[r:IMPLEMENTS]->(t)",
    }

    merge_query = queries.get(relation.relation_type)
    if not merge_query:
        return

    full_query = f"""
        MATCH (s {{name: $source_name}})
        MATCH (t {{name: $target_name}})
        {merge_query}
        SET r.confidence = CASE
            WHEN r.confidence IS NULL OR $confidence > r.confidence
            THEN $confidence ELSE r.confidence END,
            r.evidence = $evidence,
            r.passage_id = $passage_id
    """

    session.run(full_query, {
        'source_name': relation.source_entity,
        'target_name': relation.target_entity,
        'confidence': relation.confidence,
        'evidence': relation.evidence_text,
        'passage_id': extraction.passage_id
    })


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    test_text = """
    We present Img2ST, a method for predicting spatial transcriptomics from H&E images.
    Our approach uses a Vision Transformer (ViT) encoder with optimal transport for
    distribution matching. We train on Visium HD data from colorectal cancer samples
    and evaluate using Pearson correlation (PCC) and structural similarity (SSIM).

    The model predicts expression of key genes including EGFR and TP53 in T-cells and
    macrophages within the tumor microenvironment. We use Poisson NLL loss to handle
    the count nature of gene expression data.
    """

    print("=" * 60)
    print("Domain-Specific Entity Extraction Test")
    print("=" * 60)

    # Pattern extraction
    print("\n[Pattern Extraction]")
    entities = extract_entities_pattern(test_text)
    for e in entities:
        print(f"  {e.entity_type}: {e.name} (conf={e.confidence:.2f})")

    # Full extraction (sync)
    print("\n[Full Extraction (sync)]")
    result = extract_entities_sync(test_text, "test-passage-1")
    print(f"  Version: {result.extractor_version}")
    print(f"  Entities: {len(result.entities)}")
    print(f"  Relations: {len(result.relations)}")

    for e in result.entities:
        print(f"    {e.entity_type}: {e.name}")
    for r in result.relations:
        print(f"    {r.source_entity} --[{r.relation_type}]--> {r.target_entity}")
