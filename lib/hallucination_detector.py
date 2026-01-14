#!/usr/bin/env python3
"""
Hallucination Detection for RAG Memory Systems

Based on findings from:
- "HaluMem: Evaluating Hallucinations in Memory Systems of Agents" (2026)
  DOI: 10.48550/arxiv.2511.03506

Key insight: Memory systems hallucinate at THREE stages:
1. Extraction: Incorrect entity/relation extraction from source text
2. Updating: Contradictions when merging new information with existing memory
3. Question Answering: Fabricated answers not grounded in retrieved passages

This module provides validation at each stage to detect and prevent hallucination
propagation through the RAG pipeline.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from difflib import SequenceMatcher

from lib.config import config


@dataclass
class ValidationResult:
    """Result of hallucination validation."""
    is_valid: bool
    stage: str  # 'extraction', 'updating', 'qa'
    confidence: float
    issues: List[str]
    suggestions: List[str]


# =============================================================================
# Stage 1: Extraction Validation
# =============================================================================

def validate_extraction(
    extracted_entity: str,
    entity_type: str,
    source_text: str,
    source_span: str,
    similarity_threshold: float = 0.6
) -> ValidationResult:
    """
    Validate that an extracted entity is grounded in the source text.

    Checks:
    1. Source span exists in text (exact or fuzzy match)
    2. Entity name is derivable from source span
    3. Entity type is plausible given context

    Args:
        extracted_entity: The entity name extracted
        entity_type: The assigned entity type
        source_text: Full passage text
        source_span: Claimed source text for entity
        similarity_threshold: Minimum similarity for fuzzy matching

    Returns:
        ValidationResult with is_valid and issues
    """
    issues = []
    suggestions = []

    # Check 1: Source span exists in text
    if source_span not in source_text:
        # Try fuzzy matching
        best_match = None
        best_ratio = 0

        # Sliding window search
        words = source_text.split()
        span_words = len(source_span.split())

        for i in range(len(words) - span_words + 1):
            candidate = ' '.join(words[i:i + span_words])
            ratio = SequenceMatcher(None, source_span.lower(), candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate

        if best_ratio < similarity_threshold:
            issues.append(f"Source span not found in text: '{source_span}'")
            if best_match:
                suggestions.append(f"Did you mean: '{best_match}' (similarity: {best_ratio:.2f})?")
        else:
            suggestions.append(f"Fuzzy match found: '{best_match}' (similarity: {best_ratio:.2f})")

    # Check 2: Entity derivable from source span
    entity_lower = extracted_entity.lower()
    span_lower = source_span.lower()

    if entity_lower not in span_lower and span_lower not in entity_lower:
        # Check for abbreviation relationship
        if not _is_abbreviation(extracted_entity, source_span):
            issues.append(f"Entity '{extracted_entity}' not derivable from span '{source_span}'")

    # Check 3: Entity type plausibility (basic heuristics)
    if entity_type == 'GENE':
        if not re.match(r'^[A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]+)?$', extracted_entity):
            issues.append(f"'{extracted_entity}' doesn't match gene naming convention")

    if entity_type == 'METRIC':
        valid_metrics = {'PCC', 'SSIM', 'AUC', 'AUROC', 'R2', 'MAE', 'MSE', 'RMSE', 'F1'}
        if extracted_entity.upper() not in valid_metrics:
            # Check if it's a described metric
            if not any(m in extracted_entity.upper() for m in ['CORRELATION', 'ACCURACY', 'PRECISION', 'RECALL']):
                suggestions.append(f"Verify '{extracted_entity}' is a standard metric")

    is_valid = len(issues) == 0
    confidence = 1.0 - (len(issues) * 0.3)  # Deduct 0.3 for each issue

    return ValidationResult(
        is_valid=is_valid,
        stage='extraction',
        confidence=max(0.0, confidence),
        issues=issues,
        suggestions=suggestions
    )


def _is_abbreviation(short: str, long: str) -> bool:
    """Check if 'short' is a plausible abbreviation of 'long'."""
    if len(short) >= len(long):
        return False

    # Check if first letters of words match
    long_words = long.split()
    if len(short) == len(long_words):
        initials = ''.join(w[0].upper() for w in long_words if w)
        if short.upper() == initials:
            return True

    return False


# =============================================================================
# Stage 2: Update Validation (Contradiction Detection)
# =============================================================================

@dataclass
class MemoryEntry:
    """Represents an existing memory entry."""
    entity: str
    attribute: str
    value: str
    source_doc: str
    confidence: float


def validate_update(
    new_entry: MemoryEntry,
    existing_entries: List[MemoryEntry],
    contradiction_threshold: float = 0.8
) -> ValidationResult:
    """
    Validate that a new memory entry doesn't contradict existing knowledge.

    Checks:
    1. No direct contradictions with high-confidence entries
    2. Temporal consistency (if timestamps available)
    3. Source reliability comparison

    Args:
        new_entry: The entry to be added
        existing_entries: Existing entries for the same entity
        contradiction_threshold: Confidence threshold for contradiction detection

    Returns:
        ValidationResult with contradiction details
    """
    issues = []
    suggestions = []

    for existing in existing_entries:
        # Same entity, same attribute, different value = potential contradiction
        if (existing.entity.lower() == new_entry.entity.lower() and
            existing.attribute.lower() == new_entry.attribute.lower()):

            if existing.value.lower() != new_entry.value.lower():
                # Check if it's a true contradiction or just different granularity
                if _values_contradict(existing.value, new_entry.value):
                    if existing.confidence >= contradiction_threshold:
                        issues.append(
                            f"Contradiction: {existing.entity}.{existing.attribute} = "
                            f"'{existing.value}' (existing, conf={existing.confidence:.2f}) vs "
                            f"'{new_entry.value}' (new, conf={new_entry.confidence:.2f})"
                        )
                        suggestions.append(
                            f"Source comparison: existing from '{existing.source_doc}', "
                            f"new from '{new_entry.source_doc}'"
                        )
                else:
                    suggestions.append(
                        f"Non-contradicting update: '{existing.value}' → '{new_entry.value}' "
                        f"(refinement or different context)"
                    )

    is_valid = len(issues) == 0
    confidence = 1.0 - (len(issues) * 0.4)

    return ValidationResult(
        is_valid=is_valid,
        stage='updating',
        confidence=max(0.0, confidence),
        issues=issues,
        suggestions=suggestions
    )


def _values_contradict(value1: str, value2: str) -> bool:
    """
    Determine if two values are contradictory (not just different).

    Examples of non-contradictions:
    - "high" and "very high" (refinement)
    - "2023" and "2023-01" (more specific)
    - "liver cancer" and "hepatocellular carcinoma" (synonyms)
    """
    v1, v2 = value1.lower(), value2.lower()

    # Check for subset relationship (one contains the other)
    if v1 in v2 or v2 in v1:
        return False

    # Check for numeric contradiction
    nums1 = re.findall(r'\d+\.?\d*', v1)
    nums2 = re.findall(r'\d+\.?\d*', v2)

    if nums1 and nums2:
        # If both have numbers and they're very different, likely contradiction
        try:
            n1, n2 = float(nums1[0]), float(nums2[0])
            if abs(n1 - n2) / max(n1, n2, 1) > 0.5:  # >50% difference
                return True
        except (ValueError, IndexError):
            pass

    # Check for semantic opposition
    opposites = [
        ('high', 'low'), ('increase', 'decrease'), ('positive', 'negative'),
        ('yes', 'no'), ('true', 'false'), ('active', 'inactive'),
        ('expressed', 'not expressed'), ('significant', 'not significant')
    ]

    for pos, neg in opposites:
        if (pos in v1 and neg in v2) or (neg in v1 and pos in v2):
            return True

    return False


# =============================================================================
# Stage 3: QA Validation (Grounding Check)
# =============================================================================

def validate_answer(
    answer: str,
    retrieved_passages: List[str],
    question: str,
    grounding_threshold: float = 0.3
) -> ValidationResult:
    """
    Validate that an answer is grounded in retrieved passages.

    Checks:
    1. Key claims in answer have supporting evidence in passages
    2. No fabricated entities or relations
    3. Numerical values match source

    Args:
        answer: The generated answer
        retrieved_passages: Passages used to generate the answer
        question: The original question
        grounding_threshold: Minimum evidence overlap required

    Returns:
        ValidationResult with grounding analysis
    """
    issues = []
    suggestions = []

    # Concatenate passages for searching
    all_text = ' '.join(retrieved_passages).lower()

    # Extract claims from answer (simple heuristic: sentences with entities)
    claims = _extract_claims(answer)

    ungrounded_claims = []
    for claim in claims:
        # Check if claim keywords appear in passages
        claim_words = set(claim.lower().split())
        # Remove common words
        claim_words -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                       'could', 'should', 'may', 'might', 'must', 'and', 'or', 'but',
                       'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}

        if not claim_words:
            continue

        overlap = sum(1 for w in claim_words if w in all_text) / len(claim_words)

        if overlap < grounding_threshold:
            ungrounded_claims.append((claim, overlap))

    if ungrounded_claims:
        for claim, overlap in ungrounded_claims:
            issues.append(f"Ungrounded claim (overlap={overlap:.2f}): '{claim[:100]}...'")

    # Check for fabricated numbers
    answer_numbers = re.findall(r'\b\d+\.?\d*%?\b', answer)
    for num in answer_numbers:
        if num not in all_text:
            # Allow small variations
            base_num = re.sub(r'%', '', num)
            try:
                n = float(base_num)
                # Check if any similar number exists
                passage_numbers = re.findall(r'\b\d+\.?\d*%?\b', all_text)
                similar_found = any(
                    abs(float(re.sub(r'%', '', pn)) - n) / max(n, 1) < 0.1
                    for pn in passage_numbers
                    if re.sub(r'%', '', pn).replace('.', '').isdigit()
                )
                if not similar_found:
                    issues.append(f"Potentially fabricated number: {num}")
            except ValueError:
                pass

    # Suggestions for improving grounding
    if issues:
        suggestions.append("Consider rephrasing answer to more closely match passage language")
        suggestions.append("Add citations: [1], [2] etc. to indicate which passage supports each claim")

    is_valid = len(issues) == 0
    confidence = 1.0 - (len(issues) * 0.2)

    return ValidationResult(
        is_valid=is_valid,
        stage='qa',
        confidence=max(0.0, confidence),
        issues=issues,
        suggestions=suggestions
    )


def _extract_claims(text: str) -> List[str]:
    """Extract individual claims from text."""
    # Split by sentence
    sentences = re.split(r'[.!?]\s+', text)

    claims = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20:  # Skip very short sentences
            # Check if sentence makes a claim (has verb + object pattern)
            if re.search(r'\b(is|are|was|were|has|have|had|shows?|demonstrates?|indicates?|suggests?|found|discovered|reveals?)\b', sent, re.IGNORECASE):
                claims.append(sent)

    return claims


# =============================================================================
# Unified Validation Pipeline
# =============================================================================

class HallucinationDetector:
    """
    Unified hallucination detection across all RAG stages.

    Usage:
        detector = HallucinationDetector()

        # Validate extraction
        result = detector.validate_extraction(entity, type, source_text, span)

        # Validate update
        result = detector.validate_update(new_entry, existing_entries)

        # Validate QA
        result = detector.validate_answer(answer, passages, question)
    """

    def __init__(
        self,
        extraction_threshold: float = 0.6,
        contradiction_threshold: float = 0.8,
        grounding_threshold: float = 0.3
    ):
        self.extraction_threshold = extraction_threshold
        self.contradiction_threshold = contradiction_threshold
        self.grounding_threshold = grounding_threshold

    def validate_extraction(self, entity: str, entity_type: str,
                          source_text: str, source_span: str) -> ValidationResult:
        return validate_extraction(
            entity, entity_type, source_text, source_span,
            self.extraction_threshold
        )

    def validate_update(self, new_entry: MemoryEntry,
                       existing_entries: List[MemoryEntry]) -> ValidationResult:
        return validate_update(
            new_entry, existing_entries,
            self.contradiction_threshold
        )

    def validate_answer(self, answer: str, passages: List[str],
                       question: str) -> ValidationResult:
        return validate_answer(
            answer, passages, question,
            self.grounding_threshold
        )

    def full_pipeline_validation(
        self,
        extractions: List[Tuple[str, str, str, str]],  # [(entity, type, text, span), ...]
        memory_updates: List[Tuple[MemoryEntry, List[MemoryEntry]]],  # [(new, existing), ...]
        qa_pairs: List[Tuple[str, List[str], str]]  # [(answer, passages, question), ...]
    ) -> Dict[str, Any]:
        """
        Run full validation pipeline and aggregate results.

        Returns:
            {
                'extraction': {'valid': N, 'invalid': M, 'issues': [...]},
                'updating': {'valid': N, 'invalid': M, 'issues': [...]},
                'qa': {'valid': N, 'invalid': M, 'issues': [...]},
                'overall_score': 0.0-1.0
            }
        """
        results = {
            'extraction': {'valid': 0, 'invalid': 0, 'issues': []},
            'updating': {'valid': 0, 'invalid': 0, 'issues': []},
            'qa': {'valid': 0, 'invalid': 0, 'issues': []},
        }

        # Validate extractions
        for entity, etype, text, span in extractions:
            r = self.validate_extraction(entity, etype, text, span)
            if r.is_valid:
                results['extraction']['valid'] += 1
            else:
                results['extraction']['invalid'] += 1
                results['extraction']['issues'].extend(r.issues)

        # Validate updates
        for new_entry, existing in memory_updates:
            r = self.validate_update(new_entry, existing)
            if r.is_valid:
                results['updating']['valid'] += 1
            else:
                results['updating']['invalid'] += 1
                results['updating']['issues'].extend(r.issues)

        # Validate QA
        for answer, passages, question in qa_pairs:
            r = self.validate_answer(answer, passages, question)
            if r.is_valid:
                results['qa']['valid'] += 1
            else:
                results['qa']['invalid'] += 1
                results['qa']['issues'].extend(r.issues)

        # Calculate overall score
        total_valid = sum(r['valid'] for r in results.values())
        total_invalid = sum(r['invalid'] for r in results.values())
        total = total_valid + total_invalid

        results['overall_score'] = total_valid / total if total > 0 else 1.0

        return results


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HaluMem-Style Hallucination Detection Test")
    print("=" * 60)

    detector = HallucinationDetector()

    # Test extraction validation
    print("\n[Stage 1: Extraction Validation]")
    result = detector.validate_extraction(
        entity="EGFR",
        entity_type="GENE",
        source_text="We found that EGFR expression was elevated in tumor samples.",
        source_span="EGFR expression"
    )
    print(f"  Valid: {result.is_valid}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Issues: {result.issues}")

    # Test with hallucinated extraction
    result2 = detector.validate_extraction(
        entity="BRCA1",
        entity_type="GENE",
        source_text="We found that EGFR expression was elevated in tumor samples.",
        source_span="BRCA1 mutation"  # Not in text!
    )
    print(f"\n  [Hallucinated extraction]")
    print(f"  Valid: {result2.is_valid}")
    print(f"  Issues: {result2.issues}")

    # Test update validation
    print("\n[Stage 2: Update Validation]")
    existing = MemoryEntry(
        entity="Img2ST",
        attribute="performance",
        value="PCC = 0.85",
        source_doc="paper_a.pdf",
        confidence=0.9
    )
    new_entry = MemoryEntry(
        entity="Img2ST",
        attribute="performance",
        value="PCC = 0.42",  # Contradicts!
        source_doc="paper_b.pdf",
        confidence=0.8
    )
    result3 = detector.validate_update(new_entry, [existing])
    print(f"  Valid: {result3.is_valid}")
    print(f"  Issues: {result3.issues}")

    # Test QA validation
    print("\n[Stage 3: QA Validation]")
    passages = [
        "Img2ST achieves a Pearson correlation of 0.85 on the Visium HD dataset.",
        "The method uses a Vision Transformer encoder with optimal transport."
    ]
    answer = "Img2ST achieves 0.85 PCC using ViT and optimal transport on Visium HD."
    result4 = detector.validate_answer(answer, passages, "How does Img2ST perform?")
    print(f"  Valid: {result4.is_valid}")
    print(f"  Confidence: {result4.confidence:.2f}")

    # Test with fabricated answer
    fabricated = "Img2ST achieves 0.99 accuracy using a novel quantum computing approach."
    result5 = detector.validate_answer(fabricated, passages, "How does Img2ST perform?")
    print(f"\n  [Fabricated answer]")
    print(f"  Valid: {result5.is_valid}")
    print(f"  Issues: {result5.issues}")
