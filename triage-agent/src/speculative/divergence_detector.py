"""
Multi-Stage Clinical Invariant Divergence Detector.
Implements Innovation 1 from the OmniSpec-Triage Master Plan:
  - Stage 1: Hard Negation / Polarity Invariant (Forces 100% rollback on polarity flips)
  - Stage 2: Critical Anatomical & Emergency Keyword Invariant
  - Stage 3: High-Confidence Lexical Similarity (difflib SequenceMatcher + word overlap)
"""

import re
import difflib
from dataclasses import dataclass
from typing import Set

# Clinical negation and polarity tokens
NEGATION_TOKENS: Set[str] = {
    "no", "not", "never", "without", "denies", "stopped", "resolved",
    "barely", "neither", "nor", "none", "hardly", "negative", "zero"
}

# Critical anatomical and high-acuity emergency keywords
CRITICAL_ENTITIES: Set[str] = {
    "chest", "head", "breath", "breathing", "bleed", "bleeding", "blood",
    "stroke", "allergic", "allergy", "anaphylaxis", "seizure", "seizures",
    "paralysis", "unconscious", "collapse", "overdose", "airway"
}


@dataclass
class DivergenceResult:
    similarity: float
    diverged: bool
    partial_text: str
    final_text: str
    reason: str = "passed"
    polarity_shifted: bool = False
    entity_shifted: bool = False


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_tokens(norm_text: str) -> Set[str]:
    return set(norm_text.split())


def _sequence_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _word_overlap_similarity(words_a: Set[str], words_b: Set[str]) -> float:
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def compute_divergence(
    partial_text: str,
    final_text: str,
    threshold: float = 0.70,
    check_negation: bool = True,
    check_entities: bool = True,
) -> DivergenceResult:
    """
    Evaluates cross-modal speculative divergence through the 3-Stage Clinical Invariant Gate:

    Stage 1: Hard Polarity Invariant
             If negation tokens mutate between partial and final hypotheses,
             force instant rollback (diverged=True).
    Stage 2: Critical Entity Invariant
             If high-risk anatomical/emergency keywords mutate,
             force instant rollback (diverged=True).
    Stage 3: Lexical Similarity
             Combined difflib SequenceMatcher + Jaccard word overlap >= threshold.
    """
    norm_partial = _normalize(partial_text)
    norm_final = _normalize(final_text)

    words_partial = _extract_tokens(norm_partial)
    words_final = _extract_tokens(norm_final)

    seq_sim = _sequence_similarity(norm_partial, norm_final)
    word_sim = _word_overlap_similarity(words_partial, words_final)
    combined_similarity = (seq_sim + word_sim) / 2.0

    # Stage 1: Hard Negation / Polarity Invariant
    if check_negation:
        neg_partial = words_partial & NEGATION_TOKENS
        neg_final = words_final & NEGATION_TOKENS
        if neg_partial != neg_final:
            return DivergenceResult(
                similarity=combined_similarity,
                diverged=True,
                partial_text=partial_text,
                final_text=final_text,
                reason="polarity_mismatch",
                polarity_shifted=True,
                entity_shifted=False,
            )

    # Stage 2: Critical Anatomical & Emergency Entity Invariant
    if check_entities:
        crit_partial = words_partial & CRITICAL_ENTITIES
        crit_final = words_final & CRITICAL_ENTITIES
        if crit_partial != crit_final:
            return DivergenceResult(
                similarity=combined_similarity,
                diverged=True,
                partial_text=partial_text,
                final_text=final_text,
                reason="critical_entity_mismatch",
                polarity_shifted=False,
                entity_shifted=True,
            )

    # Stage 3: Lexical Similarity
    is_diverged = combined_similarity < threshold
    reason = "low_similarity" if is_diverged else "passed"

    return DivergenceResult(
        similarity=combined_similarity,
        diverged=is_diverged,
        partial_text=partial_text,
        final_text=final_text,
        reason=reason,
        polarity_shifted=False,
        entity_shifted=False,
    )

