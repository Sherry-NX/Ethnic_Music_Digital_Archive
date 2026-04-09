"""Item-to-item metadata similarity engine.

Computes similarity between archive tracks based on structured metadata
fields. Designed for a small catalogue (tens of tracks) where precomputation
is unnecessary.

Weight tiers (structured-field-first):
  HIGH   - primary_theme, emotion, usage_scene, ethnic_group
  MEDIUM - secondary_theme, secondary_emotion, performance_form, region,
           subgroup, language
  LOW    - imagery_keywords (token overlap)
  AUX    - title, retrieval_notes (very low, tiebreaker only)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from search_pipeline import (
    normalize_text,
    split_keywords,
)

# ------------------------------------------------------------------
# Weight configuration
# ------------------------------------------------------------------

# Fields compared via exact match on normalised canonical values.
EXACT_MATCH_WEIGHTS: Dict[str, float] = {
    # High weight
    "primary_theme": 6.0,
    "emotion": 6.0,
    "usage_scene": 5.0,
    "ethnic_group": 5.0,
    # Medium weight
    "secondary_theme": 3.0,
    "secondary_emotion": 3.0,
    "performance_form": 3.0,
    "region": 2.5,
    "subgroup": 2.5,
    "language": 2.0,
    # Auxiliary (low weight, categorical but less discriminative)
    "tempo": 1.0,
}

# Cross-field bonus: when one record's primary field matches the other's
# secondary field (or vice versa), award a partial credit.
CROSS_FIELD_PAIRS: List[Tuple[str, str, float]] = [
    ("primary_theme", "secondary_theme", 2.0),
    ("emotion", "secondary_emotion", 2.0),
]

# Keyword overlap (imagery_keywords): per-keyword hit weight.
KEYWORD_HIT_WEIGHT: float = 1.0
KEYWORD_MAX_HITS: int = 5

# Auxiliary free-text fields: token overlap scored at very low weight.
AUX_TEXT_FIELDS: Dict[str, float] = {
    "title": 0.3,
    "retrieval_notes": 0.2,
}

# Minimum similarity score to consider a recommendation useful.
DEFAULT_MIN_SIMILARITY: float = 4.0

# Default number of similar tracks to return.
DEFAULT_TOP_K: int = 3


# ------------------------------------------------------------------
# Core similarity computation
# ------------------------------------------------------------------

def compute_similarity(record_a: Dict[str, Any], record_b: Dict[str, Any]) -> float:
    """Return a similarity score between two metadata records.

    Higher is more similar.  The score is not normalised to [0,1] --
    it is an unbounded weighted sum designed for ranking.
    """
    score = 0.0

    # --- Exact-match categorical fields ---
    for field, weight in EXACT_MATCH_WEIGHTS.items():
        val_a = normalize_text(record_a.get(field, ""))
        val_b = normalize_text(record_b.get(field, ""))
        if val_a and val_b and val_a == val_b:
            score += weight

    # --- Cross-field partial credit ---
    for field_primary, field_secondary, weight in CROSS_FIELD_PAIRS:
        a_pri = normalize_text(record_a.get(field_primary, ""))
        b_sec = normalize_text(record_b.get(field_secondary, ""))
        a_sec = normalize_text(record_a.get(field_secondary, ""))
        b_pri = normalize_text(record_b.get(field_primary, ""))
        if a_pri and b_sec and a_pri == b_sec:
            score += weight
        if a_sec and b_pri and a_sec == b_pri:
            score += weight

    # --- Imagery keyword overlap ---
    kw_a: Set[str] = set(split_keywords(record_a.get("imagery_keywords", "")))
    kw_b: Set[str] = set(split_keywords(record_b.get("imagery_keywords", "")))
    overlap = kw_a & kw_b
    score += min(len(overlap), KEYWORD_MAX_HITS) * KEYWORD_HIT_WEIGHT

    # --- Auxiliary free-text token overlap (very low weight) ---
    for field, weight in AUX_TEXT_FIELDS.items():
        tokens_a = set(normalize_text(record_a.get(field, "")).split())
        tokens_b = set(normalize_text(record_b.get(field, "")).split())
        common = tokens_a & tokens_b - {"", "the", "a", "an", "of", "and", "in", "for", "to"}
        score += min(len(common), 3) * weight

    return round(score, 2)


# ------------------------------------------------------------------
# Top-k similar track finder
# ------------------------------------------------------------------

def find_similar(
    target_id: str,
    all_records: List[Dict[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    exclude_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Return up to *top_k* records most similar to the target.

    Each returned dict contains:
      - "record": the full metadata record
      - "similarity_score": float
      - "explanation": English explanation string
    """
    exclude: Set[str] = set(exclude_ids or [])
    exclude.add(target_id)

    target = None
    for rec in all_records:
        rid = rec.get("track_id") or rec.get("id") or ""
        if rid == target_id:
            target = rec
            break
    if target is None:
        return []

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for rec in all_records:
        rid = rec.get("track_id") or rec.get("id") or ""
        if rid in exclude:
            continue
        sim = compute_similarity(target, rec)
        if sim >= min_similarity:
            scored.append((sim, rec))

    scored.sort(key=lambda x: (-x[0], x[1].get("track_id", "")))
    results: List[Dict[str, Any]] = []
    for sim_score, rec in scored[:top_k]:
        results.append({
            "record": rec,
            "similarity_score": sim_score,
            "explanation": explain_similarity(target, rec),
        })
    return results


# ------------------------------------------------------------------
# Explanation generator
# ------------------------------------------------------------------

_FIELD_LABELS: Dict[str, str] = {
    "primary_theme": "theme",
    "secondary_theme": "theme",
    "emotion": "emotion",
    "secondary_emotion": "emotion",
    "usage_scene": "usage scene",
    "ethnic_group": "ethnic group",
    "performance_form": "performance form",
    "region": "region",
    "subgroup": "subgroup",
    "language": "language",
    "tempo": "tempo",
}


def explain_similarity(record_a: Dict[str, Any], record_b: Dict[str, Any]) -> str:
    """Return a short English explanation of why two tracks are similar."""
    shared_labels: List[str] = []
    seen: Set[str] = set()

    # Check exact-match fields.
    for field in EXACT_MATCH_WEIGHTS:
        val_a = normalize_text(record_a.get(field, ""))
        val_b = normalize_text(record_b.get(field, ""))
        if val_a and val_b and val_a == val_b:
            label = _FIELD_LABELS.get(field, field)
            if label not in seen:
                seen.add(label)
                shared_labels.append(f"{label} ({val_a.replace('_', ' ')})")

    # Check keyword overlap.
    kw_a = set(split_keywords(record_a.get("imagery_keywords", "")))
    kw_b = set(split_keywords(record_b.get("imagery_keywords", "")))
    overlap = kw_a & kw_b
    if overlap:
        sample = ", ".join(sorted(overlap)[:3])
        shared_labels.append(f"imagery ({sample})")

    if not shared_labels:
        return "Weak thematic overlap."

    return "Shared: " + "; ".join(shared_labels[:4]) + "."
