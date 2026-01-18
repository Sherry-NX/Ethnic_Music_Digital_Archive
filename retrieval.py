from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional, Tuple

NORMALIZE_FIELDS = [
    "primary_theme",
    "secondary_theme",
    "emotion",
    "tempo",
    "vocal_style",
    "performance_context",
]


def _normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return value.strip().lower()


def _parse_keywords(value: Optional[str]) -> List[str]:
    if not value:
        return []
    parts = value.split(",")
    tokens: List[str] = []
    for part in parts:
        token = part.strip().lower()
        if token:
            tokens.append(token)
    return tokens


def load_metadata(path: str) -> List[Dict[str, Any]]:
    """Load metadata from CSV and return list of records."""
    records: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    value = ""
                if isinstance(value, str):
                    value = value.strip()
                record[key] = value
            record["imagery_keywords_norm"] = _parse_keywords(
                record.get("imagery_keywords", "")
            )
            record["norm"] = {k: _normalize_text(record.get(k, "")) for k in NORMALIZE_FIELDS}
            records.append(record)
    return records


def normalize_query(q: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize query dict for matching."""
    if q is None:
        q = {}
    normalized: Dict[str, Any] = {}
    for key in NORMALIZE_FIELDS:
        raw = q.get(key, None)
        if raw is None:
            normalized[key] = None
            continue
        if isinstance(raw, str):
            value = raw.strip()
            normalized[key] = value.lower() if value else None
        else:
            normalized[key] = None

    raw_keywords = q.get("imagery_keywords", None)
    if raw_keywords is None:
        normalized["imagery_keywords"] = None
    elif isinstance(raw_keywords, str):
        tokens = _parse_keywords(raw_keywords)
        normalized["imagery_keywords"] = tokens if tokens else None
    else:
        tokens: List[str] = []
        for item in raw_keywords:
            if item is None:
                continue
            token = str(item).strip().lower()
            if token:
                tokens.append(token)
        normalized["imagery_keywords"] = tokens if tokens else None

    return normalized


def score_record(
    record: Dict[str, Any],
    query: Dict[str, Any],
    weights: Optional[Dict[str, int]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Score a record against a normalized query and return score + explanation."""
    default_weights = {
        "emotion": 3,
        "primary_theme": 2,
        "secondary_theme": 1,
        "tempo": 1,
        "vocal_style": 1,
        "performance_context": 1,
        "imagery_keywords": 1,
    }
    if weights:
        default_weights.update(weights)

    norm = record.get("norm", {})
    explanation: Dict[str, Any] = {}

    def get_record_value(field: str) -> str:
        value = norm.get(field, None)
        if value is None:
            return _normalize_text(record.get(field, ""))
        return value

    score = 0

    for field in ["emotion", "primary_theme", "secondary_theme", "tempo"]:
        query_value = query.get(field)
        record_value = get_record_value(field)
        points = 0
        matched = False
        if query_value is not None and record_value and query_value == record_value:
            points = default_weights[field]
            matched = True
        explanation[field] = {
            "points": points,
            "matched": matched,
            "value": query_value,
        }
        score += points

    for field in ["vocal_style", "performance_context"]:
        query_value = query.get(field)
        record_value = get_record_value(field)
        points = 0
        matched = False
        if query_value is not None and record_value and query_value in record_value:
            points = default_weights[field]
            matched = True
        explanation[field] = {
            "points": points,
            "matched": matched,
            "value": query_value,
        }
        score += points

    query_keywords = query.get("imagery_keywords") or []
    record_keywords = record.get("imagery_keywords_norm")
    if record_keywords is None:
        record_keywords = _parse_keywords(record.get("imagery_keywords", ""))
    matched_keywords = sorted(set(query_keywords) & set(record_keywords))
    keyword_points = min(len(matched_keywords), 3) * default_weights["imagery_keywords"]
    explanation["imagery_keywords"] = {
        "points": keyword_points,
        "matched_keywords": matched_keywords,
    }
    score += keyword_points

    explanation["final_score"] = score
    return score, explanation


CORE_FIELDS = ["emotion", "primary_theme", "secondary_theme"]


def compute_adaptive_min_score(query: Dict[str, Any]) -> int:
    constraint_fields = [
        "emotion",
        "primary_theme",
        "secondary_theme",
        "tempo",
        "vocal_style",
        "performance_context",
    ]
    constraints = 0
    for field in constraint_fields:
        value = query.get(field)
        if value is not None and str(value).strip():
            constraints += 1
    keywords = query.get("imagery_keywords") or []
    constraints += min(len(keywords), 3)
    if constraints >= 3:
        return 3
    if constraints == 2:
        return 2
    return 1


def record_matches_core(record: Dict[str, Any], query: Dict[str, Any]) -> bool:
    for field in CORE_FIELDS:
        query_value = query.get(field)
        if query_value is None:
            continue
        record_value = record.get("norm", {}).get(field, _normalize_text(record.get(field, "")))
        if record_value and record_value == query_value:
            return True
    return False


def retrieve(
    metadata: List[Dict[str, Any]],
    query: Dict[str, Any],
    top_k: int = 5,
    min_score: Optional[int] = 1,
) -> List[Dict[str, Any]]:
    """Score all records, filter by min_score and return ranked results."""
    normalized_query = normalize_query(query)
    if min_score is None:
        min_score = compute_adaptive_min_score(normalized_query)
    results: List[Dict[str, Any]] = []
    core_query_present = any(normalized_query.get(field) is not None for field in CORE_FIELDS)

    for record in metadata:
        core_match = record_matches_core(record, normalized_query)
        if core_query_present and not core_match:
            continue
        score, breakdown = score_record(record, normalized_query)
        if score < min_score:
            continue
        matched_fields = [
            field
            for field in [
                "emotion",
                "primary_theme",
                "secondary_theme",
                "tempo",
                "vocal_style",
                "performance_context",
            ]
            if breakdown.get(field, {}).get("points", 0) > 0
        ]
        matched_keywords = breakdown.get("imagery_keywords", {}).get("matched_keywords", [])
        item: Dict[str, Any] = {
            "id": record.get("id", ""),
            "score": score,
            "matched_explanation": breakdown,
            "core_match": core_match,
            "primary_theme": record.get("primary_theme", ""),
            "emotion": record.get("emotion", ""),
            "matched_fields": matched_fields,
            "matched_keywords": matched_keywords,
        }
        title = record.get("title", "")
        if title:
            item["title"] = title
        confidence = (record.get("confidence_level") or "").strip().lower()
        item["confidence_level"] = confidence
        results.append(item)

    confidence_rank = {"high": 3, "medium": 2, "low": 1, "": 0}

    def sort_key(item: Dict[str, Any]) -> Tuple[int, int, str]:
        confidence = item.get("confidence_level", "")
        return (-item["score"], -confidence_rank.get(confidence, 0), str(item.get("id", "")))

    results.sort(key=sort_key)
    return results[:top_k]
