from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple


CODEBASE_DIR = Path(__file__).resolve().parent
DEFAULT_MASTER_METADATA_PATH = CODEBASE_DIR / "data" / "metadata" / "metadata_master.csv"
TOKEN_SPLIT_RE = re.compile(r"[\s,;|/]+")

SEARCH_FIELDS = [
    "title",
    "ethnic_group",
    "subgroup",
    "region",
    "primary_theme",
    "secondary_theme",
    "emotion",
    "secondary_emotion",
    "usage_scene",
    "imagery_keywords",
    "performance_form",
    "tempo",
    "retrieval_notes",
]

FIELD_WEIGHTS = {
    "title": 4,
    "ethnic_group": 3,
    "subgroup": 2,
    "region": 2,
    "primary_theme": 3,
    "secondary_theme": 2,
    "emotion": 3,
    "secondary_emotion": 2,
    "usage_scene": 2,
    "imagery_keywords": 3,
    "performance_form": 2,
    "tempo": 1,
    "retrieval_notes": 1,
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _split_keywords(value: Any) -> List[str]:
    text = _normalize_text(value)
    if not text:
        return []
    parts = TOKEN_SPLIT_RE.split(text.replace("，", ",").replace("；", ";"))
    return [part for part in parts if part]


def _tokenize_query(text: str) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    raw_tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized)
    tokens: List[str] = []
    for token in raw_tokens:
        if token.isascii() and len(token) >= 2:
            tokens.append(token)
        elif not token.isascii() and len(token) >= 2:
            tokens.append(token)
    # Dedupe but preserve order.
    seen = set()
    result: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


class MetadataRepository(Protocol):
    def load_records(self) -> List[Dict[str, Any]]:
        ...


@dataclass
class CSVMetadataRepository:
    metadata_path: Path

    def load_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.metadata_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record: Dict[str, Any] = {}
                for key, value in row.items():
                    record[key] = value.strip() if isinstance(value, str) else (value or "")
                record["id"] = record.get("track_id") or record.get("id") or ""
                record["_norm"] = {field: _normalize_text(record.get(field, "")) for field in SEARCH_FIELDS}
                record["_imagery_tokens"] = _split_keywords(record.get("imagery_keywords", ""))
                records.append(record)
        return records


@dataclass
class MetadataSearchEngine:
    repository: MetadataRepository

    def load(self) -> List[Dict[str, Any]]:
        return self.repository.load_records()

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        structured_query: Optional[Dict[str, Any]] = None,
        min_score: int = 1,
    ) -> List[Dict[str, Any]]:
        records = self.load()
        normalized_query = _normalize_text(query_text)
        query_tokens = _tokenize_query(query_text)
        structured_query = structured_query or {}

        results: List[Dict[str, Any]] = []
        for record in records:
            score, reasons, matched_fields = _score_record(
                record=record,
                query_text=normalized_query,
                query_tokens=query_tokens,
                structured_query=structured_query,
            )
            if score < min_score:
                continue

            result = {
                "id": record.get("id", ""),
                "track_id": record.get("track_id", ""),
                "title": record.get("title", ""),
                "score": score,
                "match_reasons": reasons[:3],
                "matched_fields": matched_fields,
                "record": record,
                "audio_path": (record.get("audio_path") or "").strip(),
                "video_path": (record.get("video_path") or "").strip(),
            }
            results.append(result)

        results.sort(key=lambda x: (-int(x["score"]), str(x.get("id", ""))))
        return results[:top_k]


def _score_record(
    *,
    record: Dict[str, Any],
    query_text: str,
    query_tokens: Sequence[str],
    structured_query: Dict[str, Any],
) -> Tuple[int, List[str], List[str]]:
    norm = record.get("_norm", {})
    score = 0
    reasons: List[str] = []
    matched_fields: List[str] = []

    for field in SEARCH_FIELDS:
        value = str(norm.get(field, "") or "")
        if not value:
            continue

        field_matched = False
        field_weight = FIELD_WEIGHTS.get(field, 1)

        if query_text and query_text in value:
            score += field_weight * 2
            reasons.append(f"{field}: 命中完整短语")
            field_matched = True

        matched_tokens: List[str] = []
        for token in query_tokens:
            if token and token in value:
                matched_tokens.append(token)
        if matched_tokens:
            score += field_weight * len(matched_tokens)
            sample = ", ".join(matched_tokens[:3])
            reasons.append(f"{field}: 命中关键词 {sample}")
            field_matched = True

        query_field_value = structured_query.get(field)
        if isinstance(query_field_value, str):
            normalized_query_value = _normalize_text(query_field_value)
            if normalized_query_value and normalized_query_value in value:
                score += field_weight * 2
                reasons.append(f"{field}: 匹配结构化条件 {normalized_query_value}")
                field_matched = True
        elif isinstance(query_field_value, list):
            normalized_list = [_normalize_text(item) for item in query_field_value if _normalize_text(item)]
            list_hits = [item for item in normalized_list if item in value]
            if list_hits:
                score += field_weight * len(list_hits)
                reasons.append(f"{field}: 匹配结构化列表 {', '.join(list_hits[:3])}")
                field_matched = True

        if field_matched:
            matched_fields.append(field)

    # Make keyword overlap explicit for explainability.
    query_keyword_values = structured_query.get("imagery_keywords", [])
    if isinstance(query_keyword_values, list):
        record_keywords = record.get("_imagery_tokens", [])
        hits = [k for k in query_keyword_values if _normalize_text(k) in record_keywords]
        if hits:
            score += len(hits) * FIELD_WEIGHTS.get("imagery_keywords", 1)
            reasons.append(f"imagery_keywords: 关键词重合 {', '.join(hits[:3])}")
            if "imagery_keywords" not in matched_fields:
                matched_fields.append("imagery_keywords")

    return score, reasons, matched_fields


def resolve_metadata_path(path: Optional[str] = None) -> Path:
    if not path:
        return DEFAULT_MASTER_METADATA_PATH
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (CODEBASE_DIR / p).resolve()


def normalize_text(value: Any) -> str:
    """Public wrapper around internal text normalization."""
    return _normalize_text(value)


def tokenize_query(text: str) -> List[str]:
    """Public wrapper around internal query tokenization."""
    return _tokenize_query(text)


def split_keywords(value: Any) -> List[str]:
    """Public wrapper around internal keyword splitting."""
    return _split_keywords(value)


def create_search_engine(path: Optional[str] = None) -> MetadataSearchEngine:
    metadata_path = resolve_metadata_path(path)
    return MetadataSearchEngine(repository=CSVMetadataRepository(metadata_path=metadata_path))
