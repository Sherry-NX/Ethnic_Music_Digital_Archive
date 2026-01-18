from __future__ import annotations

import tempfile
from typing import Any, Dict

from retrieval import load_metadata, normalize_query, retrieve, score_record


def _make_record(**kwargs: Any) -> Dict[str, Any]:
    return {
        "id": kwargs.get("id", ""),
        "primary_theme": kwargs.get("primary_theme", ""),
        "secondary_theme": kwargs.get("secondary_theme", ""),
        "emotion": kwargs.get("emotion", ""),
        "tempo": kwargs.get("tempo", ""),
        "vocal_style": kwargs.get("vocal_style", ""),
        "performance_context": kwargs.get("performance_context", ""),
        "imagery_keywords": kwargs.get("imagery_keywords", ""),
        "confidence_level": kwargs.get("confidence_level", ""),
    }


def test_keyword_parsing() -> None:
    content = (
        "id,primary_theme,secondary_theme,emotion,tempo,vocal_style,performance_context,"
        "imagery_keywords,confidence_level\n"
        "1,aging,,sad,slow,,ritual,\" Flower, mist , ,Rain \",high\n"
    )
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", newline="", delete=True) as f:
        f.write(content)
        f.flush()
        records = load_metadata(f.name)
    assert records[0]["imagery_keywords_norm"] == ["flower", "mist", "rain"]


def test_scoring_exact_and_keywords() -> None:
    record = _make_record(
        id="1",
        emotion="sad",
        primary_theme="aging",
        imagery_keywords="time, regret, wind",
    )
    query = normalize_query(
        {"emotion": "sad", "primary_theme": "aging", "imagery_keywords": ["time", "regret"]}
    )
    score, breakdown = score_record(record, query)
    assert score == 7
    assert breakdown["imagery_keywords"]["matched_keywords"] == ["regret", "time"]


def test_sorting_tiebreaks() -> None:
    records = [
        _make_record(id="2", emotion="sad", confidence_level="medium"),
        _make_record(id="1", emotion="sad", confidence_level="high"),
        _make_record(id="3", emotion="sad", confidence_level=""),
    ]
    results = retrieve(records, {"emotion": "sad"}, top_k=3, min_score=1)
    ids = [result["id"] for result in results]
    assert ids == ["1", "2", "3"]
