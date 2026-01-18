from __future__ import annotations

import sys
from typing import Any, Dict, List

from retrieval import load_metadata, retrieve


def _short_explanation(breakdown: Dict[str, Any]) -> str:
    fields = []
    for field in [
        "emotion",
        "primary_theme",
        "secondary_theme",
        "tempo",
        "vocal_style",
        "performance_context",
    ]:
        if breakdown.get(field, {}).get("points", 0) > 0:
            fields.append(field)
    keywords = breakdown.get("imagery_keywords", {}).get("matched_keywords", [])
    parts = []
    if fields:
        parts.append("matched_fields=" + ",".join(fields))
    if keywords:
        parts.append("matched_keywords=" + ",".join(keywords))
    if not parts:
        parts.append("matched_fields=none")
    return "; ".join(parts)


def _print_results(query: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    print("Query:", query)
    if not results:
        print("  No results\n")
        return
    for result in results:
        confidence = result.get("confidence_level", "") or "blank"
        explanation = _short_explanation(result["matched_explanation"])
        print(
            f"  - id={result['id']} score={result['score']} "
            f"confidence={confidence} | {explanation}"
        )
    print("")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "metadata_template.csv"
    metadata = load_metadata(path)

    queries = [
        {"emotion": "sad", "primary_theme": "aging", "imagery_keywords": ["time", "regret"]},
        {"emotion": "joyful", "primary_theme": "festival", "tempo": "fast"},
        {
            "primary_theme": "ritual",
            "secondary_theme": "agriculture",
            "tempo": "slow",
            "imagery_keywords": ["terrace", "rice"],
        },
    ]

    for query in queries:
        results = retrieve(metadata, query, top_k=5, min_score=None)
        _print_results(query, results)


if __name__ == "__main__":
    main()
