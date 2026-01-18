from __future__ import annotations

import sys
from typing import Any, Dict, List

from retrieval import load_metadata, retrieve


def _format_keywords(keywords: Any) -> str:
    if not keywords:
        return ""
    if isinstance(keywords, list):
        return ",".join(str(k) for k in keywords)
    return str(keywords)


def _print_results(query: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    print("Query:", query)
    if not results:
        print("  No results\n")
        return
    for result in results:
        confidence = result.get("confidence_level", "") or "blank"
        matched_fields = result.get("matched_fields", [])
        matched_keywords = result.get("matched_keywords", [])
        matched_fields_text = ",".join(matched_fields) if isinstance(matched_fields, list) else str(matched_fields)
        matched_keywords_text = _format_keywords(matched_keywords)
        theme = result.get("primary_theme", "")
        emotion = result.get("emotion", "")
        print(
            f"{result['id']} | score={result['score']} | conf={confidence} | "
            f"theme={theme} | emotion={emotion} | matched_fields={matched_fields_text} | "
            f"matched_keywords={matched_keywords_text}"
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
        {"emotion": "joyful"},
        {"primary_theme": "agriculture", "imagery_keywords": ["rice_planting", "field"]},
    ]

    for query in queries:
        results = retrieve(metadata, query, top_k=5, min_score=None)
        _print_results(query, results)


if __name__ == "__main__":
    main()
