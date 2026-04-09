"""Smoke-test evaluation script for retrieval + similarity.

Runs a small set of representative queries, prints:
  - top retrieved results (Layer 1)
  - similar tracks for the first result (Layer 2)
  - explanation text

Intended for project-report use and quick sanity checking.
Usage:
    python eval_smoke_test.py
"""

from __future__ import annotations

import json
from pathlib import Path

from search_pipeline import create_search_engine
from similarity import find_similar

CODEBASE_DIR = Path(__file__).resolve().parent

SMOKE_QUERIES = [
    {
        "name": "Hani labor polyphonic",
        "query_text": "Hani rice terrace labor polyphonic",
        "structured_query": {
            "primary_theme": "agriculture",
            "emotion": "joyful",
            "usage_scene": "labor",
            "performance_form": "polyphonic",
        },
    },
    {
        "name": "Sad ritual farewell",
        "query_text": "sad ritual farewell mourning",
        "structured_query": {
            "emotion": "melancholic",
            "usage_scene": "ceremony",
        },
    },
    {
        "name": "Mongolian grassland nature",
        "query_text": "Mongolian grassland nature horse",
        "structured_query": {
            "primary_theme": "nature",
            "ethnic_group": "Mongol",
        },
    },
    {
        "name": "Dong choral festival",
        "query_text": "Dong choral festival joyful",
        "structured_query": {
            "performance_form": "choral",
            "emotion": "joyful",
        },
    },
    {
        "name": "Love courtship antiphonal (fallback test)",
        "query_text": "love courtship antiphonal duet",
        "structured_query": {
            "primary_theme": "courtship",
            "performance_form": "antiphonal",
        },
    },
]


def run_smoke_test() -> None:
    engine = create_search_engine()
    all_records = engine.load()
    separator = "-" * 60

    for sq in SMOKE_QUERIES:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {sq['name']}")
        print(f"  text: {sq['query_text']}")
        print(f"  structured: {json.dumps(sq.get('structured_query', {}), ensure_ascii=False)}")
        print(separator)

        results = engine.search(
            query_text=sq["query_text"],
            top_k=5,
            structured_query=sq.get("structured_query"),
            min_score=1,
        )

        if not results:
            print("  (no results)")
            continue

        print(f"  Top {len(results)} retrieved results:")
        for i, r in enumerate(results, 1):
            rid = r.get("id", "?")
            title = r.get("title", "?")
            score = r.get("score", 0)
            fields = r.get("matched_fields", [])
            print(f"    {i}. [{rid}] {title}  (score={score}, fields={fields})")

        # Layer 2: similar tracks for the first result.
        first_id = results[0].get("id", "")
        print(separator)
        print(f"  Similar tracks for [{first_id}]:")
        similar = find_similar(first_id, all_records, top_k=3)
        if similar:
            for j, s in enumerate(similar, 1):
                sm = s["record"]
                sid = sm.get("track_id", "?")
                stitle = sm.get("title", "?")
                sim_score = s["similarity_score"]
                explanation = s["explanation"]
                print(f"    {j}. [{sid}] {stitle}  (sim={sim_score})")
                print(f"       {explanation}")
        else:
            print("    No sufficiently similar tracks found.")

    print(f"\n{'=' * 60}")
    print("Smoke test complete.")


if __name__ == "__main__":
    run_smoke_test()
