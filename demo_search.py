from __future__ import annotations

import argparse
from typing import Any, Dict, List

from retrieval import load_metadata, retrieve
from llm_parse import parse_nl_query_openai


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
    parser = argparse.ArgumentParser(description="Rule-based retrieval demo")
    parser.add_argument("metadata_path", nargs="?", default="metadata_template.csv")
    parser.add_argument("--nl", dest="nl", help="Natural-language query text")
    parser.add_argument("--model", default="deepseek-chat", help="LLM model name")
    parser.add_argument("--vocab", default="vocab.json", help="Path to vocab.json")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    path = args.metadata_path
    metadata = load_metadata(path)

    if args.nl:
        parsed_query = parse_nl_query_openai(
            args.nl, vocab_path=args.vocab, model=args.model, debug=args.debug
        )
        print("NL Query:", args.nl)
        print("Parsed query:", parsed_query)
        results = retrieve(metadata, parsed_query, top_k=5, min_score=None)
        _print_results(parsed_query, results)
        return

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
