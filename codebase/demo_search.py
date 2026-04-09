from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from llm_parse import parse_nl_query_openai
from search_pipeline import create_search_engine, resolve_metadata_path


CODEBASE_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = CODEBASE_DIR / "data" / "metadata" / "metadata_master.csv"
DEFAULT_VOCAB_PATH = CODEBASE_DIR / "vocab.json"


def _print_results(query: str, results: List[Dict[str, object]]) -> None:
    print("Query:", query)
    if not results:
        print("  No results\n")
        return
    for result in results:
        matched_fields = result.get("matched_fields", [])
        reasons = result.get("match_reasons", [])
        title = result.get("title", "")
        audio_path = result.get("audio_path", "")
        video_path = result.get("video_path", "")
        print(
            f"{result['id']} | score={result['score']} | title={title} | "
            f"matched_fields={matched_fields} | reasons={reasons}"
        )
        if audio_path:
            print(f"  audio_path={audio_path}")
        if video_path:
            print(f"  video_path={video_path}")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Metadata-first retrieval demo")
    parser.add_argument("metadata_path", nargs="?", default=str(resolve_metadata_path(str(DEFAULT_METADATA_PATH))))
    parser.add_argument("--nl", dest="nl", help="Natural-language query text")
    parser.add_argument("--model", default="deepseek-chat", help="LLM model name")
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH), help="Path to vocab.json")
    parser.add_argument("--top_k", type=int, default=5, help="Top K results")
    parser.add_argument("--min_score", type=int, default=1, help="Minimum score threshold")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    engine = create_search_engine(args.metadata_path)

    if args.nl:
        vocab_path = Path(args.vocab).expanduser()
        if not vocab_path.is_absolute():
            vocab_path = (CODEBASE_DIR / vocab_path).resolve()
        parsed_query = parse_nl_query_openai(
            args.nl, vocab_path=str(vocab_path), model=args.model, debug=args.debug
        )
        print("NL Query:", args.nl)
        print("Parsed query:", parsed_query)
        results = engine.search(
            query_text=args.nl,
            top_k=args.top_k,
            structured_query=parsed_query,
            min_score=args.min_score,
        )
        _print_results(args.nl, results)
        return

    queries = [
        "哈尼族 梯田 劳作 多声部",
        "Mongolian horse grassland festival song",
        "ritual melancholic ceremony parting",
    ]

    for query in queries:
        results = engine.search(query_text=query, top_k=args.top_k, min_score=args.min_score)
        _print_results(query, results)


if __name__ == "__main__":
    main()
