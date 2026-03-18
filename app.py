from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from llm_parse import parse_nl_query_openai
from search_pipeline import (
    SEARCH_FIELDS,
    create_search_engine,
    resolve_metadata_path,
)


CODEBASE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODEBASE_DIR / "data"
DEFAULT_METADATA_PATH = DATA_DIR / "metadata" / "metadata_master.csv"
DEFAULT_VOCAB_PATH = CODEBASE_DIR / "vocab.json"
DEFAULT_WAV_DIR = DATA_DIR / "audio" / "wav"
DEFAULT_MP4_DIR = DATA_DIR / "video" / "mp4"
PARSED_QUERY_FIELDS = [
    "primary_theme",
    "secondary_theme",
    "emotion",
    "secondary_emotion",
    "usage_scene",
    "performance_form",
    "tempo",
    "imagery_keywords",
]


@st.cache_data
def _run_search(
    metadata_path: str,
    query_text: str,
    top_k: int,
    min_score: int,
    structured_query: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    engine = create_search_engine(metadata_path)
    return engine.search(
        query_text=query_text,
        top_k=top_k,
        structured_query=structured_query,
        min_score=min_score,
    )


def _resolve_input_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _resolve_media_path(
    raw_path: str,
    media_dir: Path,
    fallback_stem: str = "",
    fallback_suffix: str = "",
) -> Optional[Path]:
    if raw_path:
        path = Path(raw_path)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(media_dir / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate

    if fallback_stem and fallback_suffix:
        guessed = media_dir / f"{fallback_stem}{fallback_suffix}"
        if guessed.exists():
            return guessed
    return None


def main() -> None:
    st.title("Ethnic Music Metadata Retrieval Demo")
    st.caption("Metadata-first search with optional LLM query parsing.")

    default_metadata = os.getenv("METADATA_PATH", str(resolve_metadata_path(str(DEFAULT_METADATA_PATH))))
    metadata_path_raw = st.sidebar.text_input("Metadata CSV path", default_metadata)
    vocab_path_raw = st.sidebar.text_input("Vocab JSON path", str(DEFAULT_VOCAB_PATH))
    wav_dir_path_raw = st.sidebar.text_input("Audio directory", str(DEFAULT_WAV_DIR))
    mp4_dir_path_raw = st.sidebar.text_input("Video directory", str(DEFAULT_MP4_DIR))
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=20, value=5, step=1)
    use_llm_parse = st.sidebar.checkbox("Use LLM to parse query (optional)", value=False)
    model_name = st.sidebar.text_input("LLM model", "deepseek-chat")
    min_score_mode = st.sidebar.selectbox("Min score", options=["1", "2", "3"], index=0)

    st.text_area(
        "Natural-language query",
        value="哈尼族 梯田 劳作 多声部",
        key="query_text",
        height=100,
    )

    with st.expander("Example queries"):
        st.write("- joyful new year festival songs")
        st.write("- solemn ritual songs about terraces and rice")
        st.write("- sad reflective songs about aging and time")

    if not st.button("Search"):
        return

    query_text = st.session_state.get("query_text", "")
    if not str(query_text).strip():
        st.warning("Please enter a query.")
        return

    metadata_path = _resolve_input_path(metadata_path_raw, CODEBASE_DIR)
    vocab_path = _resolve_input_path(vocab_path_raw, CODEBASE_DIR)
    wav_dir = _resolve_input_path(wav_dir_path_raw, CODEBASE_DIR)
    mp4_dir = _resolve_input_path(mp4_dir_path_raw, CODEBASE_DIR)

    structured_query: Optional[Dict[str, Any]] = None
    parse_notice: Optional[str] = None
    try:
        if use_llm_parse:
            llm_result = parse_nl_query_openai(
                str(query_text),
                vocab_path=str(vocab_path),
                model=model_name,
            )
            fallback_reason = llm_result.get("__fallback_reason")
            structured_query = {field: llm_result.get(field) for field in PARSED_QUERY_FIELDS}
            if fallback_reason:
                parse_mode = "rule-fallback"
                parse_notice = (
                    "Remote LLM parsing unavailable. Using rule-based fallback. "
                    "See terminal for detailed logs."
                )
            else:
                parse_mode = "llm"
        else:
            parse_mode = "text-only"
    except Exception as exc:
        st.error(f"Query parsing failed: {exc}")
        return

    try:
        results = _run_search(
            metadata_path=str(metadata_path),
            query_text=str(query_text),
            top_k=int(top_k),
            min_score=int(min_score_mode),
            structured_query=structured_query,
        )
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    st.subheader("Parsed Query (Optional)")
    st.caption(f"Parser mode: {parse_mode}")
    if parse_notice:
        st.warning(parse_notice)
    if structured_query is not None:
        display_query = {field: structured_query.get(field) for field in PARSED_QUERY_FIELDS}
        st.code(json.dumps(display_query, ensure_ascii=False, indent=2), language="json")
    else:
        st.code("未启用 LLM 结构化解析，使用全文字段匹配。")

    st.subheader("Ranked Results")
    if not results:
        st.info("No results found. Try a broader query or lower min score.")
        return

    for idx, result in enumerate(results, start=1):
        meta = result.get("record", {})
        item_id = str(result.get("id", ""))
        title = str(result.get("title") or item_id or "unknown")
        score = result.get("score", 0)
        matched_fields = result.get("matched_fields", [])
        reasons = result.get("match_reasons", [])

        st.markdown(f"**{idx}. {title}**")
        st.write(f"id={item_id} | score={score} | matched_fields={matched_fields}")

        for field in SEARCH_FIELDS:
            if meta.get(field):
                st.write(f"{field}: {meta.get(field)}")

        if reasons:
            st.write("匹配原因：")
            for reason in reasons:
                st.write(f"- {reason}")

        audio_path = _resolve_media_path(
            str(result.get("audio_path", "")),
            wav_dir,
            fallback_stem=item_id,
            fallback_suffix=".wav",
        )
        if audio_path is not None:
            with st.expander(f"Play audio: {audio_path.name}"):
                st.audio(audio_path.read_bytes(), format="audio/wav")
        else:
            st.caption("No playable audio asset for this item.")

        video_path = _resolve_media_path(str(result.get("video_path", "")), mp4_dir)
        if video_path is not None:
            with st.expander(f"Play video: {video_path.name}"):
                st.video(str(video_path))
        else:
            st.caption("No playable video asset for this item.")


if __name__ == "__main__":
    main()
