from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _humanize_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("_", " ")


def _build_tags(meta: Dict[str, Any], min_tags: int = 2, max_tags: int = 3) -> List[str]:
    preferred_fields = [
        "primary_theme",
        "emotion",
        "performance_form",
        "usage_scene",
        "secondary_theme",
        "secondary_emotion",
    ]
    tags: List[str] = []
    seen = set()
    for field in preferred_fields:
        value = _humanize_value(meta.get(field, ""))
        if not value:
            continue
        norm = value.lower()
        if norm in seen:
            continue
        seen.add(norm)
        tags.append(value)
        if len(tags) >= max_tags:
            break

    if len(tags) >= min_tags:
        return tags[:max_tags]
    return tags


def _extract_query_parts(query_text: str, structured_query: Optional[Dict[str, Any]]) -> List[str]:
    parts: List[str] = []
    seen = set()
    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", str(query_text).lower()):
        t = token.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        parts.append(t)

    if structured_query:
        for field in PARSED_QUERY_FIELDS:
            value = structured_query.get(field)
            if isinstance(value, str):
                v = value.strip().lower()
                if v and v not in seen:
                    seen.add(v)
                    parts.append(v)
            elif isinstance(value, list):
                for item in value:
                    v = str(item).strip().lower()
                    if v and v not in seen:
                        seen.add(v)
                        parts.append(v)
    return parts[:6]


def _build_match_explanation(
    meta: Dict[str, Any],
    query_text: str,
    structured_query: Optional[Dict[str, Any]],
    matched_fields: List[str],
    reasons: List[str],
) -> Tuple[str, str, str]:
    query_parts = _extract_query_parts(query_text, structured_query)
    searchable_text = " ".join([str(meta.get(field, "")).lower() for field in SEARCH_FIELDS])
    matched_parts = [part for part in query_parts if part and part in searchable_text]
    unmatched_parts = [part for part in query_parts if part and part not in searchable_text]

    coverage = 0.0
    if query_parts:
        coverage = len(matched_parts) / len(query_parts)
    elif matched_fields:
        coverage = 0.5

    if coverage >= 0.75:
        strength = "Strong match"
    elif coverage >= 0.35:
        strength = "Partial match"
    else:
        strength = "Exploratory match"

    focus_map = {
        "primary_theme": ("theme", "主题"),
        "secondary_theme": ("theme", "主题"),
        "emotion": ("emotion", "情绪"),
        "secondary_emotion": ("emotion", "情绪"),
        "usage_scene": ("usage scene", "场景"),
        "performance_form": ("performance form", "表演形式"),
        "imagery_keywords": ("imagery keywords", "意象关键词"),
        "region": ("region", "地域"),
        "ethnic_group": ("ethnic group", "民族"),
        "title": ("title wording", "标题表述"),
    }
    focus_labels_en: List[str] = []
    focus_labels_zh: List[str] = []
    for field in matched_fields:
        label_pair = focus_map.get(field)
        if not label_pair:
            continue
        label_en, label_zh = label_pair
        if label_en not in focus_labels_en:
            focus_labels_en.append(label_en)
            focus_labels_zh.append(label_zh)
    if not focus_labels_en and reasons:
        focus_labels_en = ["text relevance"]
        focus_labels_zh = ["文本相关性"]

    matched_preview_en = ", ".join([f'"{p}"' for p in matched_parts[:2]]) or "part of your query terms"
    matched_preview_zh = "、".join([f"“{p}”" for p in matched_parts[:2]]) or "查询中的部分关键词"
    focus_preview_en = ", ".join(focus_labels_en[:2]) if focus_labels_en else "broader semantics"
    focus_preview_zh = "、".join(focus_labels_zh[:2]) if focus_labels_zh else "更宽泛的语义"

    if strength == "Strong match":
        explanation_en = (
            f"This result strongly matches {matched_preview_en}, especially in {focus_preview_en}. "
            "Most key parts of your query are covered."
        )
        explanation_zh = (
            f"这条结果与{matched_preview_zh}匹配度较高，重点体现在{focus_preview_zh}。"
            "你的主要查询意图基本被覆盖。"
        )
    elif strength == "Partial match":
        if unmatched_parts:
            missing_preview_en = ", ".join([f'"{p}"' for p in unmatched_parts[:2]])
            missing_preview_zh = "、".join([f"“{p}”" for p in unmatched_parts[:2]])
            explanation_en = (
                f"This result matches {matched_preview_en}, mainly in {focus_preview_en}. "
                f"It only weakly covers {missing_preview_en}, so this is a partial match."
            )
            explanation_zh = (
                f"这条结果对{matched_preview_zh}的匹配较明显，主要体现在{focus_preview_zh}。"
                f"但对{missing_preview_zh}覆盖较弱，因此属于部分匹配。"
            )
        else:
            explanation_en = (
                f"This result is clearly related to your query in {focus_preview_en}, "
                "but the overall coverage is incomplete."
            )
            explanation_zh = (
                f"这条结果在{focus_preview_zh}上与查询有明显相关性，"
                "但整体覆盖不够完整。"
            )
    else:
        explanation_en = (
            f"This result has limited but relevant overlap with your query, mainly around {focus_preview_en}. "
            "It is shown for exploration rather than a close match."
        )
        explanation_zh = (
            f"这条结果与查询有一定相关性，主要体现在{focus_preview_zh}。"
            "当前更偏向探索性推荐，而非高贴合匹配。"
        )

    return strength, explanation_en, explanation_zh


def _build_evidence_line(matched_fields: List[str]) -> str:
    evidence_map = {
        "primary_theme": "theme",
        "secondary_theme": "theme",
        "emotion": "emotion",
        "secondary_emotion": "emotion",
        "usage_scene": "usage scene",
        "performance_form": "performance setting",
        "imagery_keywords": "imagery keywords",
        "region": "region",
        "ethnic_group": "ethnic group",
        "title": "title wording",
    }
    labels: List[str] = []
    for field in matched_fields:
        label = evidence_map.get(field)
        if label and label not in labels:
            labels.append(label)
    if not labels:
        return "Based on: overall semantic similarity."
    return f"Based on: {', '.join(labels[:3])}."


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
    st.title("Ethnic Music Cultural Archive Search Demo")
    st.caption("Explore music records across ethnic groups, regions, and cultural contexts.")

    default_metadata = os.getenv("METADATA_PATH", str(resolve_metadata_path(str(DEFAULT_METADATA_PATH))))
    metadata_path_raw = st.sidebar.text_input("Metadata CSV path", default_metadata)
    vocab_path_raw = st.sidebar.text_input("Vocab JSON path", str(DEFAULT_VOCAB_PATH))
    wav_dir_path_raw = st.sidebar.text_input("Audio directory", str(DEFAULT_WAV_DIR))
    mp4_dir_path_raw = st.sidebar.text_input("Video directory", str(DEFAULT_MP4_DIR))
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=20, value=3, step=1)
    use_llm_parse = st.sidebar.checkbox("Use LLM to parse query (optional)", value=False)
    model_name = st.sidebar.text_input("LLM model", "deepseek-chat")
    min_score_mode = st.sidebar.selectbox("Min score", options=["1", "2", "3"], index=0)
    debug_mode = st.sidebar.checkbox("Debug mode", value=False)
    st.sidebar.caption("Enable Debug mode to inspect parser output and internal diagnostics.")

    st.text_area(
        "Search query",
        value="",
        key="query_text",
        height=70,
        placeholder="Try: Hani terrace labor polyphonic song",
    )

    with st.expander("Example queries"):
        st.write("- Joyful New Year festival songs")
        st.write("- Solemn ritual songs about terraces and rice")
        st.write("- Reflective songs about aging and time")

    if not st.button("Search"):
        return

    query_text = st.session_state.get("query_text", "")
    if not str(query_text).strip():
        st.info("Enter a query to start searching the archive.")
        return

    metadata_path = _resolve_input_path(metadata_path_raw, CODEBASE_DIR)
    vocab_path = _resolve_input_path(vocab_path_raw, CODEBASE_DIR)
    wav_dir = _resolve_input_path(wav_dir_path_raw, CODEBASE_DIR)
    mp4_dir = _resolve_input_path(mp4_dir_path_raw, CODEBASE_DIR)

    structured_query: Optional[Dict[str, Any]] = None
    parse_notice: Optional[str] = None
    try:
        with st.spinner("Analyzing your query..."):
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
                    parse_notice = "LLM parsing unavailable; using rule-based parsing."
                else:
                    parse_mode = "llm"
            else:
                parse_mode = "text-only"
    except Exception as exc:
        st.error(f"Query parsing failed: {exc}")
        return

    try:
        with st.spinner("Searching the archive..."):
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

    if parse_notice:
        st.sidebar.caption(f"Parsing status: {parse_notice}")

    if debug_mode:
        st.subheader("Debug")
        st.caption("Debug mode shows parser output and internal retrieval diagnostics.")
        st.caption(f"Parser mode: {parse_mode}")
        if structured_query is not None:
            display_query = {field: structured_query.get(field) for field in PARSED_QUERY_FIELDS}
            st.code(json.dumps(display_query, ensure_ascii=False, indent=2), language="json")
        else:
            st.code("LLM parsing is disabled. Full-text field matching is used.")

    st.subheader("Results")
    if not results:
        st.info("No results found. Try broader keywords, related themes, or a lower minimum score.")
        return

    for idx, result in enumerate(results, start=1):
        meta = result.get("record", {})
        item_id = str(result.get("id", ""))
        title = str(result.get("title") or item_id or "unknown")
        score = result.get("score", 0)
        matched_fields = result.get("matched_fields", [])
        reasons = result.get("match_reasons", [])
        ethnic_group = str(meta.get("ethnic_group", "")).strip() or "Unknown ethnic group"
        region = str(meta.get("region", "")).strip() or "Unknown region"
        tags = _build_tags(meta)
        strength, explanation_en, explanation_zh = _build_match_explanation(
            meta=meta,
            query_text=str(query_text),
            structured_query=structured_query,
            matched_fields=matched_fields,
            reasons=reasons,
        )
        evidence_line = _build_evidence_line(matched_fields=matched_fields)

        st.markdown(f"### {idx}. {title}")
        st.caption(f"{ethnic_group or 'Unknown ethnic group'} | {region or 'Unknown region'}")
        st.caption(f"Match strength: {strength}")
        if tags:
            st.markdown(" ".join([f"`{tag}`" for tag in tags]))
        st.markdown("**Why it matches**")
        st.caption(f"EN: {explanation_en}")
        st.caption(f"中文：{explanation_zh}")
        st.caption(evidence_line)

        audio_path = _resolve_media_path(
            str(result.get("audio_path", "")),
            wav_dir,
            fallback_stem=item_id,
            fallback_suffix=".wav",
        )
        if audio_path is not None:
            st.audio(audio_path.read_bytes(), format="audio/wav")

        video_path = _resolve_media_path(str(result.get("video_path", "")), mp4_dir)
        if video_path is not None:
            st.video(str(video_path))

        with st.expander("More details"):
            cultural_context = str(meta.get("cultural_context", "")).strip()
            uncertainty_note = str(meta.get("uncertainty_note", "")).strip()
            confidence_level = str(meta.get("confidence_level", "")).strip()
            source_url = str(meta.get("source_url", "")).strip()

            st.write(f"**Cultural context:** {cultural_context or 'N/A'}")
            st.write(f"**Uncertainty note:** {uncertainty_note or 'N/A'}")
            st.write(f"**Confidence level:** {confidence_level or 'N/A'}")
            if source_url:
                st.markdown(f"**Source URL:** [View source]({source_url})")
            else:
                st.write("**Source URL:** N/A")

            if debug_mode:
                st.write("---")
                st.caption("Debug details")
                st.write(f"id={item_id} | score={score} | matched_fields={matched_fields}")
                if reasons:
                    st.write("Internal reasons:")
                    for reason in reasons:
                        st.write(f"- {reason}")
                for field in SEARCH_FIELDS:
                    if meta.get(field):
                        st.write(f"{field}: {meta.get(field)}")

        st.divider()


if __name__ == "__main__":
    main()
