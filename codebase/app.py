from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

from llm_parse import parse_nl_query_openai
from search_pipeline import (
    SEARCH_FIELDS,
    create_search_engine,
    resolve_metadata_path,
)
from similarity import find_similar


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
SLOT_FIELDS = {
    "emotion": ["emotion", "secondary_emotion", "retrieval_notes", "cultural_context"],
    "usage_scene": ["usage_scene", "retrieval_notes", "cultural_context", "primary_theme", "secondary_theme"],
    "performance_form": ["performance_form", "retrieval_notes", "cultural_context"],
    "theme": ["primary_theme", "secondary_theme", "retrieval_notes", "cultural_context"],
    "suitability": ["usage_scene", "retrieval_notes", "cultural_context"],
}
SLOT_IMPORTANCE = {
    "emotion": 1.0,
    "usage_scene": 1.8,
    "performance_form": 1.1,
    "theme": 1.0,
    "suitability": 1.3,
}
SLOT_KEYWORDS = {
    "usage_scene": [
        "婚礼",
        "wedding",
        "ritual",
        "仪式",
        "harvest",
        "丰收",
        "festival",
        "节庆",
        "ceremony",
        "庆典",
        "葬礼",
        "funeral",
        "labor",
        "劳动",
    ],
    "emotion": [
        "欢快",
        "joyful",
        "happy",
        "melancholic",
        "悲伤",
        "sad",
        "sorrowful",
        "mournful",
        "lamenting",
        "plaintive",
        "grieving",
        "elegiac",
        "solemn",
        "庄重",
        "yearning",
        "wistful",
        "longing",
        "思念",
    ],
    "performance_form": [
        "chorus",
        "choir",
        "合唱",
        "对唱",
        "antiphonal",
        "polyphonic",
        "多声部",
        "solo",
        "独唱",
    ],
    "theme": [
        "love",
        "爱情",
        "nature",
        "自然",
        "migration",
        "迁徙",
        "community",
        "社区",
        "agriculture",
        "农耕",
    ],
    "suitability": [
        "适合",
        "suitable",
        "用来",
        "for",
        "当作",
        "用于",
        "can be used",
    ],
}
QUERY_TERM_EXPANSIONS = {
    "婚礼": ["wedding", "ceremony", "festival", "community_gathering"],
    "wedding": ["婚礼", "ceremony", "festival", "community_gathering"],
    "仪式": ["ritual", "ceremony"],
    "欢快": ["joyful", "energetic", "festival"],
    "joyful": ["欢快", "energetic"],
    "sad": ["sorrowful", "mournful", "melancholic", "lamenting", "grieving", "elegiac", "悲伤"],
    "sorrowful": ["sad", "mournful", "melancholic", "lamenting", "grieving", "elegiac", "悲伤"],
    "mournful": ["sad", "sorrowful", "melancholic", "lamenting", "grieving", "elegiac", "悲伤"],
    "melancholic": ["sad", "sorrowful", "mournful", "lamenting", "grieving", "elegiac", "悲伤"],
    "lamenting": ["sad", "sorrowful", "mournful", "melancholic", "grieving", "elegiac", "悲伤"],
    "plaintive": ["sad", "sorrowful", "mournful", "melancholic", "lamenting", "悲伤"],
    "grieving": ["sad", "sorrowful", "mournful", "melancholic", "lamenting", "悲伤"],
    "elegiac": ["sad", "sorrowful", "mournful", "melancholic", "lamenting", "悲伤"],
    "悲伤": ["sad", "sorrowful", "mournful", "melancholic", "lamenting", "grieving", "elegiac"],
    "yearning": ["wistful", "longing", "思念"],
    "wistful": ["yearning", "longing", "思念"],
    "longing": ["yearning", "wistful", "思念"],
    "合唱": ["choral", "chorus", "polyphonic"],
    "chorus": ["合唱", "choral", "polyphonic"],
    "劳动": ["labor", "work_song", "agriculture"],
    "节庆": ["festival", "community_gathering", "celebration"],
    "丰收": ["harvest", "agriculture", "labor"],
}
STRONG_SADNESS_TERMS = {
    "sad",
    "sorrowful",
    "mournful",
    "melancholic",
    "lamenting",
    "plaintive",
    "grieving",
    "elegiac",
    "悲伤",
    "哀伤",
    "哀婉",
    "伤感",
}
WEAK_SADNESS_TERMS = {"yearning", "wistful", "longing", "思念", "惆怅"}


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


def _collect_query_slots(query_text: str, structured_query: Optional[Dict[str, Any]]) -> Dict[str, Set[str]]:
    text = str(query_text).lower()
    query_slots: Dict[str, Set[str]] = {slot: set() for slot in SLOT_FIELDS.keys()}

    if structured_query:
        for slot, fields in {
            "emotion": ["emotion", "secondary_emotion"],
            "usage_scene": ["usage_scene"],
            "performance_form": ["performance_form"],
            "theme": ["primary_theme", "secondary_theme"],
        }.items():
            for field in fields:
                value = structured_query.get(field)
                if isinstance(value, str) and value.strip():
                    query_slots[slot].add(value.strip().lower())
                elif isinstance(value, list):
                    query_slots[slot].update([str(v).strip().lower() for v in value if str(v).strip()])

    for slot, keywords in SLOT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                query_slots[slot].add(kw.lower())
                for expanded in QUERY_TERM_EXPANSIONS.get(kw.lower(), []):
                    query_slots[slot].add(expanded.lower())

    if any(x in text for x in ["用来", "当作", "适合", "suitable", "for "]):
        query_slots["suitability"].add("suitability")

    return query_slots


def _slot_text(meta: Dict[str, Any], slot: str) -> str:
    return " ".join([str(meta.get(field, "")).lower() for field in SLOT_FIELDS.get(slot, [])])


def _rerank_results(
    results: List[Dict[str, Any]],
    query_text: str,
    structured_query: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not results:
        return results

    query_slots = _collect_query_slots(query_text=query_text, structured_query=structured_query)
    active_slots = [slot for slot, values in query_slots.items() if values]

    strong_sadness_requested = any(term in str(query_text).lower() for term in STRONG_SADNESS_TERMS)
    weak_sadness_requested = any(term in str(query_text).lower() for term in WEAK_SADNESS_TERMS)

    reranked: List[Dict[str, Any]] = []
    for result in results:
        meta = result.get("record", {})
        base_score = float(result.get("score", 0) or 0)
        slot_hits = 0
        weighted_hits = 0.0
        missing_critical = False
        slot_hit_map: Dict[str, bool] = {}

        for slot in active_slots:
            haystack = _slot_text(meta, slot)
            matched = any(term in haystack for term in query_slots[slot])
            slot_hit_map[slot] = matched
            if matched:
                slot_hits += 1
                weighted_hits += SLOT_IMPORTANCE.get(slot, 1.0)
            elif slot == "usage_scene":
                missing_critical = True

        emotion_text = _slot_text(meta, "emotion")
        strong_sadness_hit = any(term in emotion_text for term in STRONG_SADNESS_TERMS)
        weak_sadness_hit = any(term in emotion_text for term in WEAK_SADNESS_TERMS)

        slot_coverage = (slot_hits / len(active_slots)) if active_slots else 1.0
        weighted_coverage = (
            weighted_hits / sum([SLOT_IMPORTANCE.get(slot, 1.0) for slot in active_slots])
            if active_slots
            else 1.0
        )

        coverage_bonus = weighted_coverage * 4.0
        single_slot_penalty = 0.0
        if len(active_slots) >= 2 and slot_hits <= 1:
            single_slot_penalty += 3.5
        if len(active_slots) >= 3 and slot_coverage < 0.5:
            single_slot_penalty += 2.0
        if missing_critical and len(active_slots) >= 2:
            single_slot_penalty += 2.5
        if strong_sadness_requested and not strong_sadness_hit:
            single_slot_penalty += 2.8
            # If only weakly related sadness cues are present (e.g., yearning), keep it as weak evidence.
            if weak_sadness_hit:
                single_slot_penalty += 0.7
        elif weak_sadness_requested and not (strong_sadness_hit or weak_sadness_hit):
            single_slot_penalty += 1.0

        adjusted_score = base_score + coverage_bonus - single_slot_penalty
        strong_bucket = False
        if not active_slots:
            strong_bucket = True
        else:
            strong_bucket = weighted_coverage >= 0.72 and slot_coverage >= 0.6 and slot_hits >= 2
            if len(active_slots) == 1:
                strong_bucket = weighted_coverage >= 0.8 and slot_coverage >= 0.8
            if strong_sadness_requested and not strong_sadness_hit:
                strong_bucket = False

        segment = "best" if strong_bucket else "related"

        enriched = dict(result)
        enriched["_slot_hits"] = slot_hits
        enriched["_active_slots"] = len(active_slots)
        enriched["_slot_coverage"] = round(slot_coverage, 3)
        enriched["_weighted_coverage"] = round(weighted_coverage, 3)
        enriched["_adjusted_score"] = round(adjusted_score, 3)
        enriched["_segment"] = segment
        enriched["_slot_hit_map"] = slot_hit_map
        enriched["_strong_bucket"] = strong_bucket
        enriched["_strong_sadness_hit"] = strong_sadness_hit
        reranked.append(enriched)

    reranked.sort(
        key=lambda x: (
            -float(x.get("_adjusted_score", 0.0)),
            -float(x.get("_weighted_coverage", 0.0)),
            -int(x.get("_slot_hits", 0)),
            -int(x.get("score", 0)),
            str(x.get("id", "")),
        )
    )
    return reranked


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
            fetch_k = max(int(top_k) * 3, 10)
            results = _run_search(
                metadata_path=str(metadata_path),
                query_text=str(query_text),
                top_k=fetch_k,
                min_score=int(min_score_mode),
                structured_query=structured_query,
            )
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    results = _rerank_results(results=results, query_text=str(query_text), structured_query=structured_query)
    results = results[: int(top_k)]

    # Load full catalogue once for item-to-item similarity lookups.
    all_records = create_search_engine(str(metadata_path)).load()

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

    best_results = [r for r in results if r.get("_segment") == "best"]
    related_results = [r for r in results if r.get("_segment") != "best"]

    if not best_results and related_results:
        st.caption("No strong match found for all query conditions. Showing the closest related results below.")

    sections: List[Tuple[str, List[Dict[str, Any]]]] = []
    if best_results:
        sections.append(("Best matches", best_results))
    if related_results:
        sections.append(("Related / exploratory matches", related_results))

    rank_index = 1
    for section_title, section_items in sections:
        st.markdown(f"#### {section_title}")
        for result in section_items:
            idx = rank_index
            rank_index += 1
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
                    st.write(
                        "rerank: "
                        f"adjusted={result.get('_adjusted_score')} | "
                        f"coverage={result.get('_weighted_coverage')} | "
                        f"slot_hits={result.get('_slot_hits')}/{result.get('_active_slots')}"
                    )
                    if reasons:
                        st.write("Internal reasons:")
                        for reason in reasons:
                            st.write(f"- {reason}")
                    for field in SEARCH_FIELDS:
                        if meta.get(field):
                            st.write(f"{field}: {meta.get(field)}")

            # -- Similar Tracks (item-to-item recommendation) --
            shown_ids = {str(r.get("id", "")) for r in results}
            similar_tracks = find_similar(
                item_id,
                all_records,
                top_k=3,
                exclude_ids=shown_ids - {item_id},
            )
            if similar_tracks:
                with st.expander("Similar tracks"):
                    for sim in similar_tracks:
                        sim_meta = sim["record"]
                        sim_title = sim_meta.get("title") or sim_meta.get("track_id", "")
                        sim_ethnic = sim_meta.get("ethnic_group", "")
                        sim_score = sim["similarity_score"]
                        sim_explain = sim["explanation"]
                        st.markdown(
                            f"**{sim_title}**"
                            + (f"  ({sim_ethnic})" if sim_ethnic else "")
                        )
                        st.caption(f"{sim_explain}  (similarity: {sim_score})")

            st.divider()


if __name__ == "__main__":
    main()
