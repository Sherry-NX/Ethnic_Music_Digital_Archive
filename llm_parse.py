from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def load_vocab(vocab_path: str = "vocab.json") -> Dict[str, Any]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    if "emotion" not in vocab or "primary_theme" not in vocab:
        raise ValueError("vocab.json must include 'emotion' and 'primary_theme' keys.")
    if "aliases" not in vocab:
        vocab["aliases"] = {}
    return vocab


def _normalize_scalar(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text else None


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def project_to_vocab(parsed: Dict[str, Any], vocab: Dict[str, Any]) -> Dict[str, Any]:
    allowed_emotions = set(vocab.get("emotion", []))
    allowed_primary = set(vocab.get("primary_theme", []))
    aliases = vocab.get("aliases", {})

    def map_vocab_value(value: Optional[str], allowed: set) -> Optional[str]:
        normalized = _normalize_scalar(value)
        if not normalized:
            return None
        if normalized in allowed:
            return normalized
        alias_target = aliases.get(normalized)
        if alias_target in allowed:
            return alias_target
        return None

    imagery = parsed.get("imagery_keywords", [])
    if imagery is None:
        imagery = []
    if not isinstance(imagery, list):
        imagery = [str(imagery)]
    cleaned_keywords = []
    for item in imagery:
        token = _normalize_scalar(item)
        if token:
            cleaned_keywords.append(token)
    cleaned_keywords = _dedupe_preserve_order(cleaned_keywords)[:6]

    return {
        "emotion": map_vocab_value(parsed.get("emotion"), allowed_emotions),
        "primary_theme": map_vocab_value(parsed.get("primary_theme"), allowed_primary),
        "secondary_theme": _normalize_scalar(parsed.get("secondary_theme")),
        "tempo": _normalize_scalar(parsed.get("tempo")),
        "vocal_style": _normalize_scalar(parsed.get("vocal_style")),
        "performance_context": _normalize_scalar(parsed.get("performance_context")),
        "imagery_keywords": cleaned_keywords,
    }


def parse_nl_query_openai(
    text: str,
    vocab_path: str = "vocab.json",
    model: str = "gpt-4.1-mini",
    api_key_env: str = "OPENAI_API_KEY",
    debug: bool = False,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenAI SDK is required. Install with: pip install openai") from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")

    vocab = load_vocab(vocab_path)
    allowed_emotions = vocab.get("emotion", [])
    allowed_primary = vocab.get("primary_theme", [])

    system_text = (
        "You extract structured fields for ethnic music retrieval. "
        "Return JSON that matches the schema exactly. "
        "If unsure, return null for scalar fields and keep imagery_keywords minimal."
    )
    user_text = (
        "Extract a structured query from the text below.\n\n"
        f"Allowed emotions: {', '.join(allowed_emotions)}\n"
        f"Allowed primary themes: {', '.join(allowed_primary)}\n"
        "Do not invent labels outside these lists.\n\n"
        f"Text: {text}"
    )

    schema = {
        "name": "music_query",
        "schema": {
            "type": "object",
            "properties": {
                "emotion": {"type": ["string", "null"]},
                "primary_theme": {"type": ["string", "null"]},
                "secondary_theme": {"type": ["string", "null"]},
                "tempo": {"type": ["string", "null"]},
                "vocal_style": {"type": ["string", "null"]},
                "performance_context": {"type": ["string", "null"]},
                "imagery_keywords": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "emotion",
                "primary_theme",
                "secondary_theme",
                "tempo",
                "vocal_style",
                "performance_context",
                "imagery_keywords",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        text={"format": {"type": "json_schema", "json_schema": schema}},
    )

    if not getattr(response, "output_text", None):
        raise RuntimeError("OpenAI response did not include output_text.")

    try:
        parsed = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse model output as JSON: {response.output_text}") from exc

    projected = project_to_vocab(parsed, vocab)
    if debug:
        print("Parsed query (raw):", parsed)
        print("Parsed query (projected):", projected)
    return projected
