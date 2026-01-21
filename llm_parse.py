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
    model: str = "deepseek-chat",
    api_key_env: str = "DEEPSEEK_API_KEY",
    base_url_env: str = "DEEPSEEK_BASE_URL",
    debug: bool = False,
) -> Dict[str, Any]:
    try:
        from deepseek import DeepSeekClient
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("DeepSeek SDK is required. Install with: pip install deepseek-sdk") from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")
    base_url = os.getenv(base_url_env, "https://api.deepseek.com/v1")

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
    }

    client = DeepSeekClient(api_key=api_key, base_url=base_url)
    response = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
    )

    content = None
    if isinstance(response, dict):
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
    else:
        content = getattr(response.choices[0].message, "content", None)
    if not content:
        raise RuntimeError("DeepSeek response did not include message content.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse model output as JSON: {content}") from exc

    projected = project_to_vocab(parsed, vocab)
    if debug:
        print("Parsed query (raw):", parsed)
        print("Parsed query (projected):", projected)
    return projected
