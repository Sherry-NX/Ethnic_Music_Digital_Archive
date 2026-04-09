from __future__ import annotations

import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional


SCALAR_FIELDS = [
    "primary_theme",
    "secondary_theme",
    "emotion",
    "secondary_emotion",
    "usage_scene",
    "performance_form",
    "tempo",
]
OUTPUT_FIELDS = SCALAR_FIELDS + ["imagery_keywords"]


def load_vocab(vocab_path: str = "vocab.json") -> Dict[str, Any]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    fields = vocab.get("fields", {})
    required = [
        "primary_theme",
        "secondary_theme",
        "emotion",
        "secondary_emotion",
        "usage_scene",
        "performance_form",
        "tempo",
    ]
    for field in required:
        if field not in fields:
            raise ValueError(f"vocab.json must include fields.{field}.")
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


def _map_vocab_value(
    value: Optional[str],
    allowed: List[str],
    aliases: Dict[str, str],
) -> Optional[str]:
    normalized = _normalize_scalar(value)
    if not normalized:
        return None
    allowed_set = set(allowed)
    if normalized in allowed_set:
        return normalized
    alias_target = aliases.get(normalized)
    if alias_target in allowed_set:
        return alias_target
    return None


def _merge_with_rule_hints(
    projected: Dict[str, Any],
    rule_hints: Dict[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for field in SCALAR_FIELDS:
        merged[field] = projected.get(field) if projected.get(field) is not None else rule_hints.get(field)

    projected_keywords = projected.get("imagery_keywords", []) or []
    hint_keywords = rule_hints.get("imagery_keywords", []) or []
    combined_keywords = _dedupe_preserve_order(
        [str(item).strip().lower() for item in [*projected_keywords, *hint_keywords] if str(item).strip()]
    )
    merged["imagery_keywords"] = combined_keywords[:6]
    return merged


def project_to_vocab(parsed: Dict[str, Any], vocab: Dict[str, Any]) -> Dict[str, Any]:
    fields = vocab.get("fields", {})
    aliases = vocab.get("aliases", {})

    if parsed is None:
        parsed = {}

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

    projected: Dict[str, Any] = {}
    for field in SCALAR_FIELDS:
        projected[field] = _map_vocab_value(
            parsed.get(field),
            [str(item).strip().lower() for item in fields.get(field, [])],
            {str(k).strip().lower(): str(v).strip().lower() for k, v in aliases.get(field, {}).items()},
        )
    projected["imagery_keywords"] = cleaned_keywords
    return projected


def parse_nl_query_rule_based(
    text: str,
    vocab_path: str = "vocab.json",
) -> Dict[str, Any]:
    vocab = load_vocab(vocab_path)
    fields = vocab.get("fields", {})
    aliases = vocab.get("aliases", {})
    lowered = (text or "").strip().lower()

    hints: Dict[str, Any] = {field: None for field in SCALAR_FIELDS}
    hints["imagery_keywords"] = []

    for field in SCALAR_FIELDS:
        allowed = [str(item).strip().lower() for item in fields.get(field, [])]
        alias_map = {str(k).strip().lower(): str(v).strip().lower() for k, v in aliases.get(field, {}).items()}

        # Exact allowed value mentions first.
        for candidate in sorted(allowed, key=len, reverse=True):
            token = candidate.replace("_", " ")
            if token and token in lowered:
                hints[field] = candidate
                break
        if hints[field] is not None:
            continue

        for alias, target in alias_map.items():
            alias_token = alias.replace("_", " ")
            if alias_token and alias_token in lowered:
                hints[field] = target
                break

    # Domain-specific multi-field hints to improve recall.
    if any(token in lowered for token in ["farming", "plowing", "harvest", "耕田", "农耕", "种田", "丰收"]):
        if hints["primary_theme"] is None:
            hints["primary_theme"] = "agriculture"
        if hints["usage_scene"] is None:
            hints["usage_scene"] = "labor"

    if any(token in lowered for token in ["ritual", "ceremony", "仪式", "祭祀"]):
        if hints["usage_scene"] is None:
            hints["usage_scene"] = "ceremony"

    if any(token in lowered for token in ["love", "romance", "爱情", "恋爱", "恋歌", "情歌"]):
        if hints["primary_theme"] is None:
            hints["primary_theme"] = "courtship"

    if any(token in lowered for token in ["longing", "yearning", "missing", "思念", "想念", "相思"]):
        if hints["emotion"] is None:
            hints["emotion"] = "yearning"

    if any(token in lowered for token in ["duet", "antiphonal", "call-and-response", "对唱", "轮唱", "应答"]):
        if hints["performance_form"] is None:
            hints["performance_form"] = "antiphonal"

    if any(token in lowered for token in ["celebration", "festive", "庆典", "喜庆", "节庆"]):
        if hints["usage_scene"] is None:
            hints["usage_scene"] = "festival"

    # Lightweight imagery keyword extraction (CN + EN tokens).
    raw_tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", lowered)
    stop_tokens = {
        "find",
        "song",
        "music",
        "ethnic",
        "一首",
        "民族",
        "音乐",
        "有关",
        "适合",
        "我要",
        "找一首",
    }
    keywords = [token for token in raw_tokens if token not in stop_tokens and len(token) >= 2]
    hints["imagery_keywords"] = _dedupe_preserve_order(keywords)[:6]
    return hints


def parse_nl_query_openai(
    text: str,
    vocab_path: str = "vocab.json",
    model: str = "deepseek-chat",
    api_key_env: str = "DEEPSEEK_API_KEY",
    base_url_env: str = "DEEPSEEK_BASE_URL",
    debug: bool = False,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenAI SDK is required. Install with: pip install openai") from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        print(f"[llm-parse] {api_key_env} is not set; fallback to rule-based parser.")
        fallback = parse_nl_query_rule_based(text, vocab_path=vocab_path)
        fallback["__fallback_reason"] = f"{api_key_env} missing"
        return fallback
    base_url = os.getenv(base_url_env, "https://api.deepseek.com/v1")
    endpoint_path = "/responses"
    provider = "OpenAI-compatible SDK client"

    vocab = load_vocab(vocab_path)
    fields = vocab.get("fields", {})

    system_text = (
        "You extract a structured retrieval query for ethnic music metadata search. "
        "Return JSON that matches the schema exactly. "
        "Use only allowed vocabulary values when possible. "
        "If unsure, return null for scalar fields."
    )
    user_text = (
        "Extract a structured query from the text below.\n\n"
        f"Allowed primary_theme: {', '.join(fields.get('primary_theme', []))}\n"
        f"Allowed secondary_theme: {', '.join(fields.get('secondary_theme', []))}\n"
        f"Allowed emotion: {', '.join(fields.get('emotion', []))}\n"
        f"Allowed secondary_emotion: {', '.join(fields.get('secondary_emotion', []))}\n"
        f"Allowed usage_scene: {', '.join(fields.get('usage_scene', []))}\n"
        f"Allowed performance_form: {', '.join(fields.get('performance_form', []))}\n"
        f"Allowed tempo: {', '.join(fields.get('tempo', []))}\n"
        "Support both Chinese and English user input.\n\n"
        f"Text: {text}"
    )

    schema = {
        "type": "object",
        "properties": {
            "primary_theme": {"type": ["string", "null"]},
            "secondary_theme": {"type": ["string", "null"]},
            "emotion": {"type": ["string", "null"]},
            "secondary_emotion": {"type": ["string", "null"]},
            "usage_scene": {"type": ["string", "null"]},
            "performance_form": {"type": ["string", "null"]},
            "tempo": {"type": ["string", "null"]},
            "imagery_keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "primary_theme",
            "secondary_theme",
            "emotion",
            "secondary_emotion",
            "usage_scene",
            "performance_form",
            "tempo",
            "imagery_keywords",
        ],
        "additionalProperties": False,
    }

    print("[llm-parse] preparing remote parse request")
    print(f"[llm-parse] base_url={base_url}")
    print(f"[llm-parse] endpoint_path={endpoint_path}")
    print(f"[llm-parse] model={model}")
    print(f"[llm-parse] client_provider={provider}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "music_query",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
    except Exception as exc:  # pragma: no cover - network/runtime behavior
        print("[llm-parse] remote parsing request failed.")
        print(f"[llm-parse] exception={exc!r}")
        traceback.print_exc()
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            print(f"[llm-parse] status_code={status_code}")
        body = getattr(exc, "body", None)
        if body:
            print(f"[llm-parse] response_body={body}")
        response_obj = getattr(exc, "response", None)
        if response_obj is not None:
            raw_text = getattr(response_obj, "text", None)
            if raw_text:
                print(f"[llm-parse] response_text={raw_text}")
            try:
                json_body = response_obj.json()
                print(f"[llm-parse] response_json={json.dumps(json_body, ensure_ascii=False)}")
            except Exception:
                pass

        fallback = parse_nl_query_rule_based(text, vocab_path=vocab_path)
        fallback["__fallback_reason"] = f"remote llm call failed: {exc}"
        return fallback

    content = getattr(response, "output_text", None)
    if not content:
        print("[llm-parse] output_text missing; fallback to rule-based parser.")
        fallback = parse_nl_query_rule_based(text, vocab_path=vocab_path)
        fallback["__fallback_reason"] = "output_text missing"
        return fallback

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"[llm-parse] failed to parse model output as JSON: {content}")
        print(f"[llm-parse] exception={exc!r}")
        traceback.print_exc()
        fallback = parse_nl_query_rule_based(text, vocab_path=vocab_path)
        fallback["__fallback_reason"] = "invalid json output"
        return fallback

    projected = project_to_vocab(parsed, vocab)
    # Fill gaps with deterministic bilingual rule hints for better recall.
    rule_hints = parse_nl_query_rule_based(text, vocab_path=vocab_path)
    merged = _merge_with_rule_hints(projected, rule_hints)
    if debug:
        print("Parsed query (raw):", parsed)
        print("Parsed query (projected):", projected)
        print("Parsed query (merged):", merged)
    output = {field: merged.get(field) for field in OUTPUT_FIELDS}
    output["__fallback_reason"] = None
    return output
