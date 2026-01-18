from __future__ import annotations

from llm_parse import load_vocab, project_to_vocab


def test_project_to_vocab_aliases_and_keywords() -> None:
    vocab = load_vocab("vocab.json")
    parsed = {
        "emotion": "melancholy",
        "primary_theme": "farming",
        "secondary_theme": None,
        "tempo": None,
        "vocal_style": None,
        "performance_context": None,
        "imagery_keywords": [" Time ", "", "regret", "time"],
    }
    projected = project_to_vocab(parsed, vocab)
    assert projected["emotion"] == "sad"
    assert projected["primary_theme"] == "agriculture"
    assert projected["imagery_keywords"] == ["time", "regret"]
