from __future__ import annotations

from llm_parse import load_vocab, parse_nl_query_rule_based, project_to_vocab


def test_project_to_vocab_aliases_and_keywords() -> None:
    vocab = load_vocab("vocab.json")
    parsed = {
        "emotion": "melancholy",
        "primary_theme": "farming",
        "secondary_theme": None,
        "secondary_emotion": "longing",
        "usage_scene": "ritual",
        "performance_form": "duet singing",
        "tempo": None,
        "imagery_keywords": [" Time ", "", "regret", "time"],
    }
    projected = project_to_vocab(parsed, vocab)
    assert projected["emotion"] == "melancholic"
    assert projected["primary_theme"] == "agriculture"
    assert projected["secondary_emotion"] == "yearning"
    assert projected["usage_scene"] == "ceremony"
    assert projected["performance_form"] == "antiphonal"
    assert projected["imagery_keywords"] == ["time", "regret"]


def test_rule_parser_chinese_love_yearning_duet() -> None:
    query = "找一首和爱情有关、情绪思念、适合对唱的民族歌"
    parsed = parse_nl_query_rule_based(query, vocab_path="vocab.json")
    assert parsed["primary_theme"] == "courtship"
    assert parsed["emotion"] == "yearning"
    assert parsed["performance_form"] == "antiphonal"


def test_rule_parser_chinese_farming_harvest() -> None:
    query = "我要一首耕田的丰收歌"
    parsed = parse_nl_query_rule_based(query, vocab_path="vocab.json")
    assert parsed["primary_theme"] == "agriculture"
    assert parsed["usage_scene"] == "labor"


def test_rule_parser_chinese_ritual_solemn() -> None:
    query = "找一首仪式感强、庄重的民族音乐"
    parsed = parse_nl_query_rule_based(query, vocab_path="vocab.json")
    assert parsed["usage_scene"] == "ceremony"
    assert parsed["emotion"] == "solemn"


def test_rule_parser_english_love_longing_duet() -> None:
    query = "find an ethnic song about love, longing, and duet singing"
    parsed = parse_nl_query_rule_based(query, vocab_path="vocab.json")
    assert parsed["primary_theme"] == "courtship"
    assert parsed["emotion"] == "yearning"
    assert parsed["performance_form"] == "antiphonal"
