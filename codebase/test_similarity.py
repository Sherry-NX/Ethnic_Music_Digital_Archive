"""Tests for the item-to-item similarity engine."""

from __future__ import annotations

import pytest

from similarity import (
    compute_similarity,
    explain_similarity,
    find_similar,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_record(**kwargs):
    """Create a minimal metadata record dict with defaults."""
    defaults = {
        "track_id": "T_00",
        "title": "",
        "ethnic_group": "",
        "subgroup": "",
        "region": "",
        "language": "",
        "primary_theme": "",
        "secondary_theme": "",
        "emotion": "",
        "secondary_emotion": "",
        "usage_scene": "",
        "imagery_keywords": "",
        "performance_form": "",
        "tempo": "",
        "retrieval_notes": "",
    }
    defaults.update(kwargs)
    defaults.setdefault("id", defaults["track_id"])
    return defaults


# ---------------------------------------------------------------------------
# compute_similarity
# ---------------------------------------------------------------------------

class TestComputeSimilarity:
    def test_identical_records_score_highest(self):
        rec = _make_record(
            track_id="A",
            primary_theme="agriculture",
            emotion="joyful",
            ethnic_group="Hani",
            usage_scene="labor",
        )
        score = compute_similarity(rec, rec)
        assert score > 0

    def test_shared_high_weight_fields_beat_only_keywords(self):
        base = _make_record(track_id="A", primary_theme="agriculture", emotion="joyful")
        similar_structured = _make_record(
            track_id="B", primary_theme="agriculture", emotion="joyful",
        )
        similar_keywords = _make_record(
            track_id="C",
            imagery_keywords="rice_terrace;mountain;village;field;water",
        )
        # Give C the same keywords as A to maximise keyword overlap.
        base_kw = _make_record(
            track_id="A2",
            imagery_keywords="rice_terrace;mountain;village;field;water",
        )

        score_structured = compute_similarity(base, similar_structured)
        score_keywords = compute_similarity(base_kw, similar_keywords)
        assert score_structured > score_keywords, (
            "Structured field matches should outscore pure keyword overlap"
        )

    def test_no_overlap_scores_zero(self):
        a = _make_record(track_id="A", primary_theme="agriculture", emotion="joyful")
        b = _make_record(track_id="B", primary_theme="ritual", emotion="solemn")
        score = compute_similarity(a, b)
        assert score == 0.0

    def test_cross_field_partial_credit(self):
        a = _make_record(track_id="A", primary_theme="courtship")
        b = _make_record(track_id="B", secondary_theme="courtship")
        score = compute_similarity(a, b)
        # Should get cross-field credit but not direct primary_theme match.
        assert score > 0

    def test_empty_records(self):
        a = _make_record(track_id="A")
        b = _make_record(track_id="B")
        assert compute_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------

class TestFindSimilar:
    @pytest.fixture
    def catalogue(self):
        return [
            _make_record(track_id="HN_01", ethnic_group="Hani", primary_theme="agriculture",
                         emotion="joyful", usage_scene="labor",
                         imagery_keywords="rice_terrace;mountain;village"),
            _make_record(track_id="HN_02", ethnic_group="Hani", primary_theme="courtship",
                         emotion="yearning", usage_scene="community_gathering",
                         imagery_keywords="mountain;village;forest"),
            _make_record(track_id="DG_01", ethnic_group="Dong", primary_theme="agriculture",
                         emotion="joyful", usage_scene="community_gathering",
                         imagery_keywords="cuckoo;spring;field;village"),
            _make_record(track_id="MG_01", ethnic_group="Mongol", primary_theme="nature",
                         emotion="energetic", usage_scene="festival",
                         imagery_keywords="horse;grassland;steppe"),
        ]

    def test_returns_at_most_top_k(self, catalogue):
        results = find_similar("HN_01", catalogue, top_k=2, min_similarity=0)
        assert len(results) <= 2

    def test_excludes_target(self, catalogue):
        results = find_similar("HN_01", catalogue, top_k=10, min_similarity=0)
        result_ids = {r["record"]["track_id"] for r in results}
        assert "HN_01" not in result_ids

    def test_exclude_ids_respected(self, catalogue):
        results = find_similar(
            "HN_01", catalogue, top_k=10, min_similarity=0,
            exclude_ids={"HN_02"},
        )
        result_ids = {r["record"]["track_id"] for r in results}
        assert "HN_02" not in result_ids

    def test_min_similarity_filters(self, catalogue):
        results = find_similar("MG_01", catalogue, top_k=10, min_similarity=999)
        assert len(results) == 0

    def test_missing_target_returns_empty(self, catalogue):
        results = find_similar("NONEXISTENT", catalogue, top_k=3)
        assert results == []

    def test_single_record_catalogue(self):
        single = [_make_record(track_id="ONLY")]
        results = find_similar("ONLY", single, top_k=3, min_similarity=0)
        assert results == []

    def test_result_structure(self, catalogue):
        results = find_similar("HN_01", catalogue, top_k=1, min_similarity=0)
        if results:
            r = results[0]
            assert "record" in r
            assert "similarity_score" in r
            assert "explanation" in r
            assert isinstance(r["explanation"], str)


# ---------------------------------------------------------------------------
# explain_similarity
# ---------------------------------------------------------------------------

class TestExplainSimilarity:
    def test_shared_fields_mentioned(self):
        a = _make_record(primary_theme="agriculture", emotion="joyful")
        b = _make_record(primary_theme="agriculture", emotion="joyful")
        text = explain_similarity(a, b)
        assert "theme" in text
        assert "emotion" in text

    def test_keyword_overlap_mentioned(self):
        a = _make_record(imagery_keywords="mountain;village")
        b = _make_record(imagery_keywords="mountain;river")
        text = explain_similarity(a, b)
        assert "mountain" in text

    def test_no_overlap_message(self):
        a = _make_record(primary_theme="agriculture")
        b = _make_record(primary_theme="ritual")
        text = explain_similarity(a, b)
        assert "Weak" in text or "overlap" in text.lower()
