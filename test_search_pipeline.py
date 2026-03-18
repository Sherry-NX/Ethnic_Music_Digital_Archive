from __future__ import annotations

import tempfile
from pathlib import Path

from search_pipeline import (
    CODEBASE_DIR,
    DEFAULT_MASTER_METADATA_PATH,
    create_search_engine,
    resolve_metadata_path,
)


def test_resolve_metadata_path_defaults_to_master() -> None:
    path = resolve_metadata_path()
    assert path == DEFAULT_MASTER_METADATA_PATH


def test_resolve_metadata_path_relative_to_source_file() -> None:
    path = resolve_metadata_path("data/metadata/metadata_master.csv")
    assert path == (CODEBASE_DIR / "data" / "metadata" / "metadata_master.csv").resolve()


def test_search_returns_ranked_results_with_reasons_and_media() -> None:
    csv_content = (
        "track_id,title,ethnic_group,subgroup,region,primary_theme,secondary_theme,emotion,"
        "secondary_emotion,usage_scene,imagery_keywords,performance_form,tempo,retrieval_notes,"
        "audio_path,video_path\n"
        "A1,Mountain Love Song,Hani,,Honghe,courtship,nature,joyful,yearning,festival,"
        "mountain;love;night,antiphonal,medium,good for courtship search,a1.wav,a1.mp4\n"
        "A2,Ritual Chant,Dong,,Liping,ritual,community,solemn,peaceful,ceremony,"
        "drum_tower;ancestor;night,choral,slow,good for ritual search,,\n"
    )
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".csv", delete=True) as f:
        f.write(csv_content)
        f.flush()
        engine = create_search_engine(f.name)
        results = engine.search(query_text="Hani mountain love", top_k=2, min_score=1)

    assert len(results) >= 1
    assert results[0]["id"] == "A1"
    assert "match_reasons" in results[0]
    assert results[0]["audio_path"] == "a1.wav"
    assert results[0]["video_path"] == "a1.mp4"


def test_search_handles_missing_media_paths() -> None:
    csv_content = (
        "track_id,title,ethnic_group,subgroup,region,primary_theme,secondary_theme,emotion,"
        "secondary_emotion,usage_scene,imagery_keywords,performance_form,tempo,retrieval_notes,"
        "audio_path,video_path\n"
        "A2,Ritual Chant,Dong,,Liping,ritual,community,solemn,peaceful,ceremony,"
        "drum_tower;ancestor;night,choral,slow,good for ritual search,,\n"
    )
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".csv", delete=True) as f:
        f.write(csv_content)
        f.flush()
        engine = create_search_engine(f.name)
        results = engine.search(query_text="ritual", top_k=2, min_score=1)

    assert len(results) == 1
    assert results[0]["audio_path"] == ""
    assert results[0]["video_path"] == ""
