## Metadata-first retrieval demo

Run the demo:
- `python demo_search.py`
- `python demo_search.py data/metadata/metadata_master.csv`

The default demo database is `data/metadata/metadata_master.csv` (single source of truth).
The CLI runs built-in natural-language queries and prints ranked matches with reasons.

## Controlled vocabulary

We use a controlled vocabulary for optional LLM query parsing.
The vocabulary lives in `vocab.json` at the repo root.
To extend it safely, prefer adding entries to `aliases` before introducing new canonical labels, and update metadata/tests when you add new canonical values.

## LLM Query Parsing (Natural Language → Structured Query)

This mode uses an LLM only to convert a natural-language prompt into a structured query.
The parsed `emotion` and `primary_theme` are projected onto `vocab.json` canonical values; unknowns become null.

Run with:
- `export DEEPSEEK_API_KEY=your_key`
- `python demo_search.py data/metadata/metadata_master.csv --nl "joyful new year festival songs"`

## Smoke-test queries

Use these quick queries for sanity checks:
- `flower`
- `sad`
- `flower sad`
- `wedding festive`
- `Yi dance`

## Streamlit demo (metadata-first)

Run:
- `cd codebase`
- `pip install -r requirements.txt`
- `streamlit run app.py`

Demo flow:
- Input a natural-language query.
- Optional LLM parsing can provide structured hints.
- Run metadata-first field matching and weighted scoring.
- Show ranked results with short match reasons.
- Expose `audio_path` and `video_path` when present.

Configuration:
- Metadata path is configurable in sidebar (defaults to `data/metadata/metadata_master.csv`).
- Vocabulary path is configurable in sidebar (defaults to `vocab.json`).
- Audio directory is configurable in sidebar (defaults to `data/audio/wav`).
- Video directory is configurable in sidebar (defaults to `data/video/mp4`).

Search fields:
- `title`
- `ethnic_group`
- `subgroup`
- `region`
- `primary_theme`
- `secondary_theme`
- `emotion`
- `secondary_emotion`
- `usage_scene`
- `imagery_keywords`
- `performance_form`
- `tempo`
- `retrieval_notes`

## Architecture note

The search layer is designed for easy data-source replacement:

- `MetadataRepository`: an interface-like protocol that defines `load_records()`.
- `CSVMetadataRepository`: current implementation that loads `data/metadata/metadata_master.csv`.
- `MetadataSearchEngine`: applies token/field matching, weighted scoring, ranking, and explanations.

UI call pattern today:
- `app.py` -> `create_search_engine(...)` -> `engine.search(...)`

To switch to a future database:
- add a new repository (for example `PostgresMetadataRepository`) implementing `load_records()`
- keep `MetadataSearchEngine.search(...)` and UI call pattern unchanged
- only change repository wiring in engine construction

## Data folder structure

All runtime demo assets are placed under `codebase/data/`:

- `data/metadata/`: metadata CSV/XLSX files
- `data/audio/wav/`: WAV assets used for optional playback
- `data/video/mp4/`: MP4 assets for future demo extensions

Recommended locations:
- Put new editable metadata files in `data/metadata/`.
- Put corresponding audio clips in `data/audio/wav/` using `id.wav` naming when possible.
- Put related video files in `data/video/mp4/`.