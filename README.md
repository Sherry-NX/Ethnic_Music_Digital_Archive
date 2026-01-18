## Rule-based retrieval demo

Run the demo:
- `python demo_search.py metadata_template.csv`

This command runs three built-in queries:
1) `{"emotion":"sad","primary_theme":"aging","imagery_keywords":["time","regret"]}`
2) `{"emotion":"joyful","primary_theme":"festival","tempo":"fast"}`
3) `{"primary_theme":"ritual","secondary_theme":"agriculture","tempo":"slow","imagery_keywords":["terrace","rice"]}`

## Controlled vocabulary

We use a controlled vocabulary to keep labels consistent for retrieval and future LLM query parsing.
The vocabulary lives in `vocab.json` at the repo root.
To extend it safely, prefer adding entries to `aliases` before introducing new canonical labels, and update metadata/tests when you add new canonical values.