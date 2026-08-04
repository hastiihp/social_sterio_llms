# Project freeze candidate — 2026-08-04

This records the inspected filesystem state. It is **not an approved freeze** because
the pre-audit repository was dirty and critical context assets were untracked.

- Git commit: `25560b27680295b392d320214873f47c66495ade`
- Python: 3.13.7
- Active pip: 25.2
- Environment status: required analysis stack is not installed in the active
  interpreter (`import pandas` fails). Intended versions are pinned in
  `requirements.txt`, notably pandas 3.0.0, numpy 2.4.1, scipy 1.17.0,
  statsmodels 0.14.6, matplotlib 3.10.8, seaborn 0.13.2, torch 2.10.0, and
  transformers 5.1.0.

## Canonical datasets/results

- Persona grid: `data/personas.csv` — 5,400 rows.
- Topics: `data/topics.csv` — 7 rows.
- Original prompts: `data/prompts.csv` — 75,600 rows.
- Health prompts: `data_health/health_prompts_full.csv` — 75,600 rows (ignored).
- Context prompts: `data_context/{neutral,positive,negative_minor}_prompts_full.csv`
  — 75,600 rows each (ignored).
- Original results: `results/full_results_{model}.csv` — 75,600 rows/model.
- Health results: `results_health/health_full_results_{model}.csv` — 75,600 rows/model.
- Context results: `results_context/{context}_full_results_{model}.csv` — 75,600
  rows/context/model.
- Main merged result: `analysis/master_results.csv` — 378,000 rows (ignored,
  regenerable from original results).
- Main analysis endpoints: `tables/`, `figures/`.
- Health/context endpoints: `analysis_health/output/`, `analysis_context/output/`.

Models are Llama, Gemma, Qwen, Ministral, and DeepSeek. Falcon-H1 is excluded because
inference could not be completed reliably; no Falcon row appears in canonical results.
DeepSeek is handled separately or excluded from pooled/ordinal/ranking inference where
its valid sample is structurally inadequate.

## Analysis rules

- Primary rating inference uses Condition A only.
- Condition B is treated as optional-abstention/descriptive and selection-sensitive.
- Main pooled inference excludes DeepSeek; DeepSeek-specific sparse results are
  exploratory/non-inferential.
- Context primary outputs use full 5,400-persona files suffixed `_full5400`, or the
  `dataset_scope=full_5400_persona` field.
- Historical pilot context outputs are unsuffixed.

## Known limitations

- Current worktree is not clean; context study and its documentation are untracked.
- Health staging duplicates canonical health datasets/results.
- Health country-label extraction still uses unsafe character-set stripping, dormant
  for its pilot-only level set.
- The context script 05 pilot dominant-factor summary mixes full original scope with
  pilot context scope; use the full-scale summary for like-for-like comparisons.
- Active runtime lacks requirements; clean execution was not tested in this audit.
- Pre-fix context raw data, old pilot/manipulation outputs, prior v9 outputs, and the
  original health audit runner are unavailable, as documented in `AUDIT_HISTORY.md`.
- Several data-generation/inference scripts depend on invocation working directory;
  `data/build_dataset.py` also hard-codes an obsolete `/home/claude/...` path.

Freeze status: **REJECTED / NO GO until blocking issues in FINAL_REPORT.md are resolved.**
