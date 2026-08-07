# Final pre-push cleanup record

Date: 2026-08-07 (Europe/Paris)

Pre-cleanup commit: `8fe660b93264acc3e2999e6766ecb39975b5c1f1`

Branch: `main`
Remote: `origin` → `https://github.com/hastiihp/social_sterio_llms.git`

This is the concise reproducibility trail for the final cleanup requested before
the report/taxonomy/unified-analysis state is committed. No inference was run and no
scientific result file was deleted or altered during cleanup.

## Pre-cleanup state

Tracked modification: `analysis/03_compliance_table.py` (documentation/comments only).
Untracked scientific work: `FULL_PROJECT_REPORT.md`, `FULL_RESULTS_WALKTHROUGH.md`,
`analysis_report/`, `analysis_taxonomy/`, `analysis_unified/`, and `figures/report/`.
Ignored local artifacts included `.DS_Store`, Python bytecode caches, the five
regenerable prompt CSVs, and `analysis/master_results.csv`.

## Cleanup classification before deletion

Counts refer to the 332 non-Git files present before cleanup.

| Classification | Count | Meaning |
|---|---:|---|
| KEEP | 255 | Canonical inputs/results, scripts, outputs, documentation, logs, and ignored regenerable scientific data |
| SAFE_TO_DELETE | 27 | OS metadata and compiled Python caches only |
| REVIEW_REQUIRED | 50 | One modified and 49 untracked scientific/report files requiring validation before staging |

### SAFE_TO_DELETE

Every entry is ignored by `.gitignore`, unreferenced by code/documentation, contains no
unique content, and can be regenerated automatically.

| Path | Size (bytes) | Git status | Code refs | Doc refs | Unique | Classification | Reason |
|---|---:|---|---|---|---|---|---|
| `.DS_Store` | 14,340 | ignored | no | no | no | SAFE_TO_DELETE | macOS metadata |
| `analysis_report/.DS_Store` | 8,196 | ignored | no | no | no | SAFE_TO_DELETE | macOS metadata |
| `figures/.DS_Store` | 8,196 | ignored | no | no | no | SAFE_TO_DELETE | macOS metadata |
| `analysis/__pycache__/01_merge_dataset.cpython-313.pyc` | 2,652 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/02_validate_dataset.cpython-313.pyc` | 9,317 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/03_compliance_table.cpython-313.pyc` | 7,198 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/04_descriptives.cpython-313.pyc` | 5,591 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/05_hypothesis_models.cpython-313.pyc` | 24,784 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/05b2_proportional_odds_by_topic.cpython-313.pyc` | 9,004 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/05b_ordinal_robustness.cpython-313.pyc` | 20,996 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/05c_topic_specific_models.cpython-313.pyc` | 23,269 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/05d_country_set_robustness.cpython-313.pyc` | 12,790 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/05e_bh_correction.cpython-313.pyc` | 6,654 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/06_abstention_analysis.cpython-313.pyc` | 20,320 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/07_cross_model_agreement.cpython-313.pyc` | 9,094 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/07b_paired_comparison.cpython-313.pyc` | 40,549 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/08_variance_ranking.cpython-313.pyc` | 16,762 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/09_deepseek_report.cpython-313.pyc` | 13,507 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/10_figures.cpython-313.pyc` | 23,075 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/_common.cpython-313.pyc` | 6,089 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis/__pycache__/_style.cpython-313.pyc` | 2,513 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis_context/__pycache__/_common.cpython-313.pyc` | 24,847 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis_health/__pycache__/01_compare_health_vs_original.cpython-313.pyc` | 23,613 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis_health/__pycache__/02_compare_by_condition.cpython-313.pyc` | 16,057 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis_health/__pycache__/03_deepseek_health_diagnosis.cpython-313.pyc` | 17,140 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis_health/__pycache__/04_ranking_robustness.cpython-313.pyc` | 27,461 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |
| `analysis_report/__pycache__/generate_report_figures.cpython-313.pyc` | 24,132 | ignored | no | no | no | SAFE_TO_DELETE | compiled cache |

### REVIEW_REQUIRED before validation

| Path | Approx. size/status | Referenced | Unique | Classification | Reason |
|---|---|---|---|---|---|
| `analysis/03_compliance_table.py` | modified tracked file | taxonomy/docs | yes | REVIEW_REQUIRED | warning added after pooled-vs-condition-specific compliance check |
| `FULL_PROJECT_REPORT.md` | untracked report | figures and all analysis layers | yes | REVIEW_REQUIRED | candidate current project report |
| `FULL_RESULTS_WALKTHROUGH.md` | untracked report | existing outputs and taxonomy | yes | REVIEW_REQUIRED | candidate results guide |
| `analysis_report/` excluding caches | untracked script | report figures | yes | REVIEW_REQUIRED | figure-generation provenance |
| `analysis_taxonomy/` | 29 untracked scientific files, ~3.3 MB | report/unified layer | yes | REVIEW_REQUIRED | six-stage derived analysis and outputs |
| `analysis_unified/` | 8 untracked scientific files, ~1.5 MB | report | yes | REVIEW_REQUIRED | unified model code, diagnostics, and results |
| `figures/report/` | 11 untracked PNGs, <1 MB total | full report | yes | REVIEW_REQUIRED | current report figures |

These REVIEW_REQUIRED items are retained. They may be reclassified KEEP only after
source/reference checks, structural validation, claim spot-checking, syntax checks,
large-file review, and secret scanning pass.

### Post-review reclassification

| Path | Size (bytes) | Git status | Code refs | Doc refs | Unique | Classification | Reason |
|---|---:|---|---|---|---|---|---|
| `analysis_unified/output/_pooled_data.parquet` | 1,442,048 | untracked | no | no | no | SAFE_TO_DELETE | orphan intermediate from an earlier implementation; current unified script builds the same 756,000-row Condition-A pool in memory from tracked canonical results and neither reads nor writes this file |

All other REVIEW_REQUIRED scientific files passed source/reference inspection and
were reclassified KEEP for final validation.

## Canonical locations confirmed before cleanup

- Personas/topics/names and regenerable prompts: `data/`
- All 25 canonical model result files: `results/results_{framing}_{model}.csv`
- Main, health, and cross-context analysis: `analysis/`, `analysis_health/`,
  `analysis_context/`
- Derived taxonomy/unified work: `analysis_taxonomy/`, `analysis_unified/`
- Tables and figures: `tables/`, `figures/`
- Audit record: `AUDIT_HISTORY.md`; removed audit scaffolding remains recoverable in
  Git history (commits documented there)

## Structural validation result

All 25 result files contain exactly 75,600 rows, 5,400 personas, 75,600 unique
persona-topic-condition keys, seven topics, both conditions, zero duplicate keys,
and zero rows marked `technical_failure`. Five deterministic persona samples
(`P00275`, `P02719`, `P02963`, `P03659`, `P00472`) matched identity/demographic
fields and complete 14-row coverage across all five prompt framings.

## Cleanup executed

- Deleted the 27 listed OS/cache files.
- Deleted `analysis_unified/output/_pooled_data.parquet` after confirming it was an
  unreferenced, superseded intermediate reproducible from tracked result files.
- Retained all logs, result CSVs, tables, figures, source programs, reports, and
  scientifically relevant audit history.
- Corrected the dormant label-extraction defect in the active health ranking script;
  no historical output was regenerated because its pilot levels were unaffected.
- Made the taxonomy BH postprocessor idempotent so the checked-in corrected outputs
  can be reproduced without the script aborting on its own derived columns.

## Final validation evidence

- 47 Python source files parsed successfully with `ast.parse`; zero syntax errors.
- 56 local Markdown links were checked; zero broken links.
- 36 README quantitative checks matched canonical CSVs.
- All 25 tracked canonical result files across the five framing families remain
  present; no byte-identical duplicate file was found outside
  empty marker files.
- The remaining `.strip("[]T.")` strings occur only in historical explanatory text;
  no active factor-label extraction uses it.
- Full repository secret scan: 305 files, zero recognized API-key/token/private-key
  signatures.
- Files over 50 MB are limited to ignored, regenerable `data/prompts_*.csv` and
  `analysis/master_results.csv`. Every intended staged file is below 50 MB; canonical
  tracked raw result files are approximately 20–45 MB each.
- The active shell's Homebrew Python lacks the analysis stack. An Anaconda interpreter
  can import pandas 3.0.0 but does not match all pinned versions, so statistical models
  were deliberately not rerun during this integrity-only cleanup. Existing outputs were
  checked structurally and mechanically against their source CSVs instead.
