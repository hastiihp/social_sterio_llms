# Stereotype-Based Opinion Attribution in Instruction-Tuned LLMs

## Project overview

This research project tests whether instruction-tuned language models generate
different sociopolitical ratings for fictional personas described only by country,
profession, gender, and age. It separately studies whether models are willing to
answer at all when given the option to return `NA`.

The outcomes are therefore:

1. **Attributed ratings**: a generated 1–5 estimate under forced responding.
2. **Response willingness**: answering versus abstaining under optional responding.

These outputs describe model behavior under specific prompts. They do not reveal
models' internal beliefs or establish attitudes held by real demographic groups.

## Experimental design

The full factorial persona grid contains:

- 20 countries
- 30 professions
- 3 genders (`female`, `male`, `neutral`)
- 3 ages (25, 45, 65)
- 5,400 personas total

Each persona is crossed with seven sociopolitical topics and two response
conditions:

- **A_forced**: exactly one rating from 1 to 5.
- **B_optional**: one rating from 1 to 5, or `NA`.

This produces 5,400 × 7 × 2 = **75,600 prompts per model per framing**.

Five prompt framings were run:

| Framing | Structure | Preceding context |
|---|---|---|
| `original` | Single-turn | Direct friend description and rating task |
| `health` | Multi-message | Stress and sleep difficulty |
| `neutral` | Multi-message | Moving apartments |
| `positive` | Multi-message | A work promotion |
| `negative_minor` | Multi-message | Flight delay and lost luggage |

The four conversational framings use the same final sociopolitical rating task as
the original. They differ along several semantic dimensions—vulnerability,
competence/status, external adversity, and domain-neutral small talk—and should not
be interpreted as a one-dimensional emotional-valence manipulation.

## Models

Five models completed all five framings:

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-12b-it`
- `Qwen/Qwen3-8B`
- `mistralai/Ministral-8B-Instruct-2410`
- `deepseek-ai/deepseek-llm-7b-chat`

DeepSeek was run, but showed near-total strict-format non-compliance on the
original prompt: only 63 of 75,600 outputs were strictly valid. Its malformed or
salvageable prose is not treated as equivalent to clean numeric ratings. DeepSeek
is primarily reported as a compliance finding and is excluded from pooled,
ordinal, ranking, and unified analyses that require reliable numeric ratings.

Falcon-H1-7B was attempted and excluded after a reproducible inference/cache
failure made the full run infeasible. This was an environment/inference problem,
not a finding about Falcon's sociopolitical behavior. See
[FALCON_EXCLUSION.md](FALCON_EXCLUSION.md).

## Repository structure

```text
data/                    Canonical personas, topics, names, prompt renderers;
                         generated prompt CSVs are ignored and reproducible
results/                 25 canonical inference-result CSVs
inference/               Inference programs and retained SLURM provenance
analysis/                Main original-framing pipeline
analysis_health/         Health-versus-original analyses and outputs
analysis_context/        Five-framing comparisons; pilot and full-scale outputs
analysis_taxonomy/       Six-stage derived behavioral taxonomy
analysis_unified/        Full-scale unified mixed-effects model and diagnostics
analysis_report/         Report-figure generator
tables/                  Main-study processed tables
figures/                 Main and report figures
logs/                    Retained inference execution logs
```

Important top-level documentation:

- [analysis_plan.md](analysis_plan.md): original analysis plan and stated checks.
- [CONTEXT_EXPERIMENT.md](CONTEXT_EXPERIMENT.md): conversational-framing design,
  corrections, and full-versus-pilot findings.
- [FULL_PROJECT_REPORT.md](FULL_PROJECT_REPORT.md): current integrated report.
- [FULL_RESULTS_WALKTHROUGH.md](FULL_RESULTS_WALKTHROUGH.md): numbers-first guide.
- [MANIFEST.md](MANIFEST.md): main script-to-output map.
- [AUDIT_HISTORY.md](AUDIT_HISTORY.md): concise history of verification rounds and
  scientifically relevant corrections.
- [PRE_PUSH_CLEANUP.md](PRE_PUSH_CLEANUP.md): final cleanup and integrity record.

## Canonical data and results

The single canonical locations are:

- Persona definitions: `data/personas.csv` (5,400 rows)
- Topics: `data/topics.csv` (7 rows)
- Names: `data/names.csv` (60 rows)
- Generated prompts: `data/prompts_{framing}.csv` (75,600 rows each, ignored)
- Inference results: `results/results_{framing}_{model}.csv`
  (25 files, 75,600 rows each)
- Main merged analysis file: `analysis/master_results.csv` (ignored and regenerated
  from the five original result files)

All result files are tracked. Generated prompt files are 57–110 MB and intentionally
ignored because their renderers and canonical metadata reproduce them. Result rows
retain the prompt-template version and persona/topic/condition keys, but not the full
prompt text; regenerate `data/prompts_{framing}.csv` when exact rendered text is
needed.

## Analysis pipeline

The repository contains five related analysis layers:

- **Main pipeline (`analysis/`)**: structural validation, compliance, descriptive
  distributions, demographic hypothesis models, ordinal and country-set robustness,
  topic-specific models, abstention, paired A/B comparisons, cross-model agreement,
  variance/partial-R² ranking, DeepSeek diagnostics, and figures.
- **Health pipeline (`analysis_health/`)**: health-versus-original matched
  comparisons, condition-specific shifts, DeepSeek diagnostics, and profession/country
  ranking robustness.
- **Context pipeline (`analysis_context/`)**: all-five-framing comparisons,
  abstention stability, cross-framing agreement, ranking robustness, and variance
  ranking. Full-scale outputs use `_full5400`; unsuffixed outputs are the retained
  180-persona pilot analyses.
- **Behavioral taxonomy (`analysis_taxonomy/`)**: model, country/profession,
  topic-specific, context-sensitivity, stability, and consensus summaries derived
  from existing canonical outputs.
- **Unified model (`analysis_unified/`)**: a Condition-A, full-scale mixed-effects
  model pooling five framings and the four reliably compliant models, with diagnostics,
  variance decomposition, framing-by-model tests, and BH correction.

The main pipeline has an important run-order constraint: run
`analysis/05e_bh_correction.py` after both `05_hypothesis_models.py` and
`06_abstention_analysis.py`, because it adds adjusted columns to their outputs.
Detailed commands and dependencies are documented in script docstrings,
[MANIFEST.md](MANIFEST.md), and the analysis-layer summaries.

No table or figure reads from a staging directory. The old health/context staging
trees were removed after canonical files were consolidated into `data/`, `results/`,
and `inference/`.

## Current high-level findings

The following are descriptive summaries of verified outputs, not causal claims:

- Response willingness differs sharply by model. Under the original optional
  condition, Llama and Gemma never abstain, while Qwen and Ministral abstain on most
  opportunities.
- Preceding conversational context is associated with large changes in abstention
  for some models. Ministral's Condition-B abstention rate declines from 83.47% in
  the original framing to 29.80% in `negative_minor`.
- Demographic ranking structure is often more stable across framings than absolute
  rating levels, although stability varies by model and factor.
- In the unified Condition-A model, topic explains substantially more rating
  variation than any individual demographic factor.
- DeepSeek mainly contributes evidence about strict response-format compliance;
  its sparse valid ratings are not interpreted like the other four models' data.

See [FULL_PROJECT_REPORT.md](FULL_PROJECT_REPORT.md) for qualified results and source
citations rather than extending these headline statements beyond their evidence.

## Reproducibility

Create an isolated environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
python -m pip install -r requirements.txt
```

Prompt generation does not require model inference:

```bash
cd data
python build_dataset.py
python render_prompts.py
python render_prompts_health.py
python render_prompts_context.py neutral
python render_prompts_context.py positive
python render_prompts_context.py negative_minor
```

The expensive inference step does not need to be repeated to reproduce current
analyses: all 25 raw result files are already under `results/`. Inference programs
remain in `inference/` for provenance and extension. Analysis programs write to
`tables/`, `figures/`, or their pipeline-local `output/` directories.

Before relying on a regenerated result, preserve the distinction between Condition A
and B, between full-scale and pilot outputs, and between strict-valid and
salvageable/malformed responses. The repository records several earlier errors where
those distinctions mattered in [AUDIT_HISTORY.md](AUDIT_HISTORY.md).

## Important limitations

- The study covers five successfully run models; Falcon's exclusion reduces model
  and regional coverage.
- DeepSeek's near-total strict-format non-compliance makes its rating-based inference
  extremely limited.
- Generated ratings do not establish a model's internal beliefs or real-world group
  attitudes.
- Each country × gender cell uses one name, so country or gender effects may partly
  reflect name-specific associations.
- Robustness to minimal prompt paraphrases was planned but not run.
- The neutral-gender manipulation check is not independently recoverable in the
  current repository state.
- The conversational framings differ along multiple dimensions, limiting simple
  causal interpretation of any context contrast.
- Historical pilot, manipulation-check, prompt-reinforcement, pre-grammar-fix, and
  prior-v9 raw artifacts are incomplete, as documented in `AUDIT_HISTORY.md`.

## Project status

This is an active research/internship project. The full-scale experiment and current
analysis layers are present, but the report/manuscript is still being developed and
should not be described as a finished publication.
