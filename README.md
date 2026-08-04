# Stereotype-Based Opinion Attribution in Instruction-Tuned LLMs

## Research question

Do language models attribute sociopolitical opinions to fictional people based
on country, profession, gender, and age, when the *only* information available
is a third-person "friend" description containing no behavioral evidence? And
separately: does the model's willingness to attempt that attribution at all
(vs. abstain) itself depend on those same demographics?

These are two distinct outcomes:

- **Forced opinion attribution** (Condition A): the 1-5 rating selected when a
  numeric answer is required.
- **Optional abstention** (Condition B): whether the model judges the
  demographic information sufficient to offer even a tentative estimate, or
  responds `NA`.

The design is a full factorial grid: **5,400 personas** (20 countries x 3
genders x 3 ages x 30 professions) x **7 topics** x **2 response conditions**
= 75,600 prompts per model, run against **5 models** (Llama-3.1-8B-Instruct,
Gemma-3-12B-it, Qwen3-8B, Ministral-8B-Instruct, DeepSeek-LLM-7B-chat) for the
original single-turn prompt, and again for four naturalistic multi-turn
conversational framings (see below) — 25 (prompt-type x model) result files,
75,600 rows each (378,000 rows per prompt type across its 5 models, 1,890,000
rows total across all 25 files). See `analysis_plan.md` for the full
preregistered methodology (hypotheses, exclusion criteria, robustness checks)
and `AUDIT_HISTORY.md` for five rounds of independent verification against
this data.

## Environment setup

**Create and activate a virtualenv, then install the pinned requirements:**

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

This installs the exact pinned versions this project's data was generated
and analyzed with — notably pandas 3.0.0, numpy 2.4.1, scipy 1.17.0,
statsmodels 0.14.6, matplotlib 3.10.8, seaborn 0.13.2, torch 2.10.0,
transformers 5.1.0.

**Known gotcha (macOS with Homebrew Python installed):** a bare `python3`
with no venv activated can silently resolve to Homebrew's system-wide
Python instead of your project venv — and the two can report the *identical*
Python/pip version numbers (e.g. both "3.13.7 / pip 25.2") while one has
none of `requirements.txt` installed. Version-string matching alone cannot
tell them apart; only `which python3` / checking `$VIRTUAL_ENV` can. This
exact confusion produced a false "pandas is broken" finding during this
project's audit history — see `AUDIT_HISTORY.md` Round 5 for the full
root-cause trace if `import pandas` (or any pinned package) ever fails
unexpectedly despite having activated a venv.

## Data structure

```
data/
  personas.csv                    5,400 rows -- the canonical persona grid
  topics.csv                      7 rows -- the 7 topic statements
  names.csv                       60 rows -- per country x gender name + validation tier
  prompts_original.csv            75,600 rows -- original single-turn prompt (gitignored, regenerate from source)
  prompts_health.csv              75,600 rows -- health framing (gitignored, regenerate from source)
  prompts_neutral.csv             75,600 rows -- neutral framing (gitignored)
  prompts_positive.csv            75,600 rows -- positive framing (gitignored)
  prompts_negative_minor.csv      75,600 rows -- negative_minor framing (gitignored)
  build_dataset.py                generates names/topics/personas
  render_prompts.py               generates prompts_original.csv
  render_prompts_health.py        generates prompts_health.csv
  render_prompts_context.py       generates prompts_{neutral,positive,negative_minor}.csv

results/
  results_original_{model}.csv          5 files, 75,600 rows each -- original prompt
  results_health_{model}.csv            5 files, 75,600 rows each -- health framing
  results_neutral_{model}.csv           5 files, 75,600 rows each -- neutral framing
  results_positive_{model}.csv          5 files, 75,600 rows each -- positive framing
  results_negative_minor_{model}.csv    5 files, 75,600 rows each -- negative_minor framing
  ({model} in llama, gemma, qwen, ministral, deepseek)

inference/          inference scripts (one per prompt type) + sbatch_scripts/ (cluster job specs)
analysis/            main-study analysis pipeline (reads results/results_original_*.csv)
analysis_health/      health-vs-original analysis (reads results/results_{original,health}_*.csv)
analysis_context/     all-five-prompt-types analysis (reads all of results/)
tables/, figures/     main-study output tables/figures
analysis_health/output/, analysis_context/output/   health/context output tables/figures
```

This structure is the result of a same-day reorganization consolidating what
were previously eight scattered folders (`data/`, `data_health/`,
`data_context/`, `results/`, `results_health/`, `results_context/`,
`context_staging/`, `health_staging/`) into the two above. Every file move was
row-count-verified before the old folders were deleted; see the conversation
history around 2026-08-04 for the full before/after inventory, or
`AUDIT_HISTORY.md` Round 5 for the audit that preceded it.

**All of `results/` is tracked in git; none of the five `data/prompts_*.csv`
files are.** They range from ~57MB (`prompts_original.csv`) to ~110MB
(`prompts_health.csv`) and must be regenerated from source (below) or
obtained separately. This was made consistent during the Step 5 verification
pass of this reorg — `prompts_original.csv` had been an inconsistent
exception (tracked despite being the same tier of large, regenerable source
data as the other four); it is now gitignored like the rest, keeping every
staged file under the 50MB threshold.

## The five prompt types

All five share an identical final turn (the rating question) and identical
response-condition scale instructions — only what precedes it differs, so any
rating difference is attributable to that content, not to different
instructions or scales.

| Type | Structure | What it varies | Why |
|---|---|---|---|
| **original** | single-turn | Just the persona description + topic + question | The baseline: does demographic-only information produce attributed opinions at all? |
| **health** | 2-turn conversation + rating turn | Persona is stressed and sleep-deprived (personal vulnerability, internal to them) | Tests whether a "vulnerable" framing shifts attribution vs. the neutral baseline |
| **negative_minor** | 2-turn conversation + rating turn | Persona had a flight delay / lost luggage (external, impersonal event) | Tests an external adverse event *without* implying anything about the persona's internal state — the vulnerability/externality contrast with `health` |
| **positive** | 2-turn conversation + rating turn | Persona got a promotion (career success / competence signal) | Tests whether a competence/status signal shifts attribution, not just "positive mood" |
| **neutral** | 2-turn conversation + rating turn | Persona moved apartments (domain-neutral small talk, no adversity or achievement) | The multi-turn structural control — isolates "is it multi-turn at all" from "what does the conversation contain" |

**Important:** these four framings vary along several independent dimensions
at once (vulnerability vs. competence vs. externality vs. domain-neutrality) —
they are not points on a single positive-to-negative valence scale. See
`CONTEXT_EXPERIMENT.md`'s "Terminology note" for the full rationale; nowhere
in this codebase are they treated as an ordered axis.

Exact scripted text for all four framings, and the full cross-context findings
(rating shifts, abstention stability, cross-context clustering, the H1
variance-decomposition comparison), are in `CONTEXT_EXPERIMENT.md`. The
health-vs-original-only findings (a subset, audited independently first) are
in `analysis_health/output/CORRECTED_SUMMARY.md`.

## Models

**Five models are used throughout:** Llama-3.1-8B-Instruct, Gemma-3-12B-it,
Qwen3-8B, Ministral-8B-Instruct, DeepSeek-LLM-7B-chat. All bf16, deterministic
generation (`do_sample=False`, `temperature=0`, `top_p=1`,
`repetition_penalty=1.0`).

**Falcon-H1-7B is excluded** — a reproducible tensor-shape error in its
generation cache under batched inference (confirmed across batch sizes and
padding strategies; only ran cleanly at batch size 1, not viable for the full
run's runtime budget). This is an inference-environment failure, not a
data-quality or model-behavior finding — no conclusions are drawn about
Falcon's actual behavior. See `FALCON_EXCLUSION.md`. There is no Falcon row,
column, or placeholder anywhere in `results/` or downstream outputs.

**Model-specific quirks to know before touching inference or analysis code:**

- **Qwen3's "thinking mode" is explicitly disabled** (`enable_thinking=False`
  passed in the chat-template kwargs) in every inference script. Without this,
  Qwen3 emits `<think>...</think>` reasoning traces before its answer, which
  would break the strict 1-5/`NA` output-format parsing this study relies on.
- **DeepSeek's format non-compliance is severe and is treated as a per-model
  finding, not a pipeline defect** (per `analysis/09_deepseek_report.py`'s
  docstring — note that script cites this rule as "`analysis_plan.md` Section
  15," which does not exist in the current 14-section `analysis_plan.md`; a
  small pre-existing documentation inconsistency, left as-is rather than
  silently invented into a matching section). Strict-format compliance on the original
  prompt is 63/75,600 = 0.08%. A compact-text parser recovers a further slice
  of responses as containing a plausible rating beyond strict parsing (14.86%
  on the health condition, all recovered ratings anomalously the digit "4",
  flagged not explained). DeepSeek is **excluded from every pooled, ordinal,
  ranking, and H1 model fit** across `analysis/`, `analysis_health/`, and
  `analysis_context/` — its diagnostics are always reported separately
  (`analysis/09_deepseek_report.py`,
  `analysis_health/03_deepseek_health_diagnosis.py`,
  `analysis_context/04_deepseek_cross_context_diagnosis.py`). Never silently
  drop DeepSeek from a *raw* results file or re-include it in a pooled fit
  without checking why it was excluded there in the first place.

## Regenerating data from source

Run from within `data/` (all scripts read/write siblings in that directory):

```bash
cd data/
python3 build_dataset.py              # -> names.csv, topics.csv, personas.csv
python3 render_prompts.py             # -> prompts_original.csv (75,600 rows)
python3 render_prompts_health.py      # -> prompts_health.csv (75,600 rows)
python3 render_prompts_context.py neutral          # -> prompts_neutral.csv
python3 render_prompts_context.py positive         # -> prompts_positive.csv
python3 render_prompts_context.py negative_minor   # -> prompts_negative_minor.csv
```

`render_prompts_context.py` takes the context name as its one required
argument; running it with no argument (or an unrecognized one) prints usage
and exits.

## Running inference

All inference scripts assume `inference/` as the working directory (they
resolve `../data/...` and `../results/...` relatively) and require a GPU node
— see `inference/sbatch_scripts/` for the exact SLURM resource specs used
(`--gres=gpu:1`, `--mem=48G`, `--cpus-per-task=4`; original/context runs took
16-44 hours per job depending on scope). **Those `.sbatch` scripts are
preserved as historical provenance of the commands that actually produced the
checked-in data** — they reference the pre-reorg `context_staging/`/
`health_staging/` working directories and are not directly re-runnable
against the current `data/`+`results/` layout; use the commands below instead
for any future run, and treat the `.sbatch` files only as a reference for GPU
resource settings and conda environment (`paper_env`,
`HF_HOME=/mnt/beegfs/projects/ttessllm/hf_cache` on the original cluster).

```bash
cd inference/

# Original prompt, one model at a time:
python3 full_inference.py llama
python3 full_inference.py gemma
python3 full_inference.py qwen
python3 full_inference.py deepseek
python3 full_inference.py ministral
# -> ../results/results_original_{model}.csv

# Health framing:
python3 full_health_inference.py llama    # ... and gemma, qwen, deepseek, ministral
# -> ../results/results_health_{model}.csv

# Context framings (neutral / positive / negative_minor), one (context, model) pair at a time:
python3 context_full_inference.py neutral llama       # ... and gemma, qwen, deepseek, ministral
python3 context_full_inference.py positive llama
python3 context_full_inference.py negative_minor llama
# -> ../results/results_{context}_{model}.csv

# Smoketest mode (200-row subset, sanity check before a full run):
python3 context_full_inference.py neutral llama --smoketest
# -> writes to the system temp directory, not into results/; disposable.
```

All 25 (prompt-type x model) full runs already completed and verified are
checked into `results/` — these scripts are for re-running or extending, not
a required step to reproduce the existing analysis.

`inference/falconinferance.py` and `inference/pilot_inference.py` are
historical/superseded (Falcon exclusion provenance and the pre-full-run pilot,
respectively) — kept for provenance, not part of the current pipeline.

## Running the analysis pipeline

Three independent pipelines, each reading raw `results/` files directly (none
of them read each other's outputs except where noted).

**1. Main study** (`analysis/`, reads `results/results_original_*.csv`) — run
in this exact order; `05e` must run **last**, after both `05` and `06`, or it
silently loses its BH-correction columns the next time `05`/`06` are re-run
(this exact failure mode happened once — see `AUDIT_HISTORY.md` Round 2):

```bash
cd analysis/
python3 01_merge_dataset.py          # -> master_results.csv (everything below reads this)
python3 02_validate_dataset.py       # structural check, console output only
python3 03_compliance_table.py
python3 04_descriptives.py
python3 05_hypothesis_models.py      # must run before 05e
python3 05b_ordinal_robustness.py --model llama    # ... and gemma, qwen, ministral
python3 05b2_proportional_odds_by_topic.py
python3 05c_topic_specific_models.py
python3 05d_country_set_robustness.py
python3 06_abstention_analysis.py    # must run before 05e
python3 05e_bh_correction.py         # MUST BE LAST -- rewrites 05/06's outputs in place
python3 07_cross_model_agreement.py
python3 07b_paired_comparison.py
python3 08_variance_ranking.py       # -> tables/variance_ranking.csv, needed by analysis_context/05
python3 09_deepseek_report.py
python3 10_figures.py                # -> figures/fig2-fig7 (reads outputs of 05, 05e, 07, 08)
```

**2. Health-vs-original** (`analysis_health/`, self-contained — reads
`results/results_{original,health}_*.csv` directly, does not touch
`analysis/master_results.csv`): scripts `01`-`04` have no order dependency on
each other.

```bash
cd analysis_health/
python3 01_compare_health_vs_original.py
python3 02_compare_by_condition.py
python3 03_deepseek_health_diagnosis.py
python3 04_ranking_robustness.py
```

**3. All-five-prompt-types** (`analysis_context/`, reads all of `results/`):
scripts `01`-`04` are independent of each other and of the main pipeline.
**Script `05` is the one exception with a real cross-pipeline dependency** —
it reads `tables/variance_ranking.csv`, so `analysis/08_variance_ranking.py`
must have already run.

```bash
cd analysis_context/
python3 01_compare_context_vs_original.py health neutral positive negative_minor
python3 02_cross_context_agreement.py
python3 03_abstention_stability_across_conditions.py
python3 04_deepseek_cross_context_diagnosis.py
python3 05_variance_ranking_all_prompt_types.py   # requires analysis/08 to have run first
```

## Where outputs live

- `tables/` (56 CSVs) + `figures/` (6 PNGs) — main-study outputs, see
  `MANIFEST.md` for the exact script -> output mapping.
- `analysis_health/output/` — health-vs-original tables +
  `analysis_health/output/CORRECTED_SUMMARY.md`.
- `analysis_context/output/` — all-five-prompt-types tables and figures, at
  both pilot (180-persona) and full (5,400-persona) scope; see
  `CONTEXT_EXPERIMENT.md` for which scope is primary (full-scale) and which is
  a labeled historical appendix (pilot-scale).

## Audit trail

Five independent audit rounds have been run against this project over its
lifetime — adversarial validation of the health study, full-project
traceability/reproduction, adversarial validation of the context study, an
independent check of the grammar-bug fix and re-run, and a pre-Stage-1-5
repository-readiness preflight. Findings, fixes applied, and each round's
final verdict are in **`AUDIT_HISTORY.md`**; the raw scaffolding for all five
rounds has been removed from the working tree but is fully recoverable via
`git log --all -- <path>` (see that document for the exact commits).

## Known limitations

Stated plainly, not left implicit:

- **The paraphrase-robustness check was never run.** `analysis_plan.md`
  Section 8 predefines it (two minimal paraphrases of the prompt opener, on a
  matched persona subset, reporting rank correlation of effects across
  paraphrases) specifically because a single frozen prompt wording cannot by
  itself demonstrate that findings reflect underlying model behavior rather
  than this exact phrasing. This check was never executed. Every finding in
  this repository should be read with that caveat: robustness to prompt
  wording specifically has not been tested, only robustness to conversational
  framing (the five prompt types) and to scale (pilot vs. full).
- **Falcon-H1 is excluded** (inference-environment failure, not a finding
  about the model) — this removes the study's only UAE/MENA-origin model, a
  real limitation on any regional-comparison claim.
- **DeepSeek's inferential value is minimal** on the original prompt (0.08%
  strict compliance) and it is excluded from every pooled/ordinal/ranking/H1
  fit throughout; its own diagnostic reports are exploratory, not inferential.
- **Name-validity limitation** (state verbatim in any derived paper, per
  `analysis_plan.md` Section 13): each country-by-gender condition is
  represented by a single name, so country/gender effects may partly reflect
  name-specific associations rather than the demographic category alone.
- **Historical raw data gaps**: the prior v9 experiment, original pilot
  outputs, manipulation-check outputs, prompt-reinforcement test outputs, and
  the pre-grammar-fix raw `neutral`/`positive` result files are unrecoverable
  — code/mentions exist but no raw outputs remain. Documented as gaps, not
  silently assumed reproducible. See `AUDIT_HISTORY.md` Rounds 2 and 4.
- **The non-binary-condition manipulation check** (does the they/them +
  name-only signal reliably register as non-binary to the model, per
  `analysis_plan.md` Section 12) was planned but its execution/outcome is not
  independently confirmed in this repository's current state.
- **`analysis_health/04_ranking_robustness.py`'s country-label extraction**
  uses an unsafe character-set `.strip("[]T.")` (dormant for its pilot-only
  4-country level set, but would silently mangle a country name starting/
  ending in `[`, `]`, `T`, or `.` — e.g. "Turkey" — if this module's scope
  ever expanded to the full 20-country design). The equivalent bug in
  `analysis_context/_common.py` was found and fixed during the full-scale
  extension (see `CONTEXT_EXPERIMENT.md`); this one was left as out-of-scope,
  per `AUDIT_HISTORY.md` Round 5.
