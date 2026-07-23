# Stereotype-Based Opinion Attribution in Instruction-Tuned LLMs

A study of whether language models attribute sociopolitical opinions to fictional
personas based on demographics (country, profession, gender, age) when given only a
third-person "friend" description with no behavioral evidence -- and whether models'
willingness to attempt that attribution at all (vs. abstain) itself depends on the
same demographics.

**Status:** Data collection and the full analysis pipeline are complete. This is not
a pre-pilot or draft repository -- `analysis_plan.md` is the frozen, authoritative
methodology document (see its own header for amendment rules), and every script in
`analysis/` has been run end-to-end against the final dataset.

## Dataset

Five models, one frozen prompt template, deterministic generation
(`do_sample=false`, see `analysis_plan.md` Section 14):

- Llama-3.1-8B-Instruct
- Gemma-3-12B-it
- Qwen3-8B
- Ministral-8B-Instruct
- DeepSeek-LLM-7B-chat

**Falcon-H1-7B is excluded** -- a reproducible inference failure prevented data
collection under the available software environment. See `FALCON_EXCLUSION.md` for
the full explanation. It was not replaced or substituted; every reference to "all
models" or "the five models" in this repository means exactly the five above.

`analysis/master_results.csv` (378,000 rows: 5 models x 75,600 rows each -- 5,400
personas x 7 topics x 2 response conditions) is the single source of truth for every
analysis script. It is built by `analysis/01_merge_dataset.py` from the per-model
files in `results/` and should never be bypassed by reading those files directly
(see that script's docstring for a pandas NA-handling pitfall it specifically guards
against).

## Repository layout

```
analysis_plan.md       Frozen pre-registered methodology (read this first)
FALCON_EXCLUSION.md     Why Falcon-H1 isn't in the dataset
MANIFEST.md             Which script produces which table/figure
requirements.txt        Full pinned dependency freeze
data/                   Design files (personas, topics, prompts) and generation scripts
results/                Raw per-model model outputs (5 CSVs)
analysis/               All analysis scripts (see below) + master_results.csv
tables/                 Every table each script produces (CSV)
figures/                Figures 2-7 (PNG)
inference/, logs/       Generation-run artifacts
```

## Analysis pipeline (run order)

Every script reads `analysis/master_results.csv` via `analysis/_common.py`'s
`load_master()` (and `cast_formula_dtypes()` where a script fits a regression --
see that module for why both matter). Run from the `analysis/` directory.

| # | Script | What it does |
|---|---|---|
| 1 | `01_merge_dataset.py` | Merge the 5 per-model result CSVs into `master_results.csv` |
| 2 | `02_validate_dataset.py` | Structural validation: row counts, balance, no missing design cells |
| 3 | `03_compliance_table.py` | Table 1: strict-parsing compliance rate by model |
| 4 | `04_descriptives.py` | Mean rating by gender/country/profession/age/topic/condition, per model |
| 5 | `05_hypothesis_models.py` | Primary OLS models (H1): per-model + pooled, HC3/cluster/mixed-effects SEs |
| 5b | `05b_ordinal_robustness.py` | Ordinal-logit robustness check for H1 (Condition A, persona-clustered) |
| 5b2 | `05b2_proportional_odds_by_topic.py` | Proportional-odds assumption check, stratified by topic |
| 5c | `05c_topic_specific_models.py` | Topic-specific primary + abstention models (Section 7) |
| 5d | `05d_country_set_robustness.py` | Original-10 vs. added-10 country robustness check (Section 8) |
| 5e | `05e_bh_correction.py` | Benjamini-Hochberg FDR correction within predefined test families |
| 6 | `06_abstention_analysis.py` | Optional-abstention logistic regression (H2) |
| 7 | `07_cross_model_agreement.py` | Spearman/weighted-kappa agreement between models |
| 7b | `07b_paired_comparison.py` | Paired forced-vs-optional comparison (H3, Section 6's "primary contribution") |
| 8 | `08_variance_ranking.py` | Partial R^2 / variance-explained ranking per factor |
| 9 | `09_deepseek_report.py` | DeepSeek-specific compliance and inclusion report |
| 10 | `10_figures.py` | Figures 2-7 |

See `MANIFEST.md` for the exact table/figure each script writes.

## Known limitations

- **DeepSeek's near-total format non-compliance.** Its strict-valid rate is 0.08%
  (63/75,600 rows), and 0% under the optional condition specifically. It is never
  excluded from descriptive reporting (per `analysis_plan.md` Section 10) but is
  excluded from the pooled hypothesis model, the abstention regression, and the
  primary cross-model agreement matrix on structural (not judgment-call) grounds --
  see `09_deepseek_report.py` for the full accounting.
- **The H2 abstention model omits the model-interaction terms `analysis_plan.md`
  Section 5 specifies** (`topic:model`, `profession:model`, etc.). The fitted model
  in `06_abstention_analysis.py` is main-effects-only. Reason: even the main-effects
  model requires restricting to qwen/ministral and dropping topics with deterministic
  (0%/100%) response rates to avoid complete separation; a full interaction model
  would need per-model-per-topic cells, and `05c_topic_specific_models.py`'s
  topic-stratified abstention fits confirm empirically that most of those cells are
  themselves fully deterministic and unfittable. The main-effects model is reported
  as the best available fit given this constraint, not a substitute for the
  pre-registered specification.
- **Benjamini-Hochberg correction (Section 9) covers three families only**: the
  pooled model's coefficients, each per-model OLS's coefficients, and the abstention
  logistic regression's coefficients. It does not extend to the topic-specific models
  (05c), the country-set robustness check (05d), the ordinal robustness tests (05b),
  the cross-model agreement correlations (07), or the paired-comparison tests (07b).
  Those report raw p-values / cluster-robust SEs without a family-wise correction.
- **Marginal means and Tukey HSD pairwise contrasts for profession and country
  rankings** (`analysis_plan.md` Section 4) are not implemented. Profession/country
  effects are reported via regression coefficients and Spearman rank correlations
  throughout, not formal Tukey-adjusted pairwise contrasts.
- **Paraphrase robustness** (`analysis_plan.md` Section 8, robustness check #4 --
  two minimal paraphrases of the friend-frame opener on a matched persona subset) was
  never run. This requires new model generations, not a re-analysis of already-
  collected data, and is out of scope for this repository's analysis pipeline.

## Reproducing the analysis

```
pip install -r requirements.txt
cd analysis
python3 01_merge_dataset.py   # then 02, 03, 04, 05, 05b (per-model, see its --model flag),
                               # 05b2, 05c, 05d, 05e, 06, 07, 07b, 08, 09, 10, in that order
```

`05b_ordinal_robustness.py` and `05_hypothesis_models.py`'s mixed-effects fits are the
slowest steps (several minutes each); everything else runs in well under a minute.
