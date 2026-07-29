# Manifest: script -> output mapping

Every table lives in `tables/`, every figure in `figures/`. Paths below are relative
to those directories. `{model}` ranges over the models each script actually includes
(see the script's own docstring for exclusions, e.g. DeepSeek).

## `01_merge_dataset.py`
- `analysis/master_results.csv` -- the single source of truth every other script reads

## `02_validate_dataset.py`
- No table output -- prints a pass/fail structural validation summary only

## `03_compliance_table.py`
- `table1_compliance.csv`

## `04_descriptives.py`
- `descriptives_gender.csv`, `descriptives_country.csv`, `descriptives_profession.csv`,
  `descriptives_age.csv`, `descriptives_topic.csv`

## `05_hypothesis_models.py`
- `hypothesis_model_llama.csv`, `hypothesis_model_gemma.csv`, `hypothesis_model_qwen.csv`,
  `hypothesis_model_ministral.csv`, `hypothesis_model_deepseek.csv` (per-model OLS,
  HC3 + persona-clustered + mixed-effects SEs side by side; every DeepSeek coefficient
  row is explicitly marked **NON-INFERENTIAL / EXPLORATORY ONLY** because n=63 yields
  a sparse, rank-deficient, numerically unstable regression)
- `hypothesis_model_pooled.csv` (pooled model, 4 models -- DeepSeek excluded, Fix 4)

## `05b_ordinal_robustness.py` (run per model: `--model {llama,gemma,qwen,ministral}`)
- `ordinal_pseudo_r2_{model}.csv`
- `ordinal_profession_comparison_{model}.csv`
- `ordinal_proportional_odds_{model}.csv`
- `ordinal_full_coefficients_{model}.csv`

## `05b2_proportional_odds_by_topic.py`
- `ordinal_proportional_odds_by_topic.csv`

## `05c_topic_specific_models.py`
- `topic_specific_models.csv` (per-topic primary rating models, Condition A only)
- `topic_specific_profession_spread.csv`
- `topic_specific_r2.csv`
- `topic_specific_abstention_summary.csv` (per-topic abstention models, qwen/ministral)
- `topic_specific_abstention_coefficients.csv`

## `05d_country_set_robustness.py`
- `country_set_robustness.csv`
- `country_set_profession_comparison.csv`

## `05e_bh_correction.py`
- Rewrites `hypothesis_model_pooled.csv`, `hypothesis_model_{model}.csv` (all 5), and
  `abstention_model_qwen_ministral.csv` IN PLACE, adding `*_bh_adj` columns -- run this
  AFTER `05_hypothesis_models.py` and `06_abstention_analysis.py`, not before, or the
  BH-adjusted columns will be silently overwritten/lost if those scripts are re-run
  afterward (they don't know about the `_bh_adj` columns and will rewrite the file
  without them).
- `bh_correction_summary.csv`

## `06_abstention_analysis.py`
- `abstention_answered_rate_by_topic.csv`
- `abstention_answered_rate_by_gender.csv`, `..._by_country.csv`, `..._by_profession.csv`,
  `..._by_age.csv`
- `abstention_model_qwen_ministral.csv` (the well-posed regression)
- `abstention_model_full.csv` -- only written if the full 5-model regression happens to
  converge; it does not, by design (see script docstring), so this file will not exist
  in a normal run

## `07_cross_model_agreement.py`
- `cross_model_spearman_matrix.csv`, `cross_model_kappa_matrix.csv` (primary 4-model matrix)
- `cross_model_agreement_main4.csv` (long-format version of the same)
- `cross_model_agreement_deepseek_pairs.csv` (DeepSeek pairs, reported separately)

## `07b_paired_comparison.py`
- `paired_comparison_summary.csv` (per-model summary: midpoint test, group-diff test,
  confound check -- all persona-clustered)
- `paired_comparison_matched_cells_{model}.csv` (full matched-cell data, one row per
  persona x topic)
- `paired_comparison_topic_stratified_{model}.csv` (qwen/ministral only -- the only
  models with Condition-B abstention)

## `08_variance_ranking.py`
- `variance_ranking.csv` (both scopes in one file, distinguished by a `scope` column:
  `primary_conditionA` is the actual preregistered H1 test; `exploratory_pooled_AB` is
  the original A+B-pooled version, kept for comparison)

## `09_deepseek_report.py`
- `deepseek_compliance_by_condition.csv`

## `10_figures.py`
- `fig2_rating_distributions.png`
- `fig3_abstention_by_topic.png`
- `fig4_country_topic_heatmap.png`
- `fig5_pooled_coefficients.png` (LLAMA's profession contrasts specifically -- see Fix 6
  in the script's docstring for why, and why the other three models' effects aren't
  shown in this plot)
- `fig6_agreement_matrix.png` (4-model matrix; DeepSeek excluded, see `07`)
- `fig7_variance_explained.png` (Condition-A / primary scope only, per `08`'s `scope` column)
