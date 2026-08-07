# Stage 2: Country / Profession Behavioral Profiles

Synthesizes already-verified ranking-robustness outputs into two tables —
one row per country (20), one row per profession (30) — plus a companion
summary table. **No new statistical claims**: every cell is read directly
from an existing, already-audited output file. Full-scale (5,400 personas,
20 countries, 30 professions) throughout. DeepSeek excluded (see below).

Outputs: [`country_profiles.csv`](country_profiles.csv),
[`profession_profiles.csv`](profession_profiles.csv),
[`ranking_robustness_pvalue_summary.csv`](ranking_robustness_pvalue_summary.csv).
Build script: `analysis_taxonomy/02_build_country_profession_profiles.py`.

## What changed from the originally-approved column list

The approved structure included a **"cross-model rank agreement"** column.
Before building, I searched every script in this project that computes
rank-based correlations (`spearmanr`/`kendalltau` across `analysis/`,
`analysis_health/`, `analysis_context/`) and confirmed **no existing output
computes whether Llama's profession/country ranking correlates with
Gemma's, Qwen's, or Ministral's** — everything that exists is
cross-*context* (does a model's own ranking replicate across the five
prompt types), not cross-*model*. Per your decision, this column is
**dropped rather than computed fresh** — it does not appear in either
table. If cross-model rank agreement is wanted later, it would be a new
analysis, not a synthesis of existing work, and should be scoped as its
own task.

The **ranking-robustness p-value** also doesn't fit as a per-country/
per-profession column the way the approved structure implied — it's a
statistic about the *whole* ranking for a given (model, context, factor),
not about an individual country or profession. It's reported instead as
its own companion table, `ranking_robustness_pvalue_summary.csv` (32 rows:
4 contexts × 4 models × 2 factors), rather than force-fit into the main
tables as a column that doesn't actually vary by row.

## Sources (each read directly)

- **Rank position** (per model, per country/profession, under the original
  prompt): `analysis_context/output/health_ranking_robustness_{country,profession}_full5400.csv`'s
  `rank_orig`/`rank_label_orig` columns (`analysis_context/01_compare_context_vs_original.py`,
  full-scale scope).
- **Bootstrap top/bottom rank-position probability** (per model, per
  country/profession, under the original prompt): the same files'
  `*_ranking_robustness_bootstrap_full5400.csv`, filtered to
  `framing=="original"`.
- **Ranking-robustness p-value** (does the whole ranking replicate under
  each framing): `*_ranking_robustness_pvalues_full5400.csv`, all four
  contexts, concatenated as-is.

## Verification notes

1. **Cross-file consistency, checked programmatically before use, not
   assumed.** `health`/`neutral`/`positive`/`negative_minor`'s ranking-
   robustness scripts each independently recompute the *same*
   original-prompt ranking and bootstrap probabilities (since "original"
   is the shared baseline in every one of the four comparisons). The build
   script asserts byte-for-byte identical `rank_orig` and
   `p_top`/`p_bottom` (framing="original") across all four files before
   treating any one of them as canonical — confirmed identical for both
   country and profession, all four models. (This assertion would have
   raised and halted the build if any of the four disagreed — it did not.)
2. **A genuine scope inconsistency found and traced to its root cause, not
   silently normalized.** `negative_minor`'s bootstrap file contains 250
   rows (5 models × 50 levels) where `health`/`neutral`/`positive` contain
   only 200 (4 models × 50 levels) — DeepSeek is present only in
   `negative_minor`. Traced directly to
   `analysis_context/01_compare_context_vs_original.py`'s
   `section_ranking_robustness()`: it dynamically skips a (model, context)
   pair with fewer than `MIN_VALID_FOR_RANKING=60` valid Condition-A rows,
   not a hardcoded model list. DeepSeek has 63 valid rows under the
   original prompt but only 204 under `negative_minor` specifically
   (confirmed by direct count against `results/results_negative_minor_deepseek.csv`)
   — enough to clear the 60-row floor there and nowhere else, consistent
   with the already-documented "DeepSeek negative_minor anomaly"
   (`CONTEXT_EXPERIMENT.md` Step 4). This is confirmed intentional,
   documented pipeline behavior, not a bug. DeepSeek is excluded from both
   tables regardless (204 rows is still thin for a 30-level ranking model,
   and it would be an all-but-empty column in 3 of 4 source files) — but
   the reason is now stated explicitly rather than left as a silent gap.
3. **Spot-checked against `CONTEXT_EXPERIMENT.md`'s own quoted numbers**
   (written earlier this session, independently): the health-context
   country rho/p values for all four models (llama 0.598/0.0063, gemma
   0.678/0.0014, qwen 0.711/0.00067, ministral 0.811/0.00003) match this
   table's `ranking_robustness_pvalue_summary.csv` to the quoted precision.

## Reading the tables

Both tables report each model's **rank position under the original
prompt** (1 = highest attributed rating) and, at that same rank, the
**bootstrap probability** (out of 300 full-scale resamples) that this
level is genuinely the top- or bottom-ranked one — a level ranked #1 with
a low `bootstrap_top_pct` is a fragile top rank; one with a high
`bootstrap_top_pct` is a robust one. Rank *position* is directly
comparable across the four models for the same country/profession; the
p-values in the companion table tell you whether each model's full ranking
survives a change in conversational framing, not whether it agrees with
another model.

A few observations directly readable from the tables (descriptive, not new
statistical claims): Canada ranks #1 for three of the four models (gemma,
qwen, ministral) and #2 for the fourth (llama), and Egypt ranks in the
bottom 5 of 20 countries for **all four** models (llama #20, gemma #17,
qwen #19, ministral #16) — the closest things to a cross-model pattern
visible by eye, though (per the dropped column above) this has not been
formally tested for cross-model rank agreement. Social worker and doctor
recur near the top across multiple models' profession rankings; janitor
and truck driver recur near the bottom.

Holding here for review before any further stage, per the established
pattern.
