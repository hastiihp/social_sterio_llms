# Stage 4: Context Sensitivity Index

A 4-component index, one row per model, laying out four already-computed
sensitivity signals side by side — **not** collapsed into one weighted
score (no existing precedent in this project for how to weight these
against each other, so none is invented here). No new statistical claims:
every number is read directly from an existing, already-audited output
file, or is a mean of already-computed values. Full-scale throughout.

Output: [`context_sensitivity_index.csv`](context_sensitivity_index.csv).
Build script: `analysis_taxonomy/04_build_context_sensitivity_index.py`.

## ⚠ Correction (post-Stage-7 review): `avg_abs_rating_shift` was pooled-scope, now fixed

The original version of this table sourced component 2 from
`analysis_context/output/{context}_vs_original_summary_full5400.csv`,
assumed to be Condition-A-scoped because that's this project's convention
everywhere else. **It is not** — that file pools Condition A and Condition
B together (confirmed at the code level: `analysis_context/01`'s
`section_matched_cells()` never filters by `response_condition`), which
understated every model's average shift by roughly a third to a half.
Caught when Stage 7's unified mixed model, fit independently and directly
on raw Condition-A data, produced per-model coefficients that didn't match
this column. Root-caused, and confirmed by hand-recomputing the raw
Condition-A-only mean difference straight from `results/results_*.csv`
(matches both the corrected source file and the unified model to 4+
decimal places) — full investigation in `analysis_unified/UNIFIED_MODEL_SUMMARY.md`.

**Fixed**: component 2 now sources the Condition-A row of
`{context}_vs_original_by_condition_full5400.csv` instead. Every value in
the table below moved up (shifts were understated, not overstated) —
llama 0.045→0.108, gemma 0.130→0.142, qwen 0.119→0.144, ministral
0.041→0.046. **No conclusion changes**: Ministral is still, by a wide
margin, the smallest rating-shift model despite having the largest
abstention swing in the whole taxonomy — if anything the corrected numbers
make that contrast slightly sharper, not weaker.

## The four components (higher = more context-sensitive, except where noted)

| # | Column | What it measures | Source |
|---|---|---|---|
| 1 | `abstention_range_pct` | Range of Condition-B abstention rate across the 5 prompt types | Stage 1's own `model_taxonomy.csv` |
| 2 | `avg_abs_rating_shift` | Mean of \|rating shift\| (context vs. original), Condition A only, across the 4 conversational framings | `analysis_context/output/{context}_vs_original_by_condition_full5400.csv` × 4, `condition=="A_forced"` row |
| 3 | `avg_ranking_rho_country` / `avg_ranking_rho_profession` | Mean Spearman ρ (original ranking vs. context ranking) across the 4 framings — **inverted direction**: *lower* ρ = ranking reshuffles more = *more* sensitive | Stage 2's own `ranking_robustness_pvalue_summary.csv` |
| 4 | `dominant_factor_stable` | Does the single dominant demographic factor stay the same across all 5 prompt types? | Stage 1's own `model_taxonomy.csv` |

## The table

model | abstention range (pp) | avg \|rating shift\| (Condition A) | avg ranking ρ (country) | avg ranking ρ (profession) | dominant factor stable?
---|---|---|---|---|---
llama | 0.0 | 0.108 | 0.614 | 0.724 | **yes**
gemma | 0.1 | 0.142 | 0.672 | 0.947 | no
qwen | 7.7 | 0.144 | 0.706 | 0.937 | no
ministral | **53.7** | 0.046 | 0.762 | 0.957 | no
deepseek | 0.0 | NA | NA | NA | NA

## Why this reinforces, rather than merely repeats, Stage 1's anchor claim

The most striking pattern here isn't visible from any single Stage 1
number: **Ministral is simultaneously the *most* context-sensitive model
on component 1 (its 53.7-point abstention swing is by far the largest in
the table) and one of the *least* context-sensitive on components 2 and
3** — its average Condition-A rating shift (0.046) is the smallest of the
four compliant models, by a wide margin (the next-smallest, llama, is more
than double it), and its ranking stability is the highest (ρ=0.762
country, 0.957 profession, both the best in the table). The model whose
willingness to answer swings the most is, on the actual content of its
answers, one of the most consistent. This is an independent line of
evidence for the same pattern the Stage 1 anchor claim describes — using
different underlying metrics (rating-shift magnitude and ranking
stability, not abstention rate or dominant-factor identity) — not a
restatement of Stage 1's numbers.

Llama is the mirror case: **zero abstention sensitivity** (never abstains,
in any framing) but **not zero on the other three components** — a modest
0.108 average rating shift and the *lowest* country-ranking stability in
the table (ρ=0.614, noticeably behind the other three models' 0.67-0.76).
Llama's dominant factor is also the only one that never changes. So while
Llama is the most stable model on the willingness-to-answer axis
(trivially — it's always 0), it is not uniformly the most stable model
overall: its country ranking reshuffles somewhat more under a different
framing than the other three models' does.

## Caveats carried forward, not re-litigated

- **DeepSeek is NA on components 2-4**, not zero — it has zero
  both-valid matched rows in every context comparison (already established
  throughout this project) and is excluded from the ranking-robustness and
  H1 pipelines. Component 1's 0.0 for DeepSeek reflects near-total
  non-compliance, not confident non-abstention (Stage 1's caveat applies
  identically here).
- **Component 3's direction is inverted** relative to the other three —
  stated explicitly in the column description above so a reader building a
  chart from this CSV doesn't accidentally plot ρ as if higher meant "more
  sensitive."
- **No composite score is computed.** If a single rankable number is
  wanted later, the weighting scheme would need to be decided and stated
  explicitly as a new methodological choice — not inferred from this
  table.

Holding here for review before any further stage, per the established
pattern.
