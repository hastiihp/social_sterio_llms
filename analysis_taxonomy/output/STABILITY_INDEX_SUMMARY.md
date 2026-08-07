# Stage 5: Stability Index (per country / per profession)

Entity-level analog of Stage 4's model-level Context Sensitivity Index:
one row per country (20), one row per profession (30), each with 3
already-derivable stability signals per model, laid out side by side —
**not** collapsed into one score, same design choice as Stage 4. No new
statistical claims: every number is read directly from an existing,
already-audited output file, or is a max-min range over already-verified
rank positions. Full-scale throughout. DeepSeek excluded throughout,
consistent with Stages 1-4.

Outputs: [`country_stability_index.csv`](country_stability_index.csv) (20
rows), [`profession_stability_index.csv`](profession_stability_index.csv)
(30 rows). Build script: `analysis_taxonomy/05_build_stability_index.py`.

## The three components

| # | Column(s) | What it measures | Source |
|---|---|---|---|
| 1 | `topic_rank_range_{model}`, `topic_rank_n_valid_{model}` | Range of this level's rank position across the 7 topics (degenerate topics dropped — see note below; `n_valid` records how many of 7 actually contributed) | Stage 3's own `{country,profession}_topic_profiles.csv` |
| 2 | `framing_rank_range_{model}` | Range of this level's rank position across the original prompt + 4 conversational framings (5 positions total) | The same 4 `*_ranking_robustness_{country,profession}_full5400.csv` files Stage 2 used, pulling `rank_ctx` alongside `rank_orig` this time |
| 3 | `bootstrap_top_pct_{model}`, `bootstrap_bottom_pct_{model}` | Resampling-robustness of the top/bottom rank position under the original prompt | Carried forward unchanged from Stage 2's own `{country,profession}_profiles.csv` |

## Verification

- **`rank_orig` cross-checked identical across all 4 context files** before use for component 2, same assertion pattern as Stage 2 (would raise on any mismatch; did not).
- **Component 1 hand-verified**: Argentina/Llama's `topic_rank_range` of 15.0 was independently recomputed from `country_topic_profiles.csv` (ranks 3, 3, 8, 2, 17, 9 across the 6 non-degenerate topics — `economic redistribution` correctly excluded as `NaN` — range = 17-2 = 15.0, `n_valid`=6). Matches the build script's own output exactly.
- **Component 2 hand-verified**: Argentina/Llama's `framing_rank_range` of 9.0 was independently recomputed from the raw ranking-robustness files (`rank_orig`=10 in all four files; `rank_ctx` = 3/1/1/1 under health/neutral/positive/negative_minor respectively; combined set {10,3,1,1,1} → range = 10-1 = 9.0). Matches exactly.

## A pattern worth surfacing (descriptive, not a new statistical test)

Averaging each level's rank-range across all four ranked models: **topic-to-topic
rank swings are roughly 2.4x larger than framing-to-framing swings**, for
both countries and professions:

- Country: mean `topic_rank_range` = **12.76** (of a possible 0-19) vs.
  mean `framing_rank_range` = **5.40** (of a possible 0-19).
- Profession: mean `topic_rank_range` = **15.67** (of a possible 0-29) vs.
  mean `framing_rank_range` = **5.06** (of a possible 0-29).

In proportional terms, a country's rank position swings across ~67% of the
possible range depending on *which topic* it's evaluated on, but only
~28% of the possible range depending on *which conversational framing*
is used (profession: ~54% vs. ~17%). Which topic a stereotype is attached
to appears to matter substantially more to where a country or profession
lands in the ranking than whether the question arrives via a single-turn
prompt or one of the four multi-turn framings — a different, complementary
angle on the same overall finding this taxonomy keeps surfacing: framing
sensitivity, while real (Stage 4), is consistently smaller than other
sources of variation already documented elsewhere in this project (here,
topic; in Stage 1, willingness-to-answer).

Most and least topic-volatile levels (mean `topic_rank_range` across the 4
models): **Pakistan** (17.0) and **farmer** (24.25) swing the most across
topics; **Brazil** (7.75) and **doctor** (7.25) swing the least.

## Caveats carried forward

- **DeepSeek excluded from all three components**, consistent with its
  exclusion from the ranking-robustness and topic-specific pipelines
  throughout Stages 1-3.
- **Component 1's `n_valid` matters**: a level's `topic_rank_range` for
  Llama or Ministral may be computed from as few as 6 of 7 topics (the two
  known degenerate cells from Stage 3 — `economic redistribution` for
  Llama, `gender equality` for Ministral). Always check `n_valid` before
  comparing ranges across models with a different number of degenerate
  cells.
- **No composite score is computed**, same as Stage 4 — if a single
  rankable stability number is wanted later, the weighting across these
  three components (and across models) would need to be decided and
  stated explicitly as a new methodological choice.

Holding here for review before any further stage, per the established
pattern.
