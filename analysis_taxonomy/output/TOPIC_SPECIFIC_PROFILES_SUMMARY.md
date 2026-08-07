# Stage 3: Topic-Specific Country / Profession Rankings

Synthesizes already-computed, already-audited per-topic regression
coefficients into two long-format tables (one row per country×topic, one
row per profession×topic) plus a companion R² table. **No new model is
fit anywhere in this script.** The one derived step — coefficient → rank
position — is a mechanical sort using the project's own established
tie-aware ranking function, imported directly rather than reimplemented.
Full-scale (5,400 personas, 20 countries, 30 professions) throughout — the
source has no pilot/full distinction; it has always been full-scale.

Outputs: [`country_topic_profiles.csv`](country_topic_profiles.csv) (140
rows, 20 countries × 7 topics),
[`profession_topic_profiles.csv`](profession_topic_profiles.csv) (210
rows, 30 professions × 7 topics),
[`topic_r2_summary.csv`](topic_r2_summary.csv) (28 rows, 4 models × 7
topics). Build script: `analysis_taxonomy/03_build_topic_specific_profiles.py`.

## Source

`tables/topic_specific_models.csv` (`analysis/05c_topic_specific_models.py`,
Condition A only per `analysis_plan.md` Section 7, main study — always
full-scale). Confirmed to cover exactly 4 models (llama/gemma/qwen/
ministral; DeepSeek already excluded upstream, same n=63 sparsity
justification used everywhere else in this project) × 7 topics ×
19 non-reference country terms + 29 non-reference profession terms. The
omitted reference level in each factor (**Argentina** for country,
**accountant** for profession — confirmed by diffing the term list against
the full 20-country/30-profession set) is added back at coefficient 0.0
before ranking, matching the same reference-reinsertion convention Stage
2's source data already uses.

## A real anomaly the build script's own validation caught (not silently absorbed)

The first run of this script **failed its own assertion**: two
`(model, topic)` cells had only 1 level instead of 20/30. Investigated
directly rather than loosened the check — both are legitimate, already-
documented degenerate cases from the source script itself:

- **Llama / economic redistribution**, and **Ministral / gender equality**:
  every one of the 5,400 personas received the exact same rating (**4.0,
  "Agree"**) under Condition A, for that model×topic, regardless of
  country, profession, gender, or age. `analysis/05c_topic_specific_models.py`
  explicitly checks for zero-variance outcomes before fitting and writes a
  single `term=="DEGENERATE_NO_VARIANCE"` marker row instead of reporting
  meaningless coefficients — this script found and respected that marker
  rather than trying to fit around it.
- No country or profession coefficients exist for these two cells at all —
  not because a level is missing, but because the entire model is
  undefined (an R² isn't defined for a constant outcome either — confirmed
  blank in `topic_r2_summary.csv` for both cells).
- The build script's validation asserts the degenerate-cell set is
  **exactly** these two, and would raise if a third, previously-unknown
  one appeared. It didn't. `rank_llama`/`coef_llama` are `NaN` for all 20
  countries and 30 professions under `economic redistribution`;
  `rank_ministral`/`coef_ministral` are `NaN` under `gender equality` — a
  visible gap in the CSV, not a fabricated single-level "ranking."

## A second thing worth flagging, not a bug: near-total flatness in some cells

Spot-checking Llama's `climate change` profession coefficients directly
(not degenerate — R²=0.037, the lowest non-degenerate value in the whole
table) showed 26 of 29 non-reference professions with coefficients on the
order of 1e-15 (floating-point noise, far below the tie tolerance) —
functionally zero. Only `farmer`/`journalist` (~0.017) and `architect`/
`civil engineer` (~0.006) are distinguishable from the reference level at
all. The tie-aware ranker correctly groups the ~26 indistinguishable
professions into one large tied rank (rank 3) rather than assigning them
arbitrary distinct ranks — this is the correct, already-audited tie
handling doing its job, not an artifact of this script, but it means
**rank position alone can be misleading for near-flat cells** without also
reading the R² and the coefficient magnitude. This is consistent with — not
a new contradiction of — Stage 1's finding that Llama's dominant factor is
gender, not profession: on at least this topic, profession barely matters
to Llama at all.

## Reliability flag now embedded directly in the per-level CSVs

Following review, `r2_{model}` and `reliability_{model}` columns are now
embedded in every row of `country_topic_profiles.csv` and
`profession_topic_profiles.csv` themselves, not left only in the
`topic_r2_summary.csv` companion — someone pulling a number directly from
the per-level CSV (for a chart, a table in a paper) will see the
reliability flag right next to the rank/coefficient they're using, rather
than needing to know to cross-reference a separate file.
`reliability_{model}` is one of:

- **`OK`** — R² ≥ 0.15.
- **`LOW_R2`** — R² < 0.15. **This threshold is a design choice made for
  this table, not an audited project convention** — chosen at the natural
  gap in the empirical R² distribution across all 26 non-degenerate
  (model, topic) cells (the three lowest values, 0.037/0.089/0.111, sit
  well apart from the rest; the next value up is 0.228, a +0.117 jump vs.
  the +0.02-0.05 jumps within the low cluster itself). Verified: 100 of the
  140 country rows and 150 of the 210 profession rows carry a `LOW_R2` or
  `DEGENERATE_NO_VARIANCE` flag on at least one model, because 5 of the 7
  topics have a flagged cell for at least one model (Llama: `climate
  change` LOW_R2, `economic redistribution` DEGENERATE; Ministral:
  `religion and secularism` and `trust in government` LOW_R2, `gender
  equality` DEGENERATE) -- only `immigration` and `lgbtq rights` are
  flag-free across all four models. Gemma and Qwen individually have zero
  low-R2 or degenerate cells across all seven topics -- the flagged topics
  are entirely a Llama/Ministral phenomenon.
- **`DEGENERATE_NO_VARIANCE`** — the two zero-variance cells described
  above; `rank_{model}`, `coef_{model}`, and `r2_{model}` are all `NaN`.

## No robustness data at this granularity

Unlike Stage 2, no bootstrap top/bottom probability or permutation
p-value exists for individual (topic, country) or (topic, profession)
cells anywhere in this project — that level of robustness testing was
never run per-topic. Only rank position and the underlying coefficient are
reported; treat a topic-specific rank as a point estimate, not a
robustness-tested one, especially in low-R² cells like the one above.

## R² range (companion table, already-computed, not new)

Non-degenerate R² spans **0.037** (Llama/climate change) to **0.840**
(Gemma/lgbtq rights) — an order of magnitude difference in how much of a
given topic's rating variance these four demographic factors explain,
depending on model and topic. Reading a topic-specific rank position
without checking this number risks over-interpreting noise as a real
ranking, particularly for the low-R² end (Ministral/trust in government:
0.089; Ministral/religion and secularism: 0.111; Llama/climate change:
0.037).

Holding here for review before any further stage, per the established
pattern.
