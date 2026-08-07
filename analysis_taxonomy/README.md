# Model Behavioral Taxonomy

This folder synthesizes findings already established elsewhere in this
project into a single, cross-referenced taxonomy of how the five study
models behave — not a new experiment, but an organized, independently
re-verified accounting of what the existing data already shows. It was
built in six stages, each approved only after its own numbers were
hand-checked against source and reported back before the next stage
began. The whole taxonomy builds evidence for one claim, refined during
Stage 1 from an initially looser version after the data showed the plain
version overstated how stable stereotype content actually is:

> **Models differ far more in their willingness to answer at all —
> ranging from near-total refusal-to-abstain to abstaining on the large
> majority of opportunities in every framing tested, and swinging by tens
> of percentage points within a single model depending on conversational
> framing — than in the actual demographic stereotypes they attribute,
> where models agree with each other at a consistently moderate-to-strong
> level and where even the models whose "dominant factor" label changes
> across framings do so by modest margins.**

## The six stages

**Stage 1 — Model Behavioral Taxonomy.** One row per model (5 models),
covering Condition-B abstention rate across all five prompt types,
dominant demographic factor and its stability, cross-model rating
agreement, and strict format compliance. This is the stage that proposed
and then refined the anchor claim above, after finding that only 1 of 4
measurable models (Llama) actually has a stable dominant factor across
framings — the other three do shift, just by modest margins.
→ [`output/model_taxonomy.csv`](output/model_taxonomy.csv),
[`output/MODEL_TAXONOMY_SUMMARY.md`](output/MODEL_TAXONOMY_SUMMARY.md)

**Stage 2 — Country / Profession Profiles.** One row per country (20) and
one row per profession (30): rank position under the original prompt,
per model, plus bootstrap top/bottom rank-position robustness. A
requested "cross-model rank agreement" column was investigated and found
to have no existing source anywhere in the project, so it was dropped
rather than computed fresh; ranking-robustness p-values turned out to be a
model×context-level statistic, not a per-country/profession one, so they
live in a separate companion table instead of being force-fit into a
column that wouldn't actually vary by row.
→ [`output/country_profiles.csv`](output/country_profiles.csv),
[`output/profession_profiles.csv`](output/profession_profiles.csv),
[`output/ranking_robustness_pvalue_summary.csv`](output/ranking_robustness_pvalue_summary.csv),
[`output/COUNTRY_PROFESSION_PROFILES_SUMMARY.md`](output/COUNTRY_PROFESSION_PROFILES_SUMMARY.md)

**Stage 3 — Topic-Specific Rankings.** The same country/profession
ranking, stratified by each of the 7 topics (140 + 210 rows), with rank
derived by sorting already-audited per-topic coefficients rather than
fitting anything new. Two topic×model cells turned out to have zero
rating variance at all and were excluded with an explicit
`DEGENERATE_NO_VARIANCE` flag rather than silently producing a fabricated
rank; every remaining row carries its own R² and a `LOW_R2`/`OK` reliability
flag directly in the CSV, so a reader pulling one row in isolation can't
miss a low-confidence cell.
→ [`output/country_topic_profiles.csv`](output/country_topic_profiles.csv),
[`output/profession_topic_profiles.csv`](output/profession_topic_profiles.csv),
[`output/topic_r2_summary.csv`](output/topic_r2_summary.csv),
[`output/TOPIC_SPECIFIC_PROFILES_SUMMARY.md`](output/TOPIC_SPECIFIC_PROFILES_SUMMARY.md)

**Stage 4 — Context Sensitivity Index.** One row per model, four
sensitivity signals laid out side by side (not collapsed into one score):
abstention-rate range, average rating-shift magnitude, average
ranking-stability ρ, and dominant-factor stability. This is where the
anchor claim gets its sharpest independent confirmation: Ministral has the
single largest abstention swing in the whole taxonomy (53.7 points) while
simultaneously having the *smallest* average rating shift and the
*highest* ranking stability of the four compliant models — the model most
sensitive on whether it answers is among the least sensitive on what it
actually says.
→ [`output/context_sensitivity_index.csv`](output/context_sensitivity_index.csv),
[`output/CONTEXT_SENSITIVITY_INDEX_SUMMARY.md`](output/CONTEXT_SENSITIVITY_INDEX_SUMMARY.md)

**Stage 5 — Stability Index.** The entity-level counterpart to Stage 4:
one row per country/profession, measuring how much its rank position
swings across the 7 topics versus across the 5 prompt types (framings),
plus resampling robustness. Topic-to-topic swings turned out to be about
2.4x larger than framing-to-framing swings for both countries and
professions — a third, independent axis of evidence that framing
sensitivity, while real, is smaller than other sources of variation this
taxonomy keeps finding.
→ [`output/country_stability_index.csv`](output/country_stability_index.csv),
[`output/profession_stability_index.csv`](output/profession_stability_index.csv),
[`output/STABILITY_INDEX_SUMMARY.md`](output/STABILITY_INDEX_SUMMARY.md)

**Stage 6 — Consensus Index.** All 37,800 persona×topic cells, ranked by
how much the four compliant models' ratings for that cell agree (standard
deviation across models). 22% of all cells show literal unanimous 4-way
agreement, and no cell in the entire dataset shows a full 1-vs-5 split —
disagreement, where it exists, is concentrated by topic and country at
large effect sizes, and present but much more modestly by profession and
gender.
→ [`output/consensus_index.csv`](output/consensus_index.csv),
[`output/consensus_top20_highest_consensus.csv`](output/consensus_top20_highest_consensus.csv),
[`output/consensus_top20_highest_disagreement.csv`](output/consensus_top20_highest_disagreement.csv),
[`output/consensus_pattern_chisquare.csv`](output/consensus_pattern_chisquare.csv),
[`output/CONSENSUS_INDEX_SUMMARY.md`](output/CONSENSUS_INDEX_SUMMARY.md)

## Six independent lines of evidence

Taken together, these six stages converge on the same claim from six
genuinely different angles, not six restatements of the same number.
Willingness to answer varies categorically between models — from models
that never abstain to one that abstains on the large majority of
opportunities in every framing tested — while the actual ratings models
produce agree with each other at a consistently moderate-to-strong level
(Stage 1). No existing analysis in this project supports a claim about
models agreeing or disagreeing on aggregate rank *order*, and that gap was
confirmed rather than papered over (Stage 2). Which topic a stereotype is
evaluated on moves a country's or profession's rank position far more than
which conversational framing is used, both in the raw topic-specific data
and again independently in the stability comparison (Stages 3 and 5).
Ministral supplies the single clearest natural experiment in the dataset:
the model with the largest context-driven swing in willingness to answer
is simultaneously among the most stable in what it actually says once it
does answer, and Llama shows the mirror pattern — zero abstention
sensitivity but not uniform stability elsewhere (Stage 4). And at the
level of individual judgments, models reach full unanimous agreement on
over a fifth of all persona-topic pairs and never once split to opposite
ends of the rating scale, with what disagreement does occur concentrated
heavily in a handful of topics and countries rather than spread evenly
across the whole design (Stage 6).

## What we caught along the way

Every number in every stage was independently re-verified against its
source file before being trusted — not assumed correct because an earlier
part of the project had already audited the underlying pipeline. That
discipline caught several real issues, each investigated and resolved
rather than smoothed over:

- **A pooled-vs-condition-specific compliance mismatch** (Stage 1):
  `table1_compliance.csv`'s headline compliance number is pooled across
  both response conditions; for DeepSeek specifically, its Condition-A-only
  rate (0.167%) is exactly double the pooled figure (0.083%). Fixed at the
  source: a warning was added directly to the generating script
  (`analysis/03_compliance_table.py`) so this can't silently bite a future
  analysis the same way.
- **A missing column, confirmed absent rather than invented** (Stage 2):
  "cross-model rank agreement" had no existing source anywhere in the
  project; rather than fit something new under a "no new statistical
  claims" stage, it was dropped and the gap stated explicitly.
- **Two genuine zero-variance cells** (Stage 3): Llama on "economic
  redistribution" and Ministral on "gender equality" both rated every
  single persona identically (4.0, "Agree"), regardless of any demographic
  factor — caught by the build script's own row-count assertions failing
  on the first run, traced to an already-documented marker in the source
  data, and handled by leaving those cells as explicit gaps rather than
  fabricating a one-level "ranking."
- **A scope inconsistency across four parallel output files** (Stage 2):
  the `negative_minor` context's ranking-robustness bootstrap file
  uniquely included DeepSeek (250 rows vs. 200 in the other three
  contexts), traced to DeepSeek clearing a minimum-valid-rows threshold in
  that one context specifically — confirmed intentional, documented
  pipeline behavior, not a bug, but not obvious without checking.
- **An inverted-direction metric flagged explicitly** (Stage 4): the
  ranking-stability component of the Context Sensitivity Index runs the
  opposite direction from the other three (lower, not higher, means more
  sensitive) — called out directly in the column description so a future
  chart of this table doesn't plot it backwards.

## Scope

Full-scale data throughout every stage (5,400 personas, 20 countries, 30
professions — never the 180-persona pilot subset); DeepSeek is excluded
from this taxonomy's ranking and dominant-factor work specifically because
of its near-total non-compliance with the required output format
(established in Stage 1, carried forward consistently through Stage 6);
Falcon-H1 is excluded from the entire project, not just this taxonomy, for
an unrelated inference-environment failure documented in
`FALCON_EXCLUSION.md`.
