# Full Results Walkthrough

A consolidated, numbers-first tour of every finding produced by this
project, for a reader (human or AI) encountering the results cold, without
access to the underlying CSVs. Five sections: the main study, the
health-conversation study, the four-context comparison, the six-stage
model taxonomy, and known limitations. Every number below is pulled
directly from the project's `tables/`, `analysis_context/output/`,
`analysis_health/output/`, or `analysis_taxonomy/output/` CSVs, not from
memory or prior prose summaries.

**Design constants used throughout:** 5,400 personas (20 countries × 3
genders × 3 ages × 30 professions) × 7 topics × 2 response conditions
(A = forced 1-5 rating, B = optional 1-5-or-NA) = 75,600 prompts per model.
Five models: Llama-3.1-8B-Instruct, Gemma-3-12B-it, Qwen3-8B,
Ministral-8B-Instruct, DeepSeek-LLM-7B-chat. Falcon-H1-7B excluded
project-wide (inference-environment failure, not a finding — see
`FALCON_EXCLUSION.md`).

---

## 1. Main Study (Original Single-Turn Prompt)

Source: `tables/variance_ranking.csv`, `tables/table1_compliance.csv`,
`tables/paired_comparison_summary.csv`, `tables/cross_model_spearman_matrix.csv`,
`tables/deepseek_compliance_by_condition.csv`.

### The two-family behavioral split

Whether a model has any real capacity to abstain under Condition B splits
the five models cleanly into two families (`n_b_abstained` out of 37,800
Condition-B rows, `tables/paired_comparison_summary.csv`):

| model | n abstained (of 37,800) | abstention rate |
|---|---|---|
| llama | 0 | 0.0% |
| gemma | 0 | 0.0% |
| qwen | 34,009 | 89.97% |
| ministral | 31,550 | 83.47% |

Llama and Gemma never abstain at all under Condition B; Qwen and Ministral
abstain on the large majority of opportunities. This split recurs
throughout the rest of the project (see Sections 3-4).

### H1 per model: does profession dominate attributed opinion?

Partial R² (variance explained), `rating ~ topic + profession + country +
gender + age`, Condition A only, persona-clustered significance
(`tables/variance_ranking.csv`, `scope=="primary_conditionA"`; topic
excluded as a candidate factor — it's a fitted control, not part of the H1
test):

| model | 1st (partial R²) | 2nd | 3rd | 4th |
|---|---|---|---|---|
| llama | **gender** 0.115 | profession 0.021 | country 0.018 | age 0.008 |
| gemma | **country** 0.029 | gender 0.024 | profession 0.018 | age 0.003 |
| qwen | **profession** 0.141 | gender 0.049 | country 0.036 | age 0.003 |
| ministral | **profession** 0.052 | gender 0.040 | country 0.031 | age 0.025 |

H1 ("profession dominates") holds for only 2 of 4 models (qwen,
ministral); gender dominates for llama, country for gemma. All effects
significant at p<10⁻⁴⁸ or smaller (clustered).

### The covert-midpoint pattern

For Qwen and Ministral specifically — the two models with real Condition-B
abstention — does the forced Condition-A rating quietly cluster at the
scale midpoint (3 = "Neither agree nor disagree") specifically in cases
where the model would have abstained if given the choice?
(`tables/paired_comparison_summary.csv`)

| model | mean forced-A rating, given B abstained | % of those A ratings = 3 | % of A=3 when B *answered* instead | z-stat | p |
|---|---|---|---|---|---|
| qwen | 3.327 | **66.6%** | 0.0% | 79.4 | ~0 |
| ministral | 3.244 | **74.5%** | 0.0% | 111.0 | ~0 |

This is a striking, clean pattern: when these two models are given the
option to abstain and take it, their forced-answer counterpart is the
literal midpoint two-thirds to three-quarters of the time — versus
essentially never when they do choose to answer. This is read as evidence
that "3" functions as a **covert abstention channel** under forced
conditions, not as genuine substantive neutrality — exactly the confound
`analysis_plan.md` Section 6 predefined as worth checking for before this
analysis ran.

### DeepSeek's compliance finding

Strict-format compliance (produces exactly `1`-`5` or a valid `NA`),
original prompt (`tables/table1_compliance.csv`,
`tables/deepseek_compliance_by_condition.csv`):

| scope | strict-valid rate |
|---|---|
| pooled (Condition A + B) | 0.083% (63 / 75,600) |
| Condition A only | **0.167%** (63 / 37,800 — all 63 valid rows are Condition A) |
| Condition B only | 0.0% |

The remaining ~99.9% of DeepSeek's original-prompt output splits into:
salvageable-numeric prose (30.67%), explicit refusal (41.53%), other
malformed (27.72%). DeepSeek is excluded from every pooled, ordinal,
ranking, and H1 model fit throughout this entire project for this reason
— its own diagnostics are always reported separately.

### Cross-model rating agreement

Spearman ρ between models' Condition-A ratings, original prompt, n=37,800
pairs per pair (`tables/cross_model_spearman_matrix.csv`):

|  | llama | gemma | qwen | ministral |
|---|---|---|---|---|
| llama | — | 0.722 | 0.535 | 0.524 |
| gemma | 0.722 | — | 0.636 | 0.681 |
| qwen | 0.535 | 0.636 | — | 0.708 |
| ministral | 0.524 | 0.681 | 0.708 | — |

DeepSeek's pairwise correlations (n=63 each, `flagged_too_sparse=True` in
source — unreliable, not comparable to the above): all four negative
(-0.09 to -0.67).

---

## 2. Health-Conversation Study

Standalone, separately-audited study — its own dedicated pilot-scale
(180-persona: Germany/Brazil/Nigeria/South Korea × lawyer/registered
nurse/truck driver/farmer/computer programmer) comparison, matched 1:1
between original and health framings (12,600 rows/framing, merge
validated `one_to_one`, 0 unmatched). Source:
`analysis_health/output/CORRECTED_SUMMARY.md` and its underlying CSVs
(regenerated 2026-07-24; unaffected by the later grammar-fix bug, which
only ever touched `neutral`/`positive` — see Section 3).

### Condition A rating shift (primary evidence)

Every model's mean rating moves **down** under the health framing, all
highly significant:

| model | shift (health − original) | p (clustered) | n |
|---|---|---|---|
| llama | −0.174 | 7.6×10⁻⁷⁷ | 1,260 pairs / 180 clusters |
| gemma | −0.185 | 8.3×10⁻⁵² | 1,260 / 180 |
| qwen | −0.156 | 3.2×10⁻²² | 1,260 / 180 |
| ministral | −0.098 | 2.4×10⁻³¹ | 1,260 / 180 |

### Ministral's abstention drop (the headline health finding)

Denominator matters — reported at all four combinations
(`ministral_abstention_2x2.csv`):

| sample | denominator | original | health | shift |
|---|---|---|---|---|
| full 5,400 | all rows | 41.73% | 24.26% | −17.48pp |
| full 5,400 | Condition B only | 83.47% | 48.51% | −34.95pp |
| 180-pilot | all rows | 41.71% | 24.80% | −16.91pp |
| 180-pilot | Condition B only | 83.41% | 49.60% | **−33.81pp**, p=4.6×10⁻¹⁶⁷ |

Concentrated by topic, not uniform: climate change −81.1pp, economic
redistribution −64.4pp, immigration −56.1pp, vs. gender equality −2.8pp,
religion/secularism −0.6pp.

### Qwen's Condition-B sign flip (a selection-effect warning, not a finding about Qwen's opinions)

Qwen's Condition-B rating shift is **+0.325** (p=4.5×10⁻¹¹) — the
*opposite sign* from its own Condition-A shift (−0.156) — because only
6.35% of Qwen's Condition-B pairs are both-valid (a small, self-selected,
framing-dependent sample). Read as evidence Condition-B comparisons for
qwen/ministral are not measuring the same thing as Condition-A ones, not
as a genuine opinion reversal.

### Ranking stability

Profession rankings highly stable across the framing change: gemma/qwen/
ministral r=1.00/0.975/0.975 (exact permutation p=0.017-0.033); llama
r=0.872 (p=0.10). Truck driver ranks bottom in every model, both framings.
Country rankings (n=4) cannot reach conventional significance at pilot
scale by construction (minimum possible exact p = 2/24 = 0.083 regardless
of true agreement) — see Section 3 for the full-scale resolution of this
exact limitation.

### DeepSeek under health framing

Valid Condition-A ratings drop from 63/75,600 (original) to **0/75,600**
(health) — complete collapse. A compact-text parser recovers 14.86%
(11,234/75,600) of health responses as containing a plausible rating;
every one of those recovered ratings is the digit "4" (unexplained,
flagged, not treated as evidence of substantive opinion).

---

## 3. Four-Context Comparison (Full-Scale — Primary)

health, neutral, positive, negative_minor vs. original, **all four now at
full scale** (5,400 personas, 20 countries, 30 professions — the pilot
180-persona numbers above are superseded for cross-context comparison
purposes, though not contradicted). Source: `CONTEXT_EXPERIMENT.md` and
`analysis_context/output/*_full5400.csv`.

### ✓ One grammar bug found, fixed, and re-verified as immaterial

`neutral`/`positive` templates had a subject-verb agreement bug affecting
only neutral-gender ("they") personas (1,800/5,400 = 33.3% of the
dataset). Fixed, full 75,600-row re-run per affected context per model.
**No headline finding changed** after the fix — every rating shift,
abstention rate, and clustering number moved by at most a few tenths of a
percentage point, with one flagged exception (llama/positive/profession
ranking p-value crossed the 0.05 line: 0.033→0.067, a mid-ranking
reshuffle, not a reversal of the bottom-rank pattern).

### Country ranking significance — now reachable at full scale

At pilot scale (n=4 countries) p<0.05 was structurally unreachable. At
full scale (n=20, Monte Carlo permutation, 200,000 draws), every model ×
context cell reaches significance comfortably:

| context | llama | gemma | qwen | ministral |
|---|---|---|---|---|
| health | ρ=0.598, p=0.0063 | ρ=0.678, p=0.0014 | ρ=0.711, p=0.00067 | ρ=0.811, p=0.00003 |
| neutral | ρ=0.654, p=0.0023 | ρ=0.660, p=0.0019 | ρ=0.708, p=0.00065 | ρ=0.768, p=0.00014 |
| positive | ρ=0.602, p=0.0061 | ρ=0.598, p=0.0060 | ρ=0.752, p=0.00026 | ρ=0.657, p=0.0020 |
| negative_minor | ρ=0.600, p=0.0061 | ρ=0.750, p=0.00018 | ρ=0.651, p=0.0023 | ρ=0.812, p=0.00002 |

### Ministral's abstention decline chain (Condition-B rate, all five prompt types)

`analysis_context/output/abstention_stability_rate_table.csv`:

**original 83.47% → health 48.51% → neutral 41.80% → positive 37.25% →
negative_minor 29.80%** — strictly decreasing, every adjacent step
significant, a 53.7-percentage-point total range (the largest
context-sensitivity signal anywhere in this project). Qwen's equivalent
range is only 7.7pp (87-95%, stably reluctant rather than context-driven).

### Cross-context clustering: structure or content?

Full-scale mean Spearman ρ: **context-vs-original = 0.774**;
**context-vs-context = 0.874** (16 and 24 pairs respectively,
`cross_context_clustering_summary_full5400.csv`) — the four conversational
framings resemble each other more than any of them resembles the original
single-turn prompt. Most similar pair: health/negative_minor (ρ=0.898);
least similar: health/neutral (ρ=0.854) — range across all six pairs only
0.045, a modest secondary pattern on top of the larger structural effect.

### H1 across all five prompt types (dominant factor, full scale)

`analysis_taxonomy/output/model_taxonomy.csv` / `dominant_factor_by_model_full5400.csv`:

| model | original | health | neutral | positive | negative_minor | stable? |
|---|---|---|---|---|---|---|
| llama | gender | gender | gender | gender | gender | **yes** |
| gemma | country | profession | profession | gender | gender | no (3 distinct) |
| qwen | profession | gender | profession | gender | profession | no (2 distinct) |
| ministral | profession | gender | gender | gender | gender | no (2 distinct) |

Gemma's health/neutral "profession wins" cells are razor-thin (margins of
0.0003 partial R² — an order of magnitude tighter than any other
dominant-vs-runner-up gap in the whole project); Qwen and Ministral's
shifts have real, non-trivial margins (0.01-0.09).

### DeepSeek's cross-context anomaly (reconciled)

An initial "11% for original" figure was wrong (double-counted an
existing category) and corrected before use. True near-compliance rate
(strict-valid + salvageable + compact-parser-recovered, double-counting
removed):

| condition | true total |
|---|---|
| original | 30.75% |
| health | 14.86% |
| neutral | 68.62% |
| positive | **90.48%** (the actual extreme, not negative_minor) |
| negative_minor | 41.39% |

Independently confirmed via a regex-free zero-whitespace metric (0.00% →
15.70% → 73.35% → 81.52% → 96.98%, same ordering). Positive's
"recovered" rows are 99.96% the identical rating "4" — a repetitive
formatting artifact, not evidence of varied substantive opinion (contrast
with original's genuinely spread salvageable rows: 62%/23%/15%
"4"/"2"/"3").

---

## 4. Six-Stage Model Taxonomy (`analysis_taxonomy/`)

Full detail and per-number source citations in `analysis_taxonomy/README.md`
and its six stage summary docs. Headlines only, here:

**Stage 1 — Model Taxonomy.** Anchor claim, confirmed and adopted: *models
differ far more in willingness to answer (0% to 91% average abstention
between models) than in stereotype content (0.52-0.72 consistently
positive cross-model rating agreement)*. Only 1 of 4 measurable models
(llama) has a fully stable dominant factor across all 5 prompt types.

**Stage 2 — Country/Profession Profiles.** Rank position + bootstrap
robustness for all 20 countries / 30 professions, original prompt, full
scale. "Cross-model rank agreement" confirmed to have no existing source
anywhere in the project — dropped rather than fabricated.

**Stage 3 — Topic-Specific Rankings.** Per-topic rank for every country/
profession (140 + 210 rows). Two genuine zero-variance cells found:
Llama/economic-redistribution and Ministral/gender-equality both rated
literally every persona "4" regardless of any demographic factor. R²
ranges 0.037-0.840 across non-degenerate cells; a `LOW_R2`/`OK`
reliability flag is embedded directly in every row.

**Stage 4 — Context Sensitivity Index.** Ministral has the single largest
abstention swing in the taxonomy (53.7pp) while simultaneously having the
*smallest* average rating shift (0.041) and *highest* ranking stability
(ρ=0.762 country / 0.957 profession) of the four compliant models — an
independent confirmation of the Stage 1 anchor claim via entirely
different metrics. Llama is the mirror case: zero abstention sensitivity,
but its own country-ranking ρ=0.614 is the *lowest* in the table.

**Stage 5 — Stability Index.** Topic-to-topic rank swings are ~2.4x
larger than framing-to-framing swings for both countries (12.76 vs. 5.40
of a possible 19) and professions (15.67 vs. 5.06 of a possible 29) — a
third independent angle: which topic a stereotype attaches to matters
more than which conversational framing delivers it.

**Stage 6 — Consensus Index.** Across all 37,800 persona×topic cells
(4 compliant models, original prompt, Condition A): **22.0% show literal
unanimous 4-way agreement** (SD=0); **no cell in the entire dataset shows
a full 1-vs-5 split**. Disagreement is concentrated, not uniform, but
unevenly so by factor: strong for topic (climate change/religion are
2.4-3.0x over-represented in consensus; immigration/economic
redistribution 1.7-2.0x over-represented in disagreement) and country (up
to 2.7x), modest for profession (0.72-1.61x, manual/frontline professions
skew toward disagreement), weakest for gender (0.70-1.27x, though still
statistically significant at this sample size).

---

## 5. Known Limitations and Caveats

Full detail and the exact verdict each round issued: `AUDIT_HISTORY.md`
(five independent audit rounds, all raw scaffolding removed from the
working tree but git-recoverable).

### What's been audited and confirmed

- **Round 1** (health study): 7 specific fixes (sign conventions,
  abstention denominators, Condition A/B splitting, exact permutation
  tests, DeepSeek compact parsing, merge validation, sample-size
  labeling) — all incorporated into current scripts.
- **Round 2** (full-project traceability, 2026-07-28/29): clean-environment
  regeneration reproduced all main and health results; **GO**. Found and
  fixed a silently-lost BH correction (script run-order bug) and an
  obsolete stale table.
- **Round 3** (context study adversarial audit): found and reported the
  neutral/positive grammar bug (Section 3 above) and three hand-transcription
  errors in an earlier findings table (since corrected); **CONDITIONAL
  GO**.
- **Round 4** (grammar-fix verification): confirmed the fix genuinely
  reached the model-facing text and the re-run data; **CONDITIONAL GO**,
  conditions since met.
- **Round 5** (pre-Stage-1-5 preflight, this session): **NO GO** as
  issued, health score 62/100 — but several blocking items (data-folder
  scatter, health_staging duplicates, the pilot/full scope-mixing
  question) were resolved by the reorganization that immediately followed
  it. Also resolved: an apparent "pandas import fails" discrepancy,
  root-caused to bare Homebrew Python being confused with the project's
  pinned virtualenv (both report identical version numbers).

### What's still open, stated plainly

- **The paraphrase-robustness check (`analysis_plan.md` Section 8) was
  never run.** Two minimal paraphrases of the prompt opener, on a matched
  persona subset, were predefined specifically because a single frozen
  prompt wording cannot itself demonstrate findings reflect underlying
  model behavior rather than this exact phrasing. This has not been
  executed at any point in this project. Every finding in this document
  should be read with that caveat: robustness to prompt wording
  specifically has not been tested, only robustness to conversational
  framing and to scale.
- **The non-binary-condition manipulation check** (`analysis_plan.md`
  Section 12 — does the they/them + name-only signal reliably register as
  non-binary to the model) was planned but its execution/outcome is not
  independently confirmed anywhere in this project's current state.
- **Historical raw-data gaps**: the prior v9 experiment, original pilot
  outputs, manipulation-check outputs, prompt-reinforcement test outputs,
  and the pre-grammar-fix raw `neutral`/`positive` result files are
  unrecoverable — code/mentions exist but no raw outputs remain.
- **A formerly dormant bug, now fixed in source**:
  `analysis_health/04_ranking_robustness.py` used unsafe character-set
  `.strip("[]T.")` label extraction. Its pilot-only level set meant no
  saved output was affected. The active source now removes only the exact
  trailing bracket, matching the earlier full-scale context fix; historical
  outputs were not rewritten.
- **Name-validity limitation** (state verbatim in any derived paper, per
  `analysis_plan.md` Section 13): each country×gender condition is
  represented by a single name, so country/gender effects may partly
  reflect name-specific associations rather than the demographic category
  alone.
- **DeepSeek and Falcon are excluded** from essentially all inferential
  work in this project (DeepSeek: near-total format non-compliance;
  Falcon: inference-environment failure) — neither exclusion reflects a
  finding about the model's actual behavior or opinions.
