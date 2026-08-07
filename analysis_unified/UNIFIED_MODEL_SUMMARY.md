# Stage 7: The Unified Variance-Decomposition Model

**This is the single statistical backbone this project was missing** —
one formal mixed-effects model, fit once, pooling all five prompt types
and four compliant models, replacing the scattered per-model, per-context
R² values from Stage 1 / `analysis_context/` / `analysis_taxonomy/` with
one central table carrying proper confidence intervals and a predefined
multiple-comparison correction.

**Validated analysis state.** The diagnostics and discrepancy checks below
were completed before this analysis was included in the repository's current
reporting layer. This remains research output for an active project, not a
finished publication.

## Model and scope

```
rating ~ topic + prompt_type + profession + country + gender + age + model
  + topic:model + prompt_type:model
  + (1 | persona_id)
```

Full-scale (5,400 personas, 20 countries, 30 professions), Condition A
only, all five prompt types pooled (original, health, neutral, positive,
negative_minor), four models (llama, gemma, qwen, ministral — DeepSeek
excluded per its established near-total non-compliance; Falcon excluded
project-wide). **756,000 rows** (5,400 × 7 topics × 5 prompt types × 4
models), each with exactly 4 valid strict-parsed ratings verified before
fitting anything. Reference levels set explicitly: `prompt_type=original`,
`model=llama` (patsy's alphabetical default would otherwise silently make
`health` the prompt_type reference, breaking every "context minus
original" comparison this project has used throughout).

## Model diagnostics — read this before trusting any coefficient below

- **Converged**: yes, on the first (default) optimizer attempt, for all
  three fits required (primary REML, and both ML fits for the LRT). No
  fallback to `lbfgs`/`powell` was needed — a notable contrast with
  `analysis/05_hypothesis_models.py`'s per-model fits, two of which
  (llama, ministral) needed a fallback optimizer at the smaller
  single-prompt-type scale. Fit times: 17.0s (primary REML), 13.1s (ML
  full), 12.4s (ML reduced) — all fast; total pipeline runtime under 2
  minutes.
- **A real, worth-explaining diagnostic**: every fit raised `"The MLE may
  be on the boundary of the parameter space"` and `slogdet` divide-by-zero/
  overflow warnings during the optimizer's search path. This is not a
  convergence failure — `res.converged == True` in every case, and the
  final estimates are stable — but it does mean the persona-level
  random-intercept variance is being estimated very close to its lower
  boundary (0). This has a clean substantive explanation, not just a
  numerical quirk: **`persona_id` is a bijective function of (country,
  gender, age, profession)** in this study's design — every unique
  combination of those four already-included fixed effects maps to
  exactly one persona. Once those four fixed effects (plus topic,
  prompt_type, and model) are already in the model, there is very little
  *additional* persona-specific signal left for a random intercept to
  capture — only whatever correlation exists among a given persona's 35
  repeated rows (7 topics × 5 prompt types) beyond what the fixed effects
  already explain. The data confirm this reading directly: **random-effect
  (persona) variance = 0.00109, residual variance = 0.174, ICC = 0.0062**
  — only 0.6% of leftover variance is attributable to between-persona
  clustering. This is a legitimate finding, not a fitting problem: it says
  cluster-robust SEs (used throughout the rest of this project) and this
  formal random-intercept approach should give quite similar inferential
  conclusions in this pooled setting, which is reassuring rather than
  concerning.
- **Marginal R² (fixed effects only) = 0.6981; Conditional R² (fixed +
  random) = 0.7000.** The two are nearly identical — a direct consequence
  of the near-zero ICC above: almost all explained variance comes from the
  fixed effects, essentially none from the random intercept.

## Sanity checks against prior findings (as requested — checked directly, not asserted)

**Does topic dominate, matching every prior per-model finding of 0.56-0.84
R²?** Yes, decisively: **topic's partial R² = 0.674**, by far the largest
of the seven candidate terms (age, the smallest, is 0.0034 — a ~200x gap).
Confirmed independently: dropping `topic` (and its model-interaction)
crashes the OLS R² from 0.698 to 0.075 — the single largest drop of any
term tested.

**Does profession/gender/country importance match Stage 1's per-model H1
findings?** **Partially checkable, and I want to be precise about the
limit here rather than overclaim a match.** The formula as specified
interacts *topic* and *prompt_type* with `model`, but not profession,
country, gender, or age — so this model cannot directly test whether
profession beats gender for qwen specifically the way Stage 1's per-model
fits could; it only estimates one *shared, averaged* profession/country/
gender/age effect across all four models. What it *can* check: Stage 1
found llama and gemma gender/country-dominant while qwen and ministral are
profession-dominant (a real per-model split). The pooled model's
**profession (0.0339) and gender (0.0338) partial R² come out almost
exactly equal** — which is precisely the pattern you'd expect from
averaging two gender-dominant models with two profession-dominant models
together, not a contradiction of Stage 1's per-model result.

**Documented scope boundary, not a gap that blocks anything current**
(confirmed as a deliberate decision, not deferred by oversight): a formal
joint test of per-model profession/country/gender importance would need
`profession:model`/`country:model`/`gender:model` interactions added — a
substantially larger model (each adds roughly as many parameters as
`topic:model`'s 18) than the one specified here. Decided not to build that
now: the current model already answers the primary questions (unified
variance decomposition, formal framing-sensitivity test) cleanly, and
Stage 1's separately-fit per-model H1 results already provide a defensible
— if not jointly statistically tested against each other — answer to which
factor dominates for which model. If a future stage specifically needs a
formal cross-model significance test of *that* question (e.g. "is qwen's
profession-dominance significantly different from llama's gender-
dominance"), it should be scoped as its own model extension rather than
folded into this one after the fact.

## A significant discrepancy, investigated and root-caused (not silently resolved)

The per-model `prompt_type` "simple slopes" computed here (Section below)
came out **roughly double** the previously-cited rating-shift numbers for
the same cells — e.g. Llama/health: this model says **−0.158**, but
`CONTEXT_EXPERIMENT.md` and this session's own `analysis_taxonomy` Stage 4
cite **−0.084** for the same comparison. Per your explicit instruction,
this was investigated rather than either number being silently preferred.

**Root cause, confirmed directly against raw data:**
`analysis_context/output/health_vs_original_summary_full5400.csv` — the
file Stage 4 sourced `avg_abs_rating_shift` from — reports its
`clustered_mean_shift_ctx_minus_orig` **pooled across both Condition A and
Condition B** (`n_matched_cells=75,600`, i.e. 37,800 A + 37,800 B rows;
confirmed at the code level: `analysis_context/01_compare_context_vs_original.py`'s
`section_matched_cells()` never filters by `response_condition`). This
project's own stated convention throughout (`analysis_plan.md`,
`analysis_health/CORRECTED_SUMMARY.md`: *"Condition A (forced-choice,
primary evidence)"*) treats Condition A as primary — but the pooled number
lives in the more prominently-named "summary" file, while the correctly
Condition-A-scoped number sits in a separate, less obvious companion file
(`health_vs_original_by_condition_full5400.csv`, `condition=="A_forced"`
row).

**Directly verified, both ways:**
- Recomputing the raw mean difference for Llama/health, Condition A only,
  from `results/results_original_llama.csv` and `results/results_health_llama.csv`
  directly: **−0.1582** — matches this unified model's coefficient to 4
  decimal places (both are, in fact, the exact same quantity: in this
  perfectly balanced factorial design, the OLS `prompt_type` coefficient
  *is* the simple/matched-pairs mean difference).
- Recomputing the same raw difference pooling Condition A+B: **−0.0843**
  — matches the "summary" file's cited −0.084 almost exactly.
- Condition B alone: −0.0103 — much smaller, confirming Condition B's
  inclusion is what dilutes the pooled figure.
- **Confirmed this pattern holds for all 4 models × all 4 contexts** (16
  cells checked via `n_matched_cells`/`n_both_valid_numeric`, all showing
  75,600 = both-conditions pooled).

**Which number is right depends on the question being asked, and both
have legitimate uses — but they are not interchangeable, and prior work in
this project (specifically Stage 4's `context_sensitivity_index.csv`) used
the pooled number where the project's own stated convention says
Condition A should be primary.** `CONTEXT_EXPERIMENT.md` itself appears
unaffected — spot-checking, its prose already cites the correct
Condition-A-scoped value for this exact cell (health/llama: −0.158) — this
error is specific to this session's own Stage 4 taxonomy build, not an
error in the underlying project documentation.

**This is exactly the same category of issue as Stage 1's DeepSeek
compliance-pooling catch** (`table1_compliance.csv` pooled vs.
Condition-A-specific) — a headline-named file silently pooling both
conditions where the project's convention expects Condition A alone.

**Fixed, per your go-ahead**: `analysis_taxonomy/04_build_context_sensitivity_index.py`
now sources `avg_abs_rating_shift` from the `by_condition` file's
Condition-A row instead, `context_sensitivity_index.csv` was regenerated,
and `CONTEXT_SENSITIVITY_INDEX_SUMMARY.md` was updated with an explicit
correction note (both files now state plainly what changed and why, same
pattern as every other correction tonight). Every value moved up (shifts
were understated, not overstated): llama 0.045→0.108, gemma 0.130→0.142,
qwen 0.119→0.144, ministral 0.041→0.046. No conclusion changed — Ministral
remains, by a wide margin, the smallest-rating-shift model despite having
the largest abstention swing in the taxonomy.

## Headline table: variance decomposition (`variance_decomposition_model.csv`)

| term | partial R² (OLS) | joint Wald p (raw) | p (BH) | n params |
|---|---|---|---|---|
| **topic** | **0.674** | <10⁻³⁰⁰ | <10⁻³⁰⁰ | 24 |
| **model** | **0.412** | <10⁻³⁰⁰ | <10⁻³⁰⁰ | 33 |
| **topic:model** | 0.355 | <10⁻³⁰⁰ | <10⁻³⁰⁰ | 18 |
| profession | 0.034 | <10⁻³⁰⁰ | <10⁻³⁰⁰ | 29 |
| gender | 0.034 | <10⁻³⁰⁰ | <10⁻³⁰⁰ | 2 |
| prompt_type | 0.021 | <10⁻³⁰⁰ | <10⁻³⁰⁰ | 16 |
| country | 0.009 | <10⁻³⁰⁰ | <10⁻³⁰⁰ | 19 |
| prompt_type:model | 0.005 | <10⁻³⁰⁰ | <10⁻³⁰⁰ | 12 |
| age | 0.003 | 1.7×10⁻³⁰⁰ | 2.6×10⁻³⁰⁰ | 2 |

All nine terms remain significant after BH correction across the full
26-test family (9 term-level tests + the LRT + 16 simple slopes) — at
n=756,000, even the smallest effect (age, partial R²=0.003) is estimated
with enough precision to be non-zero beyond any reasonable doubt. **The
p-values here answer "is this effect exactly zero," not "is this effect
large"** — partial R² is the number that speaks to magnitude, and the
~200x gap between topic (0.674) and age (0.003) is the actually
informative comparison, not the uniformly-tiny p-values.

Note on `topic` and `model`'s partial R² (0.674, 0.412): these are each
tested *together with their model-interaction* (topic+topic:model;
model+both its interactions) — stated explicitly because dropping only a
main effect while leaving its interaction in the formula turns out to be a
methodological trap: patsy's full-rank coding lets the interaction alone
silently reabsorb the main effect's entire contribution, making that
comparison test nothing. (Caught directly: an earlier version of this
script's string-based formula construction accidentally did exactly this,
producing a nonsensical `model:model` self-interaction and two terms with
suspiciously identical R² — rewritten using explicit term-list
construction instead of string surgery, and cross-checked that the OLS
parameter-count drop matches the Wald test's degrees of freedom for all
nine terms before trusting either.)

## Formal test: does framing sensitivity really differ between models?

**Yes — likelihood-ratio test, full model vs. model without
`prompt_type:model`: χ²=4080.5, df=12, p≈0** (ML fits, as required for a
valid fixed-effects LRT comparison — REML likelihoods aren't comparable
across models with different fixed effects). The taxonomy's "Ministral is
the most framing-sensitive model" claim now has a formal test behind it,
not just an eyeballed comparison of point estimates.

**Per-model simple slopes** (`model_framing_sensitivity_test.csv`, vs.
original, Condition A, BH-corrected across all 16 + the LRT + 9 term
tests):

| prompt_type | llama | gemma | qwen | ministral |
|---|---|---|---|---|
| health | −0.158 (p≈0) | −0.195 (p≈0) | −0.181 (p≈0) | −0.105 (p≈0) |
| neutral | −0.106 (p≈0) | −0.140 (p≈0) | −0.069 (p≈0) | +0.003 (p=0.37, **n.s.**) |
| positive | −0.075 (p≈0) | −0.063 (p≈0) | −0.186 (p≈0) | −0.0004 (p=0.90, **n.s.**) |
| negative_minor | −0.092 (p≈0) | −0.171 (p≈0) | −0.140 (p≈0) | −0.075 (p≈0) |

Ministral is the only model with any non-significant cells (neutral,
positive) — its rating-level response to framing is the most selectively
null, even though (per Stage 4) its *abstention* response to framing is
the largest in the whole project. This is now a formally tested
confirmation of that exact pattern, not just consistent with it: **the
model whose willingness to answer swings the most is, on the content of
its answers specifically, the model most likely to show no reliable shift
at all.** These two findings — ministral's neutral/positive cells being
genuinely null here, matching `CONTEXT_EXPERIMENT.md`'s own prior
description of the neutral/ministral cell as "genuinely stays null, not
just underpowered" — agree directionally with prior work even though the
model here is adjusted for profession/country/gender/age/topic rather than
a raw matched-pairs comparison.

## BH correction gap on `analysis_taxonomy/` (secondary task, done)

Searched every file in `analysis_taxonomy/output/` for an actual p-value
column (not a substring false-positive like `bootstrap_top_pct`) — exactly
two carry one: `ranking_robustness_pvalue_summary.csv` (32 permutation
p-values) and `consensus_pattern_chisquare.csv` (8 chi-square p-values).
Treated as one predefined family (40 p-values), BH-corrected, `p_bh` +
`significant_bh_0.05` columns added **in place, raw p-values untouched**.
Result: **all 40 remain significant after correction** — every one of
these p-values was already so small (driven by the 5,400-persona /
37,800-cell sample sizes) that BH correction changes no conclusion.

## Plain-language verdict

**This confirms and formalizes the taxonomy's findings — it does not
overturn them — and it surfaced one real, previously-uncaught scope error
in this session's own prior work, which has now been corrected at its
source.**

- Topic dominance, the near-equal profession/gender split (consistent with
  averaging two gender-dominant and two profession-dominant models), and
  Ministral's distinctive pattern (largest abstention swing, most
  selectively-null rating shifts) all replicate cleanly with proper
  confidence intervals and a real significance test behind them now.
- The one discrepancy found — pooled vs. Condition-A-only rating shifts —
  was fully explained, not left a mystery: a scope-labeling issue in
  `analysis_context/01`'s "summary" output (present since that script was
  written), whose only *within-this-project* downstream consequence was
  Stage 4's `avg_abs_rating_shift` column. **Fixed**: that column now
  sources the Condition-A row of the `_by_condition` file, `context_sensitivity_index.csv`
  was regenerated, and both it and its summary doc state the correction
  plainly. No conclusion changed.
- The requested per-model profession/country/gender importance check is
  only partially answerable by the model as specified (it doesn't include
  those interactions). **Confirmed as a deliberate, documented scope
  boundary** — the current model already answers the primary questions
  cleanly, and Stage 1's per-model H1 fits already provide a defensible
  (if not jointly statistically tested) answer to per-model demographic
  importance. Noted as a possible future extension, not a gap blocking
  anything current.

**This closes out the unified model work.** Diagnostics reviewed, the
discrepancy investigated and resolved at its source, and the scope
boundary documented rather than silently worked around — per your
confirmation.
