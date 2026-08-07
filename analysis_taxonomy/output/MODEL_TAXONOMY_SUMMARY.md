# Stage 1: Model Behavioral Taxonomy

Synthesizes already-verified findings from across this project into one
master table, one row per model. **No new statistical claims are made
here** — every cell is either read directly from an existing, already-
audited output file, or a trivial arithmetic derivation (max-min range,
count of distinct values, mean of already-computed correlations) over
already-verified numbers. All full-scale (5,400 personas, 20 countries, 30
professions) throughout — no 180-persona pilot subset used anywhere.

**Anchor claim, adopted and confirmed for the taxonomy work going
forward** (refined from the originally-proposed sentence below after this
table's evidence showed the plain version overstated stereotype-content
stability):

> **Models differ far more in their willingness to answer at all — ranging
> from near-total refusal-to-abstain to abstaining on the large majority of
> opportunities in every framing tested, and swinging by tens of
> percentage points within a single model depending on conversational
> framing — than in the actual demographic stereotypes they attribute,
> where models agree with each other at a consistently moderate-to-strong
> level and where even the models whose "dominant factor" label changes
> across framings do so by modest margins.**

Originally-proposed target sentence this analysis was built to test (see
"Does the data support..." below for the full comparison that motivated the
refinement above): *"Models differ more in willingness to answer than in
the stereotypes they produce."*

Table: [`model_taxonomy.csv`](model_taxonomy.csv). Build script and full
source citations: `analysis_taxonomy/01_build_model_taxonomy.py`.

## Sources (each opened and read directly, not taken from a prior summary)

| Column(s) | Source file | Producing script |
|---|---|---|
| Abstention rate (5 prompt types) + range | `analysis_context/output/abstention_stability_rate_table.csv` | `analysis_context/03_abstention_stability_across_conditions.py` (always full-scale) |
| Dominant factor (5 prompt types) + stability | `analysis_context/output/dominant_factor_by_model_full5400.csv`, cross-checked against the raw per-factor partial R² in `analysis_context/output/variance_ranking_all_prompt_types.csv` | `analysis_context/05_variance_ranking_all_prompt_types.py` (full-scale scope only) |
| Cross-model rating agreement | `tables/cross_model_spearman_matrix.csv` + `tables/cross_model_agreement_deepseek_pairs.csv` | `analysis/07_cross_model_agreement.py` (main study, always full-scale) |
| Strict compliance, Condition A | `tables/table1_compliance.csv` (pooled A+B) + `tables/deepseek_compliance_by_condition.csv` (Condition-A-specific, needed only for DeepSeek — see below) | `analysis/03_compliance_table.py`, `analysis/09_deepseek_report.py` |

## Verification notes (discrepancies caught, not silently smoothed over)

1. **`table1_compliance.csv` is pooled Condition A+B, not Condition-A-only
   as the task requested it be read as.** Confirmed by reading
   `analysis/03_compliance_table.py`'s source: it filters only by model,
   never by condition. For llama/gemma/qwen/ministral this doesn't matter
   — their pooled rate is exactly 100.0%, which makes the Condition-A-only
   rate 100.0% too by logical necessity (100% of all rows valid implies
   100% of any subset, including Condition A alone). For **DeepSeek it does
   matter**: pooled = 0.0833% but Condition-A-only = **0.1667%**, exactly
   double, because all 63 of DeepSeek's strict-valid rows across the whole
   dataset happen to fall under Condition A (confirmed directly in
   `deepseek_compliance_by_condition.csv`'s `B_optional` row, whose `none`
   column is exactly 0.0). Using the pooled number for DeepSeek would have
   been silently wrong by a factor of 2.
2. **Dominant factor independently recomputed from raw partial R² values**
   in `variance_ranking_all_prompt_types.csv` (taking the max per model x
   prompt type x full-scale scope) and compared cell-by-cell against
   `dominant_factor_by_model_full5400.csv` — exact match on all 20 cells
   (4 models x 5 prompt types).
3. **DeepSeek has no dominant-factor row** in the source file — it is
   excluded from `analysis_context/05`'s H1 fit throughout (too few valid
   Condition-A rows to fit meaningfully in any prompt type). Recorded as
   `NA` with an explicit note in the table, not left blank or imputed.
4. **DeepSeek's cross-model agreement numbers are reported with an explicit
   reliability caveat**, not presented as equivalent to the other four
   models': each of its 4 pairwise correlations is computed on only n=63
   both-valid rows and is individually flagged `flagged_too_sparse=True` in
   its source file.

## The master table

model | abstention: original / health / neutral / positive / negative_minor (%) | abstention range (pp) | dominant factor: original / health / neutral / positive / negative_minor | stable? | avg. cross-model ρ | Cond-A compliance (%)
---|---|---|---|---|---|---
llama | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 | 0.0 | gender / gender / gender / gender / gender | **yes** | 0.594 | 100.0
gemma | 0.0 / 0.0 / 0.0 / 0.1 / 0.0 | 0.1 | country / profession / profession / gender / gender | no (3 distinct) | 0.680 | 100.0
qwen | 90.0 / 92.1 / 86.9 / 94.7 / 90.9 | 7.7 | profession / gender / profession / gender / profession | no (2 distinct) | 0.626 | 100.0
ministral | 83.5 / 48.5 / 41.8 / 37.3 / 29.8 | **53.7** | profession / gender / gender / gender / gender | no (2 distinct) | 0.638 | 100.0
deepseek | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 | 0.0 | NA (excluded from H1 fit) | NA | −0.366 (unreliable, n=63) | 0.167

## Narrative profiles

**Llama** never abstains, in any of the five prompt types (0.0% across the
board, and 100.0% strict-format compliance under the original prompt's
Condition A) — of the five models, it is the only one for which the
question "does context change its willingness to answer" simply does not
arise, because it always answers. Its dominant demographic factor is
**gender**, consistently, in all five prompt types, and by a wide margin
every time (partial R² gap over the runner-up ranges from 0.012 to 0.094) —
the most behaviorally stable model in the study on both axes. It agrees
most closely with Gemma among the other three compliant models (ρ=0.722,
the single highest pairwise correlation in the whole matrix) and least with
Ministral (ρ=0.524); its own average agreement with the other three is
0.594.

**Gemma** is also a near-total non-abstainer (0.0-0.1% across all five
conditions, a 0.1-percentage-point range that is functionally noise), and
is 100.0% strict-format compliant. Its dominant-factor picture is the one
genuinely mixed case in this table: **country** wins under the original
single-turn prompt, **profession** wins under health and neutral, and
**gender** wins under positive and negative_minor — three distinct factors
across five prompt types. But two of those three "wins" (health and
neutral) are decided by a partial R² margin of just 0.0003 against gender —
an order of magnitude tighter than any other model's dominant-vs-runner-up
gap in this table, and close enough to be read as a coin-flip rather than a
robust behavioral difference. Gemma agrees most closely with Llama
(ρ=0.722) and, on average, is the most cross-model-agreeable model in the
table (avg. ρ=0.680).

**Qwen** sits at the opposite extreme from Llama and Gemma on willingness
to answer: it abstains on **87-95% of Condition-B rows in every single one
of the five prompt types** — the least willing model to commit to an
opinion when given the option, in every framing tested, with only a 7.7
percentage-point range across framings (i.e., stably reluctant, not
context-sensitive in its abstention). Its dominant factor is **profession**
under original, neutral, and negative_minor, but **gender** takes over
specifically under health and positive — the two framings that add
personal content about the persona (vulnerability or achievement) rather
than an impersonal event — with real, non-trivial margins (0.012-0.091).
Its average cross-model agreement is 0.626, and its strict compliance under
Condition A is 100.0%.

**Ministral** is the model whose behavior most directly demonstrates the
target sentence's claim: its Condition-B abstention rate swings from
**83.5% under the original prompt down to 29.8% under negative_minor** — a
**53.7-percentage-point range**, by far the largest context-sensitivity in
this table (7x qwen's range, and the only "NOT STABLE" verdict in the
underlying abstention-stability analysis). Its dominant factor, by
contrast, only differs in one of five prompt types: **profession** wins
under the original prompt alone, and **gender** wins under every one of the
four conversational framings, with no further movement among those four
despite abstention swinging by tens of points across those same four
framings. Average cross-model agreement is 0.638, Condition-A compliance is
100.0%.

**DeepSeek** does not cleanly fit either side of the target sentence's
dichotomy, because its failure mode is neither "answers" nor "abstains" in
the sense the other four models' data supports — it is near-total
non-compliance with the required output format. Strict-format compliance
under the original prompt's Condition A is **0.167%** (63 of 37,800 rows;
note this is double the often-quoted pooled A+B figure of 0.083%, since all
63 of its valid rows happen to fall under Condition A). It essentially
never produces a valid explicit "NA" either (0.0% abstention rate in every
one of the five prompt types), so its 0.0% "abstention rate" is not
evidence of confident opinion-attribution the way Llama's is — it reflects
a near-total inability to produce parseable output of any kind, not a
behavioral choice. It is excluded from the dominant-factor analysis
entirely (too few valid rows to fit the model meaningfully), and its
cross-model rating agreement (avg. ρ=−0.366, individually negative and
weak/moderate against all four other models) is explicitly flagged
unreliable in its own source file (n=63 per pair, `flagged_too_sparse=True`
throughout) — not a genuine finding about DeepSeek's stereotype content,
just noise from a near-empty sample.

## Does the data support "models differ more in willingness to answer than in the stereotypes they produce"?

**Yes, and by a wide margin — with one honest caveat.**

**Willingness-to-answer evidence (the larger effect, by every measure
here):**
- Between-model spread in mean abstention rate across the five prompt
  types: **0.0% (Llama/DeepSeek) to 90.9% (Qwen) — a 90.9-percentage-point
  spread.** The single widest cell-to-cell range in the whole table is
  0.0% to 94.7% (Qwen/positive).
- Within a single model, willingness to answer is itself highly
  context-sensitive: Ministral's own abstention rate swings by 53.7
  percentage points depending on conversational framing alone, with the
  underlying per-context shifts individually significant at p<10⁻⁴⁴ to
  p=0 (already-audited numbers, not recomputed here).
- These are not small, marginal, or borderline differences — Llama/Gemma/
  DeepSeek functionally never abstain (≤0.1%) while Qwen abstains on the
  large majority of opportunities (87-95%) in *every* framing tested, a
  near-categorical split between models, not a gradient.

**Stereotype-content evidence (the smaller effect):**
- Cross-model rating agreement — i.e., how similarly the four compliant
  models actually rate personas — clusters in a moderate-to-strong,
  **consistently positive** 0.524-0.722 band (spread of only 0.086 in the
  per-model averages; 0.198 across all six raw pairwise values). Every
  pair agrees more than chance, and none is dramatically more or less
  aligned than another.
- Dominant demographic factor does change across framings for 3 of the 4
  measurable models (only Llama is fully stable) — so it would be
  inaccurate to claim stereotype *content* never shifts. But the
  magnitude of that shift is generally modest (partial R² margins mostly
  0.01-0.09) and in Gemma's case specifically sits at 0.0003 — an order of
  magnitude tighter than any other dominant-vs-runner-up gap in this
  table, close enough to the noise floor that "3 distinct dominant
  factors" overstates how different Gemma's actual attribution pattern is
  across framings. Ministral, the model with the single largest
  abstention swing (53.7pp), has the *smallest possible* number of
  dominant-factor changes (1 of 5) — the model most sensitive to framing
  on the answer-or-not axis is among the least sensitive to framing on the
  which-factor-dominates axis.

**Conclusion:** the target sentence is supported by this table, not just
directionally but by a large margin — the between-model and within-model
spread on willingness-to-answer (tens of percentage points, categorical
splits, high statistical significance throughout) dwarfs the spread on
stereotype content (a consistently-positive 0.52-0.72 agreement band, and
dominant-factor shifts that are real for most models but generally
modest-to-marginal in size). The one place the plain version of the
sentence overstates the evidence is implying stereotype content is
*stable* — it is stable for only 1 of 4 measurable models (Llama); a more
precise version would be:

> **Models differ far more in their willingness to answer at all — ranging
> from near-total refusal-to-abstain to abstaining on the large majority of
> opportunities in every framing tested, and swinging by tens of
> percentage points within a single model depending on conversational
> framing — than in the actual demographic stereotypes they attribute,
> where models agree with each other at a consistently moderate-to-strong
> level and where even the models whose "dominant factor" label changes
> across framings do so by modest margins.**
