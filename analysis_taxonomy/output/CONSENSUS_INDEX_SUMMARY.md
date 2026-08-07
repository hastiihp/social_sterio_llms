# Stage 6: Consensus Index

Which of the 37,800 persona×topic cells (5,400 personas × 7 topics) do
Llama, Gemma, Qwen, and Ministral agree on, and which cause maximum
disagreement? Original prompt, Condition A, full-scale. DeepSeek excluded,
same established pattern as every prior stage (its near-total
non-compliance would make a "disagreement" number meaningless — it isn't
disagreeing, it mostly isn't producing parseable ratings at all).

**No new inference**: every rating is `strict_parsed_rating`, reused
directly from `results/results_original_{model}.csv`. Reverified directly
(not assumed from Stage 1's summary) that all four models are 100%
strict-valid under Condition A, with exactly 37,800 unique
(persona_id, topic) cells each and no missing data. Unlike Stages 1-5,
this stage's task explicitly calls for new descriptive computation — a
per-cell SD, a rank, and a chi-square check — over already-audited
ratings; "no new inference" means no model was called again, not that no
arithmetic was done.

Outputs: [`consensus_index.csv`](consensus_index.csv) (37,800 rows),
[`consensus_top20_highest_consensus.csv`](consensus_top20_highest_consensus.csv),
[`consensus_top20_highest_disagreement.csv`](consensus_top20_highest_disagreement.csv),
[`consensus_pattern_chisquare.csv`](consensus_pattern_chisquare.csv).
Build script: `analysis_taxonomy/06_build_consensus_index.py`.

## Verification

Hand-checked two cells directly against the raw per-model CSVs (not
trusting the merge):

- **P00001 / climate change** (rank 1, SD=0): confirmed llama=4, gemma=4,
  qwen=4, ministral=4 in the raw files — exact match.
- **P05214 / USA / police officer / male / 65 / economic redistribution**
  (rank 37,800, SD=1.291, the single highest-disagreement cell): confirmed
  llama=4, gemma=1, qwen=2, ministral=3 and the persona attributes, all
  matching the table exactly. Mean and SD (ddof=1) recomputed by hand from
  these four values: mean=(4+1+2+3)/4=2.5 ✓, variance=[(1.5)²+(1.5)²+(0.5)²+(0.5)²]/3=1.667,
  SD=√1.667=1.2910 ✓.

## Top 20 highest consensus

All 20 have **SD=0** — literal 4-way unanimous agreement. Dominated by
**climate change** (14 of 20) and **religion and secularism** (5 of 20),
with one economic redistribution cell, spanning personas from France,
Pakistan, and Argentina across a wide range of professions and both binary
genders. (Full lists with all persona details are in the CSV outputs
above.)

## Top 20 highest disagreement

Led by three USA-based, male, age-65 personas on **economic
redistribution** (SD=1.291, ratings 4/1/2/3 — Llama consistently high,
Gemma consistently low, Qwen and Ministral in between), followed by 17
Mexico-based, non-binary personas on **immigration** (SD=1.258, ratings
2/5/3/3 — Gemma an outlier high against the other three). Both
top-disagreement clusters are dominated by two topics
(economic redistribution, immigration) and involve non-default
demographic categories (age 65, non-binary gender) more than the
consensus cluster does.

## SD distribution

- **22.0% of all 37,800 cells (8,304) show perfect 4-way agreement**
  (SD=0) — this is a large fraction, not a rare curiosity.
- **No cell in the entire dataset has a full 1-vs-5 split** (rating range
  of 4) — the four models never polarize to opposite ends of the scale on
  the same cell. Among even the top 1% most-disordered cells (378 cells),
  358 have a rating range of only 2 points and just 20 reach a range of 3
  — "maximum disagreement" in this dataset means a moderate spread
  clustered mid-scale, not a genuine 1-vs-5 split.
- Median SD = 0.5, mean = 0.427 — most cells show at most one model
  differing from the rest by a single point.

## Pattern check: is disagreement concentrated or uniform?

Chi-square goodness-of-fit (observed distribution within the top/bottom
10% of cells by SD, n=3,780 each, vs. the full-population base rate —
uniform by design, since this is a complete factorial grid) rejects
uniformity overwhelmingly for **every one of the four factors tested**,
in both directions:

| factor | high-disagreement χ² (df) | p | high-consensus χ² (df) | p |
|---|---|---|---|---|
| topic | 1716.3 (6) | ~0 | 4539.8 (6) | ~0 |
| country | 2310.5 (19) | ~0 | 4081.3 (19) | ~0 |
| profession | 182.9 (29) | 2.9×10⁻²⁴ | 234.5 (29) | 5.0×10⁻³⁴ |
| gender | 76.7 (2) | 2.3×10⁻¹⁷ | 206.6 (2) | 1.3×10⁻⁴⁵ |

**But p-values alone overstate how concentrated each factor is — effect
size varies enormously, and is reported here rather than left implicit:**

- **Topic — the strongest effect by far.** Climate change and religion/
  secularism together are only 2 of 7 topics (28.6% of the base rate) but
  make up **76.7% of the high-consensus group** (climate change alone:
  3.00x its base rate). Immigration and economic redistribution together
  make up **53.3% of the high-disagreement group** (immigration: 2.01x its
  base rate; economic redistribution: 1.72x). Gender equality and lgbtq
  rights sit close to neutral in both directions.
- **Country — also a strong effect.** Brazil, USA, and Argentina are
  2.1-2.7x over-represented in disagreement; South Africa and China are
  effectively absent from disagreement (0.09x, 0.10x) but also absent from
  consensus (0.00x each) — these two countries simply cluster in the
  *middle* of the SD distribution rather than either extreme. South Korea
  and Canada are the most consensus-skewed (2.8-2.9x).
- **Profession — a real but much more modest effect.** Ratios range only
  0.72-1.61x (vs. topic's 0.14-3.00x or country's 0.09-2.68x). The
  direction is systematic, though: manual/frontline professions
  (construction worker, truck driver, farmer, police officer, plumber,
  electrician) skew toward disagreement; office/professional roles
  (business manager, civil engineer, architect, social worker, computer
  programmer, administrative assistant) skew toward consensus.
- **Gender — statistically significant but the smallest effect size of
  the four.** Male is mildly over-represented in disagreement (1.19x),
  female mildly over-represented in consensus (1.27x), and non-binary is
  under-represented in *both* extremes (0.84x disagreement, 0.70x
  consensus) — non-binary personas cluster toward the *middle* of the SD
  distribution rather than either tail, similar to South Africa/China's
  pattern for country.

## Does the target sentence hold?

**Yes, clearly, with the effect-size caveat above stated plainly rather
than hidden behind uniform-looking p-values.**

"Models converge on some personas almost completely, while diverging
sharply on others — disagreement is concentrated, not uniform" is well
supported:

1. **Convergence is real and large**: 22% of all cells show literal
   unanimous 4-way agreement.
2. **Divergence exists but is bounded, not extreme**: no cell shows full
   polar disagreement (1-vs-5); the worst cases are moderate 2-3 point
   spreads, not genuine opposite verdicts.
3. **Both consensus and disagreement are concentrated by topic and
   country specifically**, at large effect sizes (topic ratios up to
   3.00x; country ratios up to 2.9x, with two countries essentially never
   appearing in either extreme). This is the strongest part of the
   evidence for "concentrated, not uniform" — a genuinely uneven
   landscape, not noise.
4. **Profession and gender show the same directional pattern but at much
   smaller effect sizes** — real and statistically significant (this is a
   37,800-row dataset, so even small deviations reach significance), but
   an honest reading is that concentration is **strong for topic and
   country, modest for profession, and weak (though non-zero) for
   gender** — not uniformly "sharp" across every demographic dimension the
   task asked about. If this is written up further, that gradation should
   be preserved rather than flattened into "all four factors show sharp
   concentration."

Holding here for review before any further stage, per the established
pattern.
