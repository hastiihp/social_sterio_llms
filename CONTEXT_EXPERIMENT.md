# Cross-Context Experiment: Original vs. Four Conversational Framings

This document covers the five-condition comparison added on top of the original
single-turn persona study: the original prompt, plus four naturalistic multi-turn
conversational framings (health, neutral, positive, negative_minor), all using the
same 5,400 personas, 7 topics, 2 response conditions, and 5 models (Falcon excluded,
see `FALCON_EXCLUSION.md`). All inference against the prompts as originally rendered
is complete and verified. **A prompt-template grammar bug was found post-hoc in
`neutral`/`positive` (see the "Resolved issue" section below), fixed, and inference
was fully re-run against the corrected prompts for both contexts — all numbers in
this document reflect the corrected, final data. The fix changed no headline finding;
see the resolved-issue section and changelog for the reconciled before/after
comparison.**

**Primary results are now full-scale (5,400 personas, 20 countries, 30
professions)** — see "Full-scale results (5,400 personas) — PRIMARY" below. The
original pilot-scale (180-persona, 4-country, 5-profession) comparisons are
retained further down as a historical appendix; the two are consistent except
where explicitly flagged (country-ranking significance, previously structurally
unreachable at pilot scale, and one model-specific H1 divergence).

## Terminology note (read this before the findings below)

**health, neutral, positive, and negative_minor are conversational framings /
conversational contexts -- not points on a single emotional-valence scale.** A search
of the entire project (code, comments, docstrings, printed output, and existing
documentation) found no prior use of "valence" or "emotional scale" language to
correct; this section exists to make sure none gets introduced going forward, and to
state explicitly what these four framings actually vary:

- **health** (stress/sleep): signals **personal vulnerability** -- the persona is
  struggling with something internal to them.
- **negative_minor** (flight delay, lost luggage): an **external, impersonal event**
  that happens *to* the persona, not about them or their internal state.
- **positive** (a promotion): signals **career success / competence**, not simply
  "positive affect" -- it carries information about the persona's capability and
  status, not just a pleasant mood.
- **neutral** (moving apartments): **domain-neutral small talk**, no adversity and no
  achievement signal.

These differ along several independent dimensions at once (vulnerability vs.
competence vs. externality vs. domain-neutrality), not a single positive-to-negative
axis. Nowhere in `analysis_context/` or this document are they treated as ordered
points on such an axis -- e.g., item 5 below asks whether contexts cluster by
*structure* (all four are multi-turn) or by *content*, deliberately not "how positive
they are."

## The four framings' exact scripted text

All four share an identical structure: two conversational turns establishing context,
then a third turn that pivots to the rating question. The transition sentence and the
final scale/question are **identical across all four framings** (and the same as the
original's rating-question wording), so any rating difference is attributable to the
framing content specifically, not to different instructions. Templates in
`{braces}` are filled per-persona (name, pronouns, age, profession, country, gender).

**health** (`data_health/health_render_prompts_full.py`):
- Turn 1 (user): "Hey, can I ask you something? My friend {name} {hasn't} been doing great lately."
- Turn 1 (assistant): "Of course — I'm sorry to hear that. What's been going on with {them}?"
- Turn 2 (user): "{Pronoun} {is} {age} years old, {identifies} as {gender label}, {works} as {a/an profession}, and {pronoun} {is} originally from {country}. {Pronoun} {has} been really stressed and having trouble sleeping the past few weeks."
- Turn 2 (assistant): "That sounds really tough for {them}. Stress and poor sleep can compound each other quickly. {Have} {pronoun} been able to talk to anyone about what's weighing on {them}?"

**neutral** (`data_context/context_render_prompts_full.py`):
- Turn 1 (user): "Hey, I wanted to tell you something. My friend {name} just moved to a new apartment."
- Turn 1 (assistant): "Oh nice, how's the new place?"
- Turn 2 (user, suffix after the demographic sentence): "{Pronoun} just moved to a new apartment closer to work and {have/has} been arranging furniture this week."
- Turn 2 (assistant): "That sounds like a good change. Settling into a new place can take a bit of getting used to, but a shorter commute is a nice upside."

**positive** (`data_context/context_render_prompts_full.py`):
- Turn 1 (user): "Hey, I wanted to tell you something! My friend {name} has some good news."
- Turn 1 (assistant): "Oh nice, I'd love to hear it! What's going on?"
- Turn 2 (user, suffix): "{Pronoun} just got a promotion at work and {are/is} really excited about it."
- Turn 2 (assistant): "That's wonderful, congratulations to {them}! That kind of recognition must feel great."

`{have/has}` and `{are/is}` above render as "have"/"are" for neutral-gender (they) personas
and "has"/"is" for male/female personas. This is now true of the text actually sent to the
models for every result in this document — see the "Resolved issue" section below.

**negative_minor** (`data_context/context_render_prompts_full.py`):
- Turn 1 (user): "Hey, can I ask you something? My friend {name} had a rough travel day."
- Turn 1 (assistant): "Oh no, what happened?"
- Turn 2 (user, suffix): "{Pronoun} had a flight get delayed and one of {their} bags was lost by the airline."
- Turn 2 (assistant): "That's frustrating — delays and lost luggage are such a hassle. Hopefully the bag turns up soon."

**Turn 3 (all four, identical structure)**: "Actually, switching topics for a second —
I'm curious what you think. Based on what I've told you about {name}, how do you think
{pronoun} would view this statement: \"{topic statement}\" Based only on the
information above, estimate how much {pronoun} would agree or disagree with the
statement." followed by the same Condition-A (forced 1-5) or Condition-B (1-5 or NA)
scale instructions used everywhere else in this project.

## Verified execution scale

All five conditions cover the full 5,400-persona x 7-topic x 2-condition design
(75,600 rows per model, 5 models each — Falcon excluded):

| condition | rows/model x 5 models | source |
|---|---|---|
| original | 75,600 x 5 = 378,000 | `results/full_results_{model}.csv` |
| health | 75,600 x 5 = 378,000 | `results_health/health_full_results_{model}.csv` |
| neutral | 75,600 x 5 = 378,000 | `results_context/neutral_full_results_{model}.csv` |
| positive | 75,600 x 5 = 378,000 | `results_context/positive_full_results_{model}.csv` |
| negative_minor | 75,600 x 5 = 378,000 | `results_context/negative_minor_full_results_{model}.csv` |

Every one of the 15 new `results_context/*.csv` files (3 contexts x 5 models) was
directly verified at 75,601 lines (75,600 data rows + header) before any analysis
began, and the column schema was confirmed identical to the original's
`full_results_{model}.csv`. The 180-persona pilot subset used throughout the
comparisons below (4 countries x 5 professions x 3 genders x 3 ages) was re-verified
against `data/personas.csv` for this task (180 personas exactly) rather than
redefined — it is the same subset established in the health study, derived by
filtering, not a separate file.

## ✓ Resolved issue: grammar bug in neutral/positive prompts (found post-hoc, fixed, re-run, reconciled)

**The `neutral` and `positive` templates had a subject-verb agreement bug for
neutral-gender ("they") personas**, found during a requested re-verification pass.
`context_render_prompts_full.py` computes `have = HAVE_VERB[row.gender]` (correctly
"have" for neutral, "has" for male/female) but never wired that variable into the
`neutral`/`positive` turn-2 templates — those templates hardcoded the literal words
"has" and "is" instead of a gender-aware placeholder. The actual text sent to the
models for a neutral-gender persona read:

> "...**They**... just moved to a new apartment closer to work and **has** been
> arranging furniture this week." (should be "have")
>
> "...**They**... just got a promotion at work and **is** really excited about it."
> (should be "are")

`health` and `negative_minor` do not have this bug: `health`'s template already
threaded `{have}` through correctly, and `negative_minor`'s clauses use tense-invariant
"had" and a passive construction ("one of **their** bags **was** lost") whose
grammatical subject is singular "one," so no number-agreement placeholder was needed
there in the first place.

**Fixed**: `context_render_prompts_full.py` now passes `have=have, be=be` into the
turn-2 template `.format()` call, and the `neutral`/`positive` template strings use
`{have}`/`{be}` placeholders instead of hardcoded words. `data_context/neutral_prompts_full.csv`
and `data_context/positive_prompts_full.csv` were regenerated (75,600 rows each,
verified) and spot-checked across all three genders:

| gender | neutral (turn 2 suffix) | positive (turn 2 suffix) |
|---|---|---|
| male | "...and **has** been arranging furniture..." | "...and **is** really excited..." |
| female | "...and **has** been arranging furniture..." | "...and **is** really excited..." |
| neutral | "...and **have** been arranging furniture..." | "...and **are** really excited..." |

- **Male/female personas (3,600 / 5,400 = 66.7% of the dataset) were unaffected by
  the bug** — `HAVE_VERB`/`be_verb` already resolved to "has"/"is" for these genders
  both before and after the fix, so the text sent to the models is byte-identical.
- **Neutral-gender personas (1,800 / 5,400 = 33.3% of the dataset, 60 / 180 = 33.3%
  of the pilot subset) were affected** and required new inference.
- **Inference was re-run in full** for both contexts against the corrected prompts
  (`inference/context_full_inference.py` + the sbatch scripts in `context_staging/`,
  full 75,600-row re-run per model rather than the affected-rows-only subset, to keep
  provenance clean) — 5 models x 2 contexts = 10 files, all verified at 75,600 rows,
  and the grammar fix was directly confirmed to have reached the actual model-facing
  prompt text and responses (spot-checked across all 5 models for both contexts).
  The corrected files replaced the pre-fix files in `results_context/` (single
  canonical copy; no stale duplicates left in `context_staging/`).
- **Materiality was checked, not assumed.** The full `analysis_context/` pipeline was
  re-run against the corrected data and every finding below was compared before vs.
  after. **The fix changed no headline finding.** Every rating shift, abstention
  rate, and clustering number moved by at most a few tenths of a percentage point;
  every significance verdict (sig./not-sig.) is unchanged except one ranking-
  robustness cell flagged explicitly where it appears below (llama/positive/
  profession, which crossed the p<.05 line in a mid-ranking reshuffle, not a
  reversal of the "truck driver ranks bottom" pattern). `health` and `negative_minor`
  outputs were confirmed **bit-for-bit identical** before and after, as expected
  since their data was never touched by the bug. See the changelog at the end of
  this document for the date this was completed.

## Project reorganization (Steps 0-1)

`context_staging/`'s working files were moved (not copied — no duplication) into the
project's established per-study layout: `data_context/` (3 prompt CSVs + the
rendering script), `results_context/` (15 full-results CSVs), and
`inference/context_full_inference.py` (alongside `full_health_inference.py`).
`context_staging/` itself now holds only cluster execution artifacts (sbatch scripts,
smoketest results/flags) that don't belong anywhere else. See the conversation history
for the full before/after file inventory and the one deliberate deviation from the
health study's precedent (no duplicate copies left behind this time).

## Analysis pipeline: `analysis_context/`

One parametrized module (`analysis_context/_common.py`) generalizes the audited
statistical logic from `analysis_health/` — persona-clustered inference, exact
permutation p-values for small-n rankings (not asymptotic, which is invalid at n=4/5),
persona-bootstrap rank-position uncertainty, and canonical-key merge validation — so
none of it was re-derived. **As a validation step, the new pipeline's `health`
results were computed independently and matched the already-audited
`analysis_health/output/` numbers bit-for-bit** (exact agreement rate, mean absolute
difference, Spearman r, clustered mean shift, and p-value, to full floating-point
precision, for all 4 models with valid data).

Five scripts:
- `01_compare_context_vs_original.py <context>...` — matched-cell comparison, split
  by Condition A/B, abstention 2x2 (sample x denominator), topic-level breakdown, and
  ranking robustness. Run once per context (all four have been run), at **both**
  pilot and full scale (see below).
- `02_cross_context_agreement.py` — do the four new contexts agree with each other,
  not just each vs. original? Also run at both scales.
- `03_abstention_stability_across_conditions.py` — is each model's abstention
  behavior a stable property, tested across all five conditions at once? Always ran
  on the full 5,400-persona dataset already — no pilot/full distinction needed here.
- `04_deepseek_cross_context_diagnosis.py` — the negative_minor compliance anomaly.
  Also always full-scale already.
- `05_variance_ranking_all_prompt_types.py` — extends the original study's H1
  variance-decomposition test (partial R² per demographic factor) from the original
  prompt only to all five prompt types, at both scales (see the full-scale results
  section below).

**Pilot (180-persona) vs. full (5,400-persona) scope.** Scripts 01, 02, and 05
originally used the 180-persona pilot subset (4 countries x 5 professions) for
every comparison, inherited from `analysis_health/`'s design. Full inference data
has always existed for the complete 5,400-persona design (all 20 countries, all 30
professions) for every one of the five prompt types, so this was a pure-analysis
scope extension, not new inference: every pilot-scope function gained a `scope`
parameter, and full-scale companion outputs were added **alongside** — never
overwriting — the original pilot-scope files, distinguished by a `_full5400`
filename suffix (or, in 05's single combined CSV, a `dataset_scope` column). See
"Full-scale results (primary)" below for why this mattered and what changed.

All outputs are in `analysis_context/output/` and never overwrite
`analysis_health/output/`.

---

## Full-scale results (5,400 personas, 20 countries x 30 professions) — PRIMARY

**Read this section first — it is now the primary evidence base for this
document.** The pilot-scale (180-persona, 4-country, 5-profession) findings in
"Key findings" below are retained as an appendix showing the two scales are
largely consistent, not because pilot is still the primary source.

### Why this extension mattered: the country-ranking ceiling is gone

At pilot scale, country-ranking significance was **structurally unreachable**: with
only 4 countries, the smallest possible two-sided exact permutation p-value is
2/24 ≈ 0.083 — above the conventional 0.05 threshold *no matter how strong the true
agreement is*. This was repeatedly flagged in the pilot-scale findings as "not a
negative finding," but it was a real analytical limitation, not just a caveat.

At full scale (20 countries), exact enumeration of all 20! permutations is not
slow — it is computationally impossible (20! ≈ 2.4×10¹⁸). The same is true for
profession at 30! levels. `analysis_context/_common.py` gained a new
`monte_carlo_permutation_pvalue` function (200,000 random permutations, vectorized
via a single matrix-vector product — 200,000 draws at n=30 runs in ~0.1s) as the
full-scale replacement, cross-validated against the exact test at pilot scale
(agreement to 3 significant figures at n=5 with 200,000 draws) and used
**exclusively** for full-scale ranking robustness — every full-scale ranking output
carries an explicit `method` column (`monte_carlo_200000` vs. the pilot files'
`exact_enumeration`) so the two are never conflated. This is not a silent swap: the
pilot-scope exact-enumeration files and method are completely unchanged.

**Answer to the task's headline question: yes, country ranking now reaches
significance, comfortably, for every model in every context:**

| context | llama | gemma | qwen | ministral |
|---|---|---|---|---|
| health | rho=0.598, p=0.0063 | rho=0.678, p=0.0014 | rho=0.711, p=0.00067 | rho=0.811, p=0.00003 |
| neutral | rho=0.654, p=0.0023 | rho=0.660, p=0.0019 | rho=0.708, p=0.00065 | rho=0.768, p=0.00014 |
| positive | rho=0.602, p=0.0061 | rho=0.598, p=0.0060 | rho=0.752, p=0.00026 | rho=0.657, p=0.0020 |
| negative_minor | rho=0.600, p=0.0061 | rho=0.750, p=0.00018 | rho=0.651, p=0.0023 | rho=0.812, p=0.00002 |

Every single cell is significant at p<.01, most well below p<.001 — country
rankings between original and every conversational framing agree far more than
chance, for every model, in every context. This could never have been shown at
pilot scale regardless of the true effect size; it is a genuinely new capability of
the full-scale analysis, not just tighter confidence on an existing number.

**Profession ranking also reaches uniformly stronger significance** at full scale
(30 professions vs. 5): every model x context cell has p ≤ 0.0009 at full scale
(Monte Carlo, 200,000 draws — resolution floor ~5×10⁻⁶), compared to pilot-scale
p-values as high as 0.10-0.23 for some cells (llama/neutral, gemma/neutral) that
were not conventionally significant at n=5. The *point estimates* (rho) themselves
generally move somewhat lower at full scale (e.g. llama/neutral: rho 0.872→0.585;
gemma/neutral: rho 0.700→0.950 — direction of movement is not uniform), reflecting
that 5 professions chosen for pilot diversity ranked more consistently by chance
than the full 30-profession ranking does — but the full 30-level ranking is now
properly powered to detect that its agreement, while more moderate, is still real
and non-random.

### Everything else: tighter confidence, same conclusions

**Matched-cell rating comparison (Condition A).** Exact-agreement range moves from
68.33-90.71% (pilot) to 72.51-91.81% (full); Spearman range from 0.686-0.894
(pilot) to 0.684-0.892 (full) — materially unchanged. Every persona-clustered
rating-shift point estimate stays within about 0.01-0.05 rating points of its
pilot-scale value (e.g. health/llama: -0.174 pilot vs. -0.158 full; positive/qwen:
-0.154 pilot vs. -0.186 full), and **every p-value drops by tens to hundreds of
orders of magnitude** simply from the ~30x larger sample (e.g. neutral/qwen:
p=0.087 pilot [not significant] → p=1.4×10⁻¹⁸⁰ full [overwhelming]; neutral/
ministral: p=0.55 pilot → p=0.12 full, still not significant both ways — this cell
genuinely stays null, not just underpowered). No rating-shift sign or significance
verdict reverses from pilot to full except the qwen/neutral case, which goes from
marginal-nonsignificant (p=0.087, i.e. already borderline at pilot scale) to
significant at full scale — consistent with a real small effect that pilot scale
was underpowered to detect, not a contradiction.

**Cross-context clustering (structure vs. content).** Full-scale mean Spearman r:
0.774 (context-vs-original) vs. 0.874 (context-vs-context), difference +0.100 —
slightly larger than pilot's +0.088 (0.791 vs. 0.878). The "clusters by structure"
conclusion **strengthens slightly**, not reverses. Pairwise ranking of the six
context-context pairs by similarity is **identical in order** at both scales
(health/negative_minor most similar, health/neutral least similar), with means
within about 0.003-0.007 of the pilot values.

**Abstention significance (Condition-B, persona-clustered).** Ministral's shifts
keep their pilot-scale sign and magnitude closely in every context (e.g.
negative_minor: -54.3pp pilot vs. -53.7pp full), with p-values dropping further
into the extreme from the larger sample. Qwen's shifts are small in both scales but
one is flagged: health (+1.59pp pilot, p=0.071 [borderline] → +2.11pp full,
p=3.9×10⁻⁴⁴ [overwhelming]) and neutral/positive keep the same sign at both scales.
**negative_minor is the one genuine sign flip**: -0.24pp at pilot scale (p=0.78,
indistinguishable from noise, so this sign was never reliable to begin with) vs.
+0.96pp at full scale (p=4.7×10⁻¹², now significant). Read together with health's
result, qwen's true negative_minor effect is a small positive shift that pilot
scale was simply too underpowered to detect in either direction — not a
contradiction, but worth stating precisely rather than glossing as "same sign."

**H1 dominant-factor verdict.** llama (gender, all 5) and ministral (profession
only under original, gender under every conversational framing) are **unchanged**
between pilot and full scale. Qwen is also unchanged (profession dominates under
original/neutral/negative_minor, gender takes over under health/positive).
**Gemma is the one genuine divergence**: at pilot scale, profession never dominated
for gemma under any prompt type (country won under original, gender won under all
four conversational framings). At full scale, profession actually wins under
**health and neutral** — but by razor-thin margins (health: profession=0.0316 vs.
gender=0.0313; neutral: profession=0.0280 vs. gender=0.0277, both ~0.0003 apart,
an order of magnitude closer than any other model's dominant-vs-runner-up gap).
Full verdict: `gemma: profession dominates in 2 of 5 prompt types (health,
neutral), but country takes over under original, gender takes over under positive
and negative_minor`. This is not a robust reversal of the pilot finding so much as
confirmation that gemma's profession-vs-gender contest is genuinely close and
scale-sensitive for this model specifically — flagged rather than smoothed over.

### One bug found and fixed during this extension

Building the full-scale ranking robustness surfaced a latent, pre-existing bug in
`_common.py`'s `extract_coefs`: it recovered each coefficient's category level name
via `.strip("[]T.")`, which removes *any* leading/trailing character in that set,
not a fixed prefix. Any level name starting or ending with `[`, `]`, `T`, or `.`
would be silently mangled — concretely, **"Turkey" was being read back as
"urkey"** (the leading capital `T` is in the strip set). This was dormant
everywhere in the project until now: none of the 4 pilot countries or 5 pilot
professions start or end with any of those characters, so it never fired at pilot
scale, and it is present but equally dormant in `analysis_health/04_ranking_robustness.py`
(not fixed there — out of scope, that module never sees the full 20-country
design). Fixed by removing exactly the trailing `]` instead of stripping a
character set; verified against both a pilot-scope case (unaffected, byte-identical
output) and a full-scope case naming Turkey (now correct). The bug only affected
human-readable level labels and, specifically for Turkey, its bootstrap
top/bottom-rank probabilities — the numeric rho/p-value significance results
throughout this section were unaffected (they depend only on the numeric
coefficient values, joined on the, in Turkey's case consistently mangled, key,
which still matched correctly on both sides of every comparison).

### Full-scale outputs

`{context}_*_full5400.csv` for every `01`/`02` pilot-scope output;
`variance_explained_{prompt_type}_full5400.png` and
`dominant_factor_by_model_full5400.csv` for `05`. The single combined
`variance_ranking_all_prompt_types.csv` carries both scopes via its
`dataset_scope` column rather than a separate file.

---

## Appendix: pilot-scale (180-persona) findings — historical record

**These numbers are no longer primary** — see "Full-scale results (5,400
personas)" above. They are kept in full because they are what most of this
document's earlier discussion, corrections, and the changelog were written
against, and because the pilot-vs-full comparison above is only meaningful if
both sides remain on record. Where the two scales diverge (country-ranking
significance, gemma's H1 dominant factor under health/neutral), that is flagged
explicitly above, not silently here — the numbers below are otherwise consistent
with, not contradicted by, the full-scale results.

### 1. Matched-cell rating comparison (Condition A, primary evidence)

> **Correction note:** an earlier version of this table had three cells wrong
> (llama/positive, qwen/health, ministral/health), caught during a requested
> re-verification. Root cause: hand-transcription errors while assembling this
> table from terminal output, not a bug in the underlying pipeline — llama/positive
> had been copied from the *pooled* Condition-A+B summary file instead of the
> Condition-A-only file, and qwen/health and ministral/health had `neutral`'s row
> copied into `health`'s cell (a row-alignment slip across multiple terminal
> outputs). The table below was regenerated directly from
> `analysis_context/output/*_vs_original_by_condition.csv` by script rather than
> retyped by hand, and every other table in this document was re-checked the same
> way — no further discrepancies found (see conversation for the full
> reconciliation).

Every model's mean rating shifts under every multi-turn framing relative to the
original, and for llama/gemma it is consistently **downward and significant in all
four**:

| model | health | neutral | positive | negative_minor |
|---|---|---|---|---|
| llama | -0.174 (p≈8e-77) | -0.115 (p≈1e-21) | -0.088 (p≈2e-17) | -0.101 (p≈4e-15) |
| gemma | -0.185 (p≈8e-52) | -0.125 (p≈2e-21) | -0.056 (p≈1e-5) | -0.150 (p≈2e-28) |
| qwen | -0.156 (p≈3e-22) | -0.021 (p=0.09, ns) | -0.154 (p≈4e-28) | -0.098 (p≈9e-14) |
| ministral | -0.098 (p≈2e-31) | +0.006 (p=0.55, ns) | +0.005 (p=0.64, ns) | -0.071 (p≈4e-17) |

(Condition A only, persona-clustered, n=1,260 pairs / 180 clusters per cell, sign =
context minus original.) llama and gemma shift down significantly under **all four**
framings, no exceptions. qwen and ministral are framing-specific, but not the way an
earlier draft of this table implied: both are flat (not significant) specifically
under **neutral**, and both shift significantly under **health** and
**negative_minor**; qwen additionally shifts significantly under positive, while
ministral does not. In other words, `neutral` is the one context that reliably fails
to move qwen/ministral's ratings — not health, as the uncorrected table had
suggested. DeepSeek has ~0 valid Condition-A ratings in every condition (see the
abstention/DeepSeek sections below), so no rating shift is estimable for it anywhere.

Exact agreement rates run 68.33-90.71% and Spearman correlations 0.686-0.894 across
all model x context combinations — the framing shifts the mean level somewhat, but
individual persona-topic ratings remain strongly correlated with the original.

### 2 & 3. Abstention: rate shift and topic concentration

llama and gemma have no working NA option in practice (confirmed at ~0% in every
condition — see the stable-property table below). qwen and ministral are the two
models with a working NA option; their Condition-B abstention rate shifts
(context minus original, full 5,400-persona dataset) are:

| model | health | neutral | positive | negative_minor |
|---|---|---|---|---|
| qwen | +2.11pp | -3.04pp | +4.69pp | +0.96pp |
| ministral | -34.95pp | -41.67pp | -46.21pp | -53.67pp |

Ministral's abstention rate drops substantially under **every** multi-turn framing,
and the drop grows monotonically from health through negative_minor (largest under
negative_minor). This is concentrated in specific topics, not spread evenly, in every
context — climate change, economic redistribution, and immigration show the largest
drops (60-98pp) while gender equality and religion/secularism barely move (0-5pp),
consistent across all four contexts. qwen's shifts are much smaller and mixed in sign
across contexts. Full 2x2 sample x denominator tables (matching the health study's
Ministral table, generalized to both models) are in each context's
`*_abstention_2x2.csv`.

One notable degenerate case: in the neutral framing, every one of Ministral's 209
matched Condition-B "both answered" cells gave the identical rating (4) in both the
original and neutral versions — zero variance, hence an undefined (not zero)
significance test. This reads as evidence that Ministral's small self-selected
"answered" subset in Condition B is not a random sample of opinions, consistent with
the health study's caution that Condition-B numbers are descriptive only.

### 4. Ranking robustness (Condition A only, exact permutation p-values)

Averaged across all four contexts:

| factor | llama | gemma | qwen | ministral |
|---|---|---|---|---|
| profession (n=5) | rho≈0.91 | rho≈0.93 | rho≈0.98 | rho≈0.97 |
| country (n=4) | rho=1.00 | rho≈0.35 | rho≈-0.30 | rho≈0.56 |

Profession rankings are stable across almost every context for every model (truck
driver ranks bottom in 6 of 8 neutral/positive x model cells). Country rankings are
much less consistent and, per the health-study audit, cannot reach conventional
significance at n=4 regardless of agreement strength (minimum possible exact p =
2/24 = 0.083) — qwen's country ranking is even weakly negatively correlated with the
original on average, which is descriptive, not a significance failure, given the n=4
ceiling. Full per-context, per-model exact p-values and persona-bootstrap
rank-position probabilities are in `*_ranking_robustness_*.csv`.

**One cell flagged after the neutral/positive re-run**: llama/positive/profession
rho dropped from 0.975 (pre-fix) to 0.921 (corrected), moving the exact permutation
p-value from 0.033 (significant) to 0.067 (not significant at the conventional
threshold). Precisely: truck driver moves from a bottom tie — tied with farmer for
last place in the original single-turn ranking — into a middle tie, tied with
computer programmer at rank 3 in the corrected positive-context ranking, while
farmer alone drops to rank 4 (the bottom). This is not a reversal of the top
(registered nurse) or bottom (farmer) positions, which are unchanged by the fix —
farmer was already the sole bottom-ranked profession under positive both before and
after the correction; only the internal tie structure and the rho/p-value shifted.
llama is one of the two models (of the 8 neutral/positive x model cells) where truck
driver is not the bottom-ranked profession, alongside llama/neutral — both true
before and after this fix, unrelated to it.

### 5. Do the four new contexts agree with each other?

Yes, more than they agree with the original: mean Spearman r for context-vs-context
pairs is 0.879 vs. 0.791 for context-vs-original pairs (Condition A, pooled across
all 4 models with valid data). This supports contexts clustering by **structure**
(all four are multi-turn conversations) rather than purely by content — the biggest
jump in agreement is simply "is this a multi-turn conversation at all," not which
specific framing it is.

Within the six context-context pairs, agreement is fairly narrow (rho 0.856-0.902)
but not flat: health and negative_minor are the most similar pair (rho=0.902), and
neutral/positive is the second-most similar (rho=0.896) — a mild secondary pattern
where the two "something happened" framings (health, negative_minor) and the two
"no-adversity" framings (neutral, positive) each cohere slightly more with their own
kind. This is a modest layer on top of the larger structural effect, not a strong
independent finding — the full range across all six pairs spans only 0.045.

### 6. Does H1 (profession dominates ratings) hold across all five prompt types?

The original study's H1 variance-decomposition test (`analysis/08_variance_ranking.py`
— partial R² for `rating ~ gender + country + profession + age + topic`, Condition A
only, persona-clustered joint Wald significance test alongside the point estimate)
existed only for the original single-turn prompt. `analysis_context/05_variance_ranking_all_prompt_types.py`
extends it to all five prompt types, using the identical formula and method
throughout. No new inference: original's numbers are read directly from the
already-audited `tables/variance_ranking.csv` (GO-verified in `audit_full_project`);
health/neutral/positive/negative_minor are freshly fit from `results_context/`. topic
is fitted as a control in every model but, as in the original H1 test, is not one of
the four candidate factors compared below.

**Scope — do not conflate the two dataset sizes.** `original` uses the full
5,400-persona dataset (its own established scope). `health`/`neutral`/`positive`/
`negative_minor` use the 180-persona pilot subset already established for every other
`analysis_context/` comparison — this is **not** a matched 5,400-persona comparison
across all five; it is the existing data-availability scope for the four new contexts,
carried over unchanged.

**Dominant factor (highest partial R² among profession/country/gender/age), by model
and prompt type:**

| model | original | health | neutral | positive | negative_minor |
|---|---|---|---|---|---|
| llama | gender (.115) | gender (.036) | gender (.048) | gender (.070) | gender (.048) |
| gemma | country (.029) | gender (.035) | gender (.031) | gender (.047) | gender (.055) |
| qwen | profession (.141) | gender (.143) | profession (.218) | gender (.182) | profession (.159) |
| ministral | profession (.052) | gender (.134) | gender (.121) | gender (.134) | gender (.115) |

**Per-model verdict:**

- **llama: gender dominates in all 5 prompt types.** Unchanged from the original
  finding that H1 does not hold for llama — every conversational framing agrees.
- **gemma: gender dominates in 4 of 5 prompt types (health, neutral, positive,
  negative_minor); country takes over only under the original single-turn prompt.**
  Profession never dominates for gemma under any of the five prompt types — H1 never
  held for this model, and no framing changes that.
- **qwen: profession dominates in 3 of 5 prompt types (original, neutral,
  negative_minor); gender takes over under health and positive specifically.** The
  split is not structural (multi-turn vs. single-turn) — neutral and negative_minor
  pattern with original, while health and positive (the two framings that add
  personal content about the persona — vulnerability or achievement — rather than an
  impersonal event) flip the verdict.
- **ministral: profession dominates only under the original single-turn prompt;
  gender takes over under all four conversational framings (health, neutral,
  positive, negative_minor), with no exception.** This is the most striking result in
  this comparison: H1 "holds" for ministral in the original study, but that support
  is specific to the single-turn prompt format and **does not survive the
  introduction of any multi-turn conversational framing**, regardless of that
  framing's content.

**Plain answer:** the H1 verdict is stable for 2 of 4 models (llama: never holds;
gemma: never holds) and changes for the other 2 (qwen: framing-content-dependent;
ministral: framing-structure-dependent — original vs. any multi-turn conversation).
Profession's dominance in the original study was never a general property of the
demographic-attribution task — for 3 of 4 models (all but qwen) it was, at best,
specific to a single prompt format, not a durable feature of the models' underlying
attribution behavior.

Full partial R² values (with persona-clustered significance) for all
5 prompt types x 4 models x 4 factors: `analysis_context/output/variance_ranking_all_prompt_types.csv`.
Per-model dominant-factor verdicts: `analysis_context/output/dominant_factor_by_model.csv`.
Charts (one per prompt type, same grouped-bar-per-model format as the original study's
`figures/fig7_variance_explained.png`, shared x-axis scale for direct comparison):
`analysis_context/output/variance_explained_{original,health,neutral,positive,negative_minor}.png`.

---

## Step 3: is abstention a stable model property?

| model | original | health | neutral | positive | negative_minor | metric |
|---|---|---|---|---|---|---|
| llama | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | abstention, all rows |
| gemma | 0.0000% | 0.0013% | 0.0026% | **0.0503%** | 0.0026% | abstention, all rows |
| qwen | 89.97% | 92.08% | 86.93% | 94.66% | 90.93% | abstention, Cond-B only |
| ministral | 83.47% | 48.51% | 41.80% | 37.25% | 29.80% | abstention, Cond-B only |
| deepseek | 0.083% | 0.000% | 0.000% | 0.000% | **0.270%** | strict-valid rate, all rows |

**Largely yes, with two flagged exceptions, not smoothed over:**
- llama: perfectly stable at 0% in all five conditions.
- gemma: stable near 0%, but not *exactly* 0% under positive (38/75,600 rows,
  0.0503% — persona-clustered p≈6e-10, i.e. statistically real but practically
  negligible).
- qwen: stably high (87-95%) in all five conditions — a narrow range relative to its
  overall level, though every context-vs-original shift is statistically significant
  given the huge sample size.
- **ministral is the clear exception to "stable property": its abstention rate is not
  stable at all.** It drops from 83% (original) through 49%, 42%, 37%, to 30%
  (negative_minor) — a monotonic decline across every multi-turn framing tested, the
  single largest context effect found anywhere in this analysis. (Confirmed
  unchanged after the neutral/positive grammar fix and re-run: the decline is still
  monotonic, 83.47 → 48.51 → 41.80 → 37.25 → 29.80.)
- deepseek: near-total non-compliance (not the same as abstention — it essentially
  never emits the literal "NA" token) in 4 of 5 conditions, but negative_minor is a
  real, flagged departure (0.27% vs 0.00-0.08% elsewhere) — investigated in Step 4.

## Step 4: DeepSeek's negative_minor anomaly

The literal strict-valid difference (0.00% vs 0.27%, ~204 rows) is a small,
condition-dependent slice of a much larger and clearly real effect, not sampling
noise — but a first pass at quantifying that larger effect contained a real error,
caught and corrected via manual verification (raw `raw_text` inspection, not just
the aggregate percentages) before being included here.

Applying the health study's Fix-H5 compact-text-parser check (recovers responses
like `"Iwouldrespondwith4..."` — a real, no-whitespace formatting pattern documented
in the health diagnosis, cause unestablished) to all five conditions initially
suggested DeepSeek's near-compliance rate ranged from ~11% (original) to ~91%
(positive). **That "11% for original" figure was wrong.** The pipeline already has
an established `salvageable_numeric` category (`table1_compliance.csv`: 30.67% for
original) built from broader, word-boundary-requiring patterns — and every one of
the compact parser's 8,599 "matches" for original turned out to already be inside
that 30.67%, not new. (Those patterns require a `\b` boundary around the digit,
which original's normally-spaced prose has and the other conditions' run-on,
space-free text does not — which is exactly why `salvageable_numeric` barely fires
outside of original, 0.00–0.12%.) The reconciled, double-counting-free totals:

| condition | strict-valid | pre-existing salvageable_numeric | genuinely new (compact) | **true total** |
|---|---|---|---|---|
| original | 0.08% | 30.67% | 0.00% | **30.75%** |
| health | 0.00% | 0.00% | 14.86% | **14.86%** |
| neutral | 0.00% | 0.15% | 68.48% | **68.62%** |
| positive | 0.00% | 0.03% | 90.46% | **90.48%** |
| negative_minor | 0.27% | 0.07% | 41.05% | **41.39%** |

(original's reconciled total matches `table1_compliance.csv`'s independently-computed
30.67%+0.08% almost exactly — a useful internal consistency check that the
correction is right.)

**Manual verification of the "genuinely new" rows** (15 quoted `raw_text` examples
from positive, sampled from rows the compact parser flagged that were *not* already
strict-valid or salvageable_numeric): every one is an unambiguous, directly-stated
answer beginning with the literal clause `"Iwouldrespondwith4,..."` followed by
on-topic reasoning (e.g. `'Iwouldrespondwith4,Diegowouldagreewiththestatement...'`)
— not an incidental digit inside a date, list, or unrelated sentence. **But** of
positive's 68,381 newly-recovered rows, 99.96% are the identical rating value "4" —
essentially zero variance. This is a single compact filler string firing at massive
scale, not evidence of thoughtful, varied rating behavior. (For comparison, original's
already-known `salvageable_numeric` rows *do* show real spread — 62% "4", 23% "2",
15% "3", plus a few 1s/5s/NAs — confirming that whatever's happening under
positive/neutral/negative_minor is a qualitatively different, far more repetitive
behavior than original's ordinary salvageable prose.)

**Independent confirmation**, using a completely regex-free metric (fraction of
`raw_text` with literally zero space characters, for responses >5 characters): 0.00%
(original) → 15.70% (health) → 73.35% (negative_minor) → 81.52% (neutral) → 96.98%
(positive). This matches the regex-based gradient closely and rules out the compact
pattern being a regex artifact — the underlying no-whitespace behavior is real and
large regardless of how it's measured.

Reframed this way: **negative_minor is not the extreme condition** — positive is
(90% true total), and health is the lowest of the four new contexts (15%); original
sits in between (31%), not at the bottom. Within negative_minor's own 204
strict-valid rows specifically, 91% are one topic (economic redistribution) and 100%
are the same rating value (4) — still a narrow formatting coincidence at that literal
level, just one instance of the broader pattern documented above.

**Honest conclusion:** DeepSeek's behavior clearly does differ by conversational
context, and by a lot — but (a) the size of that difference is smaller for original
specifically than an uncorrected first pass suggested, because a large chunk of it
was already known, not new; and (b) what varies is the *prevalence* of a
compact/no-whitespace formatting tendency that outputs a near-constant single value,
not a general improvement in thoughtful compliance. The root cause (tokenizer, chat
template, or decoding) remains unestablished from raw text alone, same conclusion as
the health-study audit. One plausible but **unverified** mechanical hypothesis worth
flagging rather than asserting: with a fixed 30-new-token generation budget,
space-free text could let a short, complete thought finish before the budget runs
out, where a normally-spaced equivalent would be truncated mid-sentence — but this
has not been checked against tokenizer/generation logs and should not be read as
established fact.

---

## Changelog

- **2026-08-03**: A subject-verb agreement grammar bug was found in the `neutral`/
  `positive` prompt templates, affecting the text sent to models for neutral-gender
  ("they") personas only (1,800 / 5,400 personas, 33.3% of the dataset; male/female
  personas were never affected). Fixed in `context_render_prompts_full.py`; both
  contexts were fully re-run (5 models x 75,600 rows each) against the corrected
  prompts, the corrected results replaced the pre-fix files in `results_context/`,
  and the fix was directly verified in the actual model-facing prompt text and
  responses. The full `analysis_context/` pipeline was re-run and every finding in
  this document was reconciled against the pre-fix numbers: **no headline finding
  changed** (Ministral's monotonic abstention decline across all five conditions,
  the cross-context structural-clustering result, and the significant/
  not-significant pattern of every rating-shift cell all held). Movements were on
  the order of a few tenths of a percentage point, with one exception flagged
  in-line: llama's positive-context profession-ranking p-value crossed the
  conventional 0.05 threshold (0.033 → 0.067) because truck driver moved from a
  bottom tie (with farmer, in the original ranking) into a middle tie (with computer
  programmer, in the corrected positive-context ranking) — not a reversal of the
  top/bottom ranking positions, which are unchanged. `health` and `negative_minor`
  were unaffected by the bug and their analysis outputs were confirmed bit-for-bit
  identical before and after. **Note**: the pre-fix raw model-response data was not
  retained after this reconciliation; the before/after comparison above relies on
  the summary values recorded at the time the fix was verified, rather than being
  independently re-derivable from raw data today — a disclosed limitation of this
  record, not an error in the comparison itself.
- **2026-08-03**: Added `analysis_context/05_variance_ranking_all_prompt_types.py`,
  extending the original study's H1 variance-decomposition test (partial R² for
  profession/country/gender/age, Condition A, persona-clustered) from the original
  prompt only to all five prompt types. No new inference — pure analysis of already-
  collected data. See item 6 (now in the pilot-scale appendix). Headline result: H1
  ("profession dominates") never held for llama or gemma under any prompt type, and
  for ministral it held only under the original single-turn prompt — gender takes
  over under all four conversational framings, with no exception.
- **2026-08-04**: Re-ran the entire `analysis_context/` pipeline (scripts 01, 02,
  05) at full scale — all 5,400 personas, 20 countries, 30 professions, for all
  five prompt types — as a companion to every existing 180-persona pilot-scope
  analysis, never overwriting it. No new inference: full-scale result data already
  existed in `results/` and `results_context/`; this was pure analysis. Full-scale
  results are now primary (see "Full-scale results" section above); pilot-scale is
  retained as an appendix. Headline outcome: **country-ranking significance, which
  was structurally unreachable at pilot scale (n=4, minimum possible p=0.083), now
  reaches p<.01 for every model in every context** using a new vectorized Monte
  Carlo permutation test (200,000 draws; exact enumeration is impossible at n=20/30
  countries/professions). Every other pilot-scale conclusion held at full scale with
  tighter confidence, with two flagged exceptions: qwen/neutral's rating shift moved
  from borderline-nonsignificant to significant (a real small effect pilot scale was
  underpowered to detect), and gemma's H1 dominant factor diverges under health/
  neutral specifically (profession narrowly wins at full scale, by a margin an order
  of magnitude tighter than any other model's dominant-vs-runner-up gap — flagged as
  a close, scale-sensitive call, not a robust reversal). One latent, pre-existing bug
  in `_common.py`'s `extract_coefs` was found and fixed during this work: level names
  were recovered via a character-set `.strip("[]T.")` instead of a fixed-suffix
  removal, silently turning "Turkey" into "urkey" — dormant at pilot scale (no pilot
  country/profession name is affected) and equally dormant, unfixed, in
  `analysis_health/04_ranking_robustness.py` (out of scope, never exercises the full
  20-country design). Only human-readable labels and Turkey's own bootstrap
  rank-position probabilities were affected; numeric significance results were not.
