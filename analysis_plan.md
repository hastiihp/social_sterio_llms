# Analysis Plan — Stereotype-Based Opinion Attribution in Instruction-Tuned LLMs

**Status:** Draft, pre-pilot. Freeze after pilot acceptance criteria (Section 11) are reviewed and met.
**Design matrix:** `data/personas.csv` (5,400 rows) × `data/topics.csv` (7 topics) × 2 response conditions = `data/prompts.csv` (75,600 rows).
**Models:** Llama-3.1-8B-Instruct, Gemma-3-12B-it, Qwen3-8B, DeepSeek-LLM-7B-chat, Ministral-8B-Instruct, Falcon-H1-7B. All bf16, no quantization mixing.

**Scope note:** This preregistration covers the demographic-only friend-framing study with two response conditions. Cue experiments, generational comparisons with predecessor models, and likelihood-space analyses are explicitly out of scope and reserved as follow-up studies; any such analysis performed on this data will be labeled exploratory.

---

## 1. Research question

Do language models attribute sociopolitical opinions to fictional people based on country, profession, gender, and age, when the *only* information available is a third-person "friend" description containing no behavioral evidence? And separately: does the model's willingness to attempt that attribution at all (vs. abstain) itself depend on those same demographics?

These are treated as two distinct outcomes, not one:

- **Forced opinion attribution** (Condition A): the rating selected when a numeric answer is required.
- **Optional abstention** (Condition B): whether the model judges the demographic information sufficient to offer even a tentative estimate.

## 2. Hypotheses

- **H1 (demographic attribution):** Under forced rating, profession will explain more variance in attributed opinion than country, age, or gender.
- **H2 (abstention behavior):** Under the optional condition, abstention rates will differ systematically across models, with abstention tendency varying by model family.
- **H3 (forced ≠ optional):** For persona-topic-model combinations where a model abstains under the optional condition, the same model's forced-condition rating will not be randomly distributed around the midpoint — i.e., abstention can coexist with a latent directional attribution rather than reflecting genuine indifference.

**Exploratory (not pre-specified as hypotheses):**
- Comparison of friend-frame effects with prior first-person v9 results (prompt, models, and country set all differ; no directional prediction is justified).
- Effects within the non-binary condition specifically, given the smaller evidence base for name validity in that condition (see `data/names.csv` validation tiers).

## 3. Parsing rules

From raw generation, extract:

| field | description |
|---|---|
| `raw_text` | unmodified model output |
| `normalized_text` | whitespace/case normalized |
| `parsed_rating` | 1–5 or NA if valid |
| `is_abstention` | True only for explicit NA under Condition B |
| `is_valid` | True if output matches the required format exactly |
| `parse_failure_reason` | one of: none / malformed_output / empty_output / explanatory_refusal / safety_refusal / technical_failure |

**Rules:**
- Do not convert refusals or malformed output into NA. NA is only ever the model's explicit stated choice under Condition B.
- Only `technical_failure` (e.g., inference crash, empty output due to infrastructure error) may be automatically rerun.
- A semantic refusal (model declines to speculate, or objects to the premise) is a valid data point and must never be rerun to "get a number."
- Log parse-failure rate per model per condition; if any model exceeds 5% non-technical invalid rate, flag for manual review before proceeding to analysis (do not silently drop).

## 4. Primary analysis — forced-rating condition (Condition A)

**Model:**
```
rating ~ topic + profession + country + gender + age + model
```
with interactions: `topic × model`, `profession × model`, `country × model`, `gender × model`.

**Methods:**
- Linear regression with HC3 robust standard errors (primary analysis, chosen for interpretability of coefficients and variance decomposition); all substantive conclusions checked with ordinal logistic regression (Section 9, check 3)
- Mixed-effects model: `rating ~ fixed effects + (1 | persona)` — each persona contributes 7 topics × 2 conditions, so the persona random intercept is identifiable
- Variance decomposition (partial R² / ANOVA-style variance shares per factor)
- Marginal means and pairwise contrasts (Tukey HSD) for profession and country rankings

**Primary outcomes:** demographic coefficient magnitudes and significance; profession/country rank order; variance share by factor; per-model divergence in the above.

## 5. Optional-abstention analysis (Condition B)

**Binary model:**
```
answered ~ topic + profession + country + gender + age + model
```
with model interactions as above.

**Report:** response (non-abstention) rate by model, by topic, by demographic factor; model × demographic interactions.

Among rows that did receive a numeric answer under Condition B, analyze conditional ratings separately — and explicitly flag that this subset is *not* a random sample of personas (selection induced by abstention), so conditional-rating comparisons across models are descriptive, not causal, unless selection is modeled.

## 6. Paired comparison (Condition A vs. B) — primary contribution

For every identical persona × topic × model combination:

- Did Condition B answer or abstain?
- What did Condition A (forced) return for that same cell?
- Where both answered: rating difference, exact agreement rate, weighted Cohen's κ, Spearman correlation, mean absolute difference, directional disagreement, profession/country rank correlation between conditions.

**Key analysis:** among persona-topic-model cells where Condition B = NA, what does Condition A return? Test whether the forced-condition rating distribution in these cells differs significantly from a midpoint-centered null. This distinguishes "no attributable signal" from "reluctance to state an attributable signal" — i.e., whether abstention masks a latent directional attribution.

**Anticipated confound to check explicitly:** elevated rate of "3 = Neither agree nor disagree" under Condition A specifically in cells where Condition B abstains, which would suggest the midpoint is being used as a covert abstention channel under forced conditions rather than genuine neutrality. If observed, report as a finding, not noise.

## 7. Topic-specific analysis

Run the primary and abstention models separately per topic (all 7). Do not label any topic subset "identity-related" or similar post-hoc groupings not defined here. If topic groupings are introduced for a specific test, define the grouping and its rationale before running the test, and report all 7 topics individually regardless.

## 8. Predefined robustness checks (exhaustive — no ad hoc additions after freeze)

1. **Original 10 vs. added 10 countries** — do profession/country effect sizes and rankings hold across `country_set`?
2. **Forced vs. optional condition** — covered in Section 6.
3. **Ordinal vs. linear model** — do substantive conclusions change under ordinal logistic regression vs. linear regression?
4. **Paraphrase robustness (friend-frame opener)** — exactly two minimal paraphrases of the prompt opener, run on a small matched persona subset with both conditions. Report rank correlation of profession/country effects across paraphrases. This check exists because a single frozen prompt cannot itself demonstrate that findings reflect underlying model behavior rather than this exact wording.

Any additional robustness analysis proposed after this document is frozen must be logged in an addendum with a stated rationale, not inserted silently.

## 9. Multiple comparisons

Apply Benjamini-Hochberg FDR correction within each family of tests, with each family defined a priori.

## 10. Exclusion criteria

- Exclude a persona-topic-model-condition cell only for `technical_failure` (after one automatic rerun).
- Never exclude on the basis of the *content* of a valid or semantically-refused response.
- If a model's non-technical invalid rate exceeds 5% (Section 3), report this explicitly per-model rather than excluding the model.

## 11. Stopping / rerun rules

- Full six-model run proceeds only after pilot acceptance criteria are met (compliance rate, parse success, pronoun/grammar correctness, runtime/storage projections within budget).
- No changes to `prompts.csv` wording after pilot sign-off. Any wording issue discovered post-pilot is documented as a limitation, not silently patched mid-run.
- Once the full run begins, no new experimental conditions are added.

## 12. Non-binary condition — manipulation check

The non-binary condition is signaled via name + they/them pronoun only, with no explicit label in the prompt text, preserving structural parity with the male/female conditions (which likewise never state gender explicitly). The pilot will include a separate manipulation-check experiment evaluating whether the intended gender signal is recognized. If the signal is not registering reliably, an explicit-label variant will be added as a predefined robustness check (not a change to the main design) before the full run.

## 13. Name-validity limitation (state verbatim in paper)

> Because each country-by-gender-condition was represented by a single name (see `data/names.csv` for per-name validation tier, evidence type, and documented usage skew where available), country and gender effects may partly reflect associations attached to the selected names rather than the demographic category alone. Names are documented as culturally plausible and, where evidence permits, as being used across genders with varying usage distributions — not claimed as neutral or as representative of an entire national population.

## 14. Deterministic inference settings (frozen)

```
do_sample: false
temperature: 0
top_p: 1
max_new_tokens: 5
repetition_penalty: 1.0
seed: fixed (recorded per run)
```

Applied identically to all six models. Each model's official tokenizer and chat template is used; chat-control tokens are never manually reproduced. Recorded per run: model repository and revision, tokenizer revision, Transformers/PyTorch/CUDA versions, GPU type, prompt template version, generation configuration, timestamp, SLURM job ID, git commit hash. Raw generated text is saved before any parsing.

---

*This document should be treated as frozen once pilot acceptance criteria are reviewed and confirmed. Amendments after that point require a logged addendum with rationale, per Section 8.*
