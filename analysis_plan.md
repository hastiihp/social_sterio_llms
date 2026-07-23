# Analysis Plan — Stereotype-Based Opinion Attribution in Instruction-Tuned LLMs

**Status:** Draft, pre-pilot. Freeze after pilot acceptance criteria (Section 11) are reviewed and met.
**Design matrix:** `data/personas.csv` (5,400 rows) × `data/topics.csv` (7 topics) × 2 response conditions = `data/prompts.csv` (75,600 rows).
**Models:** Llama-3.1-8B-Instruct, Gemma-3-12B-it, Qwen3-8B, DeepSeek-LLM-7B-chat, Ministral-8B-Instruct, Falcon-H1-7B. All bf16, no quantization mixing.

**Known deviation:** Falcon-H1-7B produced a reproducible tensor-shape error in its generation cache under batched inference in this transformers version, confirmed across multiple batch sizes and both dynamic and fixed-length padding, while running cleanly only at batch size 1 in smoke tests. Falcon-H1 is excluded from the study; it was not replaced or rerun. The primary analysis proceeds with the five models below (Llama, Gemma, Qwen3, DeepSeek, Ministral). This removes the study's only UAE/MENA-origin model and is reported explicitly as a limitation on the regional-comparison claim: "Falcon-H1 was excluded because inference could not be completed reliably under the available software environment despite repeated attempts."

**Naming note:** the frozen prompt template that actually produced the analyzed data uses explicit gender-identity labels, logged in provenance as `friend_v2_explicit_gender`. This is unrelated to and should not be confused with `friend_v2_strict` (Section 15), the reinforced-instruction variant that was tested and rejected. Renaming the adopted template to `friend_final` in all provenance strings and manuscript text is recommended to avoid ambiguity.

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

**Exploratory (not preregistered as hypotheses):**
- Comparison of friend-frame effects with prior first-person v9 results (prompt, models, and country set all differ; no directional prediction is justified).
- Effects within the non-binary condition specifically, given the smaller evidence base for name validity in that condition (see `data/names.csv` validation tiers).

## 3. Parsing rules

From raw generation, extract:

| field | description |
|---|---|
| `raw_text` | unmodified model output |
| `normalized_text` | whitespace/case normalized |
| `strict_parsed_rating` | 1–5 or NA, only if the trimmed output matches the required format exactly |
| `strict_is_valid` | True only if `strict_parsed_rating` was extracted under exact-format matching |
| `salvaged_rating` | a rating extracted from non-compliant text (e.g. "Based on... 4") for diagnostic use only |
| `salvage_method` | how the salvaged rating was extracted (e.g. regex first-digit-match) |
| `is_abstention` | True only for explicit NA under Condition B, under strict parsing |
| `parse_failure_reason` | one of: none / malformed_output / empty_output / explanatory_refusal / safety_refusal / technical_failure |

**Rules:**
- `strict_parsed_rating` is the only field used in primary analyses (Sections 4-6). A response that embeds the right answer in explanatory prose (e.g. "I would estimate 4") is malformed under the exact-output instruction and must not be silently treated as valid — this would violate the instruction-following measurement the format itself is designed to test.
- `salvaged_rating` exists for diagnostics only (e.g. distinguishing "model refused entirely" from "model explained before answering") and is reported separately, never substituted into the primary rating variable.
- Do not convert refusals or malformed output into NA. NA is only ever the model's explicit stated choice under Condition B, under strict parsing.
- Only `technical_failure` (e.g., inference crash, empty output due to infrastructure error) may be automatically rerun.
- A semantic refusal (model declines to speculate, or objects to the premise) is a valid data point and must never be rerun to "get a number."
- Log parse-failure rate per model per condition (strict parsing); if any model exceeds 5% non-technical invalid rate, flag for manual review before proceeding to analysis (do not silently drop). A model with a high strict-invalid rate but low salvage-rate gap (i.e. it usually explains before giving the same answer) is a different finding than one that gives inconsistent or absent answers, and both should be reported.

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

## 12. Non-binary condition — resolved via manipulation check (pre-pilot amendment)

The non-binary condition was originally signaled via name + they/them pronoun only, with no explicit label, to preserve structural parity with the male/female conditions. A manipulation check (60 personas: 20 countries x 3 genders, one per cell) tested this directly by asking each of the two pilot models to state the perceived gender of the described friend.

**Result:** the implicit (pronoun-only) signal did not reliably produce a distinct non-binary interpretation — 0/20 correct for Llama-3.1-8B, 4/20 correct for Qwen3-8B, with both models defaulting to a binary gender guess based on the name in the remaining cases. A second, symmetric comparison then tested an explicit-label version ("{name} identifies as a man / a woman / non-binary") applied identically to all three gender conditions, on the same 60 personas. Explicit labeling produced 20/20 correct non-binary identification on both models, while leaving already-correct male/female identification completely unchanged (20/20 on both models, both versions) — confirming the fix does not introduce an asymmetric confound between conditions.

**Decision:** the frozen prompt template was amended before the pilot to include an explicit identity clause for all three gender conditions ("{name} identifies as {a man / a woman / non-binary}."), applied uniformly. This is judged the more faithful operationalization of the research question in any case: the study measures how models attribute opinions given a demographic attribute, not whether models can correctly infer that attribute from a name. `prompts.csv` was regenerated in full under this amendment prior to the pilot run.

## 13. Name-validity limitation (state verbatim in paper)

> Because each country-by-gender-condition was represented by a single name (see `data/names.csv` for per-name validation tier, evidence type, and documented usage skew where available), country and gender effects may partly reflect associations attached to the selected names rather than the demographic category alone. Names are documented as culturally plausible and, where evidence permits, as being used across genders with varying usage distributions — not claimed as neutral or as representative of an entire national population.

## 14. Deterministic inference settings (frozen after smoke testing)

```
do_sample: false
temperature: 0
top_p: 1
max_new_tokens: 30
repetition_penalty: 1.0
seed: fixed (recorded per run)
```

The generation limit was increased from an initial smoke-test setting of 5 tokens to 30 after smoke testing demonstrated that shorter limits artificially truncated non-compliant responses mid-sentence, preventing correct classification of instruction-following behavior (a model attempting an explanation could not be distinguished from one that had been cut off before reaching an answer). The finalized limit does not measurably affect compliant models, which terminate after one token regardless of the ceiling.

Applied identically to all six models. Each model's official tokenizer and chat template is used; chat-control tokens are never manually reproduced. Recorded per run: model repository and revision, tokenizer revision, Transformers/PyTorch/CUDA versions, GPU type, prompt template version, generation configuration, timestamp, SLURM job ID, git commit hash. Raw generated text is saved before any parsing.

## 15. Prompt-reinforcement comparison (pre-pilot, predefined and completed)

Before freezing the friend-frame prompt, a reinforced closing-instruction variant (`friend_v2_strict`) was tested against the original (`friend_v1`) on all six models, using the same 12 matched persona-topic-condition cases per version. The adoption rule was written before results were reviewed: adopt `friend_v2_strict` only if it raised DeepSeek's strict-compliance rate to at least 9/12 AND left the other five models' strict-compliance and rating values unchanged.

**Result: `friend_v2_strict` was not adopted.** DeepSeek's strict compliance improved from 0/12 to 6/12 -- a real but insufficient improvement relative to the predefined threshold. Independently of that threshold, `friend_v2_strict` also changed rating values for four of the five previously-compliant models on identical persona-topic pairs (e.g. Qwen3's climate-change rating shifted 4->3), and changed Falcon-H1's abstention decision outright (NA -> 4 on the immigration/optional prompt). This second finding is treated as the primary reason for rejection, independent of DeepSeek's result: a wording change that alters abstention behavior confounds the exact variable the optional condition is designed to measure. `friend_v1` is retained as the frozen prompt.

DeepSeek's non-compliance is documented as a per-model finding, not treated as a pipeline defect: DeepSeek remained substantially less format-compliant than the other five models under both evaluated prompt variants, while its responses showed clear content engagement with persona-specific details (e.g. correctly reasoning about a persona's country and inferred religious context) rather than generic refusal or failure to parse the prompt. Per Section 3, DeepSeek's non-technical invalid rate will be reported explicitly per-model in results rather than triggering exclusion or prompt modification.

---

*This document should be treated as frozen once pilot acceptance criteria are reviewed and confirmed. Amendments after that point require a logged addendum with rationale, per Section 8.*
