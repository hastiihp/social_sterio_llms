# Adversarial validation audit: `analysis_context/`

## Executive verdict

`analysis_context/` is computationally valid for the principal raw-data summaries and audited statistical routines, but `CONTEXT_EXPERIMENT.md` is not internally accurate as written and two prompt templates contain a material grammar confound. Verdict: **CONDITIONAL GO**.

The two headline numerical patterns survive independent reproduction:

- Ministral Condition-B abstention is 83.4656% (original), 48.5132% (health), 42.0079% (neutral), 37.2222% (positive), and 29.7989% (negative_minor). The four context-minus-original shifts reproduce exactly. Every adjacent step in the displayed ordering is negative and persona-cluster significant: -34.9524pp, -6.5053pp, -4.7857pp, and -7.4233pp (all p < 3e-249 or numerically underflowed to 0).
- DeepSeek's reconciled, mutually exclusive strict + salvageable + newly compact totals are 30.7500% (original), 14.8598% (health), 68.3611% (neutral), 90.7976% (positive), and 41.3902% (negative_minor). Thus the corrected total range is **14.86%-90.80%, not 11%-91%**. The 11% original figure belongs to the rejected double-counting pass. Positive has 68,618 newly recovered compact rows, 68,589 of which (99.9577%) encode 4.

These are verified facts. Interpretation is narrower: Ministral demonstrates an observed framing-associated decline, not an ordered emotional-valence effect; DeepSeek demonstrates context-associated compact/no-whitespace output prevalence, not thoughtful or varied task compliance.

## Issue register

Full evidence and proposed fixes are in `outputs/issue_register.csv`.

| ID | Severity | Finding |
|---|---|---|
| CTX-01 | High | Neutral/positive templates have neutral-pronoun verb-agreement defects. |
| CTX-02 | High | Three Condition-A findings cells and their prose are materially wrong. |
| CTX-03 | Medium | Correlation difference is overinterpreted as identifying a structural cause. |
| CTX-04 | Low | Agreement/correlation ranges are misstated. |
| CTX-05 | Low | Pilot topic-drop range rounds 56.11pp upward to 60pp. |
| CTX-06 | Low | Audited helpers were copied/generalized, not directly shared; present outputs show no drift. |

## Prompt audit

The audit freshly rendered and compared two personas per gender per template (24 samples: health plus three new contexts). Every stored sample matches its renderer. Both A and B scale tails are byte-identical across original, health, and context renderers (SHA-256 recorded in `prompt_tail_byte_identity.csv`).

Health and negative_minor pass subject/object/possessive and verb-agreement checks. Neutral and positive fail for neutral-gender personas:

- neutral: “They ... and **has** been arranging furniture”;
- positive: “They ... and **is** really excited”.

The bug is in fixed suffix text, so it affects all 1,800 neutral-gender personas: 25,200 prompt/result rows in neutral and 25,200 in positive per model. This is not repaired by the existing `HAVE_VERB`/`be` variables because those variables are not interpolated into the suffixes. No existing file was modified.

Sensitivity checks are reassuring but do not erase the design defect. Ministral's displayed decline remains monotonic within female, male, and neutral-gender strata. DeepSeek positive remains about 89.06%-93.23% reconciled across gender strata, and negative_minor remains about 36.26%-47.84%. These checks support robustness of the headline direction, while neutral/positive causal attribution still requires qualification.

## Data integrity

All 15 `results_context` files contain exactly 75,600 rows and exactly 5,400 canonical persona IDs. Each has zero missing/extra persona IDs, zero duplicate canonical keys, and a complete one-to-one match to its corresponding prompt file. SHA-256 hashes are recorded in `integrity_15_files.csv`.

The canonical merge key is persona ID, country, profession, gender, age, topic, and response condition (plus model when frames are concatenated). One-to-one validation passed. `strict_is_valid`, `is_abstention`, and `salvageable_numeric` were read with literal `NA` preserved; the DeepSeek reconciliation independently confirms mutually exclusive counts.

## Ministral deep audit

The full-data per-topic diagnostic confirms concentration rather than roughly equal topic effects. Across contexts, climate, redistribution, and immigration typically show the largest declines; gender equality and religion/secularism remain small. On the study's 180-persona diagnostic, the named large-topic range is 56.11-98.33pp, slightly wider/lower than the prose's 60-98pp. On all 5,400 personas it is 64.48-99.59pp except that the same health/immigration comparison is 64.48pp. The neutral Condition-B degenerate subset also reproduces: 209 both-numeric cells, all exactly 4 in both conditions.

The monotonic claim is valid as an observed ordering and survives clustered adjacent comparisons. It should not be described as a valence dose-response: contexts were not designed as a unidimensional ordered scale.

## DeepSeek deep audit

The compact regex was independently applied to raw `raw_text`, after excluding strict-valid and pre-existing `salvageable_numeric` rows. Original has 8,599 regex matches but zero genuinely new rows, confirming the earlier double-counting correction. Independently sampled raw rows from both requested files are in `deepseek_raw_samples_independent.csv`; they are direct compact answers rather than incidental digits.

The regex-free zero-space check also reproduces: 0.00%, 15.70%, 73.35%, 81.32%, and 97.29% for original, health, negative_minor, neutral, and positive. Negative_minor has 204 strict-valid rows; 186 (91.1765%) are economic redistribution and all 204 are rating 4. This supports a narrow formatting phenomenon, not substantive response quality.

## Cross-context agreement

The raw Condition-A matrix reproduces mean rho 0.878602 for 24 context-context model-pairs and 0.790584 for 16 context-original model-pairs. Context-pair means span 0.857467-0.901570; health/negative_minor is highest (0.901570), neutral/positive second (0.894615).

The numbers are valid. The claim that they show clustering *by structure rather than content* is interpretation, not identified fact. The analysis averages dependent correlations and applies an undocumented +0.02 heuristic rather than an inferential comparison. Manuscript language must say “descriptively consistent with shared multi-turn structure,” not that structure is established as the cause.

## Inherited-method triage

- Persona-clustered mean-difference formula: matches the audited row-wise-difference/intercept design; sampled and full outputs reproduce.
- Exact permutation: independently enumerates all 120 profession and 24 country permutations. All 32 rho/p rows match exactly; no asymptotic p-value was substituted.
- Persona bootstrap: independently rerun at B=1,000 for every original/context framing and all four analyzable models. All 144 probabilities and successful-replicate counts match bit-for-bit.
- Canonical merge: one-to-one validation and zero unmatched rows confirmed.
- Parsing categories: literal `NA` preservation and strict/salvage/new exclusivity confirmed; original compact overlap is fully accounted for.

The code is a copied/generalized implementation in `_common.py`, not a direct import from the audited health module. No numerical drift was found, but the project should avoid describing this as a single shared implementation unless it is actually centralized.

## Findings-section reproduction

The complete machine-readable ledger is `findings_reproduction_ledger.csv`: 128 claims, 119 matches, 8 mismatches, and 1 qualification. The eight mismatches represent four substantive prose/table discrepancies counted as value and p-value separately where applicable:

| Claim | Reported | Independently reproduced |
|---|---:|---:|
| llama / positive Condition-A shift | -0.017, p=.06 | **-0.084127, p=6.11e-15** |
| qwen / health Condition-A shift | -0.018, p=.15 | **-0.155556, p=3.22e-22** |
| Ministral / health Condition-A shift | +0.003, p=.73 | **-0.097619, p=2.44e-31** |
| exact-agreement range | 74%-92% | **68.81%-90.71%** |
| Spearman range | 0.71-0.90 | **0.686-0.894** |

All other headline tables—including abstention shifts, ranking averages, exact p-values, correlation means, stability rates, DeepSeek reconciliation, no-space rates, and strict-slice counts—reproduce at the stated rounding. The topic “60-98pp” shorthand should be about 56-98pp for the pilot outputs.

## Manuscript readiness and final verdict

The Ministral and DeepSeek findings **cannot be presented as-is**.

- Ministral's numerical decline is manuscript-ready after qualification: state that it is a Condition-B abstention association across four heterogeneous conversational framings; preserve the self-selection caveat; do not imply a valence scale; disclose the neutral/positive neutral-pronoun grammar defect and gender-stratified robustness.
- DeepSeek's numerical reconciliation is manuscript-ready after qualification: use 14.86%-90.80% for the corrected total range, label it near-compliance/compact-format prevalence rather than compliance quality, disclose the near-constant 4, and disclose the same prompt defect for neutral/positive.
- Correct the three erroneous Condition-A table cells and associated prose before circulation.
- Downgrade the structural-clustering claim to descriptive evidence consistent with shared multi-turn structure.

**Final verdict: CONDITIONAL GO.** No new experiment is required to establish the audited numerical facts. A clean causal comparison of the intended neutral/positive templates for neutral-gender personas would require a separately authorized, versioned replication; the existing completed data must remain unchanged.
