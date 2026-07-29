# Audit Sections 13–23: confirm-or-flag pass

Date: 2026-07-28

## Method

I copied the current repository to `/tmp/social-sections13-23.FpkDBC/repo` and reran the complete main-analysis sequence (01–10, including 05b for all four main models, 05b2–05e, 07b, and 09) in the pinned clean environment at `/tmp/social-cleanenv.WBOofc/venv`. I then reran all four health-analysis scripts against the copied inputs. Every regenerated CSV cell was compared with its checked-in counterpart. Exact matches and numerically equivalent floating-point results are recorded below as **CONFIRMED, matches saved value**; discrepancies were investigated.

The health staging outputs, prompts, and scheduler logs were also found under the git-ignored `health_staging/` and `logs/` directories. The staged health prompts and raw outputs match the retained `data_health/` and `results_health/` files exactly. This corrects the earlier checkpoint wording that health execution artifacts were absent; they exist locally, although most of the health pipeline remains untracked by Git.

## 13. Main descriptive results

**CONFIRMED, matches saved value.**

All cells in the regenerated descriptive, distribution, compliance, and Table 1 CSVs matched the saved values. This includes response distributions, valid-rating and abstention rates, rating means and medians, midpoint frequencies, topic distributions, and Condition A/B counts.

Key checks:

- Llama and Gemma: 75,600/75,600 strict-valid responses.
- Qwen and Ministral: all outputs conform to the requested response format, with overall abstention rates of 44.98545% and 41.732804%, respectively.
- DeepSeek: 63/75,600 strict-valid responses (0.083333%); 99.916667% are non-strict-valid. DeepSeek remains descriptive only.
- Literal `"NA"` abstentions remain strings in the source CSV and are classified as abstentions, not pandas missing values, by the analysis loader.

No discrepancy was found in the saved descriptive statistics.

## 14. H1

**CONFIRMED, matches saved value**, with the DeepSeek caveat below.

The implementation uses Condition A only, as planned. Model-specific OLS models use persona-clustered covariance; mixed models use a persona random intercept. The factor-importance calculation and rankings were reproduced, and all four main-model rankings matched exactly. BH-adjusted columns were regenerated after all regression-producing scripts.

The reproduced primary H1 rankings are:

- Llama: topic, gender, profession, country, age — H1 not supported.
- Gemma: topic, country, gender, profession, age — H1 not supported.
- Qwen: topic, profession, gender, country, age — H1 supported under the prespecified ranking rule.
- Ministral: topic, profession, gender, country, age — H1 supported under the prespecified ranking rule.

Reference categories, formulas, samples, and the 5,400-persona clustering variable were unchanged. No main-model significance decision changed on rerun.

Flag: DeepSeek’s exploratory 63-row HC3 regression is rank-deficient and numerically unstable. Ten HC3 p-values moved materially and the Mexico coefficient changed from p=.835951 to p=.042736; clustered and mixed-model significance decisions did not flip. This does not affect H1 because DeepSeek is excluded from pooled/main inference, but its per-model inferential columns must not be interpreted substantively.

## 15. H2

**CONFIRMED, matches saved value.**

The planned full five-model abstention interaction model remains structurally non-estimable: Llama, Gemma, and DeepSeek have zero variance in the strict abstention outcome. The implementation does not silently present this model as fitted or H2 as supported. The saved restricted Qwen-versus-Ministral model regenerated successfully and matched its table, but it is explicitly a restricted descriptive/inferential comparison rather than the planned full H2 test.

The abstention script also reproduces and reports the failed separated full fit instead of swallowing it. Therefore H2 is not established and must continue to be described as non-estimable under the planned specification.

## 16. H3

**CONFIRMED, matches saved value.**

Pairing is exact on model × persona × topic. The reproduced paired tables include matched counts, both-valid counts, exact agreement, weighted kappa, Spearman correlation, mean absolute difference, signed/directional disagreement, and conditional analyses of forced ratings for Condition-B abstainers. Conditional tests use persona-clustered inference.

Key reproduced values:

- Llama: 37,800 pairs; exact agreement .802116; weighted kappa .776992; Spearman .829210; MAD .203307.
- Gemma: 37,800 pairs; exact agreement .869048; weighted kappa .922234; Spearman .922220; MAD .130952.
- Qwen: 3,791 both-valid pairs and 34,009 Condition-B abstentions; exact agreement .757056 among both-valid pairs; forced-rating mean for abstainers 3.326708; abstention is concentrated at forced rating 3 (.666294 versus 0 among voluntary answerers).
- Ministral: 6,250 both-valid pairs and 31,550 Condition-B abstentions; exact agreement 1.0 and MAD 0 among both-valid pairs; forced-rating mean for abstainers 3.244342; abstention is concentrated at forced rating 3 (.745135 versus 0 among voluntary answerers).

The data support strong descriptive associations between midpoint forced ratings and optional-condition abstention. They do not identify an internal psychological state such as “uncertainty.”

## 17. Clustered and mixed-model inference

**CONFIRMED, matches saved value**, with optimizer-level numerical variation.

Clustered covariance uses `persona_id` (5,400 clusters for the full main-model samples) and the statsmodels cluster small-sample correction. Mixed models use a persona random intercept. Repeated observations within persona are therefore accounted for in the reported clustered and mixed-model inference, although topic/condition are not modeled as additional random effects.

All saved main-model coefficient conclusions were stable across HC3, clustered, and mixed-model columns on rerun. Gemma and Qwen converged with the default optimizer; saved Ministral converged with L-BFGS. Llama’s rerun reported convergence on the default attempt instead of the saved L-BFGS label, with maximum coefficient and SE changes of only 0.000258 and 0.000140 and no significance flips. This is optimizer-dependent last-digit variation, not a changed scientific result.

The sparse DeepSeek model remains non-identifiable/unstable as described in Section 14. The pooled model correctly excludes DeepSeek and uses persona-clustered SEs; it does not claim a pooled mixed-model fit.

## 18. Ordinal robustness

**CONFIRMED, matches saved value**, subject to the explicitly reported proportional-odds limitation.

All full and reduced ordinal models converged for Llama, Gemma, Qwen, and Ministral. Factor rankings were identical to the saved rankings; partial pseudo-R² differences were at approximately 1e-11 scale, and no coefficient significance decision changed. Profession ordinal rankings also matched exactly.

The proportional-odds diagnostic is not falsely reported as successful: the global test is untestable for all four models because too few threshold-specific binary fits are usable. In topic-specific checks, only 8/34 model-topic cases are testable and 7/8 reject proportional odds at .05. Thus the ordinal fits support qualitative factor-ranking robustness, but not a blanket assertion that the proportional-odds assumption holds.

The obsolete pre-current-method `ordinal_factor_ranking.csv` has been removed and is not used.

## 19. Country and topic robustness

**CONFIRMED, matches saved value.**

The original-versus-added country definitions, samples, coefficients, and profession comparison regenerated. One p-value differed by 4.5e-10 from floating-point calculation only; no decision changed. Topic-specific regression tables also regenerated cell-for-cell within numerical tolerance, include every planned topic, use persona clustering, and contain BH-adjusted p-values.

No country or topic was dropped. The robustness results support consistency of the main qualitative conclusions across the original and added country sets and across topics, while individual level coefficients remain multiple-comparison-sensitive.

## 20. Health versus original

**CONFIRMED, matches saved value.**

All model × condition rows in the health comparison tables matched: matched cells, both-valid cells, exact agreement, MAD, signed shift, Spearman, cluster-robust SE/p-value, confidence interval, and valid-pair proportion.

Condition A contains all 1,260 paired cells per main model. Reproduced health-minus-original shifts are:

- Llama: −0.173810 (cluster SE .009368, p=7.61e-77).
- Gemma: −0.184921 (cluster SE .012211, p=8.27e-52).
- Qwen: −0.155556 (cluster SE .016048, p=3.22e-22).
- Ministral: −0.097619 (cluster SE .008383, p=2.44e-31).

Condition B has only 80/1,260 valid Qwen pairs and 209/1,260 valid Ministral pairs. The Qwen selected-pair shift is +0.325, but this is conditional on the very small non-abstaining subset and cannot be generalized to all Condition-B prompts. The negative Condition-A rating-shift finding remains supported.

## 21. Abstention analyses

**CONFIRMED, matches saved value.**

Both denominator conventions reproduced:

- Ministral, full dataset, A+B denominator: 41.732804% original versus 24.256614% health, a −17.476190 percentage-point shift.
- Ministral, Condition-B-only denominator: 83.465608% original versus 48.513228% health, a −34.952381-point shift.
- Paired pilot, A+B denominator: 41.706349% versus 24.801587%.
- Paired pilot, Condition-B-only denominator: 83.412698% versus 49.603175%.

The apparent factor-of-two discrepancy is entirely explained by including non-abstaining Condition A in the denominator. Cluster-robust paired results and every reported direction matched.

Ministral’s Condition-B topic shifts also matched: climate change −81.11 points, economic redistribution −64.44, immigration −56.11, LGBTQ rights −17.22, trust in government −14.44, gender equality −2.78, and religion/secularism −0.56. The drop is therefore substantial and topic-concentrated.

## 22. Ranking robustness

**CONFIRMED, matches saved value**, except for one tie label.

The current method ranks common-distribution adjusted predictions under Condition A, not raw treatment-coded coefficients. Spearman correlations, exact permutation p-values, bootstrap results, and all numerical adjusted predictions matched.

Profession rank correlations are:

- Llama .872082 (exact p=.10).
- Gemma 1.0 (exact p=.016667).
- Qwen 1.0 (exact p=.016667).
- Ministral .974679 (exact p=.033333).

This supports the qualified statement that profession rankings are largely preserved. Bootstrap results show strong top/bottom stability for Gemma, Qwen, and Ministral, with more uncertainty for Llama.

Flag: Llama’s original bottom profession is an exact adjusted-prediction tie between `farmer` and `truck driver` at the reported precision. The saved table labels `truck driver`, while the clean rerun labels `farmer`; the choice comes from last-digit solver/order behavior. The rho, exact p-value, and underlying predictions are unchanged. Any text naming a unique Llama bottom profession is not reproducible and should say “farmer/truck driver tie.”

Country ranking correlations (.0 to 1.0 depending on model) are descriptive only. With four countries, the smallest attainable two-sided exact permutation p-value is .0833; no claim of reliable country-rank preservation or change is supported.

## 23. Figures and tables

**CONFIRMED, matches saved value at the data level.**

Every manuscript CSV table was regenerated and compared. Apart from the DeepSeek HC3 instability and Llama tied-bottom label already detailed, saved table contents reflect the current scripts and source data. Figure 5 correctly states that it shows Llama-reference profession contrasts from the four-model Condition-A pooled model, with 95% persona-clustered CIs, and explicitly excludes DeepSeek.

All six PNGs were regenerated. Their source CSVs and plotted values match. PNG bytes/pixels are not identical because of rendering/font/antialiasing differences (changed-pixel shares approximately 0.65%–2.57%; Figure 4 is one pixel wider on rerun). Direct inspection of the largest-difference plot, Figure 5, found no content, label, scale, interval, or caption change. These are render-environment differences rather than stale-data discrepancies.

No stale figure or table carrying an earlier substantive result was found.

## Overall decision before new experiments

**GO (updated 2026-07-29 after closeout fixes).**

The reported main and health-study numerical results survive clean regeneration. The ranking output now reports numerical ties explicitly at every rank position, including the Llama `farmer / truck driver, tied for lowest` result. Every row of `hypothesis_model_deepseek.csv` now states **NON-INFERENTIAL / EXPLORATORY ONLY** and warns that the n=63 sparse, rank-deficient regression has numerically unstable coefficients, SEs, and p-values; the generating scripts and manifest repeat that restriction.

Git-level provenance of the health artifacts remains a reproducibility housekeeping task, but the artifacts, staging files, prompts, outputs, logs, and deterministic regeneration path are present and were verified. It does not invalidate the audited results or block the next planned experiment.

## What this means for the next supervisor conversation

The three health-study findings still hold after this audit. Ratings are reliably lower under health-conversation framing in Condition A for all four analyzable models, and Ministral’s Condition-B abstention rate drops substantially, with most of that drop concentrated in specific topics. Profession rankings are largely preserved, although Llama has an unresolved tie for the bottom profession that should not be presented as a unique rank. Country rankings still cannot be assessed reliably with only four countries, so no strong country-ranking conclusion should be presented.
