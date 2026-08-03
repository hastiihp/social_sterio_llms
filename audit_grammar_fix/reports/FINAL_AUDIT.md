# Independent adversarial audit of the neutral/positive grammar fix

## Executive verdict

The code fix itself is genuine and the corrected current data are propagated through the current analysis outputs. Neutral uses `{have}`, positive uses `{be}`, and both are passed to the active `.format(...)` call. Exhaustive checks of all 1,800 unique non-binary personas in each canonical prompt CSV found zero bad suffixes. Health correctly threads gender-aware `have`, `have_cap`, `neg_have`, and `be`; negative_minor uses number-invariant `had` and singular-subject “one ... was,” so no analogous defect was found.

The current corrected statistics reproduce the headline post-fix values. However, the claimed *before/after* reconciliation is not fully independently auditable from raw files: the pre-fix raw neutral/positive results were overwritten and no copy exists under the matching filenames anywhere in the repository. Thus the old sides of “unchanged” claims and the reported pre-fix llama p=.033 cannot be recomputed from raw results. A prior derived audit CSV reports the old p=.033, but this audit does not treat that as raw evidence.

Final verdict: **CONDITIONAL GO**. The corrected study can be used after correcting the findings text and explicitly qualifying the historical before/after claims as not independently reproducible from retained raw artifacts.

## Evidence by requested check

1. **Generator:** `data_context/context_render_prompts_full.py` has neutral `{have}` and positive `{be}` templates and calls `.format(..., have=have, be=be)`. Neither variable is dead. Health and negative_minor are grammatically sound for singular-they personas.
2. **Canonical results:** all ten files have 75,600 rows, local mtimes from 2026-08-03 02:47 through 10:49 CEST, and embedded run timestamps consistent with those runs. Exactly ten matching result filenames exist, all in `results_context/`; no result duplicate exists in `context_staging/` or elsewhere. Separate full prompt duplicates do exist in `context_staging/`, byte-identical to the canonical corrected prompts.
3. **Actual prompts:** ten diverse required samples are in `outputs/nonbinary_prompt_samples.csv`; every sample has “they ... have/are” and none has the buggy suffix. The exhaustive unique-persona check found 0/1,800 bad neutral suffixes and 0/1,800 bad positive suffixes.
4. **Raw-result reproduction:** Ministral is 83.465608 → 48.513228 → 41.796296 → 37.251323 → 29.798942%, strictly monotonic. Context-context mean rho is .878441 versus .790519 context-original. Current pair order and all 16 current rating-shift verdicts reproduce. Historical unchangedness cannot be proven without pre-fix raw data. Corrected llama/positive profession rho=.921053, exact p=.066667.
5. **Freshness:** every output whose name directly involves neutral/positive, plus all cross-context, stability, and DeepSeek aggregate outputs, is 3,074–3,637 seconds newer than the latest corrected result file. No stale involved output was found.
6. **Markdown:** most numerical tables match, but three statements do not: agreement ranges, the 7/8 truck-driver count, and the characterization of the llama rank movement. See the mechanical ledger and issue register.

## Llama profession order

The corrected fit gives:

1. registered nurse
2. lawyer
3. truck driver / computer programmer (tie after 9-decimal tie restoration)
5. farmer

The original-condition baseline used in the comparison is registered nurse, lawyer, computer programmer, then farmer/truck driver tied at bottom. Thus the corrected positive context preserves nurse at top and farmer at bottom, but truck driver leaves the bottom tie and joins a middle tie. Calling this merely “mid-ranking” understates that the bottom set changed, although it is not a top-to-bottom reversal. The pre-fix positive order cannot be reconstructed from retained raw data.

## Plain-language materiality answer

The corrected current data do **not** overturn the main substantive story: Ministral's five-condition abstention sequence remains strictly decreasing, and the structural clustering gap is .087922 rho (.878441 minus .790519). Current rating-shift directions/significance match the document's corrected table, with 13 of 16 cells significant.

The one confirmed inferential boundary change is the corrected llama/positive profession comparison at p=.066667 instead of the reported pre-fix p=.033333, with rho reported to have fallen from .974679 to .921053. Its rank movement is limited—nurse remains top and farmer remains bottom—but truck driver moves from a bottom tie in the original baseline to a middle tie in corrected positive. Because pre-fix raw files are absent, the exact magnitude of every before/after movement and the assertion that all other verdicts stayed unchanged cannot be independently certified.

## Deliverables

- `outputs/reproduction_table.csv`
- `outputs/issue_register.csv`
- `outputs/markdown_numeric_crosscheck.csv`
- `outputs/result_provenance.csv`
- `outputs/result_duplicate_inventory.csv`
- `outputs/nonbinary_prompt_samples.csv`
- `outputs/nonbinary_prompt_exhaustive.csv`
- `outputs/analysis_output_freshness.csv`
- `outputs/rating_shifts_current_raw.csv`
- `outputs/cross_context_raw.csv`
- `outputs/llama_positive_profession_ranks_current.csv`
- `logs/reproduction.log`
