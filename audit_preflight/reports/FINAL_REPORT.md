# Final pre-analysis preflight audit

## A. Executive verdict

**NO GO.** The inspected data files are structurally complete and the sampled
cross-prompt identities match, but this filesystem cannot serve as a frozen foundation:
the working tree was already dirty, the entire context study is untracked, canonical
prompt datasets are ignored, health canonical data/results have byte-identical staging
duplicates, and the active environment cannot import the required analysis packages.

The full-scale context outputs are preferable to the pilot outputs and appear internally
organized, but `analysis_context/05_variance_ranking_all_prompt_types.py` constructs its
pilot summary using full-scale original values, so that pilot result is not scope-consistent.

## B. Repository health score

| Dimension | Score / 100 | Finding |
|---|---:|---|
| Data integrity | 86 | Expected row counts, schemas, models, keys, and sampled personas pass; duplicates reduce certainty |
| Code integrity | 72 | Canonical loaders and condition rules are mostly disciplined; one dormant strip bug and one mixed-scope summary remain |
| Documentation integrity | 58 | 24 sampled numbers match, but README/project-state claims are badly stale and context docs are untracked |
| Reproducibility | 48 | Main lineage is mapped; current environment lacks dependencies and context state is absent from HEAD |
| Analysis readiness | 45 | Full data exist, but there is no recoverable clean freeze of the inspected state |
| Overall risk | **High** | Version-control/provenance failures can make future Stage 1–5 work irreproducible |

Overall repository health: **62/100**.

## C. Issue table

| Severity | Location | Description | Recommended fix | Blocking? |
|---|---|---|---|---|
| Critical | Git worktree | Modified `.gitignore`; context source, results, outputs, docs, and logs untracked | Review and commit the intended context state; remove/archive logs as appropriate; establish a clean commit | Yes |
| High | `.gitignore`, prompt datasets | Canonical health/context prompt CSVs are ignored, so the frozen commit cannot reconstruct exact model-facing inputs without regenerating them | Track canonical prompt files or provide immutable checksummed external artifacts with a documented retrieval procedure | Yes |
| High | `health_staging/` | Byte-identical duplicates of the health prompt, all five full results, renderer, and inference script | Remove/archive staging duplicates after confirming canonical copies; retain only scratch artifacts | Yes |
| High | Active Python environment | `pandas` is unavailable despite the requirements pin; reproduction cannot start | Build and record a clean environment, install pins, and run non-statistical smoke/import checks | Yes |
| High | `analysis_context/05_variance_ranking_all_prompt_types.py` | Pilot dominant-factor result compares full-5,400 original with pilot-180 context fits | Label as mixed-scope historical output or compute a genuine pilot original companion before using pilot claims | No for full-scale work; Yes if pilot result is used |
| Medium | `README.md` | Says next steps are pilot/full six-model run and topics are draft, contradicting completed five-model project and Falcon exclusion | Update README to current frozen state | Yes for documentation freeze |
| Medium | `analysis_plan.md` | Still marked pre-pilot draft and lists six models | Add an authoritative finalized methodology/status document or reconcile this file | Yes for documentation freeze |
| Medium | `analysis_health/04_ranking_robustness.py:107` | `.strip("[]T.")` can corrupt factor levels beginning/ending with those characters | Replace with exact suffix removal; regenerate only if health scope expands | No; dormant in existing outputs |
| Medium | `data/build_dataset.py` | Writes to obsolete absolute `/home/claude/stereotype_llm_paper/...` paths | Parameterize paths relative to the repository | No for existing data; blocks clean regeneration |
| Medium | `inference/full_health_inference.py` | Docstring says file is in `health_staging/`; prompt path is cwd-relative and only works from a staging-like directory | Make location/path documentation consistent and repository-relative | No for existing results |
| Medium | Context output naming | Pilot outputs are unsuffixed; full outputs are `_full5400`. Pilot scope is not obvious from filename alone | Add explicit `_pilot180` naming or a manifest mapping every output scope | No |
| Low | `.gitignore` | Broad staging ignores conceal full canonical duplicates and source-like scripts | Narrow staging rules or audit ignored contents before freeze | No |
| Low | Historical artifacts | Prior v9/pilot/manipulation/pre-fix context raw outputs and original health-audit runner are missing | Preserve limitation statements; do not imply those historical states are reproducible | No |

## D. Canonical dataset table

| Dataset | Canonical location | Row count | Models/scope | Duplicate-free? |
|---|---|---:|---|---|
| Personas | `data/personas.csv` | 5,400 | all studies; 20 countries × 3 genders × 3 ages × 30 professions | Yes |
| Topics | `data/topics.csv` | 7 | all studies | Yes |
| Original prompts | `data/prompts.csv` | 75,600 | all models; full design | Yes |
| Original results | `results/full_results_{model}.csv` | 75,600 each / 378,000 total | five models; full design | Yes |
| Main master | `analysis/master_results.csv` | 378,000 | five models; full design | Yes, but ignored/regenerable |
| Health prompts | `data_health/health_prompts_full.csv` | 75,600 | all five models; full design | **No** — duplicate in `health_staging/` |
| Health results | `results_health/health_full_results_{model}.csv` | 75,600 each / 378,000 total | five models; full design | **No** — five duplicates in `health_staging/` |
| Neutral prompts | `data_context/neutral_prompts_full.csv` | 75,600 | all five models; full design | Yes in working tree; ignored |
| Positive prompts | `data_context/positive_prompts_full.csv` | 75,600 | all five models; full design | Yes in working tree; ignored |
| Negative-minor prompts | `data_context/negative_minor_prompts_full.csv` | 75,600 | all five models; full design | Yes in working tree; ignored |
| Context results | `results_context/{context}_full_results_{model}.csv` | 75,600 each / 1,134,000 total | 3 contexts × 5 models; full design | Yes in working tree; untracked |

Downstream main analyses read `analysis/master_results.csv` through the safe loader,
except the documented country-set join and figure/table intermediates. Health/context
analyses intentionally read canonical raw result files directly. No downstream script
was found reading `health_staging/` or `context_staging/` data.

## E. Analysis consistency findings

- Falcon is absent from canonical results and explicitly excluded in current analysis
  code. Older inference code and the stale plan still list it, which is provenance code,
  not evidence that Falcon entered outputs.
- DeepSeek is isolated/excluded consistently in pooled, ordinal, agreement-matrix, and
  context ranking work where its strict-valid sample is unsuitable. Its diagnostic
  outputs remain separate.
- Primary ratings consistently use Condition A; Condition B is split or labeled
  descriptive/selection-sensitive. No contrary active analysis path was found.
- Full context outputs are consistently marked `_full5400` or `dataset_scope`, verified
  in five spot-checked files. Unsuffixed pilot outputs are less clear.
- Context definitions in code and `CONTEXT_EXPERIMENT.md` match the rendered prompt
  families. The three sampled persona IDs matched across all five prompt types.

## Documentation cross-check

All 24 selected numeric claims in `outputs/numeric_crosscheck.csv` match their actual
CSV sources at the documented precision. However, documentation is not globally
consistent: `README.md` and `analysis_plan.md` describe a pre-pilot, six-model project,
while `CONTEXT_EXPERIMENT.md`, `AUDIT_HISTORY.md`, and current outputs describe a
completed five-model project. No stale references to deleted audit directories were
misleading: `AUDIT_HISTORY.md` explicitly says they were removed and they are fully
recoverable from commit `15710b2`.

## Priority actions before Stage 1–5

1. Create a reviewed, clean Git commit containing the intended context code, results,
   documentation, and `.gitignore`; confirm `git status` is clean.
2. Eliminate the health staging duplicates and decide how immutable canonical prompt
   CSVs will be versioned or checksummed/retrieved.
3. Reconcile README/methodology status with the actual completed five-model design.
4. Build the pinned Python environment and demonstrate imports from a clean checkout.
5. Do not use the unsuffixed context pilot dominant-factor summary as a five-way
   like-for-like comparison unless its mixed scope is corrected or explicitly labeled.

Only after these concrete issues are closed should this state be marked GO and used as
the foundation for new Stage 1–5 analyses.
