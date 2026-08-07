# Audit History

Permanent record of the five independent audit rounds run against this project.
Each section is pulled directly from that round's own saved report(s) — not
reconstructed from memory. The underlying audit scaffolding (scripts, logs, raw
audit-output CSVs) has been removed from the working tree per the cleanup
described in `CONTEXT_EXPERIMENT.md`'s changelog; full detail for every round
remains recoverable from git history (`git log --all -- <path>`), and the
outputs of the three previously-untracked rounds (`audit_context/`,
`audit_grammar_fix/`, `audit_preflight/`) were each committed to git
specifically to preserve that recoverability before removal.

---

## 1. `analysis_health/audit/` — prior adversarial audit of the health study

**Date and scope:** Original run date unrecoverable — its script
(`run_adversarial_audit.py`) and any narrative report were both lost before this
project's current state; only the compiled `.pyc` and 13 raw output CSV/JSON
files under `analysis_health/audit/output/` survived, committed to git on
2026-07-29. Those outputs were independently regenerated and reconciled against
current `analysis_health/` scripts on 2026-07-24, producing
`analysis_health/output/CORRECTED_SUMMARY.md` (retained; not part of this
cleanup) — that document is the actual source for this section, since the
original report text does not exist.

**Key findings:** The audit's surviving outputs establish, and the 2026-07-24
reconciliation confirms, seven specific fixes needed relative to a naive
comparison (labeled H1-H7 in the source document):
- **H1** — the original/health-minus rating-shift sign convention needed
  correcting.
- **H2** — Ministral/Qwen abstention shifts must be reported at two different
  denominators (all-rows vs. Condition-B-only), which differ by roughly a
  factor of two and are not interchangeable.
- **H3** — Condition A and B results were being pooled where they needed to be
  reported split.
- **H4a/H4b** — profession/country ranking robustness needed exact permutation
  p-values (not asymptotic, invalid at n=4-5) and persona-bootstrap
  rank-position probabilities.
- **H5** — DeepSeek's health-condition output required a compact-text parser to
  recover a further 14.86% (11,234/75,600) of responses as containing a
  plausible rating beyond strict parsing; every one of those recovered ratings
  is the digit "4" (unexplained, flagged, not treated as evidence of
  substantive DeepSeek opinion).
- **H6** — the original-vs-health merge needed `validate="one_to_one"`
  enforcement (12,600 rows/family, 0 unmatched, confirmed).
- **H7** — per-condition sample-size and validity-rate labeling was required
  alongside each shift number.

Also confirmed: Qwen's Condition-B rating shift (+0.325, p=4.5e-11) is the
*opposite sign* from its own Condition-A shift (-0.156) — read as evidence of
Condition-B self-selection bias, not a genuine opinion reversal, since only
6.35% of Qwen's Condition-B pairs are both-valid.

**Fixes applied:** All seven items above were incorporated into the current
`analysis_health/01-04_*.py` scripts and their outputs, cross-checked cell-for-
cell against this audit's surviving raw CSVs wherever a ground-truth value was
available.

**Final verdict:** No independent GO/NO-GO label survives from this round
specifically (the report that would have stated one is lost). Its
reconciled methodology and numbers were later independently re-verified in
full by `audit_full_project` (Sections 20-21, "CONFIRMED, matches saved
value"), which folded this round's health-study results into its own overall
**GO**.

---

## 2. `audit_full_project/` — full-project traceability and reproduction audit

**Date and scope:** 2026-07-28 to 2026-07-29. Three checkpoints: (1) full
project/experiment traceability map and design reconstruction, (2) persona-
construction integrity, batching/independence, a silent-coding-error sweep, and
a new unit-test suite, (3) a clean-environment, cell-for-cell regeneration of
every main and health analysis result (Sections 13-23).

**Key findings:**
- The prior v9 experiment, original pilot outputs, manipulation-check outputs,
  and prompt-reinforcement test outputs are **unrecoverable** — code/mentions
  exist but no raw outputs remain (reproducibility failures, left open as
  historical gaps, not fixed).
- `analysis_health/audit/`'s own source script was already missing at this
  point (independently confirming Round 1's gap above).
- **Major, found and fixed:** `05e_bh_correction.py` (Benjamini-Hochberg
  correction) had been silently overwritten by later reruns of
  `05_hypothesis_models.py`/`06_abstention_analysis.py` — the checked-in
  coefficient tables had **zero `*_bh_adj` columns** even though a stale
  summary claimed BH correction had been applied. Root cause: `README.md` had
  the script run order wrong. Fixed by rerunning `05e` last, correcting the
  documented order, and adding a regression test to guard against recurrence.
- **Major, found and fixed:** an obsolete `tables/ordinal_factor_ranking.csv`
  from a prior implementation (pooled A+B, old `chi2_per_df` ranking) was still
  present and could be mistaken for a live result. Deleted.
- **Moderate, documented (not fixed):** `02_validate_dataset.py` prints
  validation failures but doesn't raise/exit nonzero, so a future automated
  rerun could silently continue past a structural failure; and the inference
  loop's `zip(batch, results)` isn't length-asserted, a latent silent-truncation
  risk.
- Clean-environment regeneration reproduced all H1-H3 hypothesis tests,
  clustered/mixed-model inference, ordinal robustness, country/topic
  robustness, the health-vs-original comparison, abstention analyses, and
  ranking robustness — with one labeling-only discrepancy: Llama's original
  bottom profession is an exact tie between `farmer` and `truck driver`; the
  saved table had labeled only `truck driver`, later corrected to state the
  tie explicitly.
- DeepSeek's exploratory 63-row regression was confirmed rank-deficient/
  numerically unstable (ten HC3 p-values moved materially on rerun); every row
  of its output table now carries an explicit **NON-INFERENTIAL / EXPLORATORY
  ONLY** warning.

**Fixes applied:** BH-correction rerun in correct order + regression test
added; obsolete ordinal table deleted; Llama's tied-bottom profession now
reported as a tie everywhere; DeepSeek's exploratory table annotated
non-inferential; run-order documentation corrected in `README.md`.

**Final verdict: GO** (updated 2026-07-29 after closeout fixes). "The reported
main and health-study numerical results survive clean regeneration... It does
not invalidate the audited results or block the next planned experiment."

---

## 3. `audit_context/` — adversarial validation of `analysis_context/`

**Date and scope:** 2026-08-02. Read-only validation of the `neutral`,
`positive`, `negative_minor`, and `health` context-experiment prompts, results,
and `analysis_context/` outputs — no inference launched.

**Key findings** (issue register CTX-01 through CTX-06):
- **CTX-01 (High) — the original discovery of the neutral/positive grammar
  bug.** The `neutral` and `positive` prompt templates have a subject-verb
  agreement defect for neutral-gender ("they") personas: literal hardcoded
  "has"/"is" instead of the already-computed `HAVE_VERB`/`be` gender-aware
  variables. Affects all 1,800 neutral-gender personas — 25,200 prompt/result
  rows in neutral and 25,200 in positive, per model. This is the bug fixed and
  reconciled in `CONTEXT_EXPERIMENT.md`'s changelog.
- **CTX-02 (High)** — three Condition-A findings-table cells and their prose
  were materially wrong (llama/positive, qwen/health, ministral/health) —
  hand-transcription errors, not a pipeline bug. (Corrected in
  `CONTEXT_EXPERIMENT.md` prior to this cleanup.)
- **CTX-03 (Medium)** — the cross-context "clustering by structure not content"
  claim was overinterpreted as an established structural cause rather than
  descriptive evidence.
- **CTX-04 (Low)** — the exact-agreement and Spearman ranges as originally
  stated (74-92%, 0.71-0.90) were imprecise; independently reproduced as
  68.81-90.71% and 0.686-0.894.
- **CTX-05 (Low)** — the pilot topic-drop range was rounded from 56.11pp up to
  "60pp" in prose.
- **CTX-06 (Low)** — `analysis_context/`'s statistical helpers were confirmed
  copied/generalized from the audited health module rather than a single
  shared implementation, with no numerical drift found, but should not be
  described as centralized code.

Independently reproduced and confirmed: Ministral's Condition-B abstention
decline across all five conditions (83.47% → 48.51% → 42.01% → 37.22% →
29.80%, every adjacent step significant); DeepSeek's reconciled true-compliance
range (14.86%-90.80%, not the earlier double-counted "11%-91%"); the neutral
Condition-B degenerate case (209 both-numeric cells, all rating 4 in both
conditions); all 15 `results_context` files' row/persona-ID/key integrity.
The full findings-reproduction ledger checked 128 claims: 119 matched, 8
mismatched (the CTX-02/CTX-04 items above), 1 qualified.

**Fixes applied:** None applied directly by this audit round itself (read-only
by design) — it is the round that *identified* the grammar bug and the three
wrong table cells, both of which were subsequently fixed (grammar bug: full
re-run documented in `CONTEXT_EXPERIMENT.md`'s "Resolved issue" section and
changelog; table cells: corrected in the same document's Section 1 correction
note).

**Final verdict: CONDITIONAL GO.** "The Ministral and DeepSeek findings cannot
be presented as-is" without the qualifications above; no new experiment was
required to establish the audited numerical facts, but the grammar defect had
to be disclosed and the three table cells corrected before circulation. (Both
were subsequently done — see Round 4 below and `CONTEXT_EXPERIMENT.md`.)

---

## 4. `audit_grammar_fix/` — independent adversarial audit of the grammar-bug fix and re-run

**Date and scope:** 2026-08-03. Independent verification that the CTX-01 grammar
fix (found in Round 3) was genuinely applied, that the corrected neutral/
positive data actually reached `results_context/`, and that
`CONTEXT_EXPERIMENT.md`'s reconciled before/after numbers were accurate.

**Key findings:**
- The code fix is genuine: `neutral` uses `{have}`, `positive` uses `{be}`,
  both threaded into the active `.format(...)` call. Exhaustive check of all
  1,800 unique non-binary personas in each canonical prompt CSV found **zero**
  bad suffixes remaining.
- All ten corrected result files (5 models x 2 contexts) have 75,600 rows,
  mtimes from 2026-08-03 02:47-10:49 CEST, live only in `results_context/`
  (no stale duplicate found in `context_staging/` or elsewhere) — confirming
  the move-not-copy cleanup was done correctly.
- Ministral's monotonic abstention decline reproduces on corrected data:
  83.465608% → 48.513228% → 41.796296% → 37.251323% → 29.798942%. Cross-context
  structural-clustering gap reproduces: 0.878441 (context-context) vs. 0.790519
  (context-original).
- **Three markdown inaccuracies found and later corrected in
  `CONTEXT_EXPERIMENT.md`:** the exact-agreement/Spearman ranges (should be
  68.33-90.71% / 0.686-0.894, matching Round 3's CTX-04), the "truck driver
  ranks bottom in 7 of 8 cells" count (should be 6 of 8 — llama is an exception
  under *both* neutral and positive, not just one), and the llama/positive
  ranking-movement description (truck driver moves from a **bottom tie** with
  farmer in the original ranking into a **middle tie** with computer programmer
  in the corrected positive ranking — not merely a generic "mid-ranking
  reshuffle").
- **Disclosed limitation, not an error:** the pre-fix raw neutral/positive
  result files were overwritten during cleanup and no copy exists anywhere
  under matching filenames. The historical before/after reconciliation
  therefore relies on summary values recorded at the time the fix was
  verified, not on independently re-derivable raw data today. The pre-fix
  llama/positive profession p-value of .033333 comes from a prior derived audit
  CSV, not raw files, and this audit round explicitly declined to treat that as
  raw evidence.

**Fixes applied:** The three markdown inaccuracies identified here were
corrected in `CONTEXT_EXPERIMENT.md` (agreement/Spearman ranges, the 6-of-8
count, and the bottom-tie-to-middle-tie description), with the raw-data
limitation now stated explicitly in that document's changelog.

**Final verdict: CONDITIONAL GO.** "The corrected current data do not overturn
the main substantive story" (Ministral's decline remains strictly decreasing;
13 of 16 rating-shift cells remain significant in the same direction); GO is
conditional on the three markdown corrections above and on explicitly
qualifying the historical before/after claims as not independently
reproducible from retained raw artifacts — both conditions have since been
met.

---

## 5. `audit_preflight/` — pre-Stage-1-5 repository-readiness preflight

**Date and scope:** 2026-08-04. A repository-wide health check run before
starting fresh Stage 1-5 analysis work (not a validation of any specific
finding): data integrity (row counts, schemas, models, keys, sampled
personas), code integrity (canonical loaders, condition rules), documentation
integrity (24 sampled numeric claims cross-checked against their source
CSVs), reproducibility (dependency/lineage mapping, clean-environment import
check), and analysis readiness (git cleanliness, duplicate/staging state).
Covers the whole project including the then-newly-completed full-scale
(5,400-persona) context extension.

**Key findings** (repository health score **62/100**: data integrity 86,
code integrity 72, documentation integrity 58, reproducibility 48, analysis
readiness 45):
- **Critical** — the working tree was dirty and the entire context study
  (code, results, docs) was untracked, so no commit at the time could serve
  as a reproducible frozen foundation.
- **High** — canonical health/context prompt CSVs were gitignored (blocking
  exact model-input reconstruction from a commit alone); `health_staging/`
  held byte-identical duplicates of the canonical health prompt, all five
  full result files, the renderer, and the inference script; the active
  Python environment could not `import pandas` despite the pin in
  `requirements.txt`.
- **High, not itself a numerical error** —
  `analysis_context/05_variance_ranking_all_prompt_types.py`'s *pilot-scope*
  dominant-factor summary (`dominant_factor_by_model.csv`) compares the
  full-5,400-persona `original` scope against the 180-persona pilot scope for
  the other four prompt types in one table, so that specific summary is not
  scope-consistent on its own — flagged as "label as mixed-scope historical
  output or compute a genuine full-scale companion before using pilot
  claims." (See this session's Step 1: the full-scale companion already
  existed and `CONTEXT_EXPERIMENT.md` already promoted it to primary with the
  pilot table demoted to a labeled historical appendix — confirmed via a
  follow-up repo-wide stale-reference sweep that found no unlabeled use of
  the mixed-scope verdict anywhere.)
- **Medium** — `README.md`/`analysis_plan.md` still described a stale
  pre-pilot, six-model project state; `analysis_health/04_ranking_robustness.py:107`
  has a dormant `.strip("[]T.")` bug (same class as the one this session's
  Step 1 full-scale work found and fixed in `analysis_context/_common.py`,
  but explicitly out of scope here — this module never exercises the full
  20-country design); `data/build_dataset.py` wrote to an obsolete absolute
  `/home/claude/...` path; `inference/full_health_inference.py`'s docstring
  claimed it lived in `health_staging/`; pilot context outputs were
  unsuffixed while full outputs used `_full5400`, which was flagged as a
  filename-only ambiguity risk.
- **Low** — broad `.gitignore` staging rules could conceal canonical
  duplicates; documented historical-artifact gaps (v9/pilot/manipulation/
  reinforcement raw outputs, the original health audit runner) remain
  unavailable, consistent with Rounds 1-2 above.
- All 24 sampled numeric documentation claims matched their source CSVs at
  the documented precision.

**Discrepancy noted and resolved:** this round's environment check found
`import pandas` failing, reporting the fingerprint "Python 3.13.7, pip
25.2." A same-day re-check in the cleanup session that followed this report
found `pandas` importing successfully. Root-caused directly rather than
assumed: this machine has four distinct `python3` interpreters — the
project's dedicated virtualenv (`~/.venvs/vscode`, Python 3.13.7, pip
25.2), bare Homebrew Python (`/opt/homebrew/bin/python3`, **also** Python
3.13.7, pip 25.2, since the venv was built from that same Homebrew base —
but with nothing installed into it), Anaconda's `base` env (Python 3.13.5,
pip 25.1, has pandas but drifts from `requirements.txt` on 6 of 8 pins),
and macOS system Python (3.9.6, no pandas). The audit's reported
version/pip fingerprint is an exact match for bare Homebrew Python, not the
project venv — confirmed by reproducing the identical failure when
`python3` is resolved with the venv's `bin/` excluded from `PATH`. The
audit's check simply ran without the project venv activated; nothing about
the venv's package state changed between the two checks (`pandas`'s
`dist-info` predates this report by several months). The project venv
matches all 8 `requirements.txt` pins exactly (verified via `pip list
--format=freeze` diffed against `requirements.txt`) and is the environment
a new user or future session must activate — `README.md`'s setup
instructions now say so explicitly, including the bare-`python3`-resolves-
to-an-empty-interpreter gotcha this round surfaced.

**Fixes applied:** None applied directly by this audit round itself
(read-only preflight, same design as `audit_context` in Round 3) — it is the
round that triggered the data/+results/ reorganization immediately
following it in the same cleanup session. That follow-on work: confirmed the
mixed-scope H1 summary issue above was already resolved at the documentation
level (Step 1); consolidated `data/`, `data_health/`, `data_context/`,
`results/`, `results_health/`, `results_context/`, `context_staging/`, and
`health_staging/` into a single `data/`+`results/` structure, deleting the
byte-identical `health_staging/` duplicates flagged here (Step 2); fixed
`data/build_dataset.py`'s obsolete absolute-path bug (Step 2); and updated
13 scripts' path references accordingly (Step 2). `README.md`/
`analysis_plan.md` staleness is addressed in this same session's Step 4. The
dormant `analysis_health/04_ranking_robustness.py` strip bug remained
unfixed at the end of this round, per its explicit "not blocking" judgment.
During final pre-push validation on 2026-08-07, the active source was corrected
to remove only the exact trailing bracket. Existing pilot outputs were not
regenerated because their factor levels could not trigger the defect.

The same final validation checked all 25 canonical result files: each retained
75,600 rows, 5,400 personas, seven topics, both response conditions, and 75,600
unique persona-topic-condition keys, with no duplicate keys or recorded technical
failures. No inference or statistical model was run during cleanup, and no
scientific result or analysis output was changed.

**Final verdict: NO GO** (as issued — "Only after these concrete issues are
closed should this state be marked GO and used as the foundation for new
Stage 1-5 analyses"). Several of this round's blocking items were closed by
the reorganization that immediately followed within the same cleanup
session (see "Fixes applied" above); this entry records the verdict as this
round actually issued it, not as subsequently revised.
