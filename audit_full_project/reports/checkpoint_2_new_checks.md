# Full-project audit — Checkpoint 2

Scope: Section 3 (persona construction integrity), Section 8 (batching and
independence), Section 25 (silent coding errors), and Section G (unit tests).
Sections 13–23 have not been started.

## Post-checkpoint remediation

The BH/output issues found at this checkpoint were corrected on 2026-07-28:

- `05e_bh_correction.py` was rerun against the current 05/06 tables;
- every adjusted column is present, bounded in [0, 1], and non-null wherever
  its source p-value is non-null;
- the README now places 05e after 05, 06, and the other regression-producing
  scripts and requires rerunning 05e after any rerun of 05 or 06;
- obsolete `tables/ordinal_factor_ranking.csv` was deleted;
- the corresponding unit-test guards now pass.

## Executive checkpoint result

- Persona construction and identity propagation: **PASS**.
- Main and health prompt/output row alignment: **PASS**.
- Computational batching and final partial batches: **PASS in code inspection
  and mocked execution**.
- Evidence of cross-row/cross-persona contamination: **none found**.
- Silent coding-error sweep: **one major current output-regression found**, one
  obsolete table found, and several lower-severity safeguards/provenance gaps
  documented below.
- Unit tests after remediation: **24 tests executed; 23 ordinary passes and one
  intentional expected failure for the remaining diagnostic-parser edge case**.

## Section 3 — persona construction integrity

### Canonical persona grid

Independent checks on `data/personas.csv`:

| Check | Result |
|---|---:|
| Rows | 5,400 |
| Unique persona IDs | 5,400 |
| Duplicate country×profession×gender×age combinations | 0 |
| Invalid persona-ID formats (`P` + five digits) | 0 |
| Blank/missing demographic fields | 0 |
| Leading/trailing whitespace differences | 0 |
| Case-only spelling variants | 0 |
| Country levels | 20 |
| Profession levels | 30 |
| Gender levels | 3 |
| Age levels | 3 |
| Unique names | 60 |
| Country×gender cells with more than one name/pronoun bundle | 0 |

Every country×profession×gender×age combination occurs exactly once. The
persona generation loop at `data/build_dataset.py:148-175` agrees with the
saved grid.

`Nour` (Egypt) and `Noor` (Pakistan) are distinct documented transliterations,
not accidental duplicates. No exact name is reused across country×gender cells.

### Stability through pipeline stages

The canonical key checked was:

```text
persona_id, country, profession, gender, age, topic, response_condition
```

Results:

- `data/prompts.csv`: zero duplicate canonical cells.
- `data_health/health_prompts_full.csv`: zero duplicate canonical cells.
- Every main raw result file: zero duplicate canonical cells and exact
  one-to-one key coverage against the main prompt file.
- Every health raw result file: zero duplicate canonical cells and exact
  one-to-one key coverage against the health prompt file.
- `analysis/master_results.csv`: zero duplicate model×canonical-key cells.
- Every main raw result maps one-to-one to its corresponding master rows.
- Every prompt and raw result maps each persona ID to exactly the demographics
  in `data/personas.csv`.
- The 180-persona health-analysis subset uses exactly the same persona-ID set
  in all five original files and all five health files.

Stronger than set equality, the complete canonical-key sequence in every raw
result file is **position-for-position identical** to its generating prompt
file. There were zero sequence mismatches in all ten retained model runs.

### Repeated/malformed persona prompts

- Main prompt bodies: 75,600 unique strings out of 75,600 rows.
- Health conversations: 75,600 unique `messages_json` values out of 75,600.
- No exact prompt or conversation was repeated under a different persona ID.
- All 75,600 health message lists were independently regenerated from
  `health_render_prompts_full.py`; mismatches: **0**.
- Within every health persona/topic A–B pair, the first four scripted messages
  are exactly identical; mismatches: **0**.

**Persona-integrity verdict: confirmed.**

## Section 8 — batching and independence

### Main inference

The main batching path is at:

- fresh one-user-message construction:
  `inference/full_inference.py:92-101`;
- generation and completion slicing:
  `inference/full_inference.py:104-118`;
- retry behavior:
  `inference/full_inference.py:119-124`;
- row batching and output pairing:
  `inference/full_inference.py:164-191`.

For each input row, `build_batch_inputs` constructs a separate message list and
renders it independently with `apply_chat_template`. The tokenizer receives a
list of separate strings, not one concatenated conversation. It produces an
attention mask, and the tokenizer is configured for left padding. There is no
code path carrying history, KV cache, or messages from one row to another.

The returned generation tensor is indexed by batch row, and only tokens after
the common padded input width are decoded. With left padding, slicing at
`enc["input_ids"].shape[1]` is the correct decoder-only completion boundary.

The outer loop uses:

```python
range(0, len(df), BATCH_SIZE)
df.iloc[start:start + BATCH_SIZE]
```

so the final partial batch is included.

### Health inference

The health path at `inference/full_health_inference.py:102-136` receives a list
of complete message lists. Each five-message conversation is separately
rendered. There is no shared mutable message list or generated assistant turn;
the two assistant messages are scripted in the CSV, and only the final model
continuation is live-generated.

### Retry behavior

On a batch exception, the same batch is retried once recursively. Nothing is
written before `generate_batch` returns, so a successful retry cannot duplicate
already-written rows. If the retry also fails, the function returns exactly one
technical-failure placeholder per input row. The current raw files contain:

```text
technical_failure rows: 0
empty raw_text rows:     0
```

for every main and health model.

Because run logs were not retained, it is impossible to establish whether any
batches failed on their first attempt and then succeeded. This does not affect
final row completeness, but retry history is not auditable.

### Executed batching tests

Mocked execution verified:

1. outputs remain in input order;
2. prompt tokens are excluded from decoded output;
3. a failed batch is retried once;
4. a permanent batch failure produces one failure row per input;
5. no failure retry duplicates rows;
6. an 18-row input with batch size 16 writes exactly 18 ordered rows across a
   16-row batch and a two-row final batch;
7. separate health message lists remain separate and ordered.

### Remaining defensive gaps

The inference loops use:

```python
zip(batch.itertuples(), results)
```

without asserting `len(results) == len(batch)`. Normal Hugging Face generation
returns one output per input, and the current saved counts/order are complete,
so no actual loss occurred here. Nevertheless, an anomalously short `results`
list would be silently truncated rather than raising.

Raw outputs also omit a prompt hash and original input-row ordinal. Exact
canonical-key sequence equality provides strong retrospective evidence of
alignment, but a stored prompt hash would make this independently
cryptographically verifiable.

**Batching verdict:** no batching, row-order, retry-duplication, final-batch, or
cross-conversation error was found. Semantic “memory contamination” cannot be
proved absent from output text alone, but the implementation contains no
mechanism that carries conversation state between rows.

## Section 25 — silent coding-error sweep

### Major, resolved after detection: BH-adjusted columns had been erased

`analysis/05e_bh_correction.py` adds BH columns by rewriting these tables in
place:

- `hypothesis_model_pooled.csv`;
- each `hypothesis_model_{model}.csv`;
- `abstention_model_qwen_ministral.csv`.

The current saved versions contain **zero `*_bh_adj` columns**.

At the same time, `tables/bh_correction_summary.csv` remains and reports the
numbers that previously survived BH correction. The repository therefore
simultaneously presents:

- a summary claiming BH correction was applied; and
- current coefficient tables from which the adjusted columns have disappeared.

Cause:

- `05_hypothesis_models.py` and `06_abstention_analysis.py` recreate their CSVs
  from scratch;
- `05e_bh_correction.py` must run after both;
- `MANIFEST.md` correctly warns about this;
- `README.md:65-70` and `README.md:114-118` incorrectly put `05e` before `06`;
- the file timestamps and schemas show that the model scripts were rerun after
  the last BH pass.

Why it matters: the checked-in inferential tables no longer faithfully contain
the multiplicity-corrected results required by the stated pipeline, while a
stale summary can conceal the loss.

Likely effect on scientific conclusions: the saved historical summary indicates
most main-model findings survived BH, so this probably changes reporting rather
than headline directions. That must be confirmed in Sections 13–23.

Resolution: `05e` was rerun after the final outputs from both `05` and `06`,
the documented run order was corrected, and a regression test now guards
against later overwrite.

### Major/moderate, resolved after detection: obsolete ordinal ranking table

`tables/ordinal_factor_ranking.csv` has no current producer and is not listed in
the current manifest. Its contents are visibly from an older implementation:

- 24 rows = six factors × four models;
- includes `response_condition`;
- uses the old `chi2_per_df` ranking;
- therefore pools A and B.

The current ordinal script instead uses Condition A only, five factors, partial
pseudo-R², and writes per-model files. The orphan table contradicts the current
method and could be mistaken for a live result. It was deleted after this
checkpoint finding was confirmed.

### Moderate: validation script does not fail the pipeline

`analysis/02_validate_dataset.py` prints failures but does not raise or return a
nonzero exit status. The current data pass, so this caused no present corruption.
In an automated rerun, however, later analyses would continue even after a
structural validation failure.

### Moderate: result-length mismatch could be silently truncated

As described in Section 8, no assertion guards `zip(batch, results)`. This did
not occur in current data but is a preventable silent-failure path.

### Minor/moderate: ambiguous multi-rating salvage

The terminal salvage regex accepts a digit at the end of an otherwise malformed
response. Consequently:

```text
"4 and 5" -> salvaged_rating = 5
```

Across the main DeepSeek file, 1,325 outputs contain at least two distinct
standalone digits from 1–5; 221 of those receive a salvage rating. This does not
affect `strict_is_valid` or any primary/inferential rating analysis—salvage is
diagnostic only. It can, however, distort claims about what fraction of
DeepSeek’s malformed outputs contains a recoverable unambiguous rating.

Reapplying the current parser to all raw files reproduced every saved
`strict_is_valid` and `salvaged_rating` value exactly. Thus this is a parser
design limitation, not evidence of master-file drift.

### Reproducibility/fragility findings retained from Checkpoint 1

- main prompt label `friend_v1` versus raw provenance
  `friend_v2_explicit_gender`;
- hard-coded `/home/claude/...` dataset output paths;
- absolute user-specific paths in every health analysis script;
- undocumented inference-output relocation;
- no retained run/retry logs;
- health tree untracked in Git;
- prior health-audit outputs lack source;
- pilot/manipulation/reinforcement artifacts absent.

### Checks that did not find an error

- no wrong original/health sign convention in current shift helpers;
- no condition-label type mismatch;
- no model-name inconsistency in retained raw data;
- no direct unsafe `pd.read_csv(master_results.csv)` outside `_common.py`;
- literal `NA` remains protected;
- no duplicate merge inclusion in current health comparisons;
- health merges use the full canonical key with `validate="one_to_one"`;
- no stale row order or sorting mismatch between prompts and outputs;
- all Python sources compile successfully;
- generation configuration is constant within and across the five main runs,
  and constant within and across the five health runs;
- main and health use the same model IDs, Transformers version, Torch version,
  GPU, and generation configuration in retained provenance.

## Section G — unit test suite

Created:

`audit_full_project/tests/test_pipeline_integrity.py`

Run command:

```bash
python -m unittest discover -s audit_full_project/tests -v
```

Final result after remediation:

```text
Ran 24 tests
OK (expected failures=1)
```

Ordinary passing coverage:

- complete persona factorial;
- persona ID format and uniqueness;
- demographic/name/pronoun mapping;
- exact sampled main-prompt rendering;
- exact sampled health-message rendering;
- full main/health row-count and key lineage;
- exact raw-output order;
- master merge lineage;
- health subset equality;
- strict parsing and malformed cases;
- parser parity between main and health;
- denominator conventions;
- Condition-A-only ordinal filtering;
- health-minus-original sign convention;
- exact small-n permutation ranking calculation;
- batch order and completion slicing;
- retry behavior;
- permanent failure behavior;
- final partial batch;
- separate health conversations.

The one intentional expected failure encodes the unresolved ambiguous
multiple-rating salvage behavior. The BH-column, orphan-table, and README-order
guards are now ordinary passing tests.

The tests are non-destructive. The one inference-loop integration test writes
only inside a temporary directory that is deleted automatically.

## Checkpoint-2 verdict

The previously unaudited persona and batching layers are substantially sound.
No persona misassignment, prompt/output misalignment, dropped final batch,
duplicate retry row, or cross-conversation state mechanism was found.

The output-state inconsistency identified here has now been repaired and guarded
by tests. Its impact on reported scientific conclusions still needs confirmation
during the requested confirm-or-flag reproduction pass.
