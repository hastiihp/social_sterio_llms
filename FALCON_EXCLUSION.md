# Falcon-H1 Exclusion

**Status:** Falcon-H1-7B is excluded from this study. It was not replaced, substituted,
or rerun with a different configuration. The primary analysis proceeds with five
models: Llama-3.1-8B-Instruct, Gemma-3-12B-it, Qwen3-8B, DeepSeek-LLM-7B-chat, and
Ministral-8B-Instruct.

## Reason

Falcon-H1 produced a reproducible tensor-shape error in its generation cache under
batched inference, in the transformers version used for this study. This was confirmed:

- Across multiple batch sizes
- Under both dynamic and fixed-length padding
- Running cleanly *only* at batch size 1 in smoke tests -- which is not a viable
  configuration for the full six-model run given the study's runtime/storage budget
  (Section 11 of `analysis_plan.md`)

This is an inference-environment failure, not a data-quality or model-behavior finding.
No conclusions in this study are drawn about Falcon-H1's actual behavior, because no
usable data was ever collected from it.

## Consequence for the study

Excluding Falcon-H1 removes the study's only UAE/MENA-origin model. This is a real
limitation on any regional-comparison claim the study might otherwise support, and is
stated explicitly rather than left implicit:

> Falcon-H1 was excluded because inference could not be completed reliably under the
> available software environment despite repeated attempts.

This exclusion is documented here, in `analysis_plan.md`'s "Known deviation" note, and
should be repeated in any manuscript or write-up derived from this repository -- it is
not something later analysis scripts should silently work around or omit.

## What this means for the pipeline

Every script in `analysis/` operates on the five-model dataset only. There is no
Falcon-H1 column, row, or placeholder anywhere in `results/` or `analysis/master_results.csv`.
References to "all models" or "the five models" throughout this repository's code,
tables, and figures mean exactly those five -- this is not an oversight to be
reconciled against a six-model design.
