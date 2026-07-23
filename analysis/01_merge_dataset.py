"""Step 1: concatenate the five per-model result CSVs into a single master file.

Adds a "model" column (short key: llama/gemma/qwen/ministral/deepseek).
Falcon-H1 is intentionally excluded (see analysis_plan.md, Known deviation).
Downstream scripts must read analysis/master_results.csv only, never the
per-model files in results/ directly.
"""
import pandas as pd

from _common import MODEL_FILES, MODEL_ORDER, RESULTS_DIR, MASTER_CSV, display_template


def main():
    frames = []
    for model in MODEL_ORDER:
        path = f"{RESULTS_DIR}/{MODEL_FILES[model]}"
        # keep_default_na=False: the literal string "NA" is a valid, meaningful
        # value in raw_text/normalized_text/strict_parsed_rating (the model's
        # exact compliant Condition-B abstention output) and must not be
        # silently converted to a null -- pandas' default NA sentinel list
        # includes "NA" and would otherwise destroy that distinction.
        df = pd.read_csv(path, keep_default_na=False, na_values=[])
        df.insert(0, "model", model)
        frames.append(df)
        print(f"  loaded {model:10s} {df.shape[0]:>7,} rows  {df.shape[1]} cols")

    master = pd.concat(frames, axis=0, ignore_index=True)

    # Derived numeric rating for analysis convenience: valid numeric answers
    # ('1'-'5') become floats; the abstention text "NA", blank/missing
    # fields, and any other malformed content all become NaN here. This
    # column is safe to average/model directly. The original
    # strict_parsed_rating column is left untouched as the literal string
    # (still distinguishing "NA" from "" from malformed text) for audit.
    master["rating_numeric"] = pd.to_numeric(master["strict_parsed_rating"], errors="coerce")

    templates = sorted(master["prompt_template_version"].unique())
    print(f"\nPrompt template(s) found in provenance: {templates}")
    print(f"  -> reported in write-up as: {[display_template(t) for t in templates]}")

    master.to_csv(MASTER_CSV, index=False)
    print(f"\nWrote {MASTER_CSV}")
    print(f"Total rows: {master.shape[0]:,}  ({master.shape[0] // len(MODEL_ORDER):,} per model x {len(MODEL_ORDER)} models)")
    print(f"Columns ({master.shape[1]}): {list(master.columns)}")


if __name__ == "__main__":
    main()
