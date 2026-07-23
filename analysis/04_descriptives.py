"""Step 4: descriptive mean ratings by demographic factor / topic, per model.

Uses strict-valid, numeric ratings only (rating_numeric, derived from
strict_parsed_rating, restricted to strict_is_valid==True and excluding
valid NA abstentions -- per analysis_plan.md, strict_is_valid is the only
field used for primary/inferential analysis, and salvaged_rating is never
substituted in).

Condition A (forced) and Condition B (optional) are reported SEPARATELY,
never pooled or shown as directly comparable without context. For every
cell we report both n_answered (rows with a valid strict numeric rating)
and n_total (all rows in that condition/model/factor-level slice), because
for qwen and ministral, Condition B's answered subset is a small,
self-selected fraction of the total (abstention is common for exactly
those two models -- see table1_compliance.csv / step 3 output): a
Condition-B mean there describes only the ~10-17% of cases where the model
chose to answer, not a random sample of personas. Per analysis_plan.md
Section 5, this conditional-rating subset is descriptive, not causal.
"""
import pandas as pd

from _common import load_master, MODEL_ORDER, TABLES_DIR

FACTORS = ["gender", "country", "profession", "age", "topic"]
CONDITIONS = ["A_forced", "B_optional"]


def summarize(df, factor):
    rows = []
    for model in MODEL_ORDER:
        msub = df[df["model"] == model]
        for cond in CONDITIONS:
            csub = msub[msub["response_condition"] == cond]
            for level, g in csub.groupby(factor, observed=True):
                n_total = len(g)
                valid = g[g["strict_is_valid"] & g["rating_numeric"].notnull()]
                n_answered = len(valid)
                mean_rating = valid["rating_numeric"].mean()
                sd_rating = valid["rating_numeric"].std()
                rows.append({
                    "model": model,
                    "condition": cond,
                    factor: level,
                    "mean_rating": mean_rating,
                    "sd_rating": sd_rating,
                    "n_answered": n_answered,
                    "n_total": n_total,
                    "pct_answered": 100.0 * n_answered / n_total if n_total else float("nan"),
                })
    out = pd.DataFrame(rows)
    col_order = ["model", "condition", factor, "mean_rating", "sd_rating", "n_answered", "n_total", "pct_answered"]
    return out[col_order].sort_values(["model", "condition", factor]).reset_index(drop=True)


def main():
    df = load_master()

    for factor in FACTORS:
        table = summarize(df, factor)
        out_path = f"{TABLES_DIR}/descriptives_{factor}.csv"
        table.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  ({len(table)} rows)")

    print()
    print("=" * 78)
    print("SELECTION-BIAS WARNING: Condition B 'answered' subset size, by model")
    print("(this is the denominator context every Condition-B mean above depends on)")
    print("=" * 78)
    b = df[df["response_condition"] == "B_optional"]
    for model in MODEL_ORDER:
        sub = b[b["model"] == model]
        n_total = len(sub)
        n_answered = (sub["strict_is_valid"] & sub["rating_numeric"].notnull()).sum()
        pct = 100.0 * n_answered / n_total
        flag = "  <-- small, self-selected subset; NOT comparable to Condition A without caveat" if pct < 50 else ""
        print(f"  {model:10s} n_answered={n_answered:>6,} / n_total={n_total:>6,}  ({pct:5.2f}% answered){flag}")

    print()
    print("=" * 78)
    print("EXAMPLE: mean rating by topic, Condition A vs Condition B (side by side, with n)")
    print("=" * 78)
    topic_table = summarize(df, "topic")
    pivot = topic_table.pivot_table(
        index=["model", "topic"],
        columns="condition",
        values=["mean_rating", "n_answered", "n_total"],
    )
    print(pivot.to_string(float_format=lambda x: f"{x:6.2f}"))
    print()
    print("Note: Condition A and Condition B columns are shown side by side for")
    print("compactness ONLY -- they are not a matched comparison here (see step 6 for")
    print("the proper paired persona-topic-model comparison per analysis_plan.md Section 6).")
    print("Always check n_answered/n_total for Condition B before interpreting its mean,")
    print("especially for qwen and ministral.")


if __name__ == "__main__":
    main()
