"""Step 3: is each model's abstention behavior a stable MODEL property, or
does conversational framing/context change it?

Reports two distinct metrics side by side, per model, across all FIVE
conditions (original + health + neutral + positive + negative_minor) on the
full 5,400-persona dataset -- deliberately not conflated, per Fix H7's
precision in analysis_health/02_compare_by_condition.py:
  - abstention_rate: fraction of rows where the model gave the literal,
    compliant "NA" response. Only meaningful where a working NA option
    exists in the prompt (qwen, ministral -- confirmed below). llama/gemma
    are structurally near-0% in every condition: not "choosing not to
    abstain", but operating in a prompt design where abstaining barely
    happens as a matter of trained behavior, not condition.
  - strict_valid_rate: fraction of rows that are well-formed at all (a
    single digit 1-5, or NA where allowed). This is the relevant metric for
    DeepSeek, whose story across conditions is near-total FORMAT
    non-compliance, not selective abstention -- it essentially never
    outputs the literal "NA" token either way.

Terminology: health/neutral/positive/negative_minor are conversational
framings/contexts (Step 0), not valence levels -- "does context change
abstention" is asked about conversational framing broadly, not about
"more positive" or "more negative" framings specifically.

Independent of analysis/ and analysis_health/. Writes only to
analysis_context/output/.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
_common = importlib.import_module("_common")
globals().update({k: getattr(_common, k) for k in dir(_common) if not k.startswith("__")})

OUT_DIR = f"{ROOT}/analysis_context/output"
ALL_CONDITIONS = ["original"] + CONTEXTS


def build_rate_table():
    print(f"{'='*78}\nPER-MODEL ABSTENTION / COMPLIANCE RATE, ALL 5 CONDITIONS, FULL 5,400-PERSONA DATASET\n{'='*78}")
    rows = []
    for m in MODEL_ORDER:
        for c in ALL_CONDITIONS:
            df = load_model_frame(c, m)
            b = df[df["response_condition"] == "B_optional"]
            rows.append({
                "model": m, "condition": c,
                "abstention_rate_allrows_pct": 100 * df["is_abstention"].mean(),
                "abstention_rate_condB_pct": 100 * b["is_abstention"].mean(),
                "strict_valid_rate_allrows_pct": 100 * df["strict_is_valid"].mean(),
                "n_allrows": len(df), "n_condB": len(b),
            })
    table = pd.DataFrame(rows)
    table.to_csv(f"{OUT_DIR}/abstention_stability_rate_table.csv", index=False)

    print("\nAbstention rate (% of ALL rows), model x condition:")
    print(table.pivot(index="model", columns="condition", values="abstention_rate_allrows_pct")
          .reindex(MODEL_ORDER)[ALL_CONDITIONS].to_string(float_format=lambda x: f"{x:7.4f}"))

    print("\nAbstention rate (% of Condition-B rows only), model x condition:")
    print(table.pivot(index="model", columns="condition", values="abstention_rate_condB_pct")
          .reindex(MODEL_ORDER)[ALL_CONDITIONS].to_string(float_format=lambda x: f"{x:7.4f}"))

    print("\nstrict_valid rate (% of ALL rows), model x condition -- the relevant metric for DeepSeek:")
    print(table.pivot(index="model", columns="condition", values="strict_valid_rate_allrows_pct")
          .reindex(MODEL_ORDER)[ALL_CONDITIONS].to_string(float_format=lambda x: f"{x:7.4f}"))
    return table


def summarize_stability(table):
    print(f"\n{'='*78}\nSTABILITY VERDICT PER MODEL\n{'='*78}")
    rows = []
    for m in MODEL_ORDER:
        sub = table[table["model"] == m]
        allrows = sub["abstention_rate_allrows_pct"]
        condb = sub["abstention_rate_condB_pct"]
        strict = sub["strict_valid_rate_allrows_pct"]
        rng_allrows, rng_condb, rng_strict = allrows.max() - allrows.min(), condb.max() - condb.min(), strict.max() - strict.min()

        # llama/gemma/deepseek's primary story is not "abstention range" the way
        # qwen/ministral's is -- classify using the metric that actually varies
        if m in ("llama", "gemma"):
            metric_used, rng, verdict = "abstention_rate_allrows_pct", rng_allrows, (
                "STABLE (near-0% in all 5 conditions)" if rng_allrows < 1.0 else "NOT STABLE -- see exception below")
        elif m == "deepseek":
            metric_used, rng, verdict = "strict_valid_rate_allrows_pct", rng_strict, (
                "STABLE (near-total non-compliance in all 5 conditions)" if strict.max() < 1.0
                else "NOT STABLE")
        else:  # qwen, ministral
            metric_used, rng, verdict = "abstention_rate_condB_pct", rng_condb, (
                "STABLE (high abstention, narrow range across conditions)" if rng_condb < 10.0
                else "NOT STABLE -- abstention rate shifts materially with context")

        print(f"\n  {m}:")
        print(f"    metric used for verdict: {metric_used}  (range across 5 conditions = {rng:.4f})")
        print(f"    values: " + "  ".join(f"{c}={sub[sub['condition']==c][metric_used].iloc[0]:.4f}" for c in ALL_CONDITIONS))
        print(f"    -> {verdict}")
        rows.append({"model": m, "metric_used": metric_used, "range_across_conditions": rng, "verdict": verdict})

    print(f"\n{'='*78}\nEXCEPTIONS TO FLAG EXPLICITLY (not smoothed over)\n{'='*78}")
    gemma_positive = table[(table["model"] == "gemma") & (table["condition"] == "positive")]["abstention_rate_allrows_pct"].iloc[0]
    gemma_min = table[table["model"] == "gemma"]["abstention_rate_allrows_pct"].min()
    print(f"  Gemma: abstention rate is 0.0000% in original/health, but {gemma_positive:.4f}% under positive "
          f"({int(table[(table['model']=='gemma') & (table['condition']=='positive')]['n_allrows'].iloc[0] * gemma_positive/100)} "
          f"rows) -- a real, nonzero deviation from an otherwise-perfect 0%, even though the absolute rate")
    print(f"  remains negligible in practical terms.")
    deepseek_neg = table[(table["model"] == "deepseek") & (table["condition"] == "negative_minor")]["strict_valid_rate_allrows_pct"].iloc[0]
    deepseek_others = table[(table["model"] == "deepseek") & (table["condition"].isin(["health", "neutral", "positive"]))]["strict_valid_rate_allrows_pct"]
    print(f"  DeepSeek: strict-valid rate is {deepseek_neg:.4f}% under negative_minor vs "
          f"{deepseek_others.min():.4f}-{deepseek_others.max():.4f}% under health/neutral/positive (all effectively 0) "
          f"and {table[(table['model']=='deepseek') & (table['condition']=='original')]['strict_valid_rate_allrows_pct'].iloc[0]:.4f}% "
          f"under the original single-turn prompt -- negative_minor is the ONE condition (of 5) where DeepSeek's "
          f"compliance is not near-zero. See 04_deepseek_cross_context_diagnosis.py for whether this reflects a")
    print(f"  qualitative difference in behavior or is too small a sample (n~200/75,600) to draw a conclusion from.")

    verdict_df = pd.DataFrame(rows)
    verdict_df.to_csv(f"{OUT_DIR}/abstention_stability_verdict.csv", index=False)
    return verdict_df


def significance_full_dataset():
    """Persona-clustered test of each context's shift vs. original, full
    5,400-persona dataset (not pilot -- Step 3 is a full-dataset question).
    Uses is_abstention for qwen/ministral (Condition B only, the only
    denominator where abstention is structurally possible) and
    strict_is_valid (all rows) for deepseek. llama/gemma are tested on
    is_abstention (all rows) since their rates are near-constant zero and a
    test is still informative for the rare nonzero exceptions (e.g. gemma's
    positive-condition deviation)."""
    print(f"\n{'='*78}\nPERSONA-CLUSTERED SIGNIFICANCE, FULL 5,400-PERSONA DATASET, EACH CONTEXT VS ORIGINAL\n{'='*78}")
    metric_by_model = {
        "llama": ("is_abstention", "all"), "gemma": ("is_abstention", "all"),
        "qwen": ("is_abstention", "B_only"), "ministral": ("is_abstention", "B_only"),
        "deepseek": ("strict_is_valid", "all"),
    }
    rows = []
    for m in MODEL_ORDER:
        metric, scope = metric_by_model[m]
        orig = load_model_frame("original", m)
        for c in CONTEXTS:
            ctx = load_model_frame(c, m)
            merged_ind = merge_on_canonical_key(orig, ctx, extra_cols=("strict_is_valid", "is_abstention"))
            merged = merged_ind[merged_ind["_merge"] == "both"].drop(columns="_merge")
            if scope == "B_only":
                merged = merged[merged["response_condition"] == "B_optional"]
            col_ctx, col_orig = f"{metric}_ctx", f"{metric}_orig"
            try:
                gd = clustered_diff_context_minus_orig(merged, col_ctx, col_orig, "persona_id")
            except Exception as e:
                gd = dict(diff=np.nan, se_cluster=np.nan, p_value=np.nan, n_clusters=np.nan)
            print(f"  {m:10s} {c:16s} metric={metric} scope={scope:7s} shift={100*gd['diff']:+8.4f}pp  "
                  f"p={gd['p_value']:.2e}  n_clusters={gd['n_clusters']}")
            rows.append({"model": m, "context": c, "metric": metric, "scope": scope,
                         "shift_pp": 100 * gd["diff"], "p_value": gd["p_value"], "n_clusters": gd["n_clusters"]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/abstention_stability_significance.csv", index=False)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    table = build_rate_table()
    summarize_stability(table)
    significance_full_dataset()
    print(f"\n{'='*78}\nDONE. Outputs written to {OUT_DIR}/abstention_stability_*.csv\n{'='*78}")


if __name__ == "__main__":
    main()
