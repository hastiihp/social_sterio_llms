"""Exploratory: health-vs-original comparison split by response_condition.

The comparison in 01_compare_health_vs_original.py pooled Condition A and
Condition B into one set of matched cells per model. This redoes it split
by condition, matching the discipline used everywhere else in analysis/
(H1's per-model models, H3's paired comparison, and the topic-specific
models all required this same A/B separation -- see analysis_plan.md).

Fix H3 (labeling): Condition A results are the PRIMARY rating-shift evidence
(full pilot sample -- no abstention/selection mechanism exists under
Condition A at all, so every row is available for comparison; not "always
answers", see Fix H7 note below). Condition B results are labeled
"descriptive only, selected sample" throughout this script's output and its
saved CSV (a `sample_type` column makes this explicit per row), because
Condition B's answered subset is small and self-selected -- for qwen this is
80/1,260 pairs (6.35%), for ministral 209/1,260 (16.59%). No pooled A+B
number is computed or reported anywhere in this script's output; only the
per-condition numbers, so nothing here can be mistaken for a pooled result.

Fix H7 (wording): "Condition A always answers" is not quite accurate --
Condition A has no valid "NA" option in the prompt, so abstention
specifically is impossible, but malformed or refused output remains
entirely possible (DeepSeek is the proof this distinction matters: it still
produces non-compliant output under Condition A despite no NA option
existing at all -- see 03_deepseek_health_diagnosis.py). This script says
"no valid NA option" throughout, not "always answers".

For qwen/ministral, Condition A is the clean test (no selection mechanism)
of whether the negative rating shift found in 01_compare_health_vs_original.py's
pooled numbers reflects an actual opinion shift, or is partly/wholly an
artifact of WHICH cells happen to have valid Condition-B data changing
between versions (Condition B's answered subset differs in composition and
size between the original and health versions -- exactly the selection
problem Fix 3 addressed for analysis/08_variance_ranking.py).

Fix H6: merges on the full canonical key with validate="one_to_one" and
prints pre/post row counts plus any left-only/right-only rows, matching
01_compare_health_vs_original.py's fix.

Independent of analysis/ (same self-contained design as 01_*.py): reads
results/*.csv and results_health/health_full_results_*.csv directly,
filtered to the 180-persona pilot subset (4 countries x 5 professions x 3
genders x 3 ages). Does not touch analysis/ or master_results.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = "/Users/hastihosseinpour/Desktop/social_sterio_llms"
PILOT_COUNTRIES = ["Germany", "Brazil", "Nigeria", "South Korea"]
PILOT_PROFESSIONS = ["lawyer", "registered nurse", "truck driver", "farmer", "computer programmer"]
MODEL_ORDER = ["llama", "gemma", "qwen", "ministral", "deepseek"]
CONDITIONS = ["A_forced", "B_optional"]
SAMPLE_TYPE = {
    "A_forced": "primary_full_sample_no_valid_NA_option",
    "B_optional": "descriptive_only_selected_sample",
}


def load_safe(path, model):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    df.insert(0, "model", model)
    df["rating_numeric"] = pd.to_numeric(df["strict_parsed_rating"], errors="coerce")
    return df


def filter_pilot(df):
    return df[df["country"].isin(PILOT_COUNTRIES) & df["profession"].isin(PILOT_PROFESSIONS)].copy()


def cast_object(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns and isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype("object")
    return df


def clustered_diff_health_minus_orig(data, health_col, orig_col, cluster_col):
    """Row-wise (health - orig) difference, persona-clustered one-sample test against 0.
    Unambiguous by construction (no categorical reference-level dependence) -- see
    01_compare_health_vs_original.py's Fix H1 docstring for why this replaced the
    previous melt + categorical-regression pattern."""
    d = data.copy()
    d["_diff"] = d[health_col] - d[orig_col]
    n_clusters = d[cluster_col].nunique()
    if n_clusters < 2:
        return dict(diff=np.nan, se_cluster=np.nan, p_value=np.nan, n_clusters=n_clusters)
    d = cast_object(d, [cluster_col])
    res = smf.ols("_diff ~ 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster_col]})
    return dict(diff=float(res.params["Intercept"]), se_cluster=float(res.bse["Intercept"]),
                p_value=float(res.pvalues["Intercept"]), n_clusters=n_clusters)


def main():
    print("=" * 78)
    print("Load + filter to pilot subset (same 180 personas as 01_compare_health_vs_original.py)")
    print("=" * 78)
    orig_frames, health_frames = [], []
    for m in MODEL_ORDER:
        orig_frames.append(filter_pilot(load_safe(f"{ROOT}/results/full_results_{m}.csv", m)))
        health_frames.append(filter_pilot(load_safe(f"{ROOT}/results_health/health_full_results_{m}.csv", m)))
    orig_df = pd.concat(orig_frames, ignore_index=True)
    health_df = pd.concat(health_frames, ignore_index=True)

    print(f"\n{'='*78}\nMerge on the full canonical key (Fix H6)\n{'='*78}")
    key = ["model", "persona_id", "country", "profession", "gender", "age", "topic", "response_condition"]
    print(f"  pre-merge: orig_df={len(orig_df):,} rows, health_df={len(health_df):,} rows")
    merged_ind = orig_df[key + ["strict_is_valid", "is_abstention", "rating_numeric"]].merge(
        health_df[key + ["strict_is_valid", "is_abstention", "rating_numeric"]],
        on=key, suffixes=("_orig", "_health"), how="outer", validate="one_to_one", indicator=True)
    merge_counts = merged_ind["_merge"].value_counts()
    print(f"  post-merge indicator counts: {merge_counts.to_dict()}")
    left_only = merged_ind[merged_ind["_merge"] == "left_only"]
    right_only = merged_ind[merged_ind["_merge"] == "right_only"]
    if len(left_only) or len(right_only):
        print(f"  WARNING: {len(left_only)} rows only in original, {len(right_only)} rows only in health.")
    else:
        print(f"  PASS: 0 unmatched rows on either side.")
    merged = merged_ind[merged_ind["_merge"] == "both"].drop(columns="_merge")
    print(f"  Total matched cells (both conditions, full canonical key): {len(merged):,}")

    summary_rows = []
    for m in MODEL_ORDER:
        print(f"\n{'='*78}\nMODEL: {m}\n{'='*78}")
        for cond in CONDITIONS:
            sub = merged[(merged["model"] == m) & (merged["response_condition"] == cond)]
            n_cells = len(sub)
            both_valid = sub[sub["strict_is_valid_orig"] & sub["strict_is_valid_health"] &
                              sub["rating_numeric_orig"].notnull() & sub["rating_numeric_health"].notnull()]
            n_both_valid = len(both_valid)
            pct_valid = 100 * n_both_valid / n_cells if n_cells else float("nan")

            row = {"model": m, "condition": cond, "sample_type": SAMPLE_TYPE[cond],
                   "n_matched_cells": n_cells, "n_both_valid": n_both_valid, "pct_both_valid": pct_valid}
            label = "PRIMARY" if cond == "A_forced" else "DESCRIPTIVE ONLY, SELECTED SAMPLE"
            print(f"\n  --- {cond}  [{label}] ---")
            print(f"  n matched cells={n_cells:,}   n both strict-valid numeric={n_both_valid:,} "
                  f"({pct_valid:.2f}% of matched cells)" if n_cells else "  n=0")
            if cond == "B_optional" and n_cells and pct_valid < 50:
                print(f"  ** Only {pct_valid:.1f}% of Condition-B cells are valid in both framings -- this is a small,")
                print(f"     self-selected subset, not a random sample of personas. Descriptive only; do not treat")
                print(f"     as primary evidence of a rating shift on its own. **")

            if n_both_valid >= 2:
                diff = both_valid["rating_numeric_orig"] - both_valid["rating_numeric_health"]
                exact_agree = (diff == 0).mean()
                mad = diff.abs().mean()
                rho, rho_p = stats.spearmanr(both_valid["rating_numeric_orig"], both_valid["rating_numeric_health"])

                gd = clustered_diff_health_minus_orig(both_valid, "rating_numeric_health", "rating_numeric_orig", "persona_id")

                print(f"  exact agreement rate:        {exact_agree:.4f}")
                print(f"  mean absolute difference:    {mad:.4f}")
                print(f"  Spearman r (orig vs health): {rho:.4f}  (p={rho_p:.2e})")
                print(f"  persona-clustered mean shift (health-orig): {gd['diff']:.4f}  "
                      f"se={gd['se_cluster']:.4f}  p={gd['p_value']:.2e}  (n_persona_clusters={gd['n_clusters']:,})")
                row.update(exact_agreement_rate=exact_agree, mean_abs_diff=mad, spearman_r=rho, spearman_p=rho_p,
                           clustered_mean_shift_health_minus_orig=gd["diff"], clustered_shift_se=gd["se_cluster"],
                           clustered_shift_p=gd["p_value"], n_persona_clusters=gd["n_clusters"])
            else:
                print("  (fewer than 2 both-valid-numeric cells -- comparison skipped)")
                row.update(exact_agreement_rate=np.nan, mean_abs_diff=np.nan, spearman_r=np.nan, spearman_p=np.nan,
                           clustered_mean_shift_health_minus_orig=np.nan, clustered_shift_se=np.nan,
                           clustered_shift_p=np.nan, n_persona_clusters=np.nan)
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    import os
    os.makedirs(f"{ROOT}/analysis_health/output", exist_ok=True)
    out_path = f"{ROOT}/analysis_health/output/health_vs_original_by_condition.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  (sample_type column marks each row PRIMARY vs DESCRIPTIVE-ONLY explicitly)")

    print(f"\n{'='*78}\nSUMMARY TABLE\n{'='*78}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n{'='*78}\nKEY QUESTION: does the negative rating shift hold within Condition A alone")
    print(f"for qwen/ministral (no selection effects possible -- Condition A has no valid NA option,")
    print(f"so no abstention-driven selection mechanism exists there)?\n{'='*78}")
    for m in ["qwen", "ministral"]:
        a_row = summary[(summary["model"] == m) & (summary["condition"] == "A_forced")].iloc[0]
        b_row = summary[(summary["model"] == m) & (summary["condition"] == "B_optional")].iloc[0]
        a_shift, a_p = a_row["clustered_mean_shift_health_minus_orig"], a_row["clustered_shift_p"]
        b_shift, b_p = b_row["clustered_mean_shift_health_minus_orig"], b_row["clustered_shift_p"]
        holds = (a_p < 0.05) and (np.sign(a_shift) == np.sign(b_shift) if pd.notna(b_shift) else True)
        print(f"\n  {m}:")
        print(f"    Condition A [PRIMARY] (n_both_valid={int(a_row['n_both_valid'])}, "
              f"{a_row['pct_both_valid']:.1f}% of matched cells): shift={a_shift:.4f}, p={a_p:.2e}")
        print(f"    Condition B [DESCRIPTIVE ONLY, SELECTED SAMPLE] (n_both_valid={int(b_row['n_both_valid'])}, "
              f"{b_row['pct_both_valid']:.1f}% of matched cells): shift={b_shift:.4f}, p={b_p:.2e}")
        if holds:
            print(f"    -> The shift HOLDS within Condition A alone (significant, same direction as the")
            print(f"       descriptive Condition-B number). Condition A has no valid-NA/selection mechanism,")
            print(f"       so this is evidence of an actual rating change, not purely a Condition-B selection")
            print(f"       artifact -- but it is the CONDITION-A number that supports this, not the pooled or")
            print(f"       Condition-B number, which remain descriptive only.")
        else:
            print(f"    -> The shift does NOT clearly hold within Condition A alone -- the descriptive")
            print(f"       Condition-B number may be partly or wholly driven by which cells have valid")
            print(f"       Condition-B data in each version, not by opinions actually shifting. Do not treat")
            print(f"       the Condition-B/pooled result as evidence of a rating shift for this model.")

    print(f"\n{'='*78}\nllama/gemma: Condition A is their FULL sample (100% valid in both framings, no")
    print(f"abstention in either version under either condition) -- their Condition-A numbers above ARE")
    print(f"the primary, clean comparison. Their Condition-B numbers are still labeled descriptive-only")
    print(f"for consistency, even though both happen to be 100% valid here too.\n{'='*78}")


if __name__ == "__main__":
    main()
