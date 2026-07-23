"""Step 5e (Tier 2, Fix 10): Benjamini-Hochberg FDR correction --
analysis_plan.md Section 9.

Section 9 requires FDR correction within predefined test families, each
family defined a priori. Three families, exactly as specified:
  (a) the pooled model's coefficients (tables/hypothesis_model_pooled.csv)
      -- one family
  (b) each per-model OLS's coefficients (tables/hypothesis_model_{model}.csv)
      -- one family PER MODEL (5 families: llama, gemma, qwen, ministral,
      deepseek -- not pooled together)
  (c) the abstention logistic regression's coefficients
      (tables/abstention_model_qwen_ministral.csv) -- one family

BH is applied to BOTH p-value types already in each table (HC3/original and
persona-clustered) where present, producing adjusted-p columns ADDED to the
existing tables -- raw p-values are never overwritten. Reports how many
previously-significant terms (raw p<0.05) survive BH correction, per family.

Terms with a NaN p-value (e.g. deepseek's zero-variance-dropped factors,
the MixedLM "Group Var" row, degenerate fits) are excluded from the BH
correction for that family (BH cannot be computed on an undefined p-value)
and are left as NaN in the adjusted column too, not silently coerced to 0.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from _common import TABLES_DIR, MODEL_ORDER


def bh_adjust(df, pcol, family_label):
    valid = df[pcol].notna()
    n_valid = valid.sum()
    adj_col = f"{pcol}_bh_adj"
    df[adj_col] = np.nan
    if n_valid == 0:
        print(f"    {pcol}: no valid p-values in this family, skipped")
        return df
    _, p_adj, _, _ = multipletests(df.loc[valid, pcol].values, method="fdr_bh")
    df.loc[valid, adj_col] = p_adj
    n_raw_sig = (df.loc[valid, pcol] < 0.05).sum()
    n_bh_sig = (df.loc[valid, adj_col] < 0.05).sum()
    print(f"    {pcol:12s} (family={family_label}, n_terms={n_valid}): "
          f"raw p<0.05: {n_raw_sig}   survive BH q<0.05: {n_bh_sig}   "
          f"({n_raw_sig - n_bh_sig} lost to correction)")
    return df


def process_family(path, label, pcols):
    print(f"\n{'='*78}\nFAMILY: {label}\n{'='*78}")
    df = pd.read_csv(path)
    for pcol in pcols:
        if pcol not in df.columns:
            print(f"    {pcol}: column not present in this table, skipped")
            continue
        df = bh_adjust(df, pcol, label)
    df.to_csv(path, index=False)
    print(f"  Wrote {path} (BH-adjusted columns added)")
    return df


def main():
    summary_rows = []

    # (a) pooled model -- one family
    pooled_path = f"{TABLES_DIR}/hypothesis_model_pooled.csv"
    pooled_df = process_family(pooled_path, "pooled_model", ["p_hc3", "p_cluster", "p_cluster_modelpersona"])
    for pcol in ["p_hc3", "p_cluster"]:
        if f"{pcol}_bh_adj" in pooled_df.columns:
            valid = pooled_df[pcol].notna()
            summary_rows.append({"family": "pooled_model", "p_type": pcol, "n_terms": int(valid.sum()),
                                  "n_raw_sig": int((pooled_df.loc[valid, pcol] < 0.05).sum()),
                                  "n_bh_sig": int((pooled_df.loc[valid, f"{pcol}_bh_adj"] < 0.05).sum())})

    # (b) each per-model OLS -- one family per model
    for model in MODEL_ORDER:
        path = f"{TABLES_DIR}/hypothesis_model_{model}.csv"
        df = process_family(path, f"per_model_ols_{model}", ["p_hc3", "p_cluster", "p_mixedlm"])
        for pcol in ["p_hc3", "p_cluster"]:
            if f"{pcol}_bh_adj" in df.columns:
                valid = df[pcol].notna()
                summary_rows.append({"family": f"per_model_ols_{model}", "p_type": pcol, "n_terms": int(valid.sum()),
                                      "n_raw_sig": int((df.loc[valid, pcol] < 0.05).sum()),
                                      "n_bh_sig": int((df.loc[valid, f"{pcol}_bh_adj"] < 0.05).sum())})

    # (c) abstention logistic regression -- one family
    abst_path = f"{TABLES_DIR}/abstention_model_qwen_ministral.csv"
    abst_df = process_family(abst_path, "abstention_logit_qwen_ministral", ["p_original", "p_cluster"])
    for pcol in ["p_original", "p_cluster"]:
        if f"{pcol}_bh_adj" in abst_df.columns:
            valid = abst_df[pcol].notna()
            summary_rows.append({"family": "abstention_logit_qwen_ministral", "p_type": pcol, "n_terms": int(valid.sum()),
                                  "n_raw_sig": int((abst_df.loc[valid, pcol] < 0.05).sum()),
                                  "n_bh_sig": int((abst_df.loc[valid, f"{pcol}_bh_adj"] < 0.05).sum())})

    print(f"\n{'='*78}\nSUMMARY: raw-significant terms surviving BH correction, per family\n{'='*78}")
    summary = pd.DataFrame(summary_rows)
    summary["n_lost"] = summary["n_raw_sig"] - summary["n_bh_sig"]
    summary["pct_survive"] = (100 * summary["n_bh_sig"] / summary["n_raw_sig"]).round(1)
    print(summary.to_string(index=False))
    summary.to_csv(f"{TABLES_DIR}/bh_correction_summary.csv", index=False)
    print(f"\nWrote {TABLES_DIR}/bh_correction_summary.csv")


if __name__ == "__main__":
    main()
