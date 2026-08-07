"""Close the multiple-comparison correction gap on the six analysis_taxonomy/
stages. Searched every CSV in analysis_taxonomy/output/ for an actual
p-value column (not a false-positive substring match like "top_pct" or
"top_orig") -- exactly two files carry one:

  analysis_taxonomy/output/ranking_robustness_pvalue_summary.csv  (32 rows, permutation_p)
  analysis_taxonomy/output/consensus_pattern_chisquare.csv         (8 rows, p_value)

Treated as ONE predefined family (40 p-values total), matching this
project's existing 05e precedent (one BH pass per predefined family, not
per file). Adds a p_bh / significant_bh_0.05 column to each file IN PLACE
-- raw p-values are never overwritten. The script is idempotent: on a rerun,
it replaces only its two previously derived BH columns.
"""
import os

import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{ROOT}/analysis_taxonomy/output"

FILES = [
    (f"{OUT_DIR}/ranking_robustness_pvalue_summary.csv", "permutation_p"),
    (f"{OUT_DIR}/consensus_pattern_chisquare.csv", "p_value"),
]


def main():
    frames = []
    for path, col in FILES:
        df = pd.read_csv(path)
        df = df.drop(columns=["p_bh", "significant_bh_0.05"], errors="ignore")
        frames.append((path, col, df))

    all_p = pd.concat([df[col].rename("p") for _, col, df in frames], ignore_index=True)
    print(f"Total p-values in family: {len(all_p)}", flush=True)
    print(f"Significant before correction (raw p<0.05): {(all_p < 0.05).sum()}/{len(all_p)}", flush=True)

    reject, p_bh, _, _ = multipletests(all_p.values, method="fdr_bh")
    print(f"Significant after BH correction (p_bh<0.05): {reject.sum()}/{len(all_p)}", flush=True)

    offset = 0
    for path, col, df in frames:
        n = len(df)
        df["p_bh"] = p_bh[offset:offset + n]
        df["significant_bh_0.05"] = reject[offset:offset + n]
        n_lost = ((df[col] < 0.05) & (~df["significant_bh_0.05"])).sum()
        df.to_csv(path, index=False)
        print(f"  {path}: {n} rows, {n_lost} lose significance after correction. Rewrote in place with p_bh added.", flush=True)


if __name__ == "__main__":
    main()
