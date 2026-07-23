"""Step 7: cross-model agreement -- Spearman correlation and weighted (quadratic)
Cohen's kappa between each pair of models' ratings on matched persona-topic
cells, strict-valid responses only.

Matching is restricted to Condition A (forced). Rationale: a rating is only
defined per persona x topic x CONDITION, and Condition B introduces a
selection effect (qwen/ministral abstain on the large majority of Condition-B
rows -- see step 6) that would confound "do two models agree" with "did both
models happen to answer this cell". Condition A gives a clean, fully
matched design: llama, gemma, qwen, and ministral are 100% strict-valid
under Condition A (37,800 = 5,400 personas x 7 topics cells each), so every
pairwise comparison among those four uses the full, identical n=37,800.

DeepSeek has only 63 strict-valid Condition-A rows (see steps 3/5/9), so any
pair involving deepseek is matched on at most 63 cells -- far too sparse for
a meaningful correlation/kappa estimate. Per the instruction for this step,
these are computed and reported, but kept OUT of the primary 4-model
matrix/heatmap and flagged explicitly rather than presented alongside the
other pairs without caveat.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from _common import load_master, MODEL_ORDER, TABLES_DIR

MAIN_MODELS = ["llama", "gemma", "qwen", "ministral"]
SPARSE_N_THRESHOLD = 100


def matched_ratings(df, model_a, model_b):
    a = df[df["model"] == model_a].set_index(["persona_id", "topic"])["rating_numeric"]
    b = df[df["model"] == model_b].set_index(["persona_id", "topic"])["rating_numeric"]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return joined["a"].values, joined["b"].values, len(joined)


def pair_stats(x, y, n):
    if n < 2:
        return dict(n=n, spearman_r=np.nan, spearman_p=np.nan, weighted_kappa=np.nan)
    rho, p = spearmanr(x, y)
    kappa = cohen_kappa_score(x.astype(int), y.astype(int), weights="quadratic")
    return dict(n=n, spearman_r=rho, spearman_p=p, weighted_kappa=kappa)


def main():
    df = load_master()
    a_valid = df[(df["response_condition"] == "A_forced") & df["strict_is_valid"] & df["rating_numeric"].notnull()]

    print("=" * 78)
    print("MATCHED-CELL SAMPLE SIZE CHECK (Condition A, strict-valid only) -- before")
    print("computing any correlation, per instruction")
    print("=" * 78)
    all_pairs = list(itertools.combinations(MODEL_ORDER, 2))
    n_by_pair = {}
    for m1, m2 in all_pairs:
        _, _, n = matched_ratings(a_valid, m1, m2)
        n_by_pair[(m1, m2)] = n
        involves_ds = "deepseek" in (m1, m2)
        flag = "  <-- SPARSE (involves deepseek, expected ~63 or fewer)" if involves_ds and n < SPARSE_N_THRESHOLD else ""
        print(f"  {m1:10s} x {m2:10s}  n_matched = {n:>6,}{flag}")

    print()
    print("=" * 78)
    print("PRIMARY RESULT: 4-model matrix (llama, gemma, qwen, ministral) -- deepseek excluded")
    print("from this matrix, reported separately below")
    print("=" * 78)
    spearman_mat = pd.DataFrame(index=MAIN_MODELS, columns=MAIN_MODELS, dtype=float)
    kappa_mat = pd.DataFrame(index=MAIN_MODELS, columns=MAIN_MODELS, dtype=float)
    n_mat = pd.DataFrame(index=MAIN_MODELS, columns=MAIN_MODELS, dtype=float)
    rows = []
    for m in MAIN_MODELS:
        spearman_mat.loc[m, m] = 1.0
        kappa_mat.loc[m, m] = 1.0
    for m1, m2 in itertools.combinations(MAIN_MODELS, 2):
        x, y, n = matched_ratings(a_valid, m1, m2)
        stats = pair_stats(x, y, n)
        spearman_mat.loc[m1, m2] = spearman_mat.loc[m2, m1] = stats["spearman_r"]
        kappa_mat.loc[m1, m2] = kappa_mat.loc[m2, m1] = stats["weighted_kappa"]
        n_mat.loc[m1, m2] = n_mat.loc[m2, m1] = n
        rows.append({"model_a": m1, "model_b": m2, **stats})

    pairs_table = pd.DataFrame(rows)
    pairs_table.to_csv(f"{TABLES_DIR}/cross_model_agreement_main4.csv", index=False)
    spearman_mat.to_csv(f"{TABLES_DIR}/cross_model_spearman_matrix.csv")
    kappa_mat.to_csv(f"{TABLES_DIR}/cross_model_kappa_matrix.csv")

    print(f"n_matched for every pair among these 4 models: {int(n_mat.loc[MAIN_MODELS[0], MAIN_MODELS[1]]):,} (full, identical)")
    print()
    print("Spearman correlation matrix:")
    print(spearman_mat.to_string(float_format=lambda x: f"{x:6.3f}"))
    print()
    print("Weighted (quadratic) Cohen's kappa matrix:")
    print(kappa_mat.to_string(float_format=lambda x: f"{x:6.3f}"))
    print()
    print("Pairwise detail:")
    print(pairs_table.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\nWrote tables/cross_model_agreement_main4.csv, cross_model_spearman_matrix.csv, cross_model_kappa_matrix.csv")

    print()
    print("=" * 78)
    print("DEEPSEEK PAIRS -- reported separately, NOT part of the primary matrix above")
    print("=" * 78)
    ds_rows = []
    for m in MAIN_MODELS:
        x, y, n = matched_ratings(a_valid, "deepseek", m)
        stats = pair_stats(x, y, n)
        sparse = n < SPARSE_N_THRESHOLD
        ds_rows.append({"model_a": "deepseek", "model_b": m, **stats, "flagged_too_sparse": sparse})
    ds_table = pd.DataFrame(ds_rows)
    ds_table.to_csv(f"{TABLES_DIR}/cross_model_agreement_deepseek_pairs.csv", index=False)
    print(ds_table.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print()
    print(f"CAVEAT: every deepseek pair is matched on n<=63 cells (deepseek's total strict-valid")
    print(f"Condition-A count), well below the {SPARSE_N_THRESHOLD}-cell threshold used elsewhere in this")
    print("pipeline to flag unreliable estimates (see step 5). These correlations/kappas describe")
    print("agreement on a tiny, non-representative slice (only 2 of 7 topics, 13 of 20 countries --")
    print("see step 5/9) and should not be plotted alongside the main 4-model matrix or interpreted")
    print("as comparable in precision. Full deepseek treatment is in step 9.")
    print(f"\nWrote tables/cross_model_agreement_deepseek_pairs.csv")


if __name__ == "__main__":
    main()
