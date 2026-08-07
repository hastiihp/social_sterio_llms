"""Stage 4: Context Sensitivity Index.

A 4-component index, one row per model, laying out four already-computed
sensitivity signals side by side -- NOT collapsed into one weighted score
(there is no existing project precedent for how such components should be
weighted against each other, so this script does not invent one). No new
statistical claims: every number is read directly from an existing,
already-audited output file, or is a trivial arithmetic derivation (mean
of already-computed values) over already-verified numbers. Full-scale
throughout.

Components:
  1. abstention_range_pct -- range (max-min) of Condition-B abstention rate
     across the 5 prompt types. Read directly from Stage 1's own output
     (analysis_taxonomy/output/model_taxonomy.csv's abstention_range_pct
     column), not recomputed -- Stage 1 already sourced and verified this
     from analysis_context/output/abstention_stability_rate_table.csv.
  2. avg_abs_rating_shift -- mean of |clustered_mean_shift_ctx_minus_orig|
     across the 4 conversational framings (health/neutral/positive/
     negative_minor), each read directly from
     analysis_context/output/{context}_vs_original_by_condition_full5400.csv,
     condition=="A_forced" row (analysis_context/01_compare_context_vs_original.py,
     full-scale). Absolute value taken because shift direction differs by
     framing/model (e.g. ministral/neutral is a small positive shift while
     every other ministral shift is negative) -- averaging signed shifts
     would let a model with large shifts in opposite directions look
     artificially insensitive. DeepSeek has zero both-valid matched rows
     in every one of these four files (already established throughout
     this project) so this component is NA for it, not zero.

     CORRECTED (post-Stage-7 review): originally sourced from
     {context}_vs_original_summary_full5400.csv, which this script's first
     version assumed was Condition-A-scoped because the project convention
     treats Condition A as primary everywhere else -- it is not. That file
     pools Condition A and B together (confirmed at the code level:
     analysis_context/01's section_matched_cells() never filters by
     response_condition), which understates every shift by roughly half
     (e.g. llama/health: -0.084 pooled vs. -0.158 Condition-A-only). Caught
     when Stage 7's unified mixed model, fit independently on raw Condition-A
     data, produced per-model prompt_type coefficients that didn't match
     this column -- traced to source, confirmed by hand-recomputing the raw
     Condition-A-only mean difference directly from results/results_*.csv
     (matches the by_condition file and the unified model to 4+ decimal
     places), and fixed here. See UNIFIED_MODEL_SUMMARY.md's discrepancy
     section for the full investigation.
  3. avg_ranking_rho_country / avg_ranking_rho_profession -- mean Spearman
     rho (original ranking vs. context ranking) across the same 4
     framings, separately for country and profession, read directly from
     Stage 2's own companion output
     (analysis_taxonomy/output/ranking_robustness_pvalue_summary.csv,
     itself sourced from analysis_context/output/*_ranking_robustness_pvalues_full5400.csv).
     Lower rho = ranking reshuffles more under a different framing = more
     context-sensitive; this is the inverse-direction signal from the
     other three components (higher = more sensitive there), stated
     explicitly rather than left for the reader to notice.
  4. dominant_factor_stable -- read directly from Stage 1's own output
     (model_taxonomy.csv's dominant_factor_stable column).

DeepSeek is NA on components 2-4 throughout (excluded from the underlying
analyses for the same reasons established in Stages 1-2: zero valid
matched rows for component 2, excluded from ranking-robustness and H1
fits for components 3-4). Component 1 is reported as 0.0 for DeepSeek
per Stage 1, with the same caveat Stage 1 already attached: this reflects
near-total non-compliance, not confident non-abstention.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{ROOT}/analysis_taxonomy/output"

CONTEXTS = ["health", "neutral", "positive", "negative_minor"]
MODELS = ["llama", "gemma", "qwen", "ministral", "deepseek"]
RANKED_MODELS = ["llama", "gemma", "qwen", "ministral"]


def component_1_abstention_range():
    src = f"{OUT_DIR}/model_taxonomy.csv"
    df = pd.read_csv(src).set_index("model")
    return df["abstention_range_pct"].to_dict(), src


def component_2_avg_abs_rating_shift():
    # Condition-A-only row of the by_condition file, NOT the plain "summary" file --
    # the summary file pools Condition A+B, which understates every shift by roughly
    # half. See the module docstring's CORRECTED note for the full investigation.
    frames = []
    for c in CONTEXTS:
        src = f"{ROOT}/analysis_context/output/{c}_vs_original_by_condition_full5400.csv"
        df = pd.read_csv(src)
        condA = df[df["condition"] == "A_forced"]
        frames.append(condA[["model", "clustered_mean_shift_ctx_minus_orig"]].assign(context=c))
    all_df = pd.concat(frames, ignore_index=True)
    all_df["abs_shift"] = all_df["clustered_mean_shift_ctx_minus_orig"].abs()
    out = all_df.groupby("model")["abs_shift"].mean().to_dict()
    return out, [f"{ROOT}/analysis_context/output/{c}_vs_original_by_condition_full5400.csv (condition=='A_forced')" for c in CONTEXTS]


def component_3_avg_ranking_rho():
    src = f"{OUT_DIR}/ranking_robustness_pvalue_summary.csv"
    df = pd.read_csv(src)
    out = {}
    for factor in ["country", "profession"]:
        sub = df[df["factor"] == factor]
        assert set(sub["context"].unique()) == set(CONTEXTS), (
            f"{factor}: expected contexts {CONTEXTS}, found {sorted(sub['context'].unique())}")
        out[factor] = sub.groupby("model")["spearman_r"].mean().to_dict()
    return out, src


def component_4_dominant_factor_stable():
    src = f"{OUT_DIR}/model_taxonomy.csv"
    df = pd.read_csv(src).set_index("model")
    return df["dominant_factor_stable"].to_dict(), src


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    abst, abst_src = component_1_abstention_range()
    shift, shift_src = component_2_avg_abs_rating_shift()
    rho, rho_src = component_3_avg_ranking_rho()
    stable, stable_src = component_4_dominant_factor_stable()

    print("Sources used:")
    print(f"  1. abstention range:       {abst_src}")
    print(f"  2. avg |rating shift|:     {shift_src}")
    print(f"  3. avg ranking rho:        {rho_src}")
    print(f"  4. dominant factor stable: {stable_src}")
    print()

    rows = []
    for m in MODELS:
        rows.append({
            "model": m,
            "abstention_range_pct": abst.get(m),
            "avg_abs_rating_shift": shift.get(m),
            "avg_ranking_rho_country": rho["country"].get(m),
            "avg_ranking_rho_profession": rho["profession"].get(m),
            "dominant_factor_stable": stable.get(m),
        })
    out_df = pd.DataFrame(rows)

    out_path = f"{OUT_DIR}/context_sensitivity_index.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
