"""Stage 1: Model Behavioral Taxonomy.

Synthesizes already-computed, already-audited findings into one master
table, one row per model. Introduces NO new statistical claims -- every
cell is either read directly from an existing output file, or a trivial
arithmetic derivation (max-min range, count of distinct values, mean of
already-computed correlations) over already-verified numbers. Where a
requested number did not already exist in the form asked for, that gap is
resolved by reading the most granular existing audited output rather than
by fitting anything new (see the compliance-rate note below).

All context/pilot-scope choices below are FULL-SCALE (5,400 personas, 20
countries, 30 professions) throughout -- no 180-persona pilot subset is
used anywhere in this script.

Sources (each read directly, not from a prior summary):
  1+2. abstention rate (Condition-B denominator) + range across the 5
       prompt types:
       analysis_context/output/abstention_stability_rate_table.csv
       (analysis_context/03_abstention_stability_across_conditions.py --
       this script always ran at full scale; no pilot/full distinction
       applies to it). Column used: abstention_rate_condB_pct (the
       standard "abstention rate" denominator used throughout this
       project -- NOT abstention_rate_allrows_pct, which is a different,
       non-interchangeable number per AUDIT_HISTORY.md Round 1 finding
       H2). Range is max-min of the 5 values, computed here (arithmetic,
       not a new statistical claim).
  3+4. dominant demographic factor per prompt type + stability:
       analysis_context/output/dominant_factor_by_model_full5400.csv
       (analysis_context/05_variance_ranking_all_prompt_types.py, full
       5,400-persona scope). DeepSeek is absent from this file by design
       -- excluded throughout analysis_context/05 for having too few
       valid Condition-A rows to fit the model meaningfully in any
       context (see that script's own docstring). Recorded as NA with an
       explicit note, not silently blank.
  5.   cross-model rating agreement (original prompt; the main study has
       no pilot/full distinction -- always full 5,400-persona scope):
       tables/cross_model_spearman_matrix.csv (llama/gemma/qwen/ministral
       4x4 matrix) and tables/cross_model_agreement_deepseek_pairs.csv
       (deepseek's 4 pairwise correlations, each explicitly flagged
       flagged_too_sparse=True at n=63 -- reported here with that caveat
       attached, not presented as equivalent in reliability to the other
       four models' agreement numbers).
  6.   strict format compliance, original prompt, Condition A only:
       tables/table1_compliance.csv's pct_strict_valid is POOLED across
       Condition A+B, not Condition-A-only as requested -- confirmed by
       reading analysis/03_compliance_table.py's source, which filters
       only by model, not by condition. For llama/gemma/qwen/ministral,
       pooled = 100.0%, which makes the Condition-A-only rate 100.0% too
       by logical necessity (if 100% of all rows are valid, then 100% of
       any subset, including Condition A alone, must also be valid -- no
       further computation needed). For deepseek, pooled and Condition-A
       differ substantially (0.0833% vs 0.1667%), so the Condition-A-only
       number is read directly from tables/deepseek_compliance_by_condition.csv's
       A_forced row's "none" column (analysis/09_deepseek_report.py),
       which already reports it, rather than assumed equal to the pooled
       value.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{ROOT}/analysis_taxonomy/output"

MODELS = ["llama", "gemma", "qwen", "ministral", "deepseek"]
PROMPT_TYPES = ["original", "health", "neutral", "positive", "negative_minor"]


def build_abstention_columns():
    src = f"{ROOT}/analysis_context/output/abstention_stability_rate_table.csv"
    df = pd.read_csv(src)
    out = {}
    for m in MODELS:
        sub = df[df["model"] == m].set_index("condition")
        rates = [sub.loc[pt, "abstention_rate_condB_pct"] for pt in PROMPT_TYPES]
        out[m] = {
            "abstention_pct_original": rates[0],
            "abstention_pct_health": rates[1],
            "abstention_pct_neutral": rates[2],
            "abstention_pct_positive": rates[3],
            "abstention_pct_negative_minor": rates[4],
            "abstention_range_pct": max(rates) - min(rates),
        }
    return out, src


def build_dominant_factor_columns():
    src = f"{ROOT}/analysis_context/output/dominant_factor_by_model_full5400.csv"
    df = pd.read_csv(src).set_index("model")
    out = {}
    for m in MODELS:
        if m not in df.index:
            out[m] = {
                "dominant_factor_original": "NA", "dominant_factor_health": "NA",
                "dominant_factor_neutral": "NA", "dominant_factor_positive": "NA",
                "dominant_factor_negative_minor": "NA",
                "dominant_factor_n_distinct": "NA",
                "dominant_factor_stable": "NA",
                "dominant_factor_note": ("excluded from analysis_context/05's H1 fit throughout "
                                          "-- too few valid Condition-A rows to fit the model "
                                          "meaningfully in any prompt type"),
            }
            continue
        row = df.loc[m]
        vals = [row[f"dominant_{pt}"] for pt in PROMPT_TYPES]
        distinct = sorted(set(vals))
        divergent = [pt for pt, v in zip(PROMPT_TYPES, vals) if v != vals[0]]
        out[m] = {
            "dominant_factor_original": vals[0], "dominant_factor_health": vals[1],
            "dominant_factor_neutral": vals[2], "dominant_factor_positive": vals[3],
            "dominant_factor_negative_minor": vals[4],
            "dominant_factor_n_distinct": len(distinct),
            "dominant_factor_stable": len(distinct) == 1,
            "dominant_factor_note": row["verdict"],
        }
    return out, src


def build_cross_model_agreement_columns():
    src_main = f"{ROOT}/tables/cross_model_spearman_matrix.csv"
    src_ds = f"{ROOT}/tables/cross_model_agreement_deepseek_pairs.csv"
    mat = pd.read_csv(src_main, index_col=0)
    ds = pd.read_csv(src_ds)
    out = {}
    for m in ["llama", "gemma", "qwen", "ministral"]:
        others = [c for c in mat.columns if c != m]
        out[m] = {
            "avg_cross_model_spearman_r": mat.loc[m, others].astype(float).mean(),
            "cross_model_agreement_note": "average over the other 3 models' rho, main 4-model matrix",
        }
    ds_avg = ds["spearman_r"].mean()
    out["deepseek"] = {
        "avg_cross_model_spearman_r": ds_avg,
        "cross_model_agreement_note": ("UNRELIABLE -- average of 4 pairwise rho values each computed "
                                        "on only n=63 both-valid rows, every pair flagged "
                                        "flagged_too_sparse=True in the source file; not comparable in "
                                        "reliability to the other four models' matrix-based averages"),
    }
    return out, (src_main, src_ds)


def build_compliance_column():
    src_pooled = f"{ROOT}/tables/table1_compliance.csv"
    src_ds_cond = f"{ROOT}/tables/deepseek_compliance_by_condition.csv"
    pooled = pd.read_csv(src_pooled).set_index("model")
    out = {}
    for m in ["llama", "gemma", "qwen", "ministral"]:
        pooled_rate = pooled.loc[m, "pct_strict_valid"]
        assert pooled_rate == 100.0, (
            f"{m}: expected pooled pct_strict_valid == 100.0 to justify the logical-necessity "
            f"shortcut to Condition-A-only == 100.0, got {pooled_rate}")
        out[m] = {"strict_compliance_condA_pct": 100.0}
    ds_cond = pd.read_csv(src_ds_cond).set_index("scope")
    ds_a_forced_rate = ds_cond.loc["A_forced", "none"]
    out["deepseek"] = {"strict_compliance_condA_pct": ds_a_forced_rate}
    return out, (src_pooled, src_ds_cond)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    abst, abst_src = build_abstention_columns()
    dom, dom_src = build_dominant_factor_columns()
    agree, agree_src = build_cross_model_agreement_columns()
    comp, comp_src = build_compliance_column()

    print("Sources used:")
    print(f"  abstention:        {abst_src}")
    print(f"  dominant factor:   {dom_src}")
    print(f"  cross-model agree: {agree_src}")
    print(f"  compliance:        {comp_src}")
    print()

    rows = []
    for m in MODELS:
        row = {"model": m}
        row.update(abst[m])
        row.update(dom[m])
        row.update(agree[m])
        row.update(comp[m])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = f"{OUT_DIR}/model_taxonomy.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
