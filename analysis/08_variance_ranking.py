"""Step 8: partial R^2 / variance-explained ranking per factor, per model (H1).
Revised per external audit (Fixes 2 & 3).

Fix 3 (primary scope correction): analysis_plan.md Section 4/H1 defines this
under FORCED RATING ONLY (Condition A). The original version of this script
pooled Condition A and B together, which is not what H1 was pre-specified to
test -- for qwen/ministral in particular, Condition B mixes in a small,
self-selected "answered" subset (see step 4/6) that isn't a random sample of
personas. The Condition-A-only fit below is now the PRIMARY, pre-specified
H1 test. The original pooled (A+B) version is kept and reported too, but
relabeled EXPLORATORY -- not the actual H1 test -- so nothing computed
previously is silently discarded, only correctly re-labeled.

Fix 2 (clustered significance test): partial R^2 point estimates themselves
are UNCHANGED by clustering -- R^2 is a function of residual sums of squares
from the fitted coefficients, and clustering only changes the *covariance*/
inference, not the point estimates. What clustering DOES change is whether a
factor's contribution is judged "significant". For each factor, this script
now reports a classical (non-clustered, iid-assumption) nested F-test
alongside a persona-clustered joint Wald test (same full model, refit with
cov_type='cluster', cov_kwds={'groups': persona_id}, then
wald_test_terms() for the per-factor joint hypothesis that its dummy
coefficients are all zero) -- side by side, so the correction's effect on
the SIGNIFICANCE verdict is visible without touching the R^2 numbers.

DeepSeek (n=63 valid rows, all Condition A) is fit and its partial R^2
values are reported, but flagged explicitly as unreliable/exploratory
throughout, consistent with steps 6/7/9.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from _common import load_master, cast_formula_dtypes, MODEL_ORDER, TABLES_DIR

RANKED_FACTORS = ["topic", "profession", "country", "gender", "age"]
FULL_MODEL_FACTORS_CONDA = RANKED_FACTORS  # Condition A only: no condition term at all (Fix 3)
FULL_MODEL_FACTORS_POOLED = RANKED_FACTORS + ["response_condition"]  # original, now exploratory


def valid_subset(df, model, condition=None):
    sub = df[(df["model"] == model) & df["strict_is_valid"] & df["rating_numeric"].notnull()]
    if condition is not None:
        sub = sub[sub["response_condition"] == condition]
    return sub


def build_formula(factors):
    return "rating_numeric ~ " + " + ".join(f"C({f})" for f in factors)


def fit_variance_ranking(df, model, condition, full_model_factors, scope_label):
    sub = valid_subset(df, model, condition=condition)
    n = len(sub)

    dropped_zero_var = []
    factors = []
    for f in full_model_factors:
        if sub[f].nunique() <= 1:
            dropped_zero_var.append(f)
        else:
            factors.append(f)

    print(f"\n{'='*78}\nModel: {model}   scope={scope_label}   (n = {n:,} strict-valid rows)")
    if dropped_zero_var:
        print(f"  WARNING: dropped factor(s) with zero variance in this subset: {dropped_zero_var}")
        for f in dropped_zero_var:
            if f in RANKED_FACTORS:
                print(f"  -> '{f}' cannot be ranked for this model: no variance to explain.")

    full_formula = build_formula(factors)
    try:
        full_res = smf.ols(full_formula, data=sub).fit()
    except Exception as e:
        print(f"  FIT FAILED: {type(e).__name__}: {e}")
        print("  Cannot compute variance ranking for this model -- reporting failure, not forcing a fit.")
        return None

    n_params_actual = len(full_res.params)
    df_resid = full_res.df_resid
    if df_resid <= 0:
        print(f"  FIT INVALID: df_resid = {df_resid:.0f} (n_params={n_params_actual} >= n={n}).")
        print("  More parameters than usable degrees of freedom -- cannot compute a meaningful fit.")
        return None

    print(f"  full formula: {full_formula}")
    print(f"  full model R-squared: {full_res.rsquared:.4f}   df_resid: {df_resid:.0f}   n_params: {n_params_actual}")
    if n < 3 * n_params_actual:
        print(f"  WARNING: low ratio of observations to parameters (n={n}, params={n_params_actual}).")

    # Fix 2: cluster-robust refit + joint per-factor Wald test, alongside the classical nested F-test.
    cluster_res = None
    cluster_wald = None
    try:
        cluster_res = smf.ols(full_formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})
        cluster_wald = cluster_res.wald_test_terms(skip_single=False, scalar=True)
    except Exception as e:
        print(f"  Cluster-robust joint test FAILED: {type(e).__name__}: {e}")

    sse_full = float((full_res.resid ** 2).sum())
    results = []
    for f in RANKED_FACTORS:
        if f not in factors:
            results.append({"model": model, "scope": scope_label, "factor": f, "partial_r2": np.nan,
                             "f_stat_naive": np.nan, "p_naive": np.nan,
                             "chi2_cluster": np.nan, "p_cluster": np.nan, "note": "zero variance -- not estimable"})
            continue
        reduced_factors = [x for x in factors if x != f]
        reduced_formula = build_formula(reduced_factors) if reduced_factors else "rating_numeric ~ 1"
        try:
            reduced_res = smf.ols(reduced_formula, data=sub).fit()
        except Exception as e:
            results.append({"model": model, "scope": scope_label, "factor": f, "partial_r2": np.nan,
                             "f_stat_naive": np.nan, "p_naive": np.nan,
                             "chi2_cluster": np.nan, "p_cluster": np.nan, "note": f"reduced-model fit failed: {e}"})
            continue
        sse_reduced = float((reduced_res.resid ** 2).sum())
        partial_r2 = (sse_reduced - sse_full) / sse_reduced if sse_reduced > 0 else np.nan

        # classical (non-clustered, iid-assumption) nested F-test -- the "naive" baseline
        df_diff = reduced_res.df_resid - full_res.df_resid
        f_stat = ((sse_reduced - sse_full) / df_diff) / (sse_full / full_res.df_resid) if df_diff > 0 else np.nan
        p_naive = 1 - stats.f.cdf(f_stat, df_diff, full_res.df_resid) if np.isfinite(f_stat) else np.nan

        # cluster-robust joint Wald test for this factor's parameter block
        chi2_cluster, p_cluster = np.nan, np.nan
        if cluster_wald is not None:
            term_key = f"C({f})"
            if term_key in cluster_wald.table.index:
                row = cluster_wald.table.loc[term_key]
                chi2_cluster, p_cluster = float(row["statistic"]), float(row["pvalue"])

        results.append({"model": model, "scope": scope_label, "factor": f, "partial_r2": partial_r2,
                         "f_stat_naive": f_stat, "p_naive": p_naive,
                         "chi2_cluster": chi2_cluster, "p_cluster": p_cluster, "note": ""})

    result_df = pd.DataFrame(results)
    flips = result_df.dropna(subset=["p_naive", "p_cluster"])
    flips = flips[(flips["p_naive"] < 0.05) != (flips["p_cluster"] < 0.05)]
    if len(flips):
        print(f"  Terms where significance (alpha=0.05) FLIPS between naive F-test and cluster-robust Wald test: {len(flips)}")
        print(flips[["factor", "partial_r2", "f_stat_naive", "p_naive", "chi2_cluster", "p_cluster"]].to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))
    else:
        print(f"  No significance flips between naive F-test and cluster-robust Wald test for this model/scope.")

    return result_df


def run_scope(df, condition, full_model_factors, scope_label):
    all_results = []
    for model in MODEL_ORDER:
        res = fit_variance_ranking(df, model, condition, full_model_factors, scope_label)
        if res is None:
            all_results.append(pd.DataFrame([
                {"model": model, "scope": scope_label, "factor": f, "partial_r2": np.nan,
                 "f_stat_naive": np.nan, "p_naive": np.nan, "chi2_cluster": np.nan, "p_cluster": np.nan,
                 "note": "model fit failed / invalid -- see log"}
                for f in RANKED_FACTORS
            ]))
            continue
        res = res.sort_values("partial_r2", ascending=False).reset_index(drop=True)
        res["rank"] = res["partial_r2"].rank(ascending=False, method="min")
        all_results.append(res)
        print("  Partial R^2 ranking (naive F-test vs cluster-robust Wald test side by side):")
        print(res[["factor", "partial_r2", "rank", "f_stat_naive", "p_naive", "chi2_cluster", "p_cluster"]].to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))
    return pd.concat(all_results, ignore_index=True)


def h1_check(combined, scope_label):
    print(f"\n  H1 check ({scope_label}): profession vs country/age/gender, llama/gemma/qwen/ministral only")
    verdicts = {}
    for model in [m for m in MODEL_ORDER if m != "deepseek"]:
        sub = combined[(combined["model"] == model) & (combined["scope"] == scope_label)].set_index("factor")["partial_r2"]
        prof = sub.get("profession", np.nan)
        others = {k: sub.get(k, np.nan) for k in ["country", "age", "gender"]}
        beats_all = all(prof > v for v in others.values() if not np.isnan(v))
        verdicts[model] = beats_all
        print(f"    {model:10s}: profession={prof:.4f} vs country={others['country']:.4f}, "
              f"age={others['age']:.4f}, gender={others['gender']:.4f}  -> "
              f"{'SUPPORTS H1' if beats_all else 'does NOT support H1 as stated'}")
    return verdicts


def main():
    df = cast_formula_dtypes(load_master())

    print("#" * 78)
    print("PRIMARY, PRE-SPECIFIED H1 TEST -- Condition A (forced) only, per Fix 3 / Section 4")
    print("#" * 78)
    primary = run_scope(df, "A_forced", FULL_MODEL_FACTORS_CONDA, "primary_conditionA")

    print()
    print("#" * 78)
    print("EXPLORATORY -- original A+B pooled version (NOT the pre-specified H1 test; kept for")
    print("comparison only, per Fix 3)")
    print("#" * 78)
    exploratory = run_scope(df, None, FULL_MODEL_FACTORS_POOLED, "exploratory_pooled_AB")

    combined = pd.concat([primary, exploratory], ignore_index=True)
    out_path = f"{TABLES_DIR}/variance_ranking.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  (both scopes in one file, distinguished by the 'scope' column)")

    print()
    print("=" * 78)
    print("SUMMARY: factor ranking by model, PRIMARY (Condition A only) scope")
    print("=" * 78)
    for model in MODEL_ORDER:
        sub = primary[(primary["model"] == model)].dropna(subset=["partial_r2"]).sort_values("partial_r2", ascending=False)
        order = ", ".join(f"{r.factor}({r.partial_r2:.3f})" for r in sub.itertuples())
        caveat = "  [DEEPSEEK -- EXPLORATORY / UNRELIABLE, n=63, see caveat below]" if model == "deepseek" else ""
        print(f"  {model:10s}: {order}{caveat}")

    print()
    print("=" * 78)
    print("H1 CHECK -- PRIMARY (Condition A only, the pre-specified test) vs EXPLORATORY (A+B pooled,")
    print("the original version) -- does the conclusion change?")
    print("=" * 78)
    primary_verdicts = h1_check(combined, "primary_conditionA")
    exploratory_verdicts = h1_check(combined, "exploratory_pooled_AB")

    print("\n  Does the H1 verdict CHANGE between primary (Condition A) and exploratory (A+B pooled)?")
    for model in [m for m in MODEL_ORDER if m != "deepseek"]:
        changed = primary_verdicts.get(model) != exploratory_verdicts.get(model)
        print(f"    {model:10s}: primary={primary_verdicts.get(model)}  exploratory={exploratory_verdicts.get(model)}  "
              f"-> {'CHANGED' if changed else 'unchanged'}")

    print()
    print("=" * 78)
    print("DEEPSEEK CAVEAT (same treatment as steps 6/7/9)")
    print("=" * 78)
    print("DeepSeek's partial R^2 values above are computed on only 63 strict-valid rows (all Condition")
    print("A already, so the primary/exploratory distinction does not change its subset), covering 2/7")
    print("topics, 13/20 countries, 15/30 professions, and a single gender level (gender not estimable).")
    print("Reported for completeness but exploratory/unreliable, not comparable to the other four models.")


if __name__ == "__main__":
    main()
