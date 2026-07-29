"""Step 5: primary hypothesis models (H1). Revised per external audit (Fixes 2 & 4).

(a) Per-model OLS, strict-valid responses only:
      rating ~ gender + country + profession + age + topic + condition
    Fit separately for each of the five models. Reported with THREE parallel
    SE estimates so the audit's concern is checked, not asserted:
      - HC3 (original / uncorrected): treats every row as independent.
      - Cluster-robust by persona_id (Fix 2, primary correction): each
        persona contributes up to 14 rows (7 topics x 2 conditions) --
        analysis_plan.md Section 4 explicitly requires accounting for this.
        Point estimates (coef) are IDENTICAL to HC3 -- clustering only
        changes the standard errors/inference, not the coefficients.
      - Mixed-effects model with a persona random intercept (Section 4's
        literal specification: rating ~ fixed effects + (1 | persona)).
        This DOES generally shift point estimates slightly (different
        estimator, REML), not just SEs.
        FIX A: convergence is now checked explicitly (res.converged), not
        assumed from the absence of a raised exception -- the original bug.
        statsmodels' default optimizer fails to converge for some models in
        this dataset (llama and ministral, verified); a real, stable MLE
        exists nearby in both cases (confirmed via method='lbfgs' and
        method='powell', both converging to essentially the same
        log-likelihood), so those are retried with alternate optimizers
        before giving up. Any model whose MixedLM still doesn't converge
        after all attempts is marked MIXED_MODEL_NONCONVERGED_EXCLUDED in
        the output table (mixedlm columns left blank) rather than having
        unreliable coefficients silently merged in. The cluster-robust OLS
        result remains the primary reported result regardless.

(b) Pooled model exactly as specified in analysis_plan.md Section 4,
    Condition A / forced-rating only:
      rating ~ topic + profession + country + gender + age + model
    with interactions topic:model, profession:model, country:model, gender:model.
    Fit on LLAMA, GEMMA, QWEN, MINISTRAL ONLY (Fix 4). DeepSeek is excluded
    from this pooled model: with n=63 and only 15/30 professions, 13/20
    countries, and 1/3 genders represented, its interaction terms are not
    identifiable IN PRINCIPLE (not merely numerically unstable) -- pinning
    the reference level (as the original version of this script did) only
    relocates where the instability shows up, it does not fix the
    underlying identification problem. DeepSeek's own per-model regression
    in (a) is unaffected by this and remains its separately reported,
    heavily-caveated result (see step 9).

All models use rating_numeric (derived numeric rating) restricted to
strict_is_valid==True, per analysis_plan.md's rule that strict_is_valid is
the only field used for primary/inferential analysis.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from _common import load_master, cast_formula_dtypes, MODEL_ORDER, TABLES_DIR

PER_MODEL_FACTORS = ["gender", "country", "profession", "age", "response_condition", "topic"]
POOLED_FACTORS = ["topic", "profession", "country", "gender", "age"]
POOLED_MODELS = ["llama", "gemma", "qwen", "ministral"]  # Fix 4: deepseek excluded


def valid_subset(df, model=None, condition=None):
    sub = df[df["strict_is_valid"] & df["rating_numeric"].notnull()]
    if model is not None:
        sub = sub[sub["model"] == model]
    if condition is not None:
        sub = sub[sub["response_condition"] == condition]
    return sub


def _side(res, suffix):
    ci = res.conf_int()
    return pd.DataFrame({
        "term": res.params.index,
        f"coef_{suffix}": res.params.values,
        f"se_{suffix}": res.bse.values,
        f"p_{suffix}": res.pvalues.values,
        f"ci_low_{suffix}": ci[0].values,
        f"ci_high_{suffix}": ci[1].values,
    })


def combined_coef_table(res_hc3, res_cluster, res_mixed=None, mixed_status="not_attempted"):
    """One wide table: HC3 (original) vs cluster-robust (Fix 2) vs mixed-model, side by side.

    Fix A: mixed_status records whether the MixedLM fit actually converged (checked via
    res.converged, not just the absence of a raised exception -- the original bug: a
    non-converged fit's coefficients were merged into the output table with no check at
    all). Non-converged fits are NEVER merged in here -- res_mixed must be None in that
    case -- so there is nothing to accidentally include; mixed_status is what tells a
    reader of the CSV *why* the mixedlm columns are empty, instead of it looking
    identical to "mixedlm was never attempted".
    """
    t = _side(res_hc3, "hc3").merge(_side(res_cluster, "cluster"), on="term", how="outer")
    if res_mixed is not None:
        m = _side(res_mixed, "mixedlm")
        t = t.merge(m, on="term", how="outer")
    else:
        for col in ["coef_mixedlm", "se_mixedlm", "p_mixedlm", "ci_low_mixedlm", "ci_high_mixedlm"]:
            t[col] = np.nan
    t["mixedlm_status"] = mixed_status
    return t


def fit_mixedlm_with_fallback(formula, sub, label):
    """Fit MixedLM, checking res.converged explicitly (Fix A) rather than treating "no
    exception raised" as success. statsmodels' default optimizer (method=None) fails to
    converge for some models in this dataset (verified: llama and ministral both land on
    res.converged=False by default) even though a real, stable MLE exists nearby --
    verified by refitting with method='lbfgs' and method='powell', both of which converge
    and land on essentially the same log-likelihood as whatever partial optimum the
    default run found (e.g. llama: -39010.396 vs -39010.397). Tries, in order: default,
    lbfgs, powell. Returns (res, status_string) -- res is None if every attempt fails to
    converge, and the caller must never merge a None-converged result into the output.
    """
    attempts = [("default", {}), ("lbfgs", {"method": "lbfgs"}), ("powell", {"method": "powell"})]
    for name, kwargs in attempts:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                md = smf.mixedlm(formula, data=sub, groups=sub["persona_id"])
                res = md.fit(reml=True, maxiter=200, **kwargs)
                for w in caught:
                    print(f"  WARNING (MixedLM fit, {label}, attempt={name}): {w.message}")
            if res.converged:
                print(f"  MixedLM ({label}): CONVERGED on attempt '{name}' (llf={res.llf:.4f}).")
                return res, f"converged ({name})"
            else:
                print(f"  MixedLM ({label}): attempt '{name}' did NOT converge (res.converged=False).")
        except Exception as e:
            print(f"  MixedLM ({label}): attempt '{name}' FAILED: {type(e).__name__}: {e}")
    print(f"  MixedLM ({label}): ALL optimizer attempts failed to converge -- "
          f"MIXED_MODEL_NONCONVERGED, excluded from output table.")
    return None, "MIXED_MODEL_NONCONVERGED_EXCLUDED"


def fit_variants(formula, sub, label):
    """Fit HC3, cluster-robust (persona_id), and mixed-effects (persona random intercept)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_hc3 = smf.ols(formula, data=sub).fit(cov_type="HC3")
        for w in caught:
            print(f"  WARNING (HC3 fit, {label}): {w.message}")

    res_cluster = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})

    res_mixed, mixed_status = fit_mixedlm_with_fallback(formula, sub, label)

    return res_hc3, res_cluster, res_mixed, mixed_status


def print_se_comparison(ct, terms_of_interest, has_mixed):
    print(f"\n  --- SE comparison for key terms: HC3 (original) vs cluster-robust (Fix 2) "
          f"{'vs mixed-model' if has_mixed else ''} ---")
    cols = ["term", "coef_hc3", "se_hc3", "p_hc3", "se_cluster", "p_cluster"]
    if has_mixed:
        cols += ["coef_mixedlm", "se_mixedlm", "p_mixedlm"]
    sub = ct[ct["term"].isin(terms_of_interest)][cols]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))


def fit_per_model(df, model):
    sub = valid_subset(df, model=model)
    n = len(sub)

    dropped = []
    factors = []
    for f in PER_MODEL_FACTORS:
        if sub[f].nunique() <= 1:
            dropped.append(f)
        else:
            factors.append(f)

    formula = "rating_numeric ~ " + " + ".join(f"C({f})" for f in factors)
    print(f"\n{'='*78}\nModel: {model}   (n = {n:,} strict-valid rows)")
    if dropped:
        print(f"  WARNING: dropped factor(s) with zero variance in this subset: {dropped}")
    print(f"  formula: {formula}")

    res_hc3, res_cluster, res_mixed, mixed_status = fit_variants(formula, sub, model)

    n_params = len(res_hc3.params)
    if n < 3 * n_params:
        print(f"  WARNING: low ratio of observations to parameters (n={n}, params={n_params}) -- estimates may be unstable.")
    print(f"  R-squared: {res_hc3.rsquared:.4f}   df_resid: {res_hc3.df_resid:.0f}   n_params: {n_params}")
    n_persona = sub["persona_id"].nunique()
    print(f"  n_persona clusters: {n_persona:,}   avg rows/persona in this subset: {n/n_persona:.2f}")

    ct = combined_coef_table(res_hc3, res_cluster, res_mixed, mixed_status)
    # Only check terms that are actually part of the HC3/OLS fit (res_hc3.params) -- after the
    # outer merge with MixedLM's results, rows like "Group Var" (the random-intercept variance
    # component, which has no OLS counterpart) are expected to be NaN in the hc3 columns and are
    # not a fit problem.
    ols_terms = set(res_hc3.params.index)
    ols_rows = ct[ct["term"].isin(ols_terms)]
    n_nonfinite = (~np.isfinite(ols_rows["coef_hc3"])).sum() + (~np.isfinite(ols_rows["se_hc3"])).sum()
    if n_nonfinite:
        print(f"  WARNING: {n_nonfinite} non-finite HC3 coefficient/SE values among actual OLS terms "
              f"(likely near-collinearity from sparse deepseek coverage).")

    # flag terms whose significance verdict flips between HC3 and cluster-robust at alpha=0.05
    both_finite = ct.dropna(subset=["p_hc3", "p_cluster"])
    flips = both_finite[(both_finite["p_hc3"] < 0.05) != (both_finite["p_cluster"] < 0.05)]
    print(f"  Terms where significance (alpha=0.05) FLIPS between HC3 and cluster-robust SE: {len(flips)}/{len(both_finite)}")
    if len(flips):
        print(flips[["term", "coef_hc3", "se_hc3", "p_hc3", "se_cluster", "p_cluster"]].to_string(
            index=False, float_format=lambda x: f"{x:9.4f}"))

    key_terms = [t for t in ct["term"] if any(k in t for k in ["topic)[T.trust", "topic)[T.gender equality", "gender)[T.male", "condition"])][:5]
    if key_terms:
        print_se_comparison(ct, key_terms, res_mixed is not None)

    return dict(hc3=res_hc3, cluster=res_cluster, mixed=res_mixed, mixed_status=mixed_status, ct=ct, dropped=dropped, n=n)


def fit_pooled(df):
    sub = valid_subset(df, condition="A_forced")
    sub = sub[sub["model"].isin(POOLED_MODELS)]  # Fix 4: deepseek excluded
    print(f"\n{'='*78}\nPOOLED MODEL (analysis_plan.md Section 4, Condition A / forced only)")
    print(f"Fix 4: DeepSeek EXCLUDED from this pooled model (see docstring) -- {POOLED_MODELS} only.")
    print(f"n = {len(sub):,} strict-valid Condition-A rows across these {len(POOLED_MODELS)} models")
    print(sub.groupby("model", observed=True).size().rename("n_valid_condition_A").to_string())

    dropped = []
    factors = []
    for f in POOLED_FACTORS:
        if sub[f].nunique() <= 1:
            dropped.append(f)
        else:
            factors.append(f)
    if dropped:
        print(f"  WARNING: dropped factor(s) with zero variance overall: {dropped}")

    model_term = 'C(model, Treatment(reference="llama"))'
    main_terms = " + ".join(f"C({f})" for f in factors) + f" + {model_term}"
    interaction_terms = " + ".join(f"C({f}):{model_term}" for f in factors)
    formula = f"rating_numeric ~ {main_terms} + {interaction_terms}"
    print(f"  formula: {formula}")

    # MixedLM is not attempted here: Fix 2 requires it "at least for the per-model fits" (done
    # above), not the pooled model, and with ~236 fixed-effect parameters here a mixed-model fit
    # would be extremely slow for a component the audit did not require. Cluster-robust SE
    # (required for both 5a and 5b) is fit below.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_hc3 = smf.ols(formula, data=sub).fit(cov_type="HC3")
        for w in caught:
            print(f"  WARNING (HC3 fit, pooled): {w.message}")
    # Primary: cluster by persona_id alone (a given persona's rows are grouped together ACROSS
    # all 4 models, not just within one model) -- the more conservative choice for testing the
    # model-interaction terms that are H2's object, since it accounts for correlation both within
    # a model's repeated topics and across models rating the same underlying persona.
    res_cluster = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})

    # Robustness check: cluster by (model, persona_id) instead -- a given persona's rows are
    # grouped separately per model (up to 7 topic-rows per cluster, not 28). Requested as a
    # sensitivity check on whether the persona_id-only choice (which pools across models) matters.
    # .astype(str) and string concatenation both produce pandas 3.0's StringDtype regardless of
    # the input's dtype (verified: even on an already-object-dtype source) -- cast the result
    # too, not just the source columns, since this is passed straight to cov_kwds.
    model_persona_group = (sub["model"].astype(str) + "_" + sub["persona_id"].astype(str)).astype("object")
    res_cluster_mp = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": model_persona_group})
    print(f"  Robustness check: (model, persona_id) clustering -> "
          f"{model_persona_group.nunique():,} clusters (vs {sub['persona_id'].nunique():,} for persona_id-only)")

    res_mixed = None

    n_params = len(res_hc3.params)
    print(f"  R-squared: {res_hc3.rsquared:.4f}   df_resid: {res_hc3.df_resid:.0f}   n_params: {n_params}")

    ct = combined_coef_table(res_hc3, res_cluster, res_mixed, mixed_status="not_attempted (pooled model, see docstring)")
    ct_mp = _side(res_cluster_mp, "cluster_modelpersona")
    ct = ct.merge(ct_mp, on="term", how="outer")

    SE_BLOWUP_THRESHOLD = 5.0
    unstable = ct[~np.isfinite(ct["se_hc3"]) | (ct["se_hc3"] > SE_BLOWUP_THRESHOLD)]
    if len(unstable):
        print(f"  WARNING: {len(unstable)}/{len(ct)} terms have non-finite or implausibly large "
              f"(SE > {SE_BLOWUP_THRESHOLD}) HC3 standard errors.")
        print(unstable[["term", "coef_hc3", "se_hc3"]].to_string(index=False))
    else:
        print(f"  No unstable (SE > {SE_BLOWUP_THRESHOLD}) terms -- confirms Fix 4 (deepseek removal) resolved the")
        print(f"  instability that previously required pinning the reference level as a workaround.")

    both_finite = ct.dropna(subset=["p_hc3", "p_cluster"])
    flips = both_finite[(both_finite["p_hc3"] < 0.05) != (both_finite["p_cluster"] < 0.05)].copy()
    print(f"  Terms where significance (alpha=0.05) FLIPS between HC3 and cluster-robust (persona_id) SE: "
          f"{len(flips)}/{len(both_finite)}")

    both_finite_mp = ct.dropna(subset=["p_hc3", "p_cluster_modelpersona"])
    flips_mp = both_finite_mp[(both_finite_mp["p_hc3"] < 0.05) != (both_finite_mp["p_cluster_modelpersona"] < 0.05)]
    print(f"  Terms where significance (alpha=0.05) FLIPS between HC3 and cluster-robust (model,persona_id) SE: "
          f"{len(flips_mp)}/{len(both_finite_mp)}")

    if len(flips):
        flips["abs_coef"] = flips["coef_hc3"].abs()
        flips = flips.sort_values("abs_coef", ascending=False)
        flips["flips_under_modelpersona_too"] = flips["term"].isin(
            set(flips_mp["term"]) if len(flips_mp) else set()
        )
        print(f"\n  --- All {len(flips)} persona_id-cluster flips: HC3 vs persona_id-cluster (primary) vs "
              f"(model,persona_id)-cluster (robustness check) ---")
        print(flips[["term", "coef_hc3", "p_hc3", "se_cluster", "p_cluster",
                      "se_cluster_modelpersona", "p_cluster_modelpersona", "flips_under_modelpersona_too"]].to_string(
            index=False, float_format=lambda x: f"{x:9.5f}"))
        n_agree = flips["flips_under_modelpersona_too"].sum()
        print(f"\n  -> {n_agree}/{len(flips)} of the persona_id-cluster flips ALSO flip under (model,persona_id)")
        print(f"     clustering. persona_id-only (pooling across models) is reported as the primary result;")
        print(f"     (model,persona_id) clustering is the robustness check, both shown in the saved table.")

    profession_terms = [t for t in ct["term"] if t.startswith("C(profession)[T.") and ":" not in t][:5]
    print_se_comparison(ct, profession_terms, res_mixed is not None)

    return dict(hc3=res_hc3, cluster=res_cluster, cluster_modelpersona=res_cluster_mp, mixed=res_mixed, ct=ct, dropped=dropped)


def main():
    df = cast_formula_dtypes(load_master())

    per_model_results = {}
    for model in MODEL_ORDER:
        result = fit_per_model(df, model)
        per_model_results[model] = result
        out_path = f"{TABLES_DIR}/hypothesis_model_{model}.csv"
        table = result["ct"].copy()
        if model == "deepseek":
            table["inferential_status"] = "NON-INFERENTIAL / EXPLORATORY ONLY"
            table["analysis_note"] = (
                "n=63 strict-valid rows; rank-deficient sparse design and numerically "
                "unstable coefficients/SEs/p-values. Do not interpret significance."
            )
        table.to_csv(out_path, index=False)
        print(f"  Wrote {out_path} (columns: coef/se/p for hc3, cluster, and mixedlm side by side)")
        if model == "deepseek":
            print("  DEEPSEEK OUTPUT WARNING: every coefficient row is NON-INFERENTIAL / "
                  "EXPLORATORY ONLY (n=63; sparse rank-deficient design; numerically unstable).")

    pooled_result = fit_pooled(df)
    out_path = f"{TABLES_DIR}/hypothesis_model_pooled.csv"
    pooled_result["ct"].to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(pooled_result['ct'])} coefficients, deepseek excluded per Fix 4)")

    print(f"\n{'='*78}\nCOMPARISON: per-model vs pooled")
    print("=" * 78)
    print("Per-model R-squared (HC3 fit; rating ~ demographics + topic + condition, both conditions pooled):")
    for model in MODEL_ORDER:
        result = per_model_results[model]
        note = f"  (dropped: {result['dropped']})" if result["dropped"] else ""
        print(f"  {model:10s} R2={result['hc3'].rsquared:.4f}  n={int(result['hc3'].nobs):,}{note}")
    print(f"\nPooled model (Condition A only, 4 models per Fix 4) R-squared: "
          f"{pooled_result['hc3'].rsquared:.4f}  n={int(pooled_result['hc3'].nobs):,}")

    print(f"\n{'='*78}\nMIXEDLM CONVERGENCE STATUS (Fix A -- explicitly checked, not assumed)")
    print("=" * 78)
    for model in MODEL_ORDER:
        status = per_model_results[model]["mixed_status"]
        flag = "" if status.startswith("converged") else "  <-- EXCLUDED from mixedlm columns, see table"
        print(f"  {model:10s} {status}{flag}")
    print("  The cluster-robust OLS result (se_cluster/p_cluster columns) is the primary reported result")
    print("  for every model regardless of MixedLM convergence status -- MixedLM is a secondary check.")
    print()
    print("CAVEAT (unchanged from original): the per-model formulas include 'condition' as a covariate,")
    print("mixing Condition A and Condition B ratings. For qwen/ministral this conflates the condition")
    print("effect with abstention-selection. The pooled model avoids this via Condition A only.")
    print()
    print("DeepSeek caveat: 63 valid rows total, all Condition A, gender-constant subset.")
    print("Its separately emitted coefficient table is NON-INFERENTIAL / EXPLORATORY ONLY: the sparse,")
    print("rank-deficient design produces numerically unstable coefficients, SEs, and p-values. DeepSeek")
    print("is EXCLUDED from the pooled model (b) per Fix 4 -- see step 9 for full treatment.")


if __name__ == "__main__":
    main()
