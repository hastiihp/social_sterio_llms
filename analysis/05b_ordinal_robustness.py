"""Step 5b (Tier 2, Fix 7, then Fix B): ordinal-logit robustness check for
H1's OLS results.

FIX B: this script still pooled Condition A and B and used unclustered SEs
after both were corrected everywhere else in the pipeline (Fix 3 for
population scope, Fix 2 for clustering). Restricted to Condition A only now
(H1's actual scope, matching 08_variance_ranking.py's precedent and 05c's
precedent) -- response_condition is dropped from the formula entirely
(constant after the restriction). Persona-clustered SEs
(cov_type='cluster', cov_kwds={'groups': persona_id}) are now used
throughout: OrderedModel accepts this via kwargs passthrough to
GenericLikelihoodModel.fit() (verified empirically). Note on WHERE
clustering actually changes anything: cov_type is a post-estimation choice
about how to compute SEs from already-obtained MLE point estimates -- it
does not change log-likelihoods or coefficients, so the partial-pseudo-R^2
ranking (LR-test-based, log-likelihood differences only) and the profession
rank correlation (raw coefficients, not p-values) are numerically identical
whether or not clustering is requested. Clustering concretely changes only
the proportional-odds Cochran's Q test below, which directly uses each
cutpoint-specific fit's standard errors.

analysis_plan.md Section 4 requires checking the OLS results against an
ordinal logistic regression, since rating is a 5-point ordinal scale, not
truly continuous. Fits a proportional-odds (ordinal logistic) model per
model via statsmodels' OrderedModel(distr="logit"), same formula and data
scope as step 5a's Condition-A analysis (gender + country + profession +
age + topic, Condition A only, strict-valid numeric ratings only).
DeepSeek excluded (same n=63 sparsity justification as Fix 4; also already
100% Condition A, so this restriction doesn't change its own scope, it was
never included here regardless).

Two things are checked, both QUALITATIVE robustness checks (not a claim
that OLS and ordinal-logit coefficients should match numerically -- they
are on different scales):

1. Profession ranking: Spearman rank correlation between OLS and
   ordinal-logit profession coefficients (relative to 'accountant').

2. Relative factor importance (the H1 question -- does gender or
   profession explain more?). REVISED from the first run: a single-model
   joint Wald chi2/df test was tried first and rejected -- chi2 test
   statistics conflate statistical detectability with effect size and
   scale with sample size, systematically favoring low-df concentrated
   effects (gender, 2 params) over high-df dispersed ones (profession, 29
   params) regardless of actual explanatory share. This version instead
   fits a null (cutpoints-only) model and 5 reduced models (each factor
   dropped in turn) and computes a NORMALIZED partial pseudo-R^2 per
   factor:
       partial_pseudo_R2(f) = (llf_full - llf_reduced_f) / (llf_full - llf_null)
   -- the proportion of the full model's total log-likelihood improvement
   over the null that is attributable to factor f, controlling for all
   others. This is a proportion (bounded, scale-free), the correct ordinal
   analog to OLS's SSE-based partial R^2, unlike a raw chi2 statistic.

Proportional-odds assumption test: REVISED from the first run, which used
the full ~60-parameter formula for each of 4 cutpoint-specific binary
logits and failed to converge for ANY model (0/4) due to sparse cells at
extreme cutpoints combined with too many profession/country dummies. This
version uses a REDUCED formula (gender + condition + topic only, dropping
profession/country) for the cutpoint-specific fits, keeping the check
feasible while still covering the three headline terms. Reports Cochran's Q
heterogeneity test across cutpoint-specific coefficients per term; if usable
cutpoints are still fewer than 2 for a model, that is reported plainly as
"could not be tested", not glossed over as "holds".

Runs one model per process (see run_all.sh-style orchestration in
__main__) so the ~7 OrderedModel fits per model (null + full + 5 reduced)
can run in parallel across models rather than serially.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

from _common import load_master, cast_formula_dtypes, TABLES_DIR

INCLUDED_MODELS = ["llama", "gemma", "qwen", "ministral"]  # deepseek excluded, see docstring
FACTORS = ["gender", "country", "profession", "age", "topic"]  # response_condition removed, Fix B
PO_FACTORS = ["gender", "topic"]  # reduced formula for proportional-odds test, Fix B
CUTPOINTS = [1, 2, 3, 4]


def valid_subset(df, model):
    sub = df[(df["model"] == model) & (df["response_condition"] == "A_forced") &
             df["strict_is_valid"] & df["rating_numeric"].notnull()].copy()
    sub["rating_cat"] = sub["rating_numeric"].astype(int)
    return sub


def fit_ordinal(formula, sub, label, cluster_col="persona_id"):
    mod = OrderedModel.from_formula(formula, data=sub, distr="logit")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = mod.fit(method="bfgs", maxiter=300, disp=False,
                       cov_type="cluster", cov_kwds={"groups": sub[cluster_col]})
        for w in caught:
            print(f"  WARNING (OrderedModel fit, {label}): {w.message}", flush=True)
    print(f"  [{label}] converged={res.mle_retvals.get('converged','n/a')} llf={res.llf:.1f} "
          f"n_params={len(res.params)} (persona-clustered SE, n_clusters={sub[cluster_col].nunique():,})", flush=True)
    return res


def null_loglik(sub):
    """Closed-form log-likelihood of the intercept-only (no-covariate) cumulative-logit model.

    OrderedModel cannot fit a literal '~1' formula -- it always drops any constant term since
    the K-1 cutpoints/thresholds already serve as the model's baseline, so an intercept-only
    formula produces an empty design matrix and crashes. But the no-covariate cumulative-logit
    model's MLE has a closed form: with K-1 free threshold parameters for K categories, the
    best-fitting thresholds exactly reproduce the observed marginal category proportions, so its
    log-likelihood equals the multinomial log-likelihood of those proportions. No fit needed.
    """
    counts = sub["rating_cat"].value_counts()
    n = counts.sum()
    p = counts / n
    return float((counts * np.log(p)).sum())


def partial_pseudo_r2_ranking(sub, model_label):
    print(f"\n  --- Partial pseudo-R^2 per factor (closed-form null + full + 5 reduced ordinal fits) ---", flush=True)
    full_formula = "rating_cat ~ " + " + ".join(f"C({f})" for f in FACTORS)

    llf_null = null_loglik(sub)
    print(f"  [{model_label}/null] llf={llf_null:.1f} (closed-form multinomial, no fit)", flush=True)
    res_full = fit_ordinal(full_formula, sub, f"{model_label}/full")

    rows = []
    for f in FACTORS:
        reduced_factors = [x for x in FACTORS if x != f]
        reduced_formula = "rating_cat ~ " + " + ".join(f"C({x})" for x in reduced_factors)
        res_reduced = fit_ordinal(reduced_formula, sub, f"{model_label}/-{f}")
        denom = res_full.llf - llf_null
        partial_r2 = (res_full.llf - res_reduced.llf) / denom if denom > 0 else np.nan
        lr_stat = 2 * (res_full.llf - res_reduced.llf)
        df_diff = len(res_full.params) - len(res_reduced.params)
        p = 1 - stats.chi2.cdf(lr_stat, df_diff) if df_diff > 0 else np.nan
        rows.append({"model": model_label, "factor": f, "partial_pseudo_r2": partial_r2,
                     "lr_stat": lr_stat, "df": df_diff, "p": p})
        print(f"    factor={f:20s} partial_pseudo_R2={partial_r2:.4f}  LR={lr_stat:.1f}  df={df_diff}  p={p:.3g}", flush=True)

    out = pd.DataFrame(rows).sort_values("partial_pseudo_r2", ascending=False).reset_index(drop=True)
    out["rank"] = out["partial_pseudo_r2"].rank(ascending=False, method="min")
    return out, res_full


def fit_matching_ols(sub, model_label):
    """Fit the SAME formula/population as the ordinal model (Condition A only, this
    exact `sub`) via OLS with persona-clustered SEs, for a true apples-to-apples
    profession comparison. Fix B: the original version compared against
    hypothesis_model_{model}.csv's coef_hc3, which pools Condition A+B -- comparing
    a now-Condition-A-only ordinal fit against an A+B-pooled OLS fit would reintroduce
    the exact population-mismatch problem this fix addresses, just on the other side
    of the comparison. hypothesis_model_pooled.csv isn't a substitute either: its
    profession terms are LLAMA-specific main effects in an interaction model (Fix 6),
    not a per-model coefficient. Refitting internally, on the identical `sub` already
    used for the ordinal fit, is the only way to guarantee the same population.
    """
    formula = "rating_numeric ~ " + " + ".join(f"C({f})" for f in FACTORS)
    res = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})
    print(f"  [{model_label}/matching-OLS] R2={res.rsquared:.4f} n={int(res.nobs):,} "
          f"(persona-clustered SE, Condition A only -- same population as the ordinal fit)", flush=True)
    return res


def profession_rank_comparison(matching_ols_res, ordinal_res, model_label):
    ols_ct = pd.DataFrame({"term": matching_ols_res.params.index, "coef_hc3": matching_ols_res.params.values})
    ols_prof = ols_ct[ols_ct["term"].str.match(r"^C\(profession\)\[T\.[^]]+\]$")].copy()
    ols_prof["profession"] = ols_prof["term"].str.extract(r"\[T\.([^]]+)\]")
    ols_prof = ols_prof[["profession", "coef_hc3"]].rename(columns={"coef_hc3": "coef_ols"})

    ord_params = ordinal_res.params
    ord_terms = [t for t in ord_params.index if t.startswith("C(profession)[T.")]
    ord_prof = pd.DataFrame({"term": ord_terms, "coef_ordinal": [ord_params[t] for t in ord_terms]})
    ord_prof["profession"] = ord_prof["term"].str.extract(r"\[T\.([^]]+)\]")
    ord_prof = ord_prof[["profession", "coef_ordinal"]]

    merged = ols_prof.merge(ord_prof, on="profession", how="inner")
    rho, p = stats.spearmanr(merged["coef_ols"], merged["coef_ordinal"])
    print(f"\n  --- Profession ranking, OLS vs ordinal-logit (n={len(merged)}): Spearman rho={rho:.4f} (p={p:.2e}) ---", flush=True)
    merged["model"] = model_label
    return merged, rho, p


def proportional_odds_test(sub, model_label):
    print(f"\n  --- Proportional-odds check, REDUCED formula (gender+topic only), "
          f"persona-clustered SE ---", flush=True)
    formula = "answered_le_c ~ " + " + ".join(f"C({f})" for f in PO_FACTORS)
    cutpoint_results = {}
    for c in CUTPOINTS:
        cp_sub = sub.copy()
        cp_sub["answered_le_c"] = (cp_sub["rating_cat"] <= c).astype(int)
        rate = cp_sub["answered_le_c"].mean()
        if rate <= 0.001 or rate >= 0.999:
            print(f"    cutpoint <= {c}: rate={rate:.4f} -- too extreme, skipped", flush=True)
            continue
        try:
            res_c = smf.logit(formula, data=cp_sub).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": cp_sub["persona_id"]})
            if not res_c.mle_retvals.get("converged", True):
                print(f"    cutpoint <= {c}: did not converge, skipped", flush=True)
                continue
            cutpoint_results[c] = res_c
            print(f"    cutpoint <= {c}: rate={rate:.4f}, converged, n_params={len(res_c.params)} "
                  f"(persona-clustered SE)", flush=True)
        except Exception as e:
            print(f"    cutpoint <= {c}: FIT FAILED ({type(e).__name__}: {e}), skipped", flush=True)

    if len(cutpoint_results) < 2:
        print("    Fewer than 2 usable cutpoint fits -- proportional-odds assumption COULD NOT BE TESTED "
              "for this model (reported plainly, not assumed to hold).", flush=True)
        return pd.DataFrame([{"model": model_label, "term": "ALL", "testable": False}])

    all_terms = set.intersection(*[set(r.params.index) for r in cutpoint_results.values()])
    rows = []
    for term in sorted(all_terms):
        betas = np.array([cutpoint_results[c].params[term] for c in cutpoint_results])
        ses = np.array([cutpoint_results[c].bse[term] for c in cutpoint_results])
        if np.any(ses <= 0) or not np.all(np.isfinite(ses)) or not np.all(np.isfinite(betas)):
            continue
        weights = 1.0 / (ses ** 2)
        beta_bar = np.sum(betas * weights) / np.sum(weights)
        Q = np.sum(weights * (betas - beta_bar) ** 2)
        dof = len(cutpoint_results) - 1
        p = 1 - stats.chi2.cdf(Q, dof) if dof > 0 else np.nan
        rows.append({"model": model_label, "term": term, "testable": True, "n_cutpoints": len(cutpoint_results),
                      "Q_stat": Q, "df": dof, "p_value": p, "violated_at_0.05": bool(p < 0.05) if dof > 0 else None})

    result = pd.DataFrame(rows).sort_values("p_value")
    n_violated = result["violated_at_0.05"].sum()
    print(f"\n    {n_violated}/{len(result)} terms show significant heterogeneity (p<0.05) -- proportional-odds "
          f"VIOLATED for {n_violated}/{len(result)} terms in this model.", flush=True)
    print(result[["term", "Q_stat", "p_value", "violated_at_0.05"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"), flush=True)
    return result


def run_one_model(model):
    df = cast_formula_dtypes(load_master())
    sub = valid_subset(df, model)
    print(f"\n{'='*78}\nMODEL: {model}   n={len(sub):,}\n{'='*78}", flush=True)

    ranking, res_full = partial_pseudo_r2_ranking(sub, model)
    ranking.to_csv(f"{TABLES_DIR}/ordinal_pseudo_r2_{model}.csv", index=False)

    matching_ols_res = fit_matching_ols(sub, model)
    prof_comp, rho, p = profession_rank_comparison(matching_ols_res, res_full, model)
    prof_comp.to_csv(f"{TABLES_DIR}/ordinal_profession_comparison_{model}.csv", index=False)

    po_result = proportional_odds_test(sub, model)
    po_result.to_csv(f"{TABLES_DIR}/ordinal_proportional_odds_{model}.csv", index=False)

    # Full ordinal coefficient table (coef + persona-clustered SE + p), not previously saved --
    # every other model in this pipeline saves one, and now that clustered SEs are actually
    # being computed, they're worth keeping rather than discarding after use.
    ci = res_full.conf_int()
    full_ct = pd.DataFrame({
        "model": model, "term": res_full.params.index, "coef": res_full.params.values,
        "se_cluster": res_full.bse.values, "p_cluster": res_full.pvalues.values,
        "ci_low": ci[0].values, "ci_high": ci[1].values,
    })
    full_ct.to_csv(f"{TABLES_DIR}/ordinal_full_coefficients_{model}.csv", index=False)

    print(f"\n[{model}] DONE.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=INCLUDED_MODELS)
    args = parser.parse_args()
    run_one_model(args.model)
