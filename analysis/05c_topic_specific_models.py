"""Step 5c (Tier 2, Fix 8, revised): topic-specific primary models --
analysis_plan.md Section 7.

FIX APPLIED: the original version of this script pooled Condition A and
Condition B together (controlling for condition as a covariate). This
reintroduced the exact selection problem Fix 3 already solved for
08_variance_ranking.py: for qwen and ministral, Condition B's "answered"
rows are a small, self-selected subset (topic sample sizes ranged
5,400-7,625 instead of a fixed 5,400), not a random sample of personas --
mixing them with Condition A's full, complete sample per topic distorted
the topic-specific coefficients for exactly these two models. This version
restricts to response_condition=='A_forced' only, matching Section 7's
actual scope and the precedent already set in step 8. 'response_condition'
is dropped from the formula entirely (constant after the restriction, not
just auto-dropped per-cell as before).

Refits step 5a's formula, minus 'topic' (now the stratifying variable) and
minus 'response_condition' (now fixed at A_forced):

    rating ~ gender + country + profession + age

...separately within each of the 7 topics, for each of the 4 main models
(llama, gemma, qwen, ministral -- deepseek excluded, same n=63 sparsity
justification as Fix 4). Strict-valid numeric ratings only, with
persona-clustered SEs (Fix 2), reported alongside the original HC3 SEs.

VALIDATION: before fitting anything, confirms every (model, topic) cell has
exactly 5,400 Condition-A rows (the full design sample -- 5,400 personas x
1 row per topic under A), BEFORE any strict_is_valid filtering. This is the
same fixed-sample-size property step 8 already confirmed for its Condition-A
scope; if it doesn't hold here too, something is wrong with the restriction.

ADDED (this revision): Section 7 requires topic-specific models for BOTH the
primary rating models above AND the abstention models -- this was flagged as
missing in the last audit. For qwen and ministral (the only models with
Condition-B abstention variance at all, per step 6), fits
answered ~ gender + country + profession + age separately within each topic,
Condition B rows, persona-clustered SEs. Many topics are fully or
near-fully deterministic (0% or ~100% answered) and structurally unfittable
via logistic regression regardless of clustering (complete separation is a
property of the data, not the SE calculation) -- those are detected up
front and reported as a structural fact with a cluster-robust CI on the
rate (clustered_proportion_ci, the same approach used for 07b's H3
confound-check deterministic cells), not forced through a solver.

No post-hoc topic groupings (e.g. "identity-related" vs. not) are
introduced -- all 7 topics are reported individually, exactly as Section 7
requires.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from _common import load_master, cast_formula_dtypes, clustered_proportion_ci, TABLES_DIR

INCLUDED_MODELS = ["llama", "gemma", "qwen", "ministral"]
RATING_FACTORS = ["gender", "country", "profession", "age"]
ABSTENTION_MODELS = ["qwen", "ministral"]  # only models with any Condition-B abstention variance
ABSTENTION_FACTORS = ["gender", "country", "profession", "age"]
TOPIC_ORDER = ["climate change", "economic redistribution", "gender equality",
               "immigration", "lgbtq rights", "religion and secularism", "trust in government"]
EXPECTED_N_PER_CELL = 5400


def validate_fixed_sample_size(df):
    print(f"\n{'='*78}\nVALIDATION: every (model, topic) cell has exactly {EXPECTED_N_PER_CELL:,} "
          f"Condition-A rows\n(before strict_is_valid filtering -- confirms the same fixed-sample-size")
    print("property step 8 already confirmed, now that this script also restricts to Condition A)")
    print("=" * 78)
    a = df[df["response_condition"] == "A_forced"]
    all_ok = True
    for model in INCLUDED_MODELS:
        sub = a[a["model"] == model]
        counts = sub.groupby("topic", observed=True).size().reindex(TOPIC_ORDER)
        bad = counts[counts != EXPECTED_N_PER_CELL]
        status = "OK" if len(bad) == 0 else f"MISMATCH: {bad.to_dict()}"
        if len(bad):
            all_ok = False
        print(f"  {model:10s}: {status}" + ("" if len(bad) else f"  (all 7 topics = {EXPECTED_N_PER_CELL:,})"))
    print(f"  -> {'PASS' if all_ok else 'FAIL'}: fixed-sample-size property "
          f"{'holds' if all_ok else 'does NOT hold'} for the rating models below.")
    return all_ok


def valid_subset(df, model, topic):
    return df[(df["model"] == model) & (df["topic"] == topic) & (df["response_condition"] == "A_forced") &
              df["strict_is_valid"] & df["rating_numeric"].notnull()]


def fit_topic_model(df, model, topic):
    sub = valid_subset(df, model, topic)
    n = len(sub)

    # Check for a zero-variance OUTCOME before fitting anything -- this happened for
    # ministral/gender equality in the original (A+B-pooled) version. Re-verified below whether
    # it still occurs once restricted to Condition A only.
    if sub["rating_numeric"].nunique() <= 1:
        val = sub["rating_numeric"].iloc[0] if n else float("nan")
        print(f"    DEGENERATE: {model}/{topic} -- rating_numeric is CONSTANT (value={val}, n={n:,}) "
              f"under Condition A. No variance to model; skipping fit rather than reporting meaningless "
              f"coefficients or R^2.")
        return pd.DataFrame([{"model": model, "topic": topic, "term": "DEGENERATE_NO_VARIANCE",
                               "coef": val, "se_hc3": np.nan, "p_hc3": np.nan,
                               "se_cluster": np.nan, "p_cluster": np.nan,
                               "ci_low_cluster": np.nan, "ci_high_cluster": np.nan}]), np.nan, n

    dropped = [f for f in RATING_FACTORS if sub[f].nunique() <= 1]
    factors = [f for f in RATING_FACTORS if f not in dropped]
    formula = "rating_numeric ~ " + " + ".join(f"C({f})" for f in factors)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_hc3 = smf.ols(formula, data=sub).fit(cov_type="HC3")
        for w in caught:
            print(f"    WARNING (HC3, {model}/{topic}): {w.message}")
    res_cluster = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})

    ci = res_cluster.conf_int()
    ct = pd.DataFrame({
        "model": model, "topic": topic, "term": res_hc3.params.index,
        "coef": res_hc3.params.values,
        "se_hc3": res_hc3.bse.values, "p_hc3": res_hc3.pvalues.values,
        "se_cluster": res_cluster.bse.values, "p_cluster": res_cluster.pvalues.values,
        "ci_low_cluster": ci[0].values, "ci_high_cluster": ci[1].values,
    })
    if dropped:
        print(f"    NOTE: dropped zero-variance factor(s) for {model}/{topic}: {dropped}")
    return ct, res_cluster.rsquared, n


def fit_topic_abstention_model(df, model, topic):
    """answered ~ gender + country + profession + age, Condition B, within one
    topic, persona-clustered SE. Handles two levels of determinism:
      1. overall answered rate is exactly 0% or 100% -> no regression possible
         at all (zero outcome variance); report the rate + cluster-robust CI.
      2. overall rate has some variance, but the fit still fails to converge
         (near-certain given up to 30 profession / 20 country dummies against
         a small answered subset) -> report convergence failure plainly and
         fall back to the same rate + CI, rather than presenting exploded or
         non-finite coefficients.
    """
    sub = df[(df["model"] == model) & (df["topic"] == topic) & (df["response_condition"] == "B_optional")].copy()
    sub["answered"] = (sub["strict_is_valid"] & sub["rating_numeric"].notnull()).astype(int)
    n_total = len(sub)
    n_answered = int(sub["answered"].sum())
    rate = n_answered / n_total if n_total else float("nan")

    result = {"model": model, "topic": topic, "n_total": n_total, "n_answered": n_answered,
              "answered_rate": rate, "status": None, "ci_low": np.nan, "ci_high": np.nan,
              "n_clusters": sub["persona_id"].nunique()}

    if rate in (0.0, 1.0):
        pci = clustered_proportion_ci(sub, "answered", "persona_id")
        print(f"    {topic:28s} DETERMINISTIC: answered_rate={rate:.4f} ({n_answered}/{n_total}) -- "
              f"no logistic coefficient possible (zero outcome variance, not a fitting failure).")
        print(f"      cluster-robust CI on the rate: [{pci['ci_low']:.4f}, {pci['ci_high']:.4f}] "
              f"(n_persona_clusters={pci['n_clusters']:,})")
        result.update(status="deterministic", ci_low=pci["ci_low"], ci_high=pci["ci_high"])
        return result

    # Drop any factor with a fully deterministic level within this topic (mirrors step 6's
    # per-topic factor-dropping logic) before attempting the fit.
    dropped = {}
    factors = []
    for f in ABSTENTION_FACTORS:
        level_rates = sub.groupby(f, observed=True)["answered"].mean()
        bad_levels = level_rates[(level_rates == 0) | (level_rates == 1)]
        if len(bad_levels):
            dropped[f] = list(bad_levels.index)
        else:
            factors.append(f)
    if dropped:
        print(f"    {topic:28s} WARNING: factor(s) with a deterministic level within this topic, dropped "
              f"from formula: { {k: v[:3] for k, v in dropped.items()} }")

    if not factors:
        pci = clustered_proportion_ci(sub, "answered", "persona_id")
        print(f"    {topic:28s} all factors dropped (every one has a deterministic level) -- "
              f"falling back to overall rate: {rate:.4f} ({n_answered}/{n_total}), "
              f"CI=[{pci['ci_low']:.4f}, {pci['ci_high']:.4f}]")
        result.update(status="all_factors_dropped", ci_low=pci["ci_low"], ci_high=pci["ci_high"])
        return result

    formula = "answered ~ " + " + ".join(f"C({f})" for f in factors)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.logit(formula, data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["persona_id"]}, disp=0, maxiter=200)
        converged = bool(res.mle_retvals.get("converged", False))
        max_se = float(np.nanmax(res.bse.values)) if len(res.bse) else float("nan")
        if not converged or max_se > 50:
            pci = clustered_proportion_ci(sub, "answered", "persona_id")
            print(f"    {topic:28s} FIT DID NOT CONVERGE (converged={converged}, max_se={max_se:.1f}) -- "
                  f"not interpretable. Falling back to overall rate: {rate:.4f} ({n_answered}/{n_total}), "
                  f"CI=[{pci['ci_low']:.4f}, {pci['ci_high']:.4f}]")
            result.update(status="did_not_converge", ci_low=pci["ci_low"], ci_high=pci["ci_high"])
            return result
        print(f"    {topic:28s} converged: rate={rate:.4f} ({n_answered}/{n_total}), "
              f"n_params={len(res.params)}, pseudo_R2={res.prsquared:.4f}, "
              f"n_persona_clusters={sub['persona_id'].nunique():,}")
        result.update(status="converged", pseudo_r2=res.prsquared, n_params=len(res.params))
        ci = res.conf_int()
        coef_table = pd.DataFrame({
            "model": model, "topic": topic, "term": res.params.index,
            "coef": res.params.values, "se_cluster": res.bse.values, "p_cluster": res.pvalues.values,
            "ci_low": ci[0].values, "ci_high": ci[1].values,
        })
        return result, coef_table
    except Exception as e:
        pci = clustered_proportion_ci(sub, "answered", "persona_id")
        print(f"    {topic:28s} FIT FAILED ({type(e).__name__}: {e}) -- falling back to overall rate: "
              f"{rate:.4f} ({n_answered}/{n_total}), CI=[{pci['ci_low']:.4f}, {pci['ci_high']:.4f}]")
        result.update(status="fit_failed", ci_low=pci["ci_low"], ci_high=pci["ci_high"])
        return result


def main():
    df = cast_formula_dtypes(load_master())

    validate_fixed_sample_size(df)

    all_tables = []
    r2_summary = []

    print(f"\n{'='*78}\nPART 1: TOPIC-SPECIFIC PRIMARY RATING MODELS (Condition A only)\n{'='*78}")
    for model in INCLUDED_MODELS:
        print(f"\n{'='*78}\nMODEL: {model}\n{'='*78}")
        for topic in TOPIC_ORDER:
            ct, r2, n = fit_topic_model(df, model, topic)
            all_tables.append(ct)
            r2_summary.append({"model": model, "topic": topic, "r2": r2, "n": n})
            print(f"  {topic:28s} n={n:,}  R2={r2:.4f}")

    combined = pd.concat(all_tables, ignore_index=True)
    out_path = f"{TABLES_DIR}/topic_specific_models.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(combined)} coefficient rows across {len(INCLUDED_MODELS)}x7 model-topic fits)")

    print(f"\n{'='*78}\nGENDER EFFECT BY TOPIC (coefficient relative to female, persona-clustered SE)\n{'='*78}")
    for model in INCLUDED_MODELS:
        print(f"\n  {model}:")
        sub = combined[(combined["model"] == model) & combined["term"].str.startswith("C(gender)")]
        piv = sub.pivot_table(index="topic", columns="term", values="coef").reindex(TOPIC_ORDER)
        print(piv.to_string(float_format=lambda x: f"{x:8.4f}"))
        male_col = [c for c in piv.columns if "male]" in c and "neutral" not in c]
        if male_col:
            vals = piv[male_col[0]].dropna()
            print(f"    range across topics: [{vals.min():.4f}, {vals.max():.4f}]  (spread={vals.max()-vals.min():.4f})")

    print(f"\n{'='*78}\nPROFESSION EFFECT SPREAD BY TOPIC (max-min coefficient across the 30 professions)\n{'='*78}")
    prof_spread_rows = []
    for model in INCLUDED_MODELS:
        for topic in TOPIC_ORDER:
            sub = combined[(combined["model"] == model) & (combined["topic"] == topic) &
                            combined["term"].str.startswith("C(profession)")]
            if len(sub):
                spread = sub["coef"].max() - sub["coef"].min()
                prof_spread_rows.append({"model": model, "topic": topic, "profession_coef_spread": spread})
    prof_spread = pd.DataFrame(prof_spread_rows)
    for model in INCLUDED_MODELS:
        sub = prof_spread[prof_spread["model"] == model].set_index("topic").reindex(TOPIC_ORDER)
        print(f"\n  {model}:")
        print(sub[["profession_coef_spread"]].to_string(float_format=lambda x: f"{x:8.4f}"))
        vals = sub["profession_coef_spread"].dropna()
        cv = vals.std() / vals.mean() if vals.mean() else np.nan
        print(f"    across-topic spread of the spread: mean={vals.mean():.4f}, sd={vals.std():.4f}, "
              f"coefficient of variation={cv:.3f}  {'(fairly stable)' if cv < 0.3 else '(varies meaningfully by topic)'}")

    prof_spread.to_csv(f"{TABLES_DIR}/topic_specific_profession_spread.csv", index=False)

    r2_df = pd.DataFrame(r2_summary)
    r2_df.to_csv(f"{TABLES_DIR}/topic_specific_r2.csv", index=False)
    print(f"\n{'='*78}\nMODEL FIT (R^2) BY TOPIC\n{'='*78}")
    piv_r2 = r2_df.pivot_table(index="topic", columns="model", values="r2").reindex(TOPIC_ORDER)[INCLUDED_MODELS]
    print(piv_r2.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n{'='*78}\nPART 2: TOPIC-SPECIFIC ABSTENTION MODELS (Section 7, Condition B, qwen + ministral)\n{'='*78}")
    abstention_summary = []
    abstention_coefs = []
    for model in ABSTENTION_MODELS:
        print(f"\n{'='*78}\nMODEL: {model}\n{'='*78}")
        for topic in TOPIC_ORDER:
            out = fit_topic_abstention_model(df, model, topic)
            if isinstance(out, tuple):
                result, coef_table = out
                abstention_coefs.append(coef_table)
            else:
                result = out
            abstention_summary.append(result)

    abst_summary_df = pd.DataFrame(abstention_summary)
    abst_summary_df.to_csv(f"{TABLES_DIR}/topic_specific_abstention_summary.csv", index=False)
    print(f"\nWrote {TABLES_DIR}/topic_specific_abstention_summary.csv")
    if abstention_coefs:
        pd.concat(abstention_coefs, ignore_index=True).to_csv(
            f"{TABLES_DIR}/topic_specific_abstention_coefficients.csv", index=False)
        print(f"Wrote {TABLES_DIR}/topic_specific_abstention_coefficients.csv "
              f"({sum(len(c) for c in abstention_coefs)} coefficient rows)")

    print(f"\n{'='*78}\nSUMMARY: topic-specific abstention model status, by model\n{'='*78}")
    for model in ABSTENTION_MODELS:
        sub = abst_summary_df[abst_summary_df["model"] == model]
        print(f"\n  {model}:")
        print(sub[["topic", "answered_rate", "n_answered", "n_total", "status"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
        n_det = (sub["status"] == "deterministic").sum()
        n_conv = (sub["status"] == "converged").sum()
        n_fail = len(sub) - n_det - n_conv
        print(f"    {n_det}/7 topics fully deterministic (no model possible), {n_conv}/7 converged, "
              f"{n_fail}/7 other (dropped-all-factors or non-convergence, see status column)")


if __name__ == "__main__":
    main()
