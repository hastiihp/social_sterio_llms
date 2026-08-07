"""Stage 7: the unified variance-decomposition model.

Single consolidated script (statsmodels MixedLM results don't survive
cross-process pickling cleanly -- patsy's lazy design-matrix rebuild on
unpickle needs an eval_env stack frame that doesn't exist in a fresh
process, confirmed by hitting this directly on the first attempt here).
Everything -- fitting, diagnostics, R^2, the LRT, per-model simple slopes,
BH correction -- happens in one continuous run, writing final CSVs at the
end rather than reloading a saved model object.

rating ~ topic + prompt_type + profession + country + gender + age + model
  + topic:model + prompt_type:model + (1 | persona_id)

Full-scale, Condition A only, all 5 prompt types pooled, 4 models
(llama/gemma/qwen/ministral; DeepSeek excluded per established
near-total-non-compliance precedent, Falcon excluded project-wide).
Reference levels: prompt_type=original, model=llama (both set explicitly;
patsy's alphabetical default would otherwise make "health" the prompt_type
reference, which doesn't match this project's "context minus original"
convention used everywhere else).
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from _fit_helpers import fit_mixedlm_with_fallback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{ROOT}/analysis_unified/output"
PROMPT_TYPES = ["original", "health", "neutral", "positive", "negative_minor"]
NON_REF_PROMPT_TYPES = ["health", "neutral", "positive", "negative_minor"]
MODELS = ["llama", "gemma", "qwen", "ministral"]
NON_REF_MODELS = ["gemma", "qwen", "ministral"]

# Term-list construction throughout (never string-surgery on an assembled formula --
# a first attempt using .replace() on the joined formula string silently mangled
# "topic" and "model" into a nonsensical "model:model" self-interaction, because
# " + C(topic)" is also a substring of " + C(topic):C(model...)"; caught only by
# noticing "minus_topic" and "minus_topic:model" produced suspiciously identical R^2
# and n_params, then confirmed by printing the mangled formula directly).
MODEL_TERM = 'C(model, Treatment(reference="llama"))'
TOPIC_TERM = "C(topic)"
PROMPT_TYPE_TERM = 'C(prompt_type, Treatment(reference="original"))'
PROFESSION_TERM = "C(profession)"
COUNTRY_TERM = "C(country)"
GENDER_TERM = "C(gender)"
AGE_TERM = "C(age)"
TOPIC_MODEL_TERM = f"{TOPIC_TERM}:{MODEL_TERM}"
PROMPT_TYPE_MODEL_TERM = f"{PROMPT_TYPE_TERM}:{MODEL_TERM}"

ALL_TERMS = [TOPIC_TERM, PROMPT_TYPE_TERM, PROFESSION_TERM, COUNTRY_TERM, GENDER_TERM, AGE_TERM,
             MODEL_TERM, TOPIC_MODEL_TERM, PROMPT_TYPE_MODEL_TERM]


def build_formula(terms):
    return "rating ~ " + " + ".join(terms)


FULL_FORMULA = build_formula(ALL_TERMS)
REDUCED_FORMULA = build_formula([t for t in ALL_TERMS if t != PROMPT_TYPE_MODEL_TERM])

# Per-term partial R^2 convention (stated explicitly, not left implicit): for the 7
# "main" terms that participate in a model-interaction (topic, prompt_type, model),
# drop the term TOGETHER WITH its interaction -- "does this factor matter at all,
# whether shared or model-specific". For the two pure interaction terms, drop ONLY
# the interaction, keeping the corresponding main effect -- "does this factor's
# effect specifically differ by model, beyond its shared average effect". Dropping
# a main effect while LEAVING its interaction in place is deliberately never done:
# patsy's full-rank coding for an interaction whose main effect is absent lets the
# interaction alone fully reabsorb the main effect's contribution, so that
# comparison silently tests nothing (confirmed directly -- doing this by accident
# is exactly what the string-surgery bug above produced).
TERM_DROP_FORMULAS = {
    "topic": build_formula([t for t in ALL_TERMS if t not in (TOPIC_TERM, TOPIC_MODEL_TERM)]),
    "prompt_type": build_formula([t for t in ALL_TERMS if t not in (PROMPT_TYPE_TERM, PROMPT_TYPE_MODEL_TERM)]),
    "profession": build_formula([t for t in ALL_TERMS if t != PROFESSION_TERM]),
    "country": build_formula([t for t in ALL_TERMS if t != COUNTRY_TERM]),
    "gender": build_formula([t for t in ALL_TERMS if t != GENDER_TERM]),
    "age": build_formula([t for t in ALL_TERMS if t != AGE_TERM]),
    "model": build_formula([t for t in ALL_TERMS if t not in (MODEL_TERM, TOPIC_MODEL_TERM, PROMPT_TYPE_MODEL_TERM)]),
    "topic:model": build_formula([t for t in ALL_TERMS if t != TOPIC_MODEL_TERM]),
    "prompt_type:model": REDUCED_FORMULA,
}


def load_pooled():
    frames = []
    for pt in PROMPT_TYPES:
        for m in MODELS:
            src = f"{ROOT}/results/results_{pt}_{m}.csv"
            df = pd.read_csv(src, low_memory=False,
                              usecols=["persona_id", "country", "gender", "age", "profession", "topic",
                                       "response_condition", "strict_parsed_rating", "strict_is_valid"])
            sub = df[(df["response_condition"] == "A_forced") & (df["strict_is_valid"])].copy()
            assert len(sub) == 37800, f"{pt}/{m}: expected 37800 Condition-A strict-valid rows, got {len(sub)}"
            sub["prompt_type"] = pt
            sub["model"] = m
            frames.append(sub)
    full = pd.concat(frames, ignore_index=True)
    assert len(full) == 756000, f"expected 756000 pooled rows, got {len(full)}"
    assert full["persona_id"].nunique() == 5400
    full["rating"] = full["strict_parsed_rating"].astype(float)
    for c in ["topic", "prompt_type", "profession", "country", "gender", "age", "model"]:
        full[c] = full[c].astype("category")
    return full


def _scalar(x):
    """Robustly pull a single Python float out of whatever shape ContrastResults
    hands back (observed: 0-d, (1,), and (1,1) all occur across effect/sd/pvalue/
    conf_int for a single-row contrast -- statsmodels is not consistent here)."""
    return float(np.asarray(x).reshape(-1)[0])


def contrast_matrix(term_lists, fe_index):
    """Build a numeric r_matrix for MixedLMResults.t_test/wald_test, which (unlike
    OLS's t_test/wald_test) require an array, not a string formula -- confirmed by
    hitting AttributeError on a string directly. term_lists is a list of lists of
    parameter names; each inner list becomes one row (summed if it has >1 name, for
    a linear-combination contrast like "main effect + interaction")."""
    fe_list = list(fe_index)
    k_fe = len(fe_list)
    mat = np.zeros((len(term_lists), k_fe))
    for i, names in enumerate(term_lists):
        for name in names:
            mat[i, fe_list.index(name)] = 1.0
    return mat


def fit_ols(formula, df, label):
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = smf.ols(formula, data=df).fit()
        for w in caught:
            print(f"  WARNING (OLS, {label}): {w.message}", flush=True)
    print(f"  OLS ({label}): fit in {time.time()-t0:.1f}s, R2={res.rsquared:.4f}, n_params={len(res.params)}", flush=True)
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78, flush=True)
    print("Loading pooled dataset...", flush=True)
    df = load_pooled()
    print(f"Pooled dataset: {len(df):,} rows, {df['persona_id'].nunique():,} personas, "
          f"{df['topic'].nunique()} topics, {df['prompt_type'].nunique()} prompt types, "
          f"{df['model'].nunique()} models", flush=True)

    # ------------------------------------------------------------------
    # STEP 1: primary REML mixed model (headline coefficients, CIs, R^2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("STEP 1: primary REML mixed model", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    res_reml, status_reml = fit_mixedlm_with_fallback(FULL_FORMULA, df, df["persona_id"], "primary_REML", reml=True)
    reml_seconds = time.time() - t0
    if res_reml is None:
        raise RuntimeError(f"Primary REML model failed to converge: {status_reml}. Stopping.")
    print(f"Primary REML fit: {reml_seconds:.1f}s, status={status_reml}, llf={res_reml.llf:.4f}", flush=True)

    re_var = res_reml.cov_re.iloc[0, 0]
    resid_var = res_reml.scale
    icc = re_var / (re_var + resid_var)
    print(f"Random-intercept (persona_id) variance: {re_var:.6f}", flush=True)
    print(f"Residual variance: {resid_var:.6f}", flush=True)
    print(f"ICC (persona-level share of residual+random variance): {icc:.6f}", flush=True)

    # marginal / conditional R^2 (Nakagawa & Schielzeth) -- fittedvalues from MixedLM
    # already includes BLUPs, so the fixed-effects-only prediction is built directly
    # from exog @ fe_params rather than backed out of fittedvalues.
    exog = res_reml.model.exog
    fe_params = res_reml.fe_params.values
    fixed_only_pred = exog @ fe_params
    var_fixed = np.var(fixed_only_pred, ddof=1)
    r2_marginal = var_fixed / (var_fixed + re_var + resid_var)
    r2_conditional = (var_fixed + re_var) / (var_fixed + re_var + resid_var)
    print(f"Marginal R^2 (fixed effects only): {r2_marginal:.4f}", flush=True)
    print(f"Conditional R^2 (fixed + random): {r2_conditional:.4f}", flush=True)

    # ------------------------------------------------------------------
    # STEP 2: OLS baseline (for per-term partial R^2, matching analysis/08's
    # established SSE-reduction method) + sanity check vs. REML coefficients
    # ------------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("STEP 2: OLS baseline + per-term partial R^2 (SSE-reduction, matching analysis/08)", flush=True)
    print("=" * 78, flush=True)
    res_ols_full = fit_ols(FULL_FORMULA, df, "full")
    sse_full = float((res_ols_full.resid ** 2).sum())

    partial_r2 = {}
    partial_r2_n_params = {}
    for term, dropped_formula in TERM_DROP_FORMULAS.items():
        res_reduced = fit_ols(dropped_formula, df, f"minus_{term}")
        sse_reduced = float((res_reduced.resid ** 2).sum())
        partial_r2[term] = (sse_reduced - sse_full) / sse_reduced
        partial_r2_n_params[term] = len(res_ols_full.params) - len(res_reduced.params)

    print("\nPartial R^2 per term (OLS SSE-reduction):", flush=True)
    for term, v in sorted(partial_r2.items(), key=lambda x: -x[1]):
        print(f"  {term:20s} {v:.4f}", flush=True)

    # sanity check: compare OLS vs REML point estimates for a few coefficients
    print("\nSanity check: OLS vs REML point-estimate agreement (first 10 shared terms):", flush=True)
    shared_terms = [t for t in res_ols_full.params.index if t in res_reml.params.index][:10]
    for t in shared_terms:
        print(f"  {t[:60]:60s} OLS={res_ols_full.params[t]:.4f}  REML={res_reml.params[t]:.4f}", flush=True)

    # ------------------------------------------------------------------
    # STEP 3: LRT for prompt_type:model interaction (ML, not REML)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("STEP 3: LRT for C(prompt_type):C(model) interaction (ML fits)", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    res_ml_full, status_ml_full = fit_mixedlm_with_fallback(FULL_FORMULA, df, df["persona_id"], "ML_full", reml=False)
    ml_full_seconds = time.time() - t0
    if res_ml_full is None:
        raise RuntimeError(f"ML full model failed to converge: {status_ml_full}")
    print(f"ML full fit: {ml_full_seconds:.1f}s, llf={res_ml_full.llf:.4f}", flush=True)

    t0 = time.time()
    res_ml_reduced, status_ml_reduced = fit_mixedlm_with_fallback(REDUCED_FORMULA, df, df["persona_id"], "ML_reduced", reml=False)
    ml_reduced_seconds = time.time() - t0
    if res_ml_reduced is None:
        raise RuntimeError(f"ML reduced model failed to converge: {status_ml_reduced}")
    print(f"ML reduced fit: {ml_reduced_seconds:.1f}s, llf={res_ml_reduced.llf:.4f}", flush=True)

    df_diff = len(res_ml_full.fe_params) - len(res_ml_reduced.fe_params)
    lrt_stat = 2 * (res_ml_full.llf - res_ml_reduced.llf)
    lrt_p = stats.chi2.sf(lrt_stat, df_diff)
    print(f"\nLRT: chi2={lrt_stat:.4f}, df={df_diff}, p={lrt_p:.6g}", flush=True)

    # ------------------------------------------------------------------
    # STEP 4: per-model simple slopes for prompt_type (from primary REML model)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("STEP 4: per-model simple slopes for prompt_type (vs. original)", flush=True)
    print("=" * 78, flush=True)
    print("Available parameter names (sample):", flush=True)
    pt_terms = [p for p in res_reml.params.index if "prompt_type" in p]
    for p in pt_terms:
        print(f"  {p}", flush=True)

    fe_index = res_reml.fe_params.index
    slope_rows = []
    for pt in NON_REF_PROMPT_TYPES:
        main_term = f'C(prompt_type, Treatment(reference="original"))[T.{pt}]'
        # llama (reference model): just the main effect
        r = contrast_matrix([[main_term]], fe_index)
        tt = res_reml.t_test(r)
        ci = np.asarray(tt.conf_int()).reshape(-1)
        slope_rows.append({"model": "llama", "prompt_type": pt, "estimate": _scalar(tt.effect),
                            "se": _scalar(tt.sd), "ci_low": ci[0],
                            "ci_high": ci[1], "p_value": _scalar(tt.pvalue)})
        for m in NON_REF_MODELS:
            inter_term = f'{main_term}:C(model, Treatment(reference="llama"))[T.{m}]'
            r = contrast_matrix([[main_term, inter_term]], fe_index)
            tt = res_reml.t_test(r)
            ci = np.asarray(tt.conf_int()).reshape(-1)
            slope_rows.append({"model": m, "prompt_type": pt, "estimate": _scalar(tt.effect),
                                "se": _scalar(tt.sd), "ci_low": ci[0],
                                "ci_high": ci[1], "p_value": _scalar(tt.pvalue)})
    slopes_df = pd.DataFrame(slope_rows)
    print(slopes_df.to_string(index=False), flush=True)

    # ------------------------------------------------------------------
    # STEP 5: BH correction across the full predefined family
    # (9 term-level tests from the primary model's joint significance +
    # the 1 LRT test + the 16 simple-slope tests = 26 p-values, one family)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("STEP 5: assembling term-level table + BH correction", flush=True)
    print("=" * 78, flush=True)

    # joint Wald test per term from the REML model (multi-df, for the headline table's p-value)
    term_wald_p = {}
    is_topic_main = lambda p: p.startswith("C(topic)[") and ":" not in p
    is_topic_inter = lambda p: p.startswith("C(topic)[") and ":C(model" in p
    is_pt_main = lambda p: p.startswith('C(prompt_type,') and ":" not in p
    is_pt_inter = lambda p: p.startswith('C(prompt_type,') and ":C(model" in p
    is_model_main = lambda p: p.startswith('C(model,') and ":" not in p
    # Same joint-scope convention as the OLS partial-R^2 drop above (stated there):
    # topic/prompt_type/model are tested TOGETHER WITH their interaction (matching
    # what "drop this term" actually removed in Step 2), so the Wald p-value reported
    # alongside each partial R^2 tests the same null hypothesis the R^2 measures --
    # not a narrower main-effect-only test that would silently answer a different
    # question next to a headline number that looks like it matches.
    wald_groups = {
        "topic": [p for p in res_reml.params.index if is_topic_main(p) or is_topic_inter(p)],
        "prompt_type": [p for p in res_reml.params.index if is_pt_main(p) or is_pt_inter(p)],
        "profession": [p for p in res_reml.params.index if p.startswith("C(profession)[") and ":" not in p],
        "country": [p for p in res_reml.params.index if p.startswith("C(country)[") and ":" not in p],
        "gender": [p for p in res_reml.params.index if p.startswith("C(gender)[") and ":" not in p],
        "age": [p for p in res_reml.params.index if p.startswith("C(age)[") and ":" not in p],
        "model": [p for p in res_reml.params.index if is_model_main(p) or is_topic_inter(p) or is_pt_inter(p)],
        "topic:model": [p for p in res_reml.params.index if is_topic_inter(p)],
        "prompt_type:model": [p for p in res_reml.params.index if is_pt_inter(p)],
    }
    # unlike t_test (which pads with zeros for the random-effects columns itself),
    # wald_test is inherited unmodified from the base LikelihoodModelResults and
    # expects r_matrix to match len(res.params) (fe params + "Group Var"), not just
    # k_fe -- confirmed by hitting a patsy "wrong shape for coefs" error otherwise.
    n_extra = len(res_reml.params) - len(fe_index)
    for term, params in wald_groups.items():
        assert len(params) > 0, f"no matching params found for term {term}"
        r = contrast_matrix([[p] for p in params], fe_index)
        r = np.hstack([r, np.zeros((r.shape[0], n_extra))])
        wt = res_reml.wald_test(r, scalar=True)
        term_wald_p[term] = _scalar(wt.pvalue)
        print(f"  {term:20s} n_params={len(params):3d}  wald_chi2={_scalar(wt.statistic):.2f}  p={term_wald_p[term]:.4g}", flush=True)

    # cross-check: the OLS reduced-model parameter-count drop should match the Wald
    # test's degrees of freedom exactly, for every term -- an independent check that
    # the two term-drop constructions (OLS refit vs. REML contrast grouping) agree on
    # what each term actually is. This is exactly the kind of mismatch the earlier
    # string-surgery bug would have produced.
    for t in wald_groups:
        assert partial_r2_n_params[t] == len(wald_groups[t]), (
            f"{t}: OLS param-count drop ({partial_r2_n_params[t]}) != Wald test df ({len(wald_groups[t])}) "
            f"-- term construction disagrees between the two methods, investigate before trusting either.")
    print("\nCross-check passed: OLS param-count drop matches Wald-test df for every term.", flush=True)

    main_table = pd.DataFrame([
        {"term": t, "partial_r2_ols": partial_r2[t], "wald_p_raw": term_wald_p[t], "n_params": len(wald_groups[t])}
        for t in wald_groups
    ])
    main_table["raw_p_family"] = "primary_model_terms"

    lrt_row = pd.DataFrame([{"term": "LRT: prompt_type:model interaction", "partial_r2_ols": np.nan,
                              "wald_p_raw": lrt_p, "n_params": df_diff, "raw_p_family": "lrt_test"}])

    slopes_df["raw_p_family"] = "simple_slopes"
    slopes_pvals = slopes_df.rename(columns={"p_value": "wald_p_raw"})[["model", "prompt_type", "wald_p_raw", "raw_p_family"]]

    all_p_rows = pd.concat([
        main_table[["term", "wald_p_raw", "raw_p_family"]].assign(model=np.nan, prompt_type=np.nan),
        lrt_row[["term", "wald_p_raw", "raw_p_family"]].assign(model=np.nan, prompt_type=np.nan),
        slopes_pvals.assign(term=np.nan),
    ], ignore_index=True)

    reject, p_bh, _, _ = multipletests(all_p_rows["wald_p_raw"].values, method="fdr_bh")
    all_p_rows["p_bh"] = p_bh
    all_p_rows["significant_bh_0.05"] = reject
    print(f"\nBH correction applied across {len(all_p_rows)} p-values (one family: primary model's 9 "
          f"term-level tests + 1 LRT + 16 simple-slope tests).", flush=True)
    print(f"Significant before correction (raw p<0.05): {(all_p_rows['wald_p_raw'] < 0.05).sum()}/{len(all_p_rows)}", flush=True)
    print(f"Significant after BH correction (p_bh<0.05): {reject.sum()}/{len(all_p_rows)}", flush=True)

    # ------------------------------------------------------------------
    # WRITE OUTPUTS
    # ------------------------------------------------------------------
    main_table = main_table.merge(all_p_rows[all_p_rows["raw_p_family"] == "primary_model_terms"][["term", "p_bh", "significant_bh_0.05"]], on="term")
    main_table["r2_marginal_model"] = r2_marginal
    main_table["r2_conditional_model"] = r2_conditional
    main_table["icc_persona"] = icc
    main_table_path = f"{OUT_DIR}/variance_decomposition_model.csv"
    main_table.to_csv(main_table_path, index=False)
    print(f"\nWrote {main_table_path}", flush=True)

    slopes_out = slopes_df.merge(
        all_p_rows[all_p_rows["raw_p_family"] == "simple_slopes"][["model", "prompt_type", "p_bh", "significant_bh_0.05"]],
        on=["model", "prompt_type"])
    slopes_out["lrt_chi2"] = lrt_stat
    slopes_out["lrt_df"] = df_diff
    slopes_out["lrt_p_raw"] = lrt_p
    slopes_out["lrt_p_bh"] = float(all_p_rows[all_p_rows["term"] == "LRT: prompt_type:model interaction"]["p_bh"].iloc[0])
    slopes_path = f"{OUT_DIR}/model_framing_sensitivity_test.csv"
    slopes_out.to_csv(slopes_path, index=False)
    print(f"Wrote {slopes_path}", flush=True)

    diagnostics = pd.DataFrame([{
        "reml_status": status_reml, "reml_converged": res_reml.converged, "reml_llf": res_reml.llf,
        "reml_fit_seconds": reml_seconds,
        "ml_full_status": status_ml_full, "ml_reduced_status": status_ml_reduced,
        "re_variance_persona": re_var, "residual_variance": resid_var, "icc": icc,
        "r2_marginal": r2_marginal, "r2_conditional": r2_conditional,
        "n_obs": len(df), "n_personas": df["persona_id"].nunique(), "n_fixed_params": len(res_reml.fe_params),
    }])
    diag_path = f"{OUT_DIR}/model_diagnostics.csv"
    diagnostics.to_csv(diag_path, index=False)
    print(f"Wrote {diag_path}", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
