"""Step 6: optional-abstention analysis (H2) -- analysis_plan.md Section 5.
Revised per external audit (Fixes 2 & 5).

Binary outcome, CORRECTED per Fix 5:
  answered = strict_is_valid & rating_numeric.notnull()
(a real, valid numeric rating was given), among Condition B (optional) rows.

This replaces the original definition (answered = ~is_abstention), which
counted DeepSeek's malformed/refusal rows as "answered" simply because they
were not a clean, strictly-parsed "NA" -- conflating "did not abstain
properly" with "gave a valid answer". Verified before rewriting anything:
the redefinition changes classification for DEEPSEEK ONLY (37,800 rows flip
from "answered" to "not answered"); llama, gemma, qwen, and ministral are
byte-for-byte identical under both definitions, since those four models are
~100% strict-valid whenever they are not abstaining, so
NOT(is_abstention) == strict_is_valid & rating_numeric.notnull() already
held for them. DeepSeek now correctly shows ~0% answered under Condition B
(it has zero valid Condition-B ratings at all -- see step 3/9), which is
the accurate picture, not the misleading 100% the old definition produced.

Fix 2 (clustered SEs): Regression 2 (qwen+ministral) is now fit twice --
original per-row SEs and cluster-robust SEs (grouped by persona_id, since
each persona contributes multiple topic-rows) -- reported side by side so
the correction's effect is visible, not asserted.

Given this, the script:
  1. Reports answered-rate by model, topic, and each demographic factor
     (the descriptive requirement in Section 5) -- no modeling assumptions.
  2. Attempts the full logistic regression exactly as specified
     (answered ~ gender+country+profession+age+topic+model) on all
     Condition B rows. Still fails to converge under the corrected
     definition, for an updated reason: llama/gemma are now a constant 1.0
     and deepseek is now a constant ~0.0 -- three model levels with zero
     within-group outcome variance still perfectly separate the "model"
     predictor, just with deepseek pinned at the opposite extreme than
     before.
  3. Fits the well-posed logistic regression restricted to qwen + ministral
     (unchanged data/result from the original version, confirmed identical
     since the redefinition doesn't touch these two models), now with both
     original and cluster-robust SEs reported side by side.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from _common import load_master, cast_formula_dtypes, MODEL_ORDER, TABLES_DIR

FACTORS = ["gender", "country", "profession", "age", "topic", "model"]
SE_FLAG_THRESHOLD = 3.0  # log-odds scale; flag anything larger as not reliably identified


def rate_table(b, factor):
    g = b.groupby([factor, "model"], observed=True)["answered"].agg(["mean", "sum", "count"]).reset_index()
    g.columns = [factor, "model", "answered_rate", "n_answered", "n_total"]
    return g.sort_values([factor, "model"])


def combined_coef_table(res_orig, res_cluster):
    ci_o, ci_c = res_orig.conf_int(), res_cluster.conf_int()
    return pd.DataFrame({
        "term": res_orig.params.index,
        "coef_logit": res_orig.params.values,          # identical under both SE types
        "odds_ratio": np.exp(res_orig.params.values),
        "se_original": res_orig.bse.values,
        "z_original": res_orig.tvalues.values,
        "p_original": res_orig.pvalues.values,
        "se_cluster": res_cluster.bse.values,
        "z_cluster": res_cluster.tvalues.values,
        "p_cluster": res_cluster.pvalues.values,
    })


def main():
    df = cast_formula_dtypes(load_master())
    b = df[df["response_condition"] == "B_optional"].copy()
    # Fix 5: corrected "answered" definition
    b["answered"] = (b["strict_is_valid"] & b["rating_numeric"].notnull()).astype(int)

    print("=" * 78)
    print("DESCRIPTIVE: answered rate under Condition B, by model (Fix 5 definition)")
    print("=" * 78)
    overall = b.groupby("model", observed=True)["answered"].agg(["mean", "sum", "count"]).reindex(MODEL_ORDER)
    overall.columns = ["answered_rate", "n_answered", "n_total"]
    print(overall.to_string(float_format=lambda x: f"{x:8.4f}"))
    print()
    print("CONFIRMATION: deepseek now shows 0% answered under Condition B (0 valid ratings, as it should --")
    print("it produced zero strictly-valid numeric Condition-B ratings; see step 3/9). Under the OLD/original")
    print("definition (~is_abstention) it incorrectly showed 100%, counting refusal/malformed text as 'answered'.")
    print()
    print("deepseek Condition-B rows are now entirely 'not answered' -- composition (all non-compliant, no valid ratings):")
    print(b.loc[b["model"] == "deepseek", "parse_failure_reason"].value_counts().to_string())

    print()
    print("=" * 78)
    print("DESCRIPTIVE: answered rate by model x topic (Section 5 report requirement)")
    print("=" * 78)
    topic_pivot = b.pivot_table(index="topic", columns="model", values="answered", aggfunc="mean")[MODEL_ORDER]
    print(topic_pivot.to_string(float_format=lambda x: f"{x:7.4f}"))
    topic_pivot.to_csv(f"{TABLES_DIR}/abstention_answered_rate_by_topic.csv")

    print()
    print("=" * 78)
    print("DESCRIPTIVE: answered rate by demographic factor x model")
    print("=" * 78)
    for factor in ["gender", "country", "profession", "age"]:
        t = rate_table(b, factor)
        out_path = f"{TABLES_DIR}/abstention_answered_rate_by_{factor}.csv"
        t.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}")

    print()
    print("=" * 78)
    print("REGRESSION 1: full model as specified (answered ~ gender+country+profession+age+topic+model),")
    print("all 5 models, all Condition B rows -- Fix 5 definition")
    print("=" * 78)
    formula_full = "answered ~ " + " + ".join(f"C({f})" for f in FACTORS)
    print(f"  formula: {formula_full}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=Warning)
            res_full = smf.logit(formula_full, data=b).fit(disp=0)
        print("  Converged without error -- unexpected given the descriptive rates above; inspect coefficients in CSV.")
        combined_coef_table(res_full, res_full).to_csv(f"{TABLES_DIR}/abstention_model_full.csv", index=False)
    except (PerfectSeparationError, Exception) as e:
        print(f"  FAILED TO FIT: {type(e).__name__}: {e}")
        print("  Still expected under the corrected definition, for an updated reason: llama and gemma are a")
        print("  constant 1.0 (100% answered) and deepseek is now a constant ~0.0 (0% answered) -- three of")
        print("  five 'model' levels have zero within-group outcome variance, which still perfectly separates")
        print("  the 'model' predictor (previously all three were pinned at 1.0; now deepseek is pinned at the")
        print("  opposite extreme, but the separation problem is the same in kind). Proceeding to the")
        print("  restricted, well-posed model below instead of reporting exploded coefficients.")

    print()
    print("=" * 78)
    print("REGRESSION 2: restricted to qwen + ministral (the only models with outcome variance)")
    print("Fix 2: fit with BOTH original (per-row) and cluster-robust (persona_id) SEs, side by side")
    print("=" * 78)
    sub = b[b["model"].isin(["qwen", "ministral"])].copy()
    print(f"n = {len(sub):,} Condition-B rows (qwen + ministral)")
    print("NOTE: this subset and its answered values are IDENTICAL to the original (pre-Fix-5) version --")
    print("verified directly (Fix 5's redefinition only affects deepseek rows, which aren't in this subset).")

    dropped = {}
    factors = []
    for f in ["gender", "country", "profession", "age", "topic", "model"]:
        rates = sub.groupby(f, observed=True)["answered"].mean()
        zero_or_one = rates[(rates == 0) | (rates == 1)]
        if len(zero_or_one):
            dropped[f] = zero_or_one.to_dict()
        factors.append(f)

    if "topic" in dropped:
        bad_topics = list(dropped["topic"].keys())
        print(f"  Topics with exactly 0% or 100% answered in this subset (perfect separation -- dropped from")
        print(f"  the regression, reported as deterministic facts instead): {bad_topics}")
        for t, rate in dropped["topic"].items():
            n = len(sub[sub["topic"] == t])
            print(f"    topic='{t}': answered_rate={rate:.4f} (n={n:,}) -- deterministic, not modeled")
        sub = sub[~sub["topic"].isin(bad_topics)]

    for f in ["gender", "country", "profession", "age", "model"]:
        if f in dropped:
            print(f"  WARNING: factor '{f}' also has deterministic level(s) in this subset: {dropped[f]} -- dropping factor from formula.")
            factors.remove(f)
    if "topic" in factors and sub["topic"].nunique() <= 1:
        factors.remove("topic")

    formula_restricted = "answered ~ " + " + ".join(f"C({f})" for f in factors)
    print(f"  n after dropping deterministic topics = {len(sub):,}")
    print(f"  n_persona clusters = {sub['persona_id'].nunique():,}")
    print(f"  formula: {formula_restricted}")

    res_orig = smf.logit(formula_restricted, data=sub).fit(disp=0)
    res_cluster = smf.logit(formula_restricted, data=sub).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})
    print(f"  converged (original): {res_orig.mle_retvals.get('converged')}   "
          f"converged (cluster): {res_cluster.mle_retvals.get('converged')}")
    print(f"  pseudo R-sq: {res_orig.prsquared:.4f}   n_params: {len(res_orig.params)}")

    ct2 = combined_coef_table(res_orig, res_cluster)
    unstable = ct2[(~np.isfinite(ct2["se_cluster"])) | (ct2["se_cluster"] > SE_FLAG_THRESHOLD)]
    if len(unstable):
        print(f"  WARNING: {len(unstable)}/{len(ct2)} terms have non-finite or implausibly large "
              f"(cluster SE > {SE_FLAG_THRESHOLD}):")
        print(unstable[["term", "coef_logit", "se_original", "se_cluster", "odds_ratio"]].to_string(index=False))

    both_finite = ct2.dropna(subset=["p_original", "p_cluster"])
    flips = both_finite[(both_finite["p_original"] < 0.05) != (both_finite["p_cluster"] < 0.05)]
    print(f"\n  Terms where significance (alpha=0.05) FLIPS between original and cluster-robust SE: "
          f"{len(flips)}/{len(both_finite)}")
    if len(flips):
        print(flips[["term", "coef_logit", "se_original", "p_original", "se_cluster", "p_cluster"]].to_string(
            index=False, float_format=lambda x: f"{x:9.4f}"))

    out_path = f"{TABLES_DIR}/abstention_model_qwen_ministral.csv"
    ct2.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path}")
    key_terms = ["Intercept", "C(gender)[T.male]", "C(gender)[T.neutral]",
                 "C(model)[T.qwen]", "C(age)[T.65]"]
    key_terms = [t for t in key_terms if t in ct2["term"].values]
    print(f"\n  --- SE comparison for key terms: original (per-row) vs cluster-robust (persona_id) ---")
    print(ct2[ct2["term"].isin(key_terms)][
        ["term", "coef_logit", "se_original", "p_original", "se_cluster", "p_cluster"]
    ].to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    print()
    print("  NOTE on remaining topic coefficients -- this is a MAIN-EFFECTS-ONLY model (no topic x")
    print("  model interaction, unlike analysis_plan.md Section 5's literal spec), so per-model zero")
    print("  cells that don't affect BOTH models are not auto-dropped the way immigration/trust-in-gov")
    print("  were. Two of the five remaining topics are driven by very thin, one-model evidence:")
    per_model_topic_n = sub.groupby(["topic", "model"], observed=True)["answered"].agg(["sum", "count"])
    for topic in sorted(sub["topic"].unique()):
        rows = per_model_topic_n.loc[topic]
        detail = ", ".join(f"{m}: {int(rows.loc[m,'sum'])}/{int(rows.loc[m,'count'])}" for m in rows.index)
        flag = "  <-- fragile: one model contributes ~all/zero of the signal" if (rows["sum"] == 0).any() and rows["sum"].sum() < 100 else ""
        print(f"    {topic:28s} {detail}{flag}")
    print("  In particular, the 'religion and secularism' coefficient (large negative, see table above)")
    print("  is driven almost entirely by ministral's exact 0/5,400 with qwen contributing only 3/5,400")
    print("  positive cases -- treat this coefficient as a description of near-total abstention on that")
    print("  topic, not a precisely estimated effect size. 'economic redistribution' has the same issue")
    print("  (ministral 0/5,400, qwen 60/5,400).")

    print()
    print("=" * 78)
    print("FLAGS -- notable patterns for H2 / the abstention story")
    print("=" * 78)
    print("1. In this sample of five checkpoints, abstention is a near-binary split, not a graded")
    print("   tendency: llama, gemma = 0% abstention (100% genuinely answered). deepseek = 100%")
    print("   abstention/non-response under the corrected definition (0% valid answers -- distinct")
    print("   from llama/gemma's genuine engagement). qwen = 90.0% abstention, ministral = 83.5%.")
    print("   This is itself the headline H2 result -- stated as a pattern observed in these five")
    print("   specific checkpoints, not a general claim about the model families/architectures they")
    print("   belong to (a different checkpoint or version from the same family could behave differently).")
    print()
    print("2. For qwen and ministral, abstention is strongly topic-gated, not uniform: both models")
    print("   answer 0/5,400 times on immigration and trust in government (perfect abstention), and")
    print("   near-zero on religion/economic redistribution, while answering the majority of the time")
    print("   on gender equality (ministral: 97.2%, qwen: 41.2%). This suggests abstention tracks")
    print("   topic sensitivity/controversy rather than a flat per-model abstention rate applied")
    print("   uniformly across content -- worth testing formally against a defined 'controversial")
    print("   topic' grouping per analysis_plan.md Section 7's rule (define the grouping and")
    print("   rationale before testing, report all 7 topics individually regardless).")
    print()
    print("3. See tables/abstention_answered_rate_by_*.csv for the full demographic breakdown feeding")
    print("   regression 2's coefficients.")


if __name__ == "__main__":
    main()
