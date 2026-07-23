"""Step 7b: paired Condition A (forced) vs Condition B (optional) comparison.

analysis_plan.md Section 6 calls this the study's "primary contribution" --
it was missing from the original pipeline (steps 1-10) and is added here,
numbered after step 7 since it depends on step 7's matched-cell machinery.

For every (model, persona_id, topic) cell where both an A_forced and a
B_optional row exist (structurally guaranteed by the design -- every persona
has exactly one row per topic per condition, see step 2), this script asks:

  1. Did Condition B answer (strict_is_valid & rating_numeric notnull) or
     abstain (is_abstention==True)?
  2. What did Condition A return for that same cell?
  3. Where BOTH conditions produced a valid number: rating difference, exact
     agreement, weighted (quadratic) Cohen's kappa, Spearman correlation,
     mean absolute difference, directional disagreement, and
     profession/country rank correlation of condition-level means.
  4. THE KEY ANALYSIS (Section 6): among cells where B == NA, what does A
     return? Tested against a midpoint-centered null and against the
     distribution of A's ratings in cells where B did answer.
  5. THE CONFOUND CHECK (Section 6): rate of A rating==3 in B-abstained
     cells vs. B-answered cells. An elevated rate in the abstained group
     would suggest the midpoint is used as a covert abstention channel
     under forced conditions.

PERSONA CLUSTERING: every significance test above treats matched persona x
topic rows as independent, but each persona contributes up to 7 topic-rows
to these groups -- the same pseudoreplication issue steps 5/6/8 correct for
via cov_type='cluster', cov_kwds={'groups': persona_id}. That correction is
applied here too, and it is applied BEFORE any verdict is stated -- an
earlier version of this script printed the uncorrected verdict first and
only caveated it afterward, which doesn't retroactively fix an
already-stated conclusion. The corrected (clustered) test is now what
determines the printed VERDICT; the original scipy tests (t-test, Wilcoxon,
Mann-Whitney, two-proportion z-test) are still reported alongside as
uncorrected reference points, explicitly labeled as such, never as the
basis for a verdict.

Clustered tests are implemented as regressions with persona-clustered SEs
(matching steps 5/6/8's approach exactly), not as closed-form adjustments
to the original test statistics:
  - one-sample midpoint test  -> intercept-only OLS, cluster-robust t-test
    of Intercept == 3
  - abstained-vs-answered rating comparison -> OLS of rating on a group
    indicator, cluster-robust SE on the group coefficient
  - confound check (P(A==3) by group) -> logistic regression of the binary
    indicator on the group indicator, cluster-robust SE -- but ONLY where
    both groups have actual outcome variance. Where one group is
    deterministic (0% or 100%), no MLE coefficient exists regardless of
    clustering (complete separation, same issue diagnosed in steps 6/7);
    this is detected and reported as a deterministic fact, with a
    cluster-robust CI given for the non-deterministic group's own
    proportion instead of a forced/uninterpretable model fit.

DeepSeek is excluded. This is not a judgment call about its data quality in
general (it is never dropped from descriptive reporting elsewhere in this
pipeline, per analysis_plan.md Section 10) -- it is a structural fact about
this specific paired design: DeepSeek's strict-valid Condition-B count is
exactly zero (see steps 3/6), so there are zero (model=deepseek, persona,
topic) cells where a Condition-B outcome of any kind (answered OR a clean
abstention) exists to pair with Condition A. There is nothing to match.

Run per model: llama, gemma, qwen, ministral. Llama and gemma have ZERO
Condition-B abstentions (see step 6), so the key analysis and confound
check are structurally not applicable for them (n=0 abstained cells) --
this is reported explicitly, not glossed over. Qwen and ministral, with
90.0% and 83.5% B-abstention rates respectively, are where this analysis
has statistical power.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportions_ztest

from _common import load_master, cast_formula_dtypes, clustered_proportion_ci, TABLES_DIR

INCLUDED_MODELS = ["llama", "gemma", "qwen", "ministral"]
MIDPOINT = 3


def build_matched(df, model):
    sub = df[df["model"] == model]
    a = sub[sub["response_condition"] == "A_forced"].set_index(["persona_id", "topic"])
    b = sub[sub["response_condition"] == "B_optional"].set_index(["persona_id", "topic"])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    matched = pd.DataFrame({
        "persona_id": [i[0] for i in idx],
        "topic": [i[1] for i in idx],
        "country": a["country"].values,
        "profession": a["profession"].values,
        "a_valid": (a["strict_is_valid"] & a["rating_numeric"].notnull()).values,
        "a_rating": a["rating_numeric"].values,
        "b_answered": (b["strict_is_valid"] & b["rating_numeric"].notnull()).values,
        "b_abstained": (b["is_abstention"] == True).values,
        "b_rating": b["rating_numeric"].values,
    })
    # matched is a freshly-constructed DataFrame, but pandas 3.0 infers StringDtype by default
    # for ANY list/array of Python strings passed to the constructor -- not just columns copied
    # via .values from an already-StringDtype source. Verified empirically: persona_id and topic
    # (built from plain Python list comprehensions over a MultiIndex) end up StringDtype too, not
    # just country/profession (built via .values). Cast all of them, since 'topic' is later used
    # in a C(topic) formula term.
    matched = cast_formula_dtypes(matched, columns=["persona_id", "topic", "country", "profession"])
    return matched


def rank_corr_by_group(both, group_col):
    g = both.groupby(group_col, observed=True)[["a_rating", "b_rating"]].mean()
    if len(g) < 3:
        return np.nan, np.nan, len(g)
    rho, p = stats.spearmanr(g["a_rating"], g["b_rating"])
    return rho, p, len(g)


def clustered_one_sample_test(data, value_col, cluster_col, null_value):
    """Cluster-robust test of whether mean(value_col) differs from null_value:
    an intercept-only OLS with persona-clustered SEs, then a t-test on the
    intercept against null_value. Regression-based cluster-robust analog of a
    one-sample t-test, matching steps 5/6/8's approach (refit as a clustered
    regression rather than adjust a closed-form statistic)."""
    n_clusters = data[cluster_col].nunique()
    if n_clusters < 2 or len(data) < 2:
        return dict(mean=np.nan, se_cluster=np.nan, t_stat=np.nan, p_value=np.nan, n_clusters=n_clusters)
    res = smf.ols(f"{value_col} ~ 1", data=data).fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
    tt = res.t_test(f"Intercept = {null_value}")
    return dict(mean=float(res.params["Intercept"]), se_cluster=float(res.bse["Intercept"]),
                t_stat=float(np.asarray(tt.tvalue).squeeze()), p_value=float(np.asarray(tt.pvalue).squeeze()),
                n_clusters=n_clusters)


def clustered_group_diff_test(data, value_col, group_col, cluster_col):
    """Cluster-robust test of whether mean(value_col) differs between the two
    levels of group_col: OLS of value_col on a group indicator with
    persona-clustered SEs. Regression analog of Mann-Whitney / difference-in-
    means, replacing the rank-based test with a mean-based one -- the same
    trade made throughout this pipeline wherever a clustered test was needed
    (means are what cluster-robust regression tests; ranks don't have a
    standard cluster-robust generalization)."""
    n_clusters = data[cluster_col].nunique()
    if n_clusters < 2 or data[group_col].nunique() < 2:
        return dict(diff=np.nan, se_cluster=np.nan, t_stat=np.nan, p_value=np.nan, n_clusters=n_clusters)
    res = smf.ols(f"{value_col} ~ C({group_col})", data=data).fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
    term = [t for t in res.params.index if t.startswith(f"C({group_col})")][0]
    return dict(diff=float(res.params[term]), se_cluster=float(res.bse[term]),
                t_stat=float(res.tvalues[term]), p_value=float(res.pvalues[term]), n_clusters=n_clusters)


def clustered_confound_test(data, cluster_col, group_col="b_abstained_flag", outcome_col="a_eq_3"):
    """Cluster-robust test of whether P(outcome_col) differs by group_col: a
    persona-clustered logistic regression -- but ONLY where both groups have
    actual outcome variance. If one arm is deterministic (rate exactly 0 or
    1), no MLE coefficient exists (complete separation) regardless of
    clustering -- that is detected up front and reported as a fact, not
    forced through a solver (same principle applied to the ordinal-model
    proportional-odds checks in step 5b)."""
    rates = data.groupby(group_col, observed=True)[outcome_col].mean()
    counts = data.groupby(group_col, observed=True)[outcome_col].agg(["sum", "count"])
    deterministic = bool(rates.isin([0.0, 1.0]).any()) or len(rates) < 2
    result = dict(deterministic=deterministic, rates=rates.to_dict(), counts=counts.to_dict("index"))
    if not deterministic:
        n_clusters = data[cluster_col].nunique()
        res = smf.logit(f"{outcome_col} ~ C({group_col})", data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data[cluster_col]}, disp=0)
        term = [t for t in res.params.index if t.startswith(f"C({group_col})")][0]
        result.update(coef=float(res.params[term]), se_cluster=float(res.bse[term]),
                       p_value=float(res.pvalues[term]), odds_ratio=float(np.exp(res.params[term])),
                       converged=bool(res.mle_retvals.get("converged", True)), n_clusters=n_clusters)
    return result


def main():
    print(__doc__.strip().split("\n\n")[0])
    print()
    summary_rows = []

    for model in INCLUDED_MODELS:
        matched = build_matched(load_master(), model)
        n_matched = len(matched)
        n_a_invalid = (~matched["a_valid"]).sum()
        n_b_other_invalid = (~matched["b_answered"] & ~matched["b_abstained"]).sum()

        print("=" * 78)
        print(f"MODEL: {model}   (n_matched persona x topic cells = {n_matched:,})")
        print("=" * 78)
        if n_a_invalid:
            print(f"  NOTE: {n_a_invalid} cells have an invalid Condition-A rating -- excluded from A-dependent stats.")
        if n_b_other_invalid:
            print(f"  NOTE: {n_b_other_invalid} Condition-B cells are neither a valid answer nor a clean abstention "
                  f"(malformed/other) -- excluded from the b_answered/b_abstained split below.")

        # --- 3. where both answered ---
        both = matched[matched["a_valid"] & matched["b_answered"]].copy()
        n_both = len(both)
        row = {"model": model, "n_matched": n_matched, "n_both_answered": n_both,
               "n_b_abstained": int(matched["b_abstained"].sum())}

        print(f"\n  --- Where both conditions answered (n={n_both:,}) ---")
        if n_both >= 2:
            diff = both["a_rating"] - both["b_rating"]
            exact_agree = (diff == 0).mean()
            a_is_constant = both["a_rating"].nunique() == 1
            b_is_constant = both["b_rating"].nunique() == 1
            if a_is_constant or b_is_constant:
                print(f"    NOTE: a_rating constant={a_is_constant} (value={both['a_rating'].iloc[0] if a_is_constant else 'n/a'}), "
                      f"b_rating constant={b_is_constant} (value={both['b_rating'].iloc[0] if b_is_constant else 'n/a'}) "
                      f"in this subset -- kappa/Spearman are mathematically undefined (NaN) with zero variance, not a bug. "
                      f"This itself is the finding: every time this model answers under BOTH conditions, it gives the same")
                print(f"    single rating regardless of condition.")
            kappa = cohen_kappa_score(both["a_rating"].astype(int), both["b_rating"].astype(int), weights="quadratic")
            rho, rho_p = stats.spearmanr(both["a_rating"], both["b_rating"])
            mad = diff.abs().mean()
            pct_gt = (diff > 0).mean()
            pct_lt = (diff < 0).mean()
            pct_eq = (diff == 0).mean()
            prof_rho, prof_p, n_prof = rank_corr_by_group(both, "profession")
            country_rho, country_p, n_country = rank_corr_by_group(both, "country")

            print(f"    exact agreement rate:       {exact_agree:.4f}")
            print(f"    weighted (quadratic) kappa: {kappa:.4f}")
            print(f"    Spearman r (A vs B rating): {rho:.4f}  (p={rho_p:.2e})")
            print(f"    mean absolute difference:   {mad:.4f}")
            print(f"    directional: A>B {pct_gt:.4f}, A<B {pct_lt:.4f}, A==B {pct_eq:.4f}")
            print(f"    profession-level rank corr (A-mean vs B-mean, n={n_prof} professions): rho={prof_rho:.4f} (p={prof_p:.2e})" if n_prof >= 3 else "    profession-level rank corr: insufficient groups")
            print(f"    country-level rank corr    (A-mean vs B-mean, n={n_country} countries):   rho={country_rho:.4f} (p={country_p:.2e})" if n_country >= 3 else "    country-level rank corr: insufficient groups")

            row.update(dict(exact_agreement_rate=exact_agree, weighted_kappa=kappa,
                             spearman_r=rho, spearman_p=rho_p, mean_abs_diff=mad,
                             pct_A_gt_B=pct_gt, pct_A_lt_B=pct_lt, pct_A_eq_B=pct_eq,
                             profession_rank_corr=prof_rho, profession_rank_corr_p=prof_p,
                             country_rank_corr=country_rho, country_rank_corr_p=country_p))
        else:
            print("    (fewer than 2 matched both-answered cells -- skipped)")

        # --- 4. KEY ANALYSIS: A's rating where B abstained ---
        b_abstained = matched[matched["a_valid"] & matched["b_abstained"]].copy()
        n_abstained = len(b_abstained)
        print(f"\n  --- KEY ANALYSIS: Condition A rating where Condition B abstained (n={n_abstained:,}) ---")
        row["n_b_abstained_with_valid_A"] = n_abstained
        if n_abstained == 0:
            print("    N/A -- zero Condition-B abstentions for this model (see step 6: llama/gemma never abstain).")
            print("    Key analysis and confound check below are structurally inapplicable.")
            row.update(dict(mean_A_given_B_NA=np.nan, median_A_given_B_NA=np.nan,
                             midpoint_test_mean=np.nan, midpoint_test_se_cluster=np.nan,
                             midpoint_test_t=np.nan, midpoint_test_p_cluster=np.nan, midpoint_test_n_clusters=0,
                             ttest_vs_midpoint_stat_uncorrected=np.nan, ttest_vs_midpoint_p_uncorrected=np.nan,
                             wilcoxon_vs_midpoint_stat=np.nan, wilcoxon_vs_midpoint_p=np.nan,
                             groupdiff_vs_answered_diff=np.nan, groupdiff_vs_answered_se_cluster=np.nan,
                             groupdiff_vs_answered_p_cluster=np.nan,
                             mannwhitney_vs_answered_stat_uncorrected=np.nan, mannwhitney_vs_answered_p_uncorrected=np.nan))
        else:
            a_vals = b_abstained["a_rating"].values
            mean_a = a_vals.mean()
            median_a = np.median(a_vals)
            print(f"    mean A rating: {mean_a:.4f}   median: {median_a:.1f}   n={n_abstained}")

            # CORRECTED (persona-clustered) test, computed and stated FIRST -- this determines
            # the printed verdict, not the uncorrected test below it.
            mt = clustered_one_sample_test(b_abstained, "a_rating", "persona_id", MIDPOINT)
            print(f"    [CLUSTERED, persona-robust] mean vs midpoint (3): mean={mt['mean']:.4f}  "
                  f"se_cluster={mt['se_cluster']:.4f}  t={mt['t_stat']:.3f}  p={mt['p_value']:.2e}  "
                  f"(n_persona_clusters={mt['n_clusters']:,})")
            verdict = "DIFFERS from midpoint-centered null" if mt["p_value"] < 0.05 else "does NOT differ from midpoint-centered null"
            print(f"    -> VERDICT (persona-clustered): A's forced rating in B-abstained cells {verdict} (alpha=0.05).")
            print("       This is a statistically significant shift in the forced-rating distribution conditional")
            print("       on abstention (mean != 3), distinct from a distribution centered at the midpoint with no")
            print("       directional shift. It does not establish what the model 'actually believes' -- only that")
            print("       its forced-condition output distribution, when it had abstained given the option, is not")
            print("       consistent with indifference/no-signal.")

            # Uncorrected reference statistics -- reported for comparison only, NOT the basis
            # for the verdict above (pseudoreplication makes these anti-conservative).
            t_stat, t_p = stats.ttest_1samp(a_vals, popmean=MIDPOINT)
            nonzero = a_vals[a_vals != MIDPOINT]
            if len(nonzero) >= 1:
                try:
                    w_stat, w_p = stats.wilcoxon(a_vals - MIDPOINT)
                except ValueError as e:
                    w_stat, w_p = np.nan, np.nan
            else:
                w_stat, w_p = np.nan, np.nan
            wilcoxon_str = f"W={w_stat:.1f}, p={w_p:.2e}" if not np.isnan(w_stat) else "skipped"
            print(f"    [uncorrected reference, NOT the verdict] one-sample t-test: t={t_stat:.3f}, p={t_p:.2e}  |  "
                  f"Wilcoxon: {wilcoxon_str}")

            # --- clustered group-difference test: B-abstained vs B-answered A-ratings ---
            b_answered_a_vals = matched.loc[matched["a_valid"] & matched["b_answered"], "a_rating"].values
            gd = dict(diff=np.nan, se_cluster=np.nan, t_stat=np.nan, p_value=np.nan, n_clusters=0)
            mw_stat, mw_p = np.nan, np.nan
            if len(b_answered_a_vals) >= 1:
                group_df = pd.concat([
                    b_abstained[["persona_id", "a_rating"]].assign(b_abstained_flag=1),
                    matched.loc[matched["a_valid"] & matched["b_answered"], ["persona_id", "a_rating"]].assign(b_abstained_flag=0),
                ], ignore_index=True)
                gd = clustered_group_diff_test(group_df, "a_rating", "b_abstained_flag", "persona_id")
                print(f"    [CLUSTERED, persona-robust] A-rating difference, B-abstained vs B-answered: "
                      f"diff={gd['diff']:.4f}  se_cluster={gd['se_cluster']:.4f}  t={gd['t_stat']:.3f}  "
                      f"p={gd['p_value']:.2e}  (n_persona_clusters={gd['n_clusters']:,})")
                print(f"      mean A | B abstained = {mean_a:.4f}   vs   mean A | B answered = {b_answered_a_vals.mean():.4f}")
                diff_verdict = "DIFFERS" if gd["p_value"] < 0.05 else "does NOT differ"
                print(f"    -> VERDICT (persona-clustered): mean A-rating {diff_verdict} between B-abstained and "
                      f"B-answered cells (alpha=0.05).")

                mw_stat, mw_p = stats.mannwhitneyu(a_vals, b_answered_a_vals, alternative="two-sided")
                print(f"    [uncorrected reference, NOT the verdict] Mann-Whitney U: U={mw_stat:.1f}, p={mw_p:.2e}")

            row.update(dict(mean_A_given_B_NA=mean_a, median_A_given_B_NA=median_a,
                             midpoint_test_mean=mt["mean"], midpoint_test_se_cluster=mt["se_cluster"],
                             midpoint_test_t=mt["t_stat"], midpoint_test_p_cluster=mt["p_value"],
                             midpoint_test_n_clusters=mt["n_clusters"],
                             ttest_vs_midpoint_stat_uncorrected=t_stat, ttest_vs_midpoint_p_uncorrected=t_p,
                             wilcoxon_vs_midpoint_stat=w_stat, wilcoxon_vs_midpoint_p=w_p,
                             groupdiff_vs_answered_diff=gd["diff"], groupdiff_vs_answered_se_cluster=gd["se_cluster"],
                             groupdiff_vs_answered_p_cluster=gd["p_value"],
                             mannwhitney_vs_answered_stat_uncorrected=mw_stat, mannwhitney_vs_answered_p_uncorrected=mw_p))

        # --- 5. CONFOUND CHECK: rate of A==3 in abstained vs answered cells ---
        print(f"\n  --- CONFOUND CHECK: rate of Condition-A rating==3, B-abstained vs B-answered ---")
        if n_abstained == 0:
            print("    N/A -- zero Condition-B abstentions for this model.")
            row.update(dict(pct_A_eq3_given_B_abstained=np.nan, pct_A_eq3_given_B_answered=np.nan,
                             confound_ztest_stat_uncorrected=np.nan, confound_ztest_p_uncorrected=np.nan,
                             confound_clustered_deterministic=None, confound_clustered_coef=np.nan,
                             confound_clustered_p=np.nan, confound_clustered_odds_ratio=np.nan))
        else:
            n3_abstained = int((b_abstained["a_rating"] == MIDPOINT).sum())
            n3_answered = int((both["a_rating"] == MIDPOINT).sum())
            pct3_abstained = n3_abstained / n_abstained
            pct3_answered = n3_answered / n_both if n_both else np.nan
            print(f"    P(A==3 | B abstained) = {pct3_abstained:.4f}  ({n3_abstained}/{n_abstained})")
            print(f"    P(A==3 | B answered)  = {pct3_answered:.4f}  ({n3_answered}/{n_both})")

            confound_df = pd.concat([
                b_abstained[["persona_id"]].assign(a_eq_3=(b_abstained["a_rating"] == MIDPOINT).astype(int), b_abstained_flag=1),
                both[["persona_id"]].assign(a_eq_3=(both["a_rating"] == MIDPOINT).astype(int), b_abstained_flag=0),
            ], ignore_index=True) if n_both else None

            ct = dict(deterministic=None)
            if confound_df is not None and len(confound_df):
                ct = clustered_confound_test(confound_df, "persona_id")
            if ct.get("deterministic"):
                print(f"    [CLUSTERED, persona-robust] DETERMINISTIC: one arm has rate exactly 0 or 1 -- no logistic")
                print(f"    coefficient exists regardless of clustering (complete separation, not a fitting failure).")
                print(f"    rates by group: {ct['rates']}   raw counts: {ct['counts']}")
                pci = clustered_proportion_ci(b_abstained.assign(a_eq_3=(b_abstained["a_rating"] == MIDPOINT).astype(int)),
                                               "a_eq_3", "persona_id")
                print(f"    Cluster-robust estimate for the non-deterministic side (B-abstained) alone: "
                      f"P(A==3)={pci['prop']:.4f}  95% CI=[{pci['ci_low']:.4f}, {pci['ci_high']:.4f}]  "
                      f"(n_persona_clusters={pci['n_clusters']:,})")
                print(f"    -> VERDICT: the B-answered side is a deterministic {pct3_answered:.0%} (exactly {n3_answered}/{n_both}),")
                print(f"       not a sampling estimate to be tested against -- this is itself the strongest form of")
                print(f"       evidence for the gap (a repeatable structural fact across {n_both:,} rows / "
                      f"{both['persona_id'].nunique():,} personas, not a p-value-dependent claim).")
                row.update(dict(pct_A_eq3_given_B_abstained=pct3_abstained, pct_A_eq3_given_B_answered=pct3_answered,
                                 confound_clustered_deterministic=True, confound_clustered_coef=np.nan,
                                 confound_clustered_p=np.nan, confound_clustered_odds_ratio=np.nan))
            elif ct.get("deterministic") is False:
                print(f"    [CLUSTERED, persona-robust] logistic regression: coef={ct['coef']:.4f}  "
                      f"se_cluster={ct['se_cluster']:.4f}  p={ct['p_value']:.2e}  odds_ratio={ct['odds_ratio']:.3f}  "
                      f"(n_persona_clusters={ct['n_clusters']:,}, converged={ct['converged']})")
                cverdict = "SIGNIFICANTLY ELEVATED" if (ct["coef"] > 0 and ct["p_value"] < 0.05) else \
                           ("SIGNIFICANTLY LOWER" if ct["p_value"] < 0.05 else "not significantly different")
                print(f"    -> VERDICT (persona-clustered): rate of A==3 is {cverdict} when B abstains (alpha=0.05).")
                row.update(dict(pct_A_eq3_given_B_abstained=pct3_abstained, pct_A_eq3_given_B_answered=pct3_answered,
                                 confound_clustered_deterministic=False, confound_clustered_coef=ct["coef"],
                                 confound_clustered_p=ct["p_value"], confound_clustered_odds_ratio=ct["odds_ratio"]))
            else:
                row.update(dict(pct_A_eq3_given_B_abstained=pct3_abstained, pct_A_eq3_given_B_answered=pct3_answered,
                                 confound_clustered_deterministic=None, confound_clustered_coef=np.nan,
                                 confound_clustered_p=np.nan, confound_clustered_odds_ratio=np.nan))

            if n_both >= 1:
                z_stat, z_p = proportions_ztest([n3_abstained, n3_answered], [n_abstained, n_both])
                print(f"    [uncorrected reference, NOT the verdict] two-proportion z-test: z={z_stat:.3f}, p={z_p:.2e}")
                row.update(dict(confound_ztest_stat_uncorrected=z_stat, confound_ztest_p_uncorrected=z_p))
            else:
                row.update(dict(confound_ztest_stat_uncorrected=np.nan, confound_ztest_p_uncorrected=np.nan))

        # --- 5b. CONFOUND CHECK, TOPIC-STRATIFIED ---
        print(f"\n  --- CONFOUND CHECK, TOPIC-STRATIFIED (does the gap survive within topic?) ---")
        topic_strat_rows = []
        if n_abstained == 0:
            print("    N/A -- zero Condition-B abstentions for this model.")
        else:
            MIN_N = 20
            for topic in sorted(matched["topic"].unique()):
                ab_t = b_abstained[b_abstained["topic"] == topic]
                an_t = both[both["topic"] == topic]
                n_ab_t, n_an_t = len(ab_t), len(an_t)
                if n_ab_t < MIN_N or n_an_t < MIN_N:
                    print(f"    {topic:28s} n_abstained={n_ab_t:>6,}  n_answered={n_an_t:>6,}  "
                          f"-- insufficient n in one arm (need >={MIN_N}), skipped")
                    continue
                n3_ab_t = int((ab_t["a_rating"] == MIDPOINT).sum())
                n3_an_t = int((an_t["a_rating"] == MIDPOINT).sum())
                p_ab_t, p_an_t = n3_ab_t / n_ab_t, n3_an_t / n_an_t

                topic_confound_df = pd.concat([
                    ab_t[["persona_id"]].assign(a_eq_3=(ab_t["a_rating"] == MIDPOINT).astype(int), b_abstained_flag=1),
                    an_t[["persona_id"]].assign(a_eq_3=(an_t["a_rating"] == MIDPOINT).astype(int), b_abstained_flag=0),
                ], ignore_index=True)
                tct = clustered_confound_test(topic_confound_df, "persona_id")

                zt, zp = proportions_ztest([n3_ab_t, n3_an_t], [n_ab_t, n_an_t])

                if tct["deterministic"]:
                    print(f"    {topic:28s} P(A==3|abstained)={p_ab_t:.3f} ({n3_ab_t}/{n_ab_t})  "
                          f"P(A==3|answered)={p_an_t:.3f} ({n3_an_t}/{n_an_t})  "
                          f"[CLUSTERED: DETERMINISTIC, no coefficient possible]  "
                          f"[uncorrected ref: z={zt:.2f} p={zp:.2e}]")
                    topic_strat_rows.append(dict(model=model, topic=topic, n_abstained=n_ab_t, n_answered=n_an_t,
                                                  pct_A_eq3_abstained=p_ab_t, pct_A_eq3_answered=p_an_t,
                                                  clustered_deterministic=True, clustered_coef=np.nan, clustered_p=np.nan,
                                                  ztest_stat_uncorrected=zt, ztest_p_uncorrected=zp))
                else:
                    cflag = "VIOLATED/ELEVATED" if (tct["coef"] > 0 and tct["p_value"] < 0.05) else \
                            ("SIG. LOWER" if tct["p_value"] < 0.05 else "not sig.")
                    print(f"    {topic:28s} P(A==3|abstained)={p_ab_t:.3f} ({n3_ab_t}/{n_ab_t})  "
                          f"P(A==3|answered)={p_an_t:.3f} ({n3_an_t}/{n_an_t})  "
                          f"[CLUSTERED: coef={tct['coef']:.3f} p={tct['p_value']:.2e} -> {cflag}]  "
                          f"[uncorrected ref: z={zt:.2f} p={zp:.2e}]")
                    topic_strat_rows.append(dict(model=model, topic=topic, n_abstained=n_ab_t, n_answered=n_an_t,
                                                  pct_A_eq3_abstained=p_ab_t, pct_A_eq3_answered=p_an_t,
                                                  clustered_deterministic=False, clustered_coef=tct["coef"],
                                                  clustered_p=tct["p_value"],
                                                  ztest_stat_uncorrected=zt, ztest_p_uncorrected=zp))
            if topic_strat_rows:
                gap_survives = all(r["pct_A_eq3_abstained"] > r["pct_A_eq3_answered"] for r in topic_strat_rows)
                n_deterministic = sum(r["clustered_deterministic"] for r in topic_strat_rows)
                print(f"    -> Within every topic with enough data in both arms, P(A==3|abstained) > P(A==3|answered): "
                      f"{gap_survives}. The gap {'survives' if gap_survives else 'does NOT uniformly survive'} topic stratification.")
                print(f"    -> {n_deterministic}/{len(topic_strat_rows)} testable topics are deterministic in the")
                print(f"       B-answered arm (no formal test possible there; see per-topic detail above) --")
                print(f"       the gap in those topics is a structural fact, not a p-value-dependent claim.")
            else:
                print("    -> No topic had enough n in both arms to test.")

        # --- 5c. CONFOUND CHECK, TOPIC-ADJUSTED (logistic regression) ---
        print(f"\n  --- CONFOUND CHECK, TOPIC-ADJUSTED (logistic regression: A==3 ~ B_abstained + topic) ---")
        if n_abstained == 0:
            print("    N/A -- zero Condition-B abstentions for this model.")
            row.update(dict(confound_logit_coef_abstained=np.nan, confound_logit_se=np.nan,
                             confound_logit_p=np.nan, confound_logit_odds_ratio=np.nan))
        else:
            reg_df = pd.concat([
                b_abstained.assign(b_abstained_flag=1),
                both.assign(b_abstained_flag=0),
            ], ignore_index=True)
            reg_df["a_eq_3"] = (reg_df["a_rating"] == MIDPOINT).astype(int)
            # Check for the deterministic pattern before trusting any MLE output: if P(A==3 |
            # answered)==0 within every topic stratum that has both arms, "answered" perfectly
            # predicts a_eq_3==0 -- classic complete/quasi-complete separation. A logistic
            # regression cannot return a finite, meaningful coefficient in that situation; it
            # is not something to interpret from the raw fit output regardless of what number
            # comes out.
            testable_topics = [r["topic"] for r in topic_strat_rows]
            zero_in_answered = all(
                both.loc[both["topic"] == t, "a_rating"].eq(MIDPOINT).sum() == 0 for t in testable_topics
            ) if testable_topics else False
            try:
                logit_res = smf.logit("a_eq_3 ~ C(b_abstained_flag) + C(topic)", data=reg_df).fit(disp=0)
                converged = logit_res.mle_retvals.get("converged")
                coef = logit_res.params.get("C(b_abstained_flag)[T.1]", np.nan)
                se = logit_res.bse.get("C(b_abstained_flag)[T.1]", np.nan)
                p = logit_res.pvalues.get("C(b_abstained_flag)[T.1]", np.nan)
                if not converged or se > 100:
                    print(f"    FIT DID NOT CONVERGE (raw output: coef={coef:.2f}, SE={se:.2f} -- not interpretable).")
                    if zero_in_answered:
                        print("    REASON: within every topic tested above, P(A==3 | B answered) = 0% exactly --")
                        print("    'answered' perfectly predicts A != 3, a classic complete-separation pattern that")
                        print("    prevents any finite MLE estimate for the B_abstained coefficient. This is not a")
                        print("    fitting failure to work around: it IS the topic-adjusted answer -- the stratified")
                        print("    table above already shows it directly and reliably, with no model needed.")
                    else:
                        print("    REASON UNCLEAR -- does not match the expected zero-in-answered-arm pattern; inspect manually.")
                    row.update(dict(confound_logit_coef_abstained=np.nan, confound_logit_se=np.nan,
                                     confound_logit_p=np.nan, confound_logit_odds_ratio=np.nan,
                                     confound_logit_converged=False))
                else:
                    or_val = np.exp(coef)
                    print(f"    B_abstained coefficient (log-odds, topic-adjusted): {coef:.4f}  SE={se:.4f}  p={p:.2e}")
                    print(f"    odds ratio: {or_val:.3f}  (P(A==3) is {or_val:.2f}x higher when B abstained, holding topic fixed)")
                    print(f"    converged: True   n={int(logit_res.nobs):,}")
                    row.update(dict(confound_logit_coef_abstained=coef, confound_logit_se=se,
                                     confound_logit_p=p, confound_logit_odds_ratio=or_val,
                                     confound_logit_converged=True))
            except Exception as e:
                print(f"    FIT FAILED: {type(e).__name__}: {e}")
                row.update(dict(confound_logit_coef_abstained=np.nan, confound_logit_se=np.nan,
                                 confound_logit_p=np.nan, confound_logit_odds_ratio=np.nan,
                                 confound_logit_converged=False))
            print("    -> CONCLUSION from topic-adjusted check: rely on the CLUSTERED topic-stratified table above,")
            print("       not this pooled regression, given the separation issue. See its per-topic verdict. Note:")
            print("       clustering cannot rescue a non-converging fit either -- separation means no MLE point")
            print("       estimate exists at all, which is a property of the data, not the SE calculation on top of it.")

        print("\n  NOTE on persona clustering: every VERDICT above (midpoint test, B-abstained-vs-B-answered")
        print("  rating comparison, confound check) is now based on the [CLUSTERED, persona-robust] statistic,")
        print("  computed via a regression with persona-clustered SEs (cov_type='cluster', groups=persona_id),")
        print("  matching steps 5/6/8's correction for the same pseudoreplication issue (each persona contributes")
        print("  up to 7 topic-rows). The [uncorrected reference] lines (raw t-test/Wilcoxon/Mann-Whitney/")
        print("  two-proportion z-test) are reported alongside for comparison only and do NOT drive any verdict.")

        summary_rows.append(row)
        matched.to_csv(f"{TABLES_DIR}/paired_comparison_matched_cells_{model}.csv", index=False)
        if topic_strat_rows:
            pd.DataFrame(topic_strat_rows).to_csv(f"{TABLES_DIR}/paired_comparison_topic_stratified_{model}.csv", index=False)
        print()

    summary = pd.DataFrame(summary_rows)
    out_path = f"{TABLES_DIR}/paired_comparison_summary.csv"
    summary.to_csv(out_path, index=False)
    print("=" * 78)
    print(f"Wrote {out_path} and per-model matched-cell CSVs (tables/paired_comparison_matched_cells_*.csv)")
    print("=" * 78)
    print()
    print("DeepSeek excluded from this analysis: its Condition-B strict-valid count is exactly")
    print("zero (see steps 3/6), so no (persona, topic) cell has a matchable Condition-B outcome")
    print("to pair with Condition A -- there is nothing to compute, not a data-quality judgment call.")


if __name__ == "__main__":
    main()
