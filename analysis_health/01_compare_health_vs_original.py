"""Exploratory: original ("friend_final" / friend_v2_explicit_gender) vs.
health-conversation-variant (health_v1) prompt results, for the 180-persona
pilot subset (4 countries x 5 professions x 3 genders x 3 ages) used
throughout this comparison.

Independent of analysis/: does not import from or write to analysis/,
master_results.csv, or anything under tables/. Reads results/results_original_*.csv
(original) and results/results_health_*.csv (health variant) directly, each
filtered down to the pilot subset. Both source files cover the FULL 5,400
persona design (verified directly -- data/prompts_health.csv has
75,600 rows / 5,400 unique personas / all 20 countries / all 30 professions;
there is no separate pre-filtered pilot file), so the 180-persona subset is
derived here by filtering on country + profession, not loaded from a
dedicated file.

Replicates the one correctness-critical detail from analysis/_common.py's
load_master() rather than importing it (keeping this pipeline self-contained
and not coupled to analysis/ internals): the literal string "NA" is a valid,
meaningful value in raw_text/strict_parsed_rating (the model's exact
compliant Condition-B abstention output), not a missing-value marker. A
plain pd.read_csv() would silently null it out.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = "/Users/hastihosseinpour/Desktop/social_sterio_llms"
PILOT_COUNTRIES = ["Germany", "Brazil", "Nigeria", "South Korea"]
PILOT_PROFESSIONS = ["lawyer", "registered nurse", "truck driver", "farmer", "computer programmer"]
MODEL_ORDER = ["llama", "gemma", "qwen", "ministral", "deepseek"]
TOPIC_ORDER = ["climate change", "economic redistribution", "gender equality",
               "immigration", "lgbtq rights", "religion and secularism", "trust in government"]


def load_safe(path, model):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    df.insert(0, "model", model)
    df["rating_numeric"] = pd.to_numeric(df["strict_parsed_rating"], errors="coerce")
    return df


def filter_pilot(df):
    return df[df["country"].isin(PILOT_COUNTRIES) & df["profession"].isin(PILOT_PROFESSIONS)].copy()


def cast_object(df, cols):
    """Guard against pandas 3.0's StringDtype breaking statsmodels formula fitting
    (same issue documented in analysis/_common.py's cast_formula_dtypes)."""
    df = df.copy()
    for c in cols:
        if c in df.columns and isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype("object")
    return df


def clustered_group_diff(data, value_col, group_col, cluster_col):
    """DEPRECATED as of Fix H1 -- kept only so nothing else silently breaks if still
    referenced. Relied on patsy's alphabetical reference-level choice ("health" < "orig",
    so the fitted coefficient was orig-minus-health, not health-minus-orig), which the
    rating-shift call site correctly negated but the abstention-shift call site did not
    -- a silent sign bug (audit Fix H1). Use clustered_diff_health_minus_orig instead,
    which computes the row-wise (health - orig) difference directly and is immune to
    this entire class of bug by construction."""
    n_clusters = data[cluster_col].nunique()
    if n_clusters < 2 or data[group_col].nunique() < 2:
        return dict(diff=np.nan, se_cluster=np.nan, p_value=np.nan, n_clusters=n_clusters)
    data = cast_object(data, [group_col, cluster_col])
    res = smf.ols(f"{value_col} ~ C({group_col})", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
    term = [t for t in res.params.index if t.startswith(f"C({group_col})")][0]
    return dict(diff=float(res.params[term]), se_cluster=float(res.bse[term]),
                p_value=float(res.pvalues[term]), n_clusters=n_clusters)


def clustered_diff_health_minus_orig(data, health_col, orig_col, cluster_col):
    """Fix H1: cluster-robust test of the mean of (health_col - orig_col), computed as
    an explicit row-wise difference (not via a categorical reference level, which is
    what caused the sign bug this replaces). Sign convention is unambiguous by
    construction: positive = health is higher, negative = health is lower."""
    d = data.copy()
    d["_diff"] = d[health_col].astype(int) - d[orig_col].astype(int)
    n_clusters = d[cluster_col].nunique()
    if n_clusters < 2:
        return dict(diff=np.nan, se_cluster=np.nan, p_value=np.nan, n_clusters=n_clusters)
    d = cast_object(d, [cluster_col])
    res = smf.ols("_diff ~ 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster_col]})
    return dict(diff=float(res.params["Intercept"]), se_cluster=float(res.bse["Intercept"]),
                p_value=float(res.pvalues["Intercept"]), n_clusters=n_clusters)


def main():
    print("=" * 78)
    print("STEP 1: load + filter to pilot subset, confirm persona overlap")
    print("=" * 78)
    orig_frames, health_frames = [], []
    for m in MODEL_ORDER:
        orig = filter_pilot(load_safe(f"{ROOT}/results/results_original_{m}.csv", m))
        health = filter_pilot(load_safe(f"{ROOT}/results/results_health_{m}.csv", m))
        orig_frames.append(orig)
        health_frames.append(health)
        print(f"  {m:10s} original pilot rows={len(orig):,} personas={orig['persona_id'].nunique()}   "
              f"health pilot rows={len(health):,} personas={health['persona_id'].nunique()}")

    orig_df = pd.concat(orig_frames, ignore_index=True)
    health_df = pd.concat(health_frames, ignore_index=True)

    orig_personas = set(orig_df["persona_id"])
    health_personas = set(health_df["persona_id"])
    print(f"\n  Original pilot personas (union across models): {len(orig_personas)}")
    print(f"  Health pilot personas (union across models):   {len(health_personas)}")
    print(f"  Overlap: {len(orig_personas & health_personas)}   "
          f"Only in original: {len(orig_personas - health_personas)}   "
          f"Only in health: {len(health_personas - orig_personas)}")
    expected = len(PILOT_COUNTRIES) * len(PILOT_PROFESSIONS) * 3 * 3
    print(f"  Expected pilot persona count (4 countries x 5 professions x 3 genders x 3 ages): {expected}")
    if orig_personas != health_personas:
        print("  WARNING: persona sets differ between original and health pilot subsets -- "
              "results below are restricted to the intersection only.")
    print(f"  -> {'PASS' if orig_personas == health_personas == set() or orig_personas == health_personas else 'CHECK'}: "
          f"persona sets {'match exactly' if orig_personas == health_personas else 'DO NOT match'}.")

    print(f"\n{'='*78}\nSTEP 2: merge on the full canonical key (Fix H6)\n{'='*78}")
    # Fix H6: merge on the full canonical key (not just model/persona_id/topic/condition --
    # also country/profession/gender/age, which are redundant with persona_id in a correctly
    # constructed dataset but catch a persona_id collision or mislabeling silently if one ever
    # occurs), with validate="one_to_one" so a many-to-one/one-to-many merge raises instead of
    # silently duplicating or dropping rows, and explicit pre/post row counts plus any
    # left-only/right-only rows printed rather than assumed.
    key = ["model", "persona_id", "country", "profession", "gender", "age", "topic", "response_condition"]
    print(f"  pre-merge: orig_df={len(orig_df):,} rows, health_df={len(health_df):,} rows")
    merged_ind = orig_df[key + ["strict_is_valid", "is_abstention", "rating_numeric"]].merge(
        health_df[key + ["strict_is_valid", "is_abstention", "rating_numeric"]],
        on=key, suffixes=("_orig", "_health"), how="outer", validate="one_to_one", indicator=True)
    merge_counts = merged_ind["_merge"].value_counts()
    print(f"  post-merge indicator counts: {merge_counts.to_dict()}")
    left_only = merged_ind[merged_ind["_merge"] == "left_only"]
    right_only = merged_ind[merged_ind["_merge"] == "right_only"]
    if len(left_only) or len(right_only):
        print(f"  WARNING: {len(left_only)} rows only in original, {len(right_only)} rows only in health --")
        print(f"  these are dropped from the inner-join comparison below. Sample:")
        if len(left_only):
            print(left_only.head(3).to_string())
        if len(right_only):
            print(right_only.head(3).to_string())
    else:
        print(f"  PASS: 0 unmatched rows on either side -- every original row has exactly one health "
              f"counterpart on the full canonical key, and vice versa.")
    merged = merged_ind[merged_ind["_merge"] == "both"].drop(columns="_merge")
    print(f"  Total matched cells (full canonical key): {len(merged):,}")
    print(merged.groupby("model", observed=True).size().reindex(MODEL_ORDER).to_string())

    print(f"\n{'='*78}\nSTEP 3: per-model comparison\n{'='*78}")
    summary_rows = []
    for m in MODEL_ORDER:
        sub = merged[merged["model"] == m]
        n_cells = len(sub)

        both_valid = sub[sub["strict_is_valid_orig"] & sub["strict_is_valid_health"] &
                          sub["rating_numeric_orig"].notnull() & sub["rating_numeric_health"].notnull()]
        n_both_valid = len(both_valid)

        row = {"model": m, "n_matched_cells": n_cells, "n_both_valid_numeric": n_both_valid}

        print(f"\n  --- {m} ---")
        print(f"  n matched cells={n_cells:,}   n both strict-valid numeric={n_both_valid:,}")

        if n_both_valid >= 2:
            diff = both_valid["rating_numeric_orig"] - both_valid["rating_numeric_health"]
            exact_agree = (diff == 0).mean()
            mad = diff.abs().mean()
            rho, rho_p = stats.spearmanr(both_valid["rating_numeric_orig"], both_valid["rating_numeric_health"])

            rating_diff_df = both_valid[["persona_id", "rating_numeric_health", "rating_numeric_orig"]].copy()
            rating_diff_df["_diff"] = rating_diff_df["rating_numeric_health"] - rating_diff_df["rating_numeric_orig"]
            rating_diff_df = cast_object(rating_diff_df, ["persona_id"])
            res_r = smf.ols("_diff ~ 1", data=rating_diff_df).fit(
                cov_type="cluster", cov_kwds={"groups": rating_diff_df["persona_id"]})
            gd = dict(diff=float(res_r.params["Intercept"]), se_cluster=float(res_r.bse["Intercept"]),
                      p_value=float(res_r.pvalues["Intercept"]), n_clusters=rating_diff_df["persona_id"].nunique())

            print(f"  exact agreement rate:        {exact_agree:.4f}")
            print(f"  mean absolute difference:    {mad:.4f}")
            print(f"  Spearman r (orig vs health): {rho:.4f}  (p={rho_p:.2e})")
            print(f"  persona-clustered diff (health-orig, mean rating shift): "
                  f"{gd['diff']:.4f}  se={gd['se_cluster']:.4f}  p={gd['p_value']:.2e}  "
                  f"(n_persona_clusters={gd['n_clusters']:,})")
            row.update(exact_agreement_rate=exact_agree, mean_abs_diff=mad, spearman_r=rho, spearman_p=rho_p,
                       clustered_mean_shift_health_minus_orig=gd["diff"], clustered_shift_se=gd["se_cluster"],
                       clustered_shift_p=gd["p_value"], n_persona_clusters=gd["n_clusters"])
        else:
            print("  (fewer than 2 both-valid-numeric cells -- rating comparison skipped)")
            row.update(exact_agreement_rate=np.nan, mean_abs_diff=np.nan, spearman_r=np.nan, spearman_p=np.nan,
                       clustered_mean_shift_health_minus_orig=np.nan, clustered_shift_se=np.nan,
                       clustered_shift_p=np.nan, n_persona_clusters=np.nan)

        # abstention rate change (Condition B only -- abstention is structurally impossible under A)
        b_sub = sub[sub["persona_id"].notnull()]
        b_orig = orig_df[(orig_df["model"] == m) & (orig_df["response_condition"] == "B_optional")]
        b_health = health_df[(health_df["model"] == m) & (health_df["response_condition"] == "B_optional")]
        abst_orig_agg = b_orig["is_abstention"].mean()
        abst_health_agg = b_health["is_abstention"].mean()

        b_matched = merged[(merged["model"] == m) & (merged["response_condition"] == "B_optional")]
        abst_orig_paired = b_matched["is_abstention_orig"].mean()
        abst_health_paired = b_matched["is_abstention_health"].mean()

        # Fix H1: was computed via a categorical-reference-level regression that silently
        # gave orig-minus-health (patsy defaults to "health" as the alphabetical reference,
        # so the fitted coefficient was the OTHER direction), while this print/column always
        # claimed to be "health-orig". Replaced with an explicit row-wise difference, which
        # cannot have this sign ambiguity: is_abstention_health - is_abstention_orig, per row,
        # then a persona-clustered one-sample test of that difference's mean against 0.
        gd_abst = clustered_diff_health_minus_orig(b_matched, "is_abstention_health", "is_abstention_orig", "persona_id")

        print(f"  abstention rate (Condition B), AGGREGATE (unfiltered to matched pairs): "
              f"orig={100*abst_orig_agg:.2f}%  health={100*abst_health_agg:.2f}%")
        print(f"  abstention rate (Condition B), PAIRED (same {b_matched['persona_id'].nunique()} personas x topics "
              f"in both): orig={100*abst_orig_paired:.2f}%  health={100*abst_health_paired:.2f}%")
        print(f"  persona-clustered test of the shift (health-orig, positive = health MORE abstention): "
              f"diff={100*gd_abst['diff']:.2f}pp  se={100*gd_abst['se_cluster']:.2f}pp  p={gd_abst['p_value']:.2e}")
        row.update(abstention_rate_B_orig_aggregate=abst_orig_agg, abstention_rate_B_health_aggregate=abst_health_agg,
                   abstention_rate_B_orig_paired=abst_orig_paired, abstention_rate_B_health_paired=abst_health_paired,
                   abstention_shift_health_minus_orig_pp=100 * gd_abst["diff"], abstention_shift_p=gd_abst["p_value"])

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    import os
    os.makedirs(f"{ROOT}/analysis_health/output", exist_ok=True)
    summary.to_csv(f"{ROOT}/analysis_health/output/health_vs_original_summary.csv", index=False)
    print(f"\nWrote {ROOT}/analysis_health/output/health_vs_original_summary.csv")

    print(f"\n{'='*78}\nSUMMARY TABLE\n{'='*78}")
    print("NOTE (Fix H3): Condition A columns above are the primary rating-shift evidence (full pilot")
    print("sample, 100% valid in every model except deepseek -- no abstention/selection mechanism exists")
    print("under Condition A at all). The abstention columns below are Condition B specific and, for")
    print("qwen/ministral's RATING comparison (not shown in this table -- see 02_compare_by_condition.py),")
    print("rest on a small SELECTED sample (qwen: 80/1,260 pairs = 6.35%; ministral: 209/1,260 = 16.59%)")
    print("-- descriptive only, not primary evidence. See 02_compare_by_condition.py for the Condition-A")
    print("vs Condition-B split this table pools together.")
    print(summary[["model", "n_matched_cells", "n_both_valid_numeric", "exact_agreement_rate", "mean_abs_diff",
                    "spearman_r", "abstention_rate_B_orig_paired", "abstention_rate_B_health_paired",
                    "abstention_shift_health_minus_orig_pp", "abstention_shift_p"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n{'='*78}\nFIX H2: full 2x2 denominator table for Ministral abstention rate\n{'='*78}")
    print("The original report compared 'pilot/Condition-B-only' (83.41%->49.60%) against")
    print("'full-dataset/all-rows' (41.73%->24.26%) as if they were the same metric under different")
    print("labels -- but BOTH the sample (full 5,400 personas vs 180-persona pilot subset) AND the")
    print("denominator (all rows, where Condition A structurally cannot abstain, vs Condition-B-only)")
    print("changed between those two numbers, not the denominator alone. All four cells below, so the")
    print("full picture is visible in one place:\n")

    ministral_orig_full = load_safe(f"{ROOT}/results/results_original_ministral.csv", "ministral")
    ministral_health_full = load_safe(f"{ROOT}/results/results_health_ministral.csv", "ministral")
    ministral_orig_pilot = filter_pilot(ministral_orig_full)
    ministral_health_pilot = filter_pilot(ministral_health_full)

    def abst_rate(df, b_only):
        sub = df[df["response_condition"] == "B_optional"] if b_only else df
        return 100 * sub["is_abstention"].mean(), len(sub)

    rows_2x2 = []
    for sample_label, o_df, h_df in [("full_5400_personas", ministral_orig_full, ministral_health_full),
                                       ("pilot_180_personas", ministral_orig_pilot, ministral_health_pilot)]:
        for denom_label, b_only in [("all_rows", False), ("condition_B_only", True)]:
            o_rate, o_n = abst_rate(o_df, b_only)
            h_rate, h_n = abst_rate(h_df, b_only)
            rows_2x2.append({"sample": sample_label, "denominator": denom_label, "n_orig": o_n, "n_health": h_n,
                              "orig_rate_pct": o_rate, "health_rate_pct": h_rate, "shift_pp": h_rate - o_rate})
    table_2x2 = pd.DataFrame(rows_2x2)
    print(table_2x2.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    table_2x2.to_csv(f"{ROOT}/analysis_health/output/ministral_abstention_2x2.csv", index=False)
    print(f"\n  Wrote {ROOT}/analysis_health/output/ministral_abstention_2x2.csv")
    print("\n  Reading the table: moving from the pilot/B-only cell (bottom-right-equivalent) to the")
    print("  full/all-rows cell (top-left-equivalent) changes BOTH axes at once -- sample AND denominator")
    print("  -- so comparing those two cells directly, as if only one thing changed, is not a valid")
    print("  like-for-like comparison. Compare within a row (same sample, denominator changes) or within")
    print("  a column (same denominator, sample changes) instead.")

    print(f"\n{'='*78}\nSTEP 4: is Ministral's abstention change concentrated in specific topics, or spread evenly?\n{'='*78}")
    ministral_b = merged[(merged["model"] == "ministral") & (merged["response_condition"] == "B_optional")]
    topic_rows = []
    for topic in TOPIC_ORDER:
        t_sub = ministral_b[ministral_b["topic"] == topic]
        n = len(t_sub)
        if n == 0:
            continue
        orig_rate = t_sub["is_abstention_orig"].mean()
        health_rate = t_sub["is_abstention_health"].mean()
        print(f"  {topic:28s} n={n:4d}   orig={100*orig_rate:6.2f}%   health={100*health_rate:6.2f}%   "
              f"shift={100*(health_rate-orig_rate):+7.2f}pp")
        topic_rows.append({"topic": topic, "n": n, "abstention_orig": orig_rate,
                            "abstention_health": health_rate, "shift_pp": 100 * (health_rate - orig_rate)})
    topic_df = pd.DataFrame(topic_rows)
    topic_df.to_csv(f"{ROOT}/analysis_health/output/ministral_abstention_by_topic.csv", index=False)
    shifts = topic_df["shift_pp"]
    print(f"\n  Shift range across topics: [{shifts.min():.2f}pp, {shifts.max():.2f}pp]   "
          f"mean={shifts.mean():.2f}pp   sd={shifts.std():.2f}pp")
    cv = shifts.std() / shifts.mean() if shifts.mean() else np.nan
    print(f"  -> {'Concentrated in specific topics' if shifts.std() > abs(shifts.mean())*0.5 else 'Roughly spread evenly across topics'} "
          f"(coefficient of variation of the shift = {cv:.3f}); see per-topic detail above for which topics drive it.")
    print(f"  Wrote {ROOT}/analysis_health/output/ministral_abstention_by_topic.csv")


if __name__ == "__main__":
    main()
