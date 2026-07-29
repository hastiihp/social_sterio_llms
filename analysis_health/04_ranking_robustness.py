"""Exploratory: are profession and country RANKINGS (not absolute ratings)
stable between the original and health-conversation framings?

This is a distinct question from 01/02's finding that absolute rating
levels shift downward under the health framing. A stable ranking riding on
a lower baseline is entirely possible -- both can be true at once. This
script isolates the ranking question specifically: does WHO ranks higher
than WHOM (which profession, which country) survive the framing change,
independent of whether the overall rating level moved.

Fix H4: the asymptotic Spearman p-value scipy.stats.spearmanr returns is
invalid at n=4/5 -- it relies on a large-sample approximation that does not
hold with this few categories. Replaced with two things instead:
  (a) EXACT permutation p-values: with n=5 (profession) there are 5!=120
      possible pairings and with n=4 (country) there are 4!=24 -- small
      enough to enumerate completely, so the p-value is exact, not
      approximated, computed as the fraction of all possible pairings whose
      |Spearman rho| is >= the observed |rho|.
  (b) A persona bootstrap (1,000 resamples, resampling whole persona
      clusters with replacement -- not individual rows, which would break
      the within-persona correlation structure): for each resample, refit
      the model and record which level ranks TOP and which ranks BOTTOM.
      Reports p_top/p_bottom per level per framing -- a rank-POSITION
      probability, not just a single point-estimate ranking.
Stated explicitly per the audit: at n=4 (country), no ranking can reach the
conventional p<.05 threshold even with PERFECT rank agreement (rho=1.0) --
the smallest possible two-sided exact permutation p-value at n=4 is
2/24=0.0833. This is a sample-size ceiling built into having only 4
categories, not evidence against a real effect. n=5 (profession) can reach
p=2/120=.0167 at best, still not conventionally "very small" for n this low.

Condition A only (forced, no abstention/selection effects possible --
matching the discipline already established for H1 in
analysis/08_variance_ranking.py and the topic-specific models in
analysis/05c_topic_specific_models.py, and for the same reason: Condition B
introduces a self-selected "answered" subset that differs in composition
and size between the original and health versions, as already shown in
analysis_health/02_compare_by_condition.py).

180-persona pilot subset (4 countries x 5 professions x 3 genders x 3 ages),
same as every other analysis_health script. DeepSeek is skipped: it has 0
valid Condition-A rows in the health version (see
03_deepseek_health_diagnosis.py), so there is nothing to fit.

Independent of analysis/ (same self-contained design as the other
analysis_health scripts): reads results/*.csv and
results_health/health_full_results_*.csv directly. Does not touch
analysis/ or master_results.csv.
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = "/Users/hastihosseinpour/Desktop/social_sterio_llms"
PILOT_COUNTRIES = ["Germany", "Brazil", "Nigeria", "South Korea"]
PILOT_PROFESSIONS = ["lawyer", "registered nurse", "truck driver", "farmer", "computer programmer"]
MODEL_ORDER = ["llama", "gemma", "qwen", "ministral"]  # deepseek skipped, see docstring
TIE_ATOL = 1e-10
TIE_RTOL = 1e-9
FACTORS = ["gender", "country", "profession", "age", "topic"]
N_BOOT = 1000
BOOT_SEED = 0


def load_safe(path, model):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    df.insert(0, "model", model)
    df["rating_numeric"] = pd.to_numeric(df["strict_parsed_rating"], errors="coerce")
    return df


def filter_pilot_condA(df):
    sub = df[df["country"].isin(PILOT_COUNTRIES) & df["profession"].isin(PILOT_PROFESSIONS) &
             (df["response_condition"] == "A_forced") & df["strict_is_valid"] & df["rating_numeric"].notnull()]
    return sub.copy()


def cast_object(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns and isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype("object")
    return df


def fit_model(sub, label):
    dropped = [f for f in FACTORS if sub[f].nunique() <= 1]
    factors = [f for f in FACTORS if f not in dropped]
    if dropped:
        print(f"    WARNING [{label}]: dropped zero-variance factor(s): {dropped}")
    formula = "rating_numeric ~ " + " + ".join(f"C({f})" for f in factors)
    sub = cast_object(sub, ["persona_id"] + factors)
    res_cluster = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})
    print(f"    [{label}] n={len(sub):,}  n_persona_clusters={sub['persona_id'].nunique():,}  "
          f"R2={res_cluster.rsquared:.4f}")
    return res_cluster, formula


def extract_coefs(res, term_prefix):
    terms = [t for t in res.params.index if t.startswith(term_prefix)]
    out = pd.DataFrame({
        "term": terms,
        "level": [t[len(term_prefix):].strip("[]T.") for t in terms],
        "coef": [res.params[t] for t in terms],
        "se_cluster": [res.bse[t] for t in terms],
        "p_cluster": [res.pvalues[t] for t in terms],
    })
    return out


def exact_permutation_pvalue(x, y):
    """Fix H4(a): exact two-sided permutation p-value for Spearman rho at small n --
    enumerate ALL n! pairings (n<=5 here, so <=120, trivial to enumerate completely),
    not an asymptotic approximation.

    Inputs are rounded to 9 decimal places before ranking. In this fully balanced
    factorial design, two levels that are truly tied in adjusted mean (e.g. farmer and
    truck driver for llama/original) can come out of the OLS matrix solve differing only
    at the ~1e-13 level -- numerical noise, not a real difference. Unrounded, that noise
    silently breaks the tie and changes which rank (and thus which rho) scipy computes.
    Genuine differences in this data are of order 1e-2 or larger, so rounding to 1e-9
    restores true ties without touching any real distinction."""
    x = np.round(np.asarray(x, dtype=float), 9)
    y = np.round(np.asarray(y, dtype=float), 9)
    n = len(x)
    obs_rho, _ = stats.spearmanr(x, y)
    if np.isnan(obs_rho):
        obs_rho = 0.0
    count = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        total += 1
        rho, _ = stats.spearmanr(x, y[list(perm)])
        if np.isnan(rho):
            rho = 0.0
        if abs(rho) >= abs(obs_rho) - 1e-9:
            count += 1
    return obs_rho, count / total, total


def bootstrap_ranks(df, all_levels_by_factor, factors_formula, ref_levels, n_boot=N_BOOT, seed=BOOT_SEED):
    """Fix H4(b): persona bootstrap. Resamples whole persona_id clusters (not individual
    rows -- resampling rows would destroy the within-persona correlation structure and
    understate variance), refits the SAME full model each time, and records which level
    of each factor ranks top/bottom in that resample. Returns rank-position probabilities
    per level, not a single point estimate."""
    rng = np.random.default_rng(seed)
    persona_ids = df["persona_id"].unique()
    n_personas = len(persona_ids)
    groups = {pid: g for pid, g in df.groupby("persona_id", observed=True)}
    formula = "rating_numeric ~ " + " + ".join(f"C({f})" for f in factors_formula)

    counts = {factor: {lvl: {"top": 0, "bottom": 0} for lvl in levels}
              for factor, levels in all_levels_by_factor.items()}
    n_successful = 0

    for b in range(n_boot):
        sampled = rng.choice(persona_ids, size=n_personas, replace=True)
        boot_df = pd.concat([groups[pid] for pid in sampled], ignore_index=True)
        boot_df = cast_object(boot_df, factors_formula)
        # guard against a resample that happens to drop all variance in a factor
        if any(boot_df[f].nunique() <= 1 for f in factors_formula):
            continue
        try:
            res = smf.ols(formula, data=boot_df).fit()
        except Exception:
            continue
        n_successful += 1
        for factor, levels in all_levels_by_factor.items():
            coefs = extract_coefs(res, f"C({factor})[T.")
            level_coef = dict(zip(coefs["level"], coefs["coef"]))
            level_coef[ref_levels[factor]] = 0.0
            ranked = sorted(levels, key=lambda l: -level_coef.get(l, np.nan)
                             if not np.isnan(level_coef.get(l, np.nan)) else np.inf)
            counts[factor][ranked[0]]["top"] += 1
            counts[factor][ranked[-1]]["bottom"] += 1

    rows = []
    for factor, levels in all_levels_by_factor.items():
        for lvl in levels:
            rows.append({"factor": factor, "level": lvl,
                         "p_top": counts[factor][lvl]["top"] / n_successful if n_successful else np.nan,
                         "p_bottom": counts[factor][lvl]["bottom"] / n_successful if n_successful else np.nan,
                         "bootstrap_B": n_successful})
    return pd.DataFrame(rows)


def add_tie_aware_ranks(df, value_col, suffix):
    """Add descending rank groups, treating solver-level numerical ties as ties."""
    unique_values = []
    for value in sorted(df[value_col].dropna().unique(), reverse=True):
        if not any(np.isclose(value, seen, atol=TIE_ATOL, rtol=TIE_RTOL)
                   for seen in unique_values):
            unique_values.append(value)

    def rank_for(value):
        if pd.isna(value):
            return np.nan
        return next(
            rank for rank, group_value in enumerate(unique_values, start=1)
            if np.isclose(value, group_value, atol=TIE_ATOL, rtol=TIE_RTOL)
        )

    rank_col = f"rank_{suffix}"
    label_col = f"rank_label_{suffix}"
    df[rank_col] = df[value_col].map(rank_for)
    group_sizes = df[rank_col].value_counts()
    df[label_col] = df[rank_col].map(
        lambda rank: (
            f"tied for rank {int(rank)}"
            if pd.notna(rank) and group_sizes.get(rank, 0) > 1
            else f"rank {int(rank)}" if pd.notna(rank) else "not ranked"
        )
    )


def extreme_label(df, value_col, highest):
    """Return an explicit label and level set for a highest/lowest tie group."""
    extreme = df[value_col].max() if highest else df[value_col].min()
    tied = sorted(
        df.loc[np.isclose(df[value_col], extreme, atol=TIE_ATOL, rtol=TIE_RTOL), "level"]
        .astype(str)
        .tolist()
    )
    position = "highest" if highest else "lowest"
    label = " / ".join(tied)
    if len(tied) > 1:
        label += f", tied for {position}"
    return label, frozenset(tied)


def main():
    print("=" * 78)
    print("Load + filter to pilot subset, Condition A only, strict-valid")
    print("=" * 78)

    all_prof_rows = []
    all_country_rows = []
    exact_rows = []
    bootstrap_rows = []

    for m in MODEL_ORDER:
        print(f"\n{'='*78}\nMODEL: {m}\n{'='*78}")
        orig = filter_pilot_condA(load_safe(f"{ROOT}/results/full_results_{m}.csv", m))
        health = filter_pilot_condA(load_safe(f"{ROOT}/results_health/health_full_results_{m}.csv", m))
        print(f"  original: n={len(orig):,} valid Condition-A rows (pilot subset)")
        print(f"  health:   n={len(health):,} valid Condition-A rows (pilot subset)")

        res_orig, formula = fit_model(orig, f"{m}/original")
        res_health, _ = fit_model(health, f"{m}/health")

        ref_prof = sorted(PILOT_PROFESSIONS)[0]
        ref_ctry = sorted(PILOT_COUNTRIES)[0]

        # --- profession ranking comparison ---
        prof_o = extract_coefs(res_orig, "C(profession)[T.")
        prof_h = extract_coefs(res_health, "C(profession)[T.")
        prof_merged = prof_o[["level", "coef"]].rename(columns={"coef": "coef_orig"}).merge(
            prof_h[["level", "coef"]].rename(columns={"coef": "coef_health"}), on="level", how="outer")
        if ref_prof not in prof_merged["level"].values:
            prof_merged = pd.concat([prof_merged, pd.DataFrame(
                [{"level": ref_prof, "coef_orig": 0.0, "coef_health": 0.0}])], ignore_index=True)
        prof_merged["model"] = m
        all_prof_rows.append(prof_merged)

        rho_p, p_p, n_perm_p = exact_permutation_pvalue(prof_merged["coef_orig"], prof_merged["coef_health"])
        print(f"\n  --- PROFESSION ranking (n={len(prof_merged)}, reference='{ref_prof}') ---")
        print(f"  Spearman rho = {rho_p:.4f}   EXACT permutation p = {p_p:.4f}  ({n_perm_p} permutations enumerated, Fix H4a)")
        add_tie_aware_ranks(prof_merged, "coef_orig", "orig")
        add_tie_aware_ranks(prof_merged, "coef_health", "health")
        ranked = prof_merged.sort_values(["rank_orig", "level"])
        print(ranked[["level", "coef_orig", "coef_health", "rank_label_orig",
                      "rank_label_health"]].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
        top_o, top_o_set = extreme_label(prof_merged, "coef_orig", highest=True)
        bot_o, bot_o_set = extreme_label(prof_merged, "coef_orig", highest=False)
        top_h, top_h_set = extreme_label(prof_merged, "coef_health", highest=True)
        bot_h, bot_h_set = extreme_label(prof_merged, "coef_health", highest=False)
        print(f"  TOP profession:    original='{top_o}'   health='{top_h}'   "
              f"{'MATCH' if top_o_set == top_h_set else 'DIFFERENT'}")
        print(f"  BOTTOM profession: original='{bot_o}'   health='{bot_h}'   "
              f"{'MATCH' if bot_o_set == bot_h_set else 'DIFFERENT'}")

        # --- country ranking comparison ---
        ctry_o = extract_coefs(res_orig, "C(country)[T.")
        ctry_h = extract_coefs(res_health, "C(country)[T.")
        ctry_merged = ctry_o[["level", "coef"]].rename(columns={"coef": "coef_orig"}).merge(
            ctry_h[["level", "coef"]].rename(columns={"coef": "coef_health"}), on="level", how="outer")
        if ref_ctry not in ctry_merged["level"].values:
            ctry_merged = pd.concat([ctry_merged, pd.DataFrame(
                [{"level": ref_ctry, "coef_orig": 0.0, "coef_health": 0.0}])], ignore_index=True)
        ctry_merged["model"] = m
        all_country_rows.append(ctry_merged)

        rho_c, p_c, n_perm_c = exact_permutation_pvalue(ctry_merged["coef_orig"], ctry_merged["coef_health"])
        print(f"\n  --- COUNTRY ranking (n={len(ctry_merged)}, reference='{ref_ctry}') ---")
        print(f"  Spearman rho = {rho_c:.4f}   EXACT permutation p = {p_c:.4f}  ({n_perm_c} permutations enumerated, Fix H4a)")
        print(f"  (at n=4, minimum possible two-sided p is 2/24={2/24:.4f} -- p<.05 is UNREACHABLE here even with")
        print(f"   perfect rank agreement; this is a sample-size ceiling, not a negative finding, per Fix H4)")
        add_tie_aware_ranks(ctry_merged, "coef_orig", "orig")
        add_tie_aware_ranks(ctry_merged, "coef_health", "health")
        ranked_c = ctry_merged.sort_values(["rank_orig", "level"])
        print(ranked_c[["level", "coef_orig", "coef_health", "rank_label_orig",
                       "rank_label_health"]].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
        top_co, top_co_set = extreme_label(ctry_merged, "coef_orig", highest=True)
        bot_co, bot_co_set = extreme_label(ctry_merged, "coef_orig", highest=False)
        top_ch, top_ch_set = extreme_label(ctry_merged, "coef_health", highest=True)
        bot_ch, bot_ch_set = extreme_label(ctry_merged, "coef_health", highest=False)
        print(f"  TOP country:    original='{top_co}'   health='{top_ch}'   "
              f"{'MATCH' if top_co_set == top_ch_set else 'DIFFERENT'}")
        print(f"  BOTTOM country: original='{bot_co}'   health='{bot_ch}'   "
              f"{'MATCH' if bot_co_set == bot_ch_set else 'DIFFERENT'}")

        exact_rows.append({"model": m, "factor": "profession", "spearman_r": rho_p, "exact_permutation_p": p_p,
                            "n_permutations": n_perm_p, "top_orig": top_o, "top_health": top_h,
                            "bottom_orig": bot_o, "bottom_health": bot_h})
        exact_rows.append({"model": m, "factor": "country", "spearman_r": rho_c, "exact_permutation_p": p_c,
                            "n_permutations": n_perm_c, "top_orig": top_co, "top_health": top_ch,
                            "bottom_orig": bot_co, "bottom_health": bot_ch})

        # --- Fix H4(b): persona bootstrap, both framings ---
        print(f"\n  --- Persona bootstrap ({N_BOOT} resamples, Fix H4b) ---")
        for framing, sub in [("original", orig), ("health", health)]:
            factors_used = [f for f in FACTORS if sub[f].nunique() > 1]
            boot = bootstrap_ranks(
                sub,
                all_levels_by_factor={"profession": sorted(PILOT_PROFESSIONS), "country": sorted(PILOT_COUNTRIES)},
                factors_formula=factors_used,
                ref_levels={"profession": ref_prof, "country": ref_ctry},
            )
            boot["model"] = m
            boot["framing"] = framing
            bootstrap_rows.append(boot)
            prof_boot = boot[boot["factor"] == "profession"].sort_values("p_top", ascending=False)
            ctry_boot = boot[boot["factor"] == "country"].sort_values("p_top", ascending=False)
            print(f"    [{framing}] profession p_top (top rank-position probability): "
                  + ", ".join(f"{r.level}={r.p_top:.3f}" for r in prof_boot.itertuples()))
            print(f"    [{framing}] country    p_top (top rank-position probability): "
                  + ", ".join(f"{r.level}={r.p_top:.3f}" for r in ctry_boot.itertuples()))

    prof_all = pd.concat(all_prof_rows, ignore_index=True)
    country_all = pd.concat(all_country_rows, ignore_index=True)
    exact_df = pd.DataFrame(exact_rows)
    bootstrap_df = pd.concat(bootstrap_rows, ignore_index=True)

    os.makedirs(f"{ROOT}/analysis_health/output", exist_ok=True)
    prof_all.to_csv(f"{ROOT}/analysis_health/output/ranking_robustness_profession.csv", index=False)
    country_all.to_csv(f"{ROOT}/analysis_health/output/ranking_robustness_country.csv", index=False)
    exact_df.to_csv(f"{ROOT}/analysis_health/output/ranking_robustness_exact_pvalues.csv", index=False)
    bootstrap_df.to_csv(f"{ROOT}/analysis_health/output/ranking_robustness_bootstrap.csv", index=False)

    print(f"\n{'='*78}\nSUMMARY: exact permutation p-values (Fix H4a), per model\n{'='*78}")
    print(exact_df[["model", "factor", "spearman_r", "exact_permutation_p", "n_permutations"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n{'='*78}")
    print("NOTE: this tests whether the RELATIVE pattern (who ranks higher/lower than whom)")
    print("survives the framing change, separate from 01/02's finding that ABSOLUTE rating")
    print("levels shift downward under the health framing. Both can be true simultaneously --")
    print("a stable ranking riding on a lower baseline. Rank correlation and absolute-level")
    print("shift are different, non-contradictory properties of the same data.")
    print()
    print("NOTE (Fix H4): asymptotic Spearman p-values are NOT reported anywhere above -- they are")
    print("invalid at n=4/5. Exact permutation p-values and bootstrap rank-position probabilities are")
    print("the only significance evidence reported. At n=4 (country), p<.05 is mathematically")
    print("unreachable regardless of how strong the agreement is (minimum possible p=2/24=.0833) --")
    print("country rankings here can be DESCRIBED (via the bootstrap p_top/p_bottom) but not formally")
    print("significance-tested at the conventional threshold. This is a sample-size ceiling, not a")
    print("negative finding about country ranking stability.")
    print(f"\nWrote analysis_health/output/ranking_robustness_{{profession,country,exact_pvalues,bootstrap}}.csv")


if __name__ == "__main__":
    main()
