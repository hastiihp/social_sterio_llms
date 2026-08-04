"""Item 5 of Step 2: do the four naturalistic multi-turn conversational
framings (health, neutral, positive, negative_minor) agree with EACH OTHER
-- not just each vs. the original single-turn prompt? E.g. does neutral's
rating pattern correlate more with positive than with negative_minor?

This tests whether the four framings cluster by STRUCTURE (all four are
multi-turn conversations, vs. the single-turn original) or by CONTENT
similarity (e.g. neutral and positive both lack a vulnerability/adversity
cue, unlike health and negative_minor). Per the Step-0 terminology
correction, these four are not ordered on a valence scale, so "agree more"
here means correlate more in rating pattern -- not "are closer on a
positive-negative axis".

Condition A only (pilot subset, persona-clustered): no valid-NA option, so
no abstention/selection mechanism, matching the discipline used in
01_compare_context_vs_original.py's ranking-robustness section and
throughout analysis_health/.

Computes, for all 5 conditions (original + 4 contexts) pairwise -- with the
6 context-context pairs being the answer to item 5 specifically:
  1. Rating-pattern agreement: persona x topic matched-cell Spearman
     correlation and persona-clustered mean difference, per model.
  2. Ranking agreement: profession and country coefficient correlation
     (exact permutation p-values, Fix H4a) between every pair of
     conditions, per model.

Independent of analysis/ and analysis_health/. Writes only to
analysis_context/output/.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
_common = importlib.import_module("_common")
globals().update({k: getattr(_common, k) for k in dir(_common) if not k.startswith("__")})

OUT_DIR = f"{ROOT}/analysis_context/output"
ALL_CONDITIONS = ["original"] + CONTEXTS  # 5 conditions total
RANKING_MODEL_ORDER = ["llama", "gemma", "qwen", "ministral"]  # deepseek: see 01's dynamic skip, same reason applies here


def load_condA_pilot(condition, model):
    return filter_pilot_condA_valid(load_model_frame(condition, model))


def load_condA_full(condition, model):
    return filter_condA_valid(load_model_frame(condition, model))


def rating_agreement_matrix(model_order=MODEL_ORDER, scope="pilot"):
    """Item 5.1: pairwise Spearman + persona-clustered mean diff, Condition A,
    matched persona x topic cells, for every pair among the 5 conditions.
    scope="pilot" (default, unchanged, un-suffixed output) or "full" (all 5,400
    personas, writes cross_context_rating_agreement_full5400.csv)."""
    is_full = scope == "full"
    loader = load_condA_full if is_full else load_condA_pilot
    suffix = "_full5400" if is_full else ""
    scope_label = "full 5,400-persona dataset" if is_full else "180-persona pilot subset"
    print(f"\n{'='*78}\nITEM 5.1: rating-pattern agreement, all condition pairs, Condition A only "
          f"({scope_label})\n{'='*78}")
    rows = []
    frames = {c: {m: loader(c, m) for m in model_order} for c in ALL_CONDITIONS}

    for m in model_order:
        print(f"\n  --- {m} ---")
        for c1, c2 in itertools.combinations(ALL_CONDITIONS, 2):
            df1, df2 = frames[c1][m], frames[c2][m]
            key = ["persona_id", "country", "profession", "gender", "age", "topic"]
            merged = df1[key + ["rating_numeric"]].merge(
                df2[key + ["rating_numeric"]], on=key, suffixes=("_1", "_2"), how="inner")
            n = len(merged)
            if n < 2:
                print(f"    {c1:16s} vs {c2:16s}  n={n}  (too few matched cells, skipped)")
                rows.append({"model": m, "condition_1": c1, "condition_2": c2, "n_matched": n,
                             "spearman_r": np.nan, "spearman_p": np.nan,
                             "clustered_mean_diff_2_minus_1": np.nan, "clustered_diff_p": np.nan,
                             "is_context_pair": c1 in CONTEXTS and c2 in CONTEXTS})
                continue
            rho, rho_p = stats.spearmanr(merged["rating_numeric_1"], merged["rating_numeric_2"])
            gd = clustered_diff_context_minus_orig(merged, "rating_numeric_2", "rating_numeric_1", "persona_id")
            both_ctx = c1 in CONTEXTS and c2 in CONTEXTS
            tag = "  [context-context pair -- item 5]" if both_ctx else ""
            print(f"    {c1:16s} vs {c2:16s}  n={n:5d}  rho={rho:.4f} (p={rho_p:.2e})  "
                  f"diff(2-1)={gd['diff']:+.4f} (p={gd['p_value']:.2e}){tag}")
            rows.append({"model": m, "condition_1": c1, "condition_2": c2, "n_matched": n,
                         "spearman_r": rho, "spearman_p": rho_p,
                         "clustered_mean_diff_2_minus_1": gd["diff"], "clustered_diff_p": gd["p_value"],
                         "is_context_pair": both_ctx})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/cross_context_rating_agreement{suffix}.csv", index=False)
    return df


def ranking_agreement_matrix(model_order=RANKING_MODEL_ORDER, scope="pilot"):
    """Item 5.2: pairwise profession/country ranking correlation between every
    pair of conditions. scope="pilot" (default, unchanged): exact permutation
    p-values (4/5-level enumeration), un-suffixed output. scope="full": all 5,400
    personas, 20 countries / 30 professions -- exact enumeration is impossible at
    this n (see 01's section_ranking_robustness docstring), so this uses
    monte_carlo_permutation_pvalue (200,000 draws) instead, written to a separate
    cross_context_ranking_agreement_full5400.csv."""
    is_full = scope == "full"
    loader = load_condA_full if is_full else load_condA_pilot
    suffix = "_full5400" if is_full else ""
    profession_levels = ALL_PROFESSIONS if is_full else sorted(PILOT_PROFESSIONS)
    country_levels = ALL_COUNTRIES if is_full else sorted(PILOT_COUNTRIES)
    perm_fn = monte_carlo_permutation_pvalue if is_full else exact_permutation_pvalue
    perm_method_label = f"monte_carlo_{N_MONTE_CARLO}" if is_full else "exact_enumeration"
    ref_prof, ref_ctry = profession_levels[0], country_levels[0]
    scope_label = "full 5,400-persona dataset, 20 countries x 30 professions" if is_full \
        else "180-persona pilot subset"
    print(f"\n{'='*78}\nITEM 5.2: ranking agreement, all condition pairs, Condition A only "
          f"({scope_label})\n{'='*78}")
    rows = []

    for m in model_order:
        print(f"\n  --- {m} ---")
        coefs = {}
        for c in ALL_CONDITIONS:
            sub = loader(c, m)
            if len(sub) < MIN_VALID_FOR_RANKING:
                print(f"    {c}: SKIPPED (only {len(sub)} valid Condition-A rows)")
                continue
            res, _ = fit_model(sub, f"{m}/{c}")
            prof = extract_coefs(res, "C(profession)[T.").set_index("level")["coef"]
            ctry = extract_coefs(res, "C(country)[T.").set_index("level")["coef"]
            prof[ref_prof], ctry[ref_ctry] = 0.0, 0.0
            coefs[c] = {"profession": prof.reindex(profession_levels),
                        "country": ctry.reindex(country_levels)}

        for factor in ["profession", "country"]:
            available = [c for c in ALL_CONDITIONS if c in coefs]
            for c1, c2 in itertools.combinations(available, 2):
                rho, p, n_perm = perm_fn(coefs[c1][factor], coefs[c2][factor])
                both_ctx = c1 in CONTEXTS and c2 in CONTEXTS
                tag = "  [context-context pair]" if both_ctx else ""
                print(f"    {factor:11s} {c1:16s} vs {c2:16s}  rho={rho:.4f}  p={p:.4g} ({perm_method_label}){tag}")
                rows.append({"model": m, "factor": factor, "condition_1": c1, "condition_2": c2,
                             "spearman_r": rho, "permutation_p": p, "n_permutations": n_perm,
                             "method": perm_method_label, "is_context_pair": both_ctx})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/cross_context_ranking_agreement{suffix}.csv", index=False)
    return df


def summarize_clustering(rating_df, scope="pilot"):
    """Explicit answer to item 5's motivating question: do the four new
    contexts cluster together (vs. original) more than they differentiate
    from each other by content? Averages Spearman r within the 6
    context-context pairs vs. the 4 context-vs-original pairs, per model
    and pooled, and reports the single most/least similar context pair.
    scope="pilot" (default, unchanged) or "full" (writes _full5400-suffixed
    companion files instead of overwriting the pilot-scope ones)."""
    is_full = scope == "full"
    suffix = "_full5400" if is_full else ""
    scope_label = "full 5,400-persona dataset" if is_full else "180-persona pilot subset"
    print(f"\n{'='*78}\nSUMMARY: do the 4 contexts cluster by STRUCTURE (multi-turn) or CONTENT? "
          f"({scope_label})\n{'='*78}")
    valid = rating_df.dropna(subset=["spearman_r"])
    ctx_ctx = valid[valid["is_context_pair"]]
    ctx_orig = valid[~valid["is_context_pair"]]

    print(f"\n  Mean Spearman r, context-vs-original pairs (4 per model): "
          f"{ctx_orig['spearman_r'].mean():.4f}  (n={len(ctx_orig)})")
    print(f"  Mean Spearman r, context-vs-context pairs (6 per model):   "
          f"{ctx_ctx['spearman_r'].mean():.4f}  (n={len(ctx_ctx)})")
    diff = ctx_ctx['spearman_r'].mean() - ctx_orig['spearman_r'].mean()
    print(f"  Difference (context-context minus context-original): {diff:+.4f}")
    if diff > 0.02:
        print(f"  -> Contexts correlate MORE with each other than with the original -- consistent with")
        print(f"     clustering by STRUCTURE (multi-turn framing itself), not just shared content.")
    elif diff < -0.02:
        print(f"  -> Contexts do NOT correlate more with each other than with the original -- no evidence")
        print(f"     of a structural multi-turn cluster distinct from content effects.")
    else:
        print(f"  -> Essentially no difference -- context-context and context-original agreement are similar.")

    print(f"\n  Per context-pair mean Spearman r (pooled across models), most to least similar:")
    pair_means = ctx_ctx.groupby(["condition_1", "condition_2"])["spearman_r"].mean().sort_values(ascending=False)
    for (c1, c2), r in pair_means.items():
        print(f"    {c1:16s} vs {c2:16s}  mean rho={r:.4f}")
    most_similar = pair_means.index[0]
    least_similar = pair_means.index[-1]
    print(f"\n  Most similar context pair:  {most_similar[0]} vs {most_similar[1]} (rho={pair_means.iloc[0]:.4f})")
    print(f"  Least similar context pair: {least_similar[0]} vs {least_similar[1]} (rho={pair_means.iloc[-1]:.4f})")

    summary = pd.DataFrame([
        {"comparison": "context_vs_original", "mean_spearman_r": ctx_orig["spearman_r"].mean(), "n_pairs": len(ctx_orig)},
        {"comparison": "context_vs_context", "mean_spearman_r": ctx_ctx["spearman_r"].mean(), "n_pairs": len(ctx_ctx)},
    ])
    summary.to_csv(f"{OUT_DIR}/cross_context_clustering_summary{suffix}.csv", index=False)
    pair_means.reset_index().rename(columns={"spearman_r": "mean_spearman_r"}).to_csv(
        f"{OUT_DIR}/cross_context_pairwise_similarity_ranked{suffix}.csv", index=False)
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rating_df = rating_agreement_matrix(scope="pilot")
    ranking_agreement_matrix(scope="pilot")
    summarize_clustering(rating_df, scope="pilot")

    rating_df_full = rating_agreement_matrix(scope="full")
    ranking_agreement_matrix(scope="full")
    summarize_clustering(rating_df_full, scope="full")

    print(f"\n{'='*78}\nDONE. Outputs written to {OUT_DIR}/cross_context_*.csv (pilot) and "
          f"{OUT_DIR}/cross_context_*_full5400.csv (full)\n{'='*78}")


if __name__ == "__main__":
    main()
