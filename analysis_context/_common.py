"""Shared, parametrized helpers for the analysis_context/ scripts.

This module generalizes the audited logic from analysis_health/01, 02, and
04 (persona-clustered inference, exact permutation p-values for small-n
rankings, the persona bootstrap for rank-position uncertainty, tie-aware
ranking) so it is written once and reused across all four conversational
framings (health, neutral, positive, negative_minor), instead of being
re-derived per context. Every function below is a direct generalization of
its analysis_health counterpart -- same formulas, same fixes (H1, H4, H6),
renamed from "health"-specific to "context"-generic naming. See
analysis_health/01_compare_health_vs_original.py and
04_ranking_robustness.py for the original audited derivations and the
Fix H1/H4/H6 rationale, which is not repeated in full here.

Terminology note (Step 0 of the cross-context task): health, neutral,
positive, and negative_minor are CONVERSATIONAL FRAMINGS / CONTEXTS, not
points on a single emotional-valence scale. They differ along multiple
dimensions at once, not valence alone:
  - health (stress/sleep):        signals personal vulnerability
  - positive (promotion):         signals career success / competence,
                                   not just "positive affect"
  - negative_minor (travel mishap): an external, impersonal event that
                                   happens TO the persona, not about them
  - neutral (moving apartments):  domain-neutral small talk
Nowhere in this module or its callers should these be treated as ordered
points on a positive-to-negative axis.
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The four naturalistic multi-turn conversational framings compared against
# the original single-turn persona prompt. "original" is not itself a
# context in this dict -- it is always the comparison baseline.
CONTEXTS = ["health", "neutral", "positive", "negative_minor"]

RESULTS_PATH_TEMPLATE = {
    "original": f"{ROOT}/results/results_original_{{m}}.csv",
    "health": f"{ROOT}/results/results_health_{{m}}.csv",
    "neutral": f"{ROOT}/results/results_neutral_{{m}}.csv",
    "positive": f"{ROOT}/results/results_positive_{{m}}.csv",
    "negative_minor": f"{ROOT}/results/results_negative_minor_{{m}}.csv",
}

# Same 180-persona pilot subset established in analysis_health (4 countries x
# 5 professions x 3 genders x 3 ages = 180). Re-verified against
# data/personas.csv for this task (see conversation) rather than redefined:
# every results_*/*.csv covers the same full 5,400-persona design, so this
# subset is derived by filtering, not loaded from a dedicated file, exactly
# as in analysis_health.
PILOT_COUNTRIES = ["Germany", "Brazil", "Nigeria", "South Korea"]
PILOT_PROFESSIONS = ["lawyer", "registered nurse", "truck driver", "farmer", "computer programmer"]
EXPECTED_PILOT_PERSONAS = len(PILOT_COUNTRIES) * len(PILOT_PROFESSIONS) * 3 * 3  # 180

# Full-design counterpart of PILOT_COUNTRIES/PILOT_PROFESSIONS, for the full-scale
# (5,400-persona) companion analyses. Loaded from data/personas.csv -- the same
# canonical source PILOT_COUNTRIES/PILOT_PROFESSIONS are re-verified against
# elsewhere in this project -- rather than hardcoded, so this can never silently
# drift from the actual persona grid.
_personas = pd.read_csv(f"{ROOT}/data/personas.csv")
ALL_COUNTRIES = sorted(_personas["country"].unique().tolist())
ALL_PROFESSIONS = sorted(_personas["profession"].unique().tolist())
assert len(ALL_COUNTRIES) == 20 and len(ALL_PROFESSIONS) == 30, (
    f"expected 20 countries / 30 professions, got {len(ALL_COUNTRIES)}/{len(ALL_PROFESSIONS)}")
del _personas

MODEL_ORDER = ["llama", "gemma", "qwen", "ministral", "deepseek"]
CONDITIONS = ["A_forced", "B_optional"]
TOPIC_ORDER = ["climate change", "economic redistribution", "gender equality",
               "immigration", "lgbtq rights", "religion and secularism", "trust in government"]
FACTORS = ["gender", "country", "profession", "age", "topic"]
N_BOOT = 1000
# Full-scale (5,400-persona, 20 countries x 30 professions) bootstrap uses fewer
# resamples than the pilot's 1,000: each full-scale OLS refit costs ~30x more
# rows and ~10x more dummy parameters than the pilot fit, so 1,000 full-scale
# resamples would cost on the order of hours across every (model, context,
# framing) combination in 01_compare_context_vs_original.py. 300 resamples is
# still far more than the 24-120 permutations the pilot's EXACT test enumerates
# at n=4/5, and rank-position probabilities (top/bottom %) are a coarse,
# fast-converging statistic -- documented compute-budget tradeoff, not a
# precision compromise on the primary significance claims (those come from
# monte_carlo_permutation_pvalue at 200,000 draws, not this bootstrap).
N_BOOT_FULL = 300
BOOT_SEED = 0
TIE_ATOL = 1e-10
TIE_RTOL = 1e-9

# Minimum valid Condition-A pilot rows required before a ranking model is
# fit at all. deepseek never has enough (63/75,600 valid in original, 0 in
# health/neutral/positive, 204/75,600 in negative_minor -- all far below
# what's needed to fit gender+country+profession+age+topic on 180 personas
# x 7 topics = 1,260 possible pairs). Checked dynamically per (model,
# context) rather than hardcoding a skip list, so this generalizes cleanly
# and prints a warning instead of silently omitting a model.
MIN_VALID_FOR_RANKING = 60  # ~1 obs per (persona x topic) pair minimum; well below this, OLS is unstable/rank-deficient


def load_safe(path):
    """The literal string "NA" in raw_text/strict_parsed_rating is a valid,
    meaningful Condition-B abstention value, not a missing-value marker --
    keep_default_na=False prevents pandas from silently nulling it out."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    df["rating_numeric"] = pd.to_numeric(df["strict_parsed_rating"], errors="coerce")
    return df


def load_model_frame(context, model):
    """context: 'original' or one of CONTEXTS."""
    path = RESULTS_PATH_TEMPLATE[context].format(m=model)
    df = load_safe(path)
    df.insert(0, "model", model)
    return df


def filter_pilot(df):
    return df[df["country"].isin(PILOT_COUNTRIES) & df["profession"].isin(PILOT_PROFESSIONS)].copy()


def filter_condA_valid(df):
    """Full-5,400-persona counterpart of filter_pilot_condA_valid: same Condition-A
    + strict-valid filter, but no country/profession restriction -- all 20
    countries and all 30 professions. Used by the full-scale (FULL_SCOPE)
    companion sections added alongside every pilot-scope (180-persona) analysis,
    per the Step-N full-scale extension task: produce both, never silently
    replace the pilot-scope scope that the rest of this project's cross-context
    work is built on."""
    sub = df[(df["response_condition"] == "A_forced") & df["strict_is_valid"] & df["rating_numeric"].notnull()]
    return sub.copy()


def filter_pilot_condA_valid(df):
    sub = df[df["country"].isin(PILOT_COUNTRIES) & df["profession"].isin(PILOT_PROFESSIONS) &
             (df["response_condition"] == "A_forced") & df["strict_is_valid"] & df["rating_numeric"].notnull()]
    return sub.copy()


def cast_object(df, cols):
    """Guard against pandas 3.0's StringDtype breaking statsmodels formula fitting."""
    df = df.copy()
    for c in cols:
        if c in df.columns and isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype("object")
    return df


def clustered_diff_context_minus_orig(data, context_col, orig_col, cluster_col):
    """Generalization of analysis_health's Fix-H1 clustered_diff_health_minus_orig:
    persona-clustered one-sample test of the mean of (context_col - orig_col),
    computed as an explicit row-wise difference (immune by construction to the
    categorical-reference-level sign bug that Fix H1 removed). Sign convention:
    positive = context is higher, negative = context is lower."""
    d = data.copy()
    d["_diff"] = d[context_col].astype(float) - d[orig_col].astype(float)
    n_clusters = d[cluster_col].nunique()
    if n_clusters < 2:
        return dict(diff=np.nan, se_cluster=np.nan, p_value=np.nan, n_clusters=n_clusters)
    d = cast_object(d, [cluster_col])
    res = smf.ols("_diff ~ 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster_col]})
    return dict(diff=float(res.params["Intercept"]), se_cluster=float(res.bse["Intercept"]),
                p_value=float(res.pvalues["Intercept"]), n_clusters=n_clusters)


def fit_model(sub, label, factors=FACTORS):
    dropped = [f for f in factors if sub[f].nunique() <= 1]
    used = [f for f in factors if f not in dropped]
    if dropped:
        print(f"    WARNING [{label}]: dropped zero-variance factor(s): {dropped}")
    formula = "rating_numeric ~ " + " + ".join(f"C({f})" for f in used)
    sub = cast_object(sub, ["persona_id"] + used)
    res_cluster = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})
    print(f"    [{label}] n={len(sub):,}  n_persona_clusters={sub['persona_id'].nunique():,}  "
          f"R2={res_cluster.rsquared:.4f}")
    return res_cluster, used


def extract_coefs(res, term_prefix):
    """Bug found during the full-scale extension task, fixed here: the level name
    was previously recovered via `t[len(term_prefix):].strip("[]T.")` -- .strip()
    removes ANY leading/trailing characters found in its argument, treated as a
    character SET, not a fixed prefix/suffix. term_prefix already consumes the
    "[T." prefix (e.g. "C(country)[T."), so the remainder is "LevelName]" and only
    the trailing "]" should be removed -- but .strip("[]T.") also stripped a
    literal leading 'T' from any level name starting with one, silently turning
    "Turkey" into "urkey". This was dormant everywhere in the project until now:
    no pilot-scope country (Germany/Brazil/Nigeria/South Korea) or profession
    (lawyer/registered nurse/truck driver/farmer/computer programmer) starts or
    ends with '[', ']', 'T', or '.', so it never fired in any pilot-scope output,
    nor in the original health-study module this was generalized from
    (analysis_health/04_ranking_robustness.py has the identical bug, equally
    dormant there for the same reason -- not fixed there, out of scope for this
    task, and that module never touches the full 20-country design where "Turkey"
    appears). Only the full-scale (20-country) companion analyses added in this
    task could ever have hit it. Fixed by removing exactly one trailing "]"
    instead of stripping a character set."""
    terms = [t for t in res.params.index if t.startswith(term_prefix)]
    return pd.DataFrame({
        "term": terms,
        "level": [t[len(term_prefix):].removesuffix("]") for t in terms],
        "coef": [res.params[t] for t in terms],
        "se_cluster": [res.bse[t] for t in terms],
        "p_cluster": [res.pvalues[t] for t in terms],
    })


def exact_permutation_pvalue(x, y):
    """Fix H4(a), unchanged: exact two-sided permutation p-value for Spearman rho
    at small n (<=5 here) -- enumerates ALL n! pairings, not an asymptotic
    approximation. Inputs rounded to 9dp first to restore true ties that OLS
    numerical noise (~1e-13) would otherwise silently break."""
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


N_MONTE_CARLO = 200_000


def monte_carlo_permutation_pvalue(x, y, n_draws=N_MONTE_CARLO, seed=0):
    """Full-scale (20 countries / 30 professions) counterpart of
    exact_permutation_pvalue: exhaustive enumeration is exact_permutation_pvalue's
    whole point at pilot scale (n=4/5, 24/120 permutations -- trivial to enumerate
    fully), but is computationally impossible at full scale (20! and 30! are
    astronomically large, not "slow" -- there is no borderline case here). This
    function is NOT a silent substitute: every full-scale ranking-robustness output
    that uses it is written to a separate _full5400-suffixed file/column stating
    the method explicitly, never merged into or overwriting an exact-test output.

    Vectorized (not a Python-level loop over draws): ranks x and y once, generates
    n_draws random permutations of y's ranks as a (n_draws, n) matrix, and computes
    every permuted Spearman rho in one matrix-vector product (Pearson correlation on
    ranks == Spearman correlation; mean/std of the rank vector are permutation-
    invariant, so only the numerator changes per draw). Benchmarked at n=30:
    200,000 draws in ~0.1s, vs. ~30s+ for an unvectorized scipy-per-draw loop, and
    cross-validated against exact_permutation_pvalue at n=5 (agreement to 3
    significant figures with 200,000 draws). Uses add-one (Laplace) smoothing on
    the p-value, standard practice for Monte Carlo permutation tests, so p is never
    reported as exactly 0 -- floor is 1/(n_draws+1).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    obs_rho, _ = stats.spearmanr(x, y)
    if np.isnan(obs_rho):
        obs_rho = 0.0
    xr, yr = stats.rankdata(x), stats.rankdata(y)
    xc, yc = xr - xr.mean(), yr - yr.mean()
    std_x, std_y = xc.std(), yc.std()
    if std_x == 0 or std_y == 0:
        return obs_rho, np.nan, n_draws
    rng = np.random.default_rng(seed)
    perm_idx = np.argsort(rng.random((n_draws, n)), axis=1)
    yperm = yc[perm_idx]
    corrs = (yperm @ xc) / (n * std_x * std_y)
    count = int(np.sum(np.abs(corrs) >= abs(obs_rho) - 1e-9))
    return obs_rho, (count + 1) / (n_draws + 1), n_draws


def bootstrap_ranks(df, all_levels_by_factor, factors_formula, ref_levels, n_boot=N_BOOT, seed=BOOT_SEED):
    """Fix H4(b): persona bootstrap (resamples whole persona_id clusters, not
    rows) -> rank-position (top/bottom) probability per level.

    Resampling is done via precomputed positional indices (df.groupby(...).indices,
    computed once) + np.concatenate + .iloc, instead of a per-iteration
    groupby-dict-slice-then-pd.concat. This is a pure performance optimization for
    the full-scale (5,400-persona) companion analyses, where the naive approach's
    per-iteration group-slicing dominates runtime -- it draws the exact same
    persona_id sequence per iteration (same rng calls, same seed) and produces the
    same resampled rows (only their materialization order can differ, which OLS
    fitting is invariant to), so pilot-scope (180-persona) results are byte-for-byte
    unchanged from before this change."""
    rng = np.random.default_rng(seed)
    persona_ids = df["persona_id"].unique()
    n_personas = len(persona_ids)
    idx_by_persona = df.groupby("persona_id", observed=True).indices  # dict: pid -> positional index array
    formula = "rating_numeric ~ " + " + ".join(f"C({f})" for f in factors_formula)

    counts = {factor: {lvl: {"top": 0, "bottom": 0} for lvl in levels}
              for factor, levels in all_levels_by_factor.items()}
    n_successful = 0

    for _ in range(n_boot):
        sampled = rng.choice(persona_ids, size=n_personas, replace=True)
        positions = np.concatenate([idx_by_persona[pid] for pid in sampled])
        boot_df = df.iloc[positions]
        boot_df = cast_object(boot_df, factors_formula)
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
    unique_values = []
    for value in sorted(df[value_col].dropna().unique(), reverse=True):
        if not any(np.isclose(value, seen, atol=TIE_ATOL, rtol=TIE_RTOL) for seen in unique_values):
            unique_values.append(value)

    def rank_for(value):
        if pd.isna(value):
            return np.nan
        return next(rank for rank, gv in enumerate(unique_values, start=1)
                    if np.isclose(value, gv, atol=TIE_ATOL, rtol=TIE_RTOL))

    rank_col, label_col = f"rank_{suffix}", f"rank_label_{suffix}"
    df[rank_col] = df[value_col].map(rank_for)
    group_sizes = df[rank_col].value_counts()
    df[label_col] = df[rank_col].map(
        lambda rank: (f"tied for rank {int(rank)}" if pd.notna(rank) and group_sizes.get(rank, 0) > 1
                      else f"rank {int(rank)}" if pd.notna(rank) else "not ranked"))


def extreme_label(df, value_col, highest):
    extreme = df[value_col].max() if highest else df[value_col].min()
    tied = sorted(df.loc[np.isclose(df[value_col], extreme, atol=TIE_ATOL, rtol=TIE_RTOL), "level"]
                  .astype(str).tolist())
    position = "highest" if highest else "lowest"
    label = " / ".join(tied)
    if len(tied) > 1:
        label += f", tied for {position}"
    return label, frozenset(tied)


def merge_on_canonical_key(orig_df, ctx_df, extra_cols=("strict_is_valid", "is_abstention", "rating_numeric")):
    """Fix H6, unchanged: merge on the full canonical key with
    validate='one_to_one' so a many-to-one/one-to-many merge raises instead
    of silently duplicating or dropping rows."""
    key = ["model", "persona_id", "country", "profession", "gender", "age", "topic", "response_condition"]
    cols = key + list(extra_cols)
    merged_ind = orig_df[cols].merge(ctx_df[cols], on=key, suffixes=("_orig", "_ctx"),
                                      how="outer", validate="one_to_one", indicator=True)
    return merged_ind
