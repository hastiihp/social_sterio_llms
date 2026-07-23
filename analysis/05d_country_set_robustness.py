"""Step 5d (Tier 2, Fix 9): original-10 vs. added-10 country-set robustness
check -- analysis_plan.md Section 8, robustness check #1.

Tests whether profession and country effect sizes/rankings are stable
between personas from the original 10 countries and the 10 added
countries, using the pooled model (llama/gemma/qwen/ministral, Fix 4's
version, Condition A only).

APPROACH CHOSEN: two separately-fit models (original_10 subset vs
added_10 subset) with a rank-correlation comparison of profession effects,
rather than adding a country_set x profession interaction term to the
existing 236-parameter pooled model. Reasons: (1) this mirrors the
already-validated approach from Fix 7's OLS-vs-ordinal-logit profession
comparison; (2) country_set is a deterministic function of country (each
country belongs to exactly one set), so a country_set x country
interaction term would be degenerate/redundant -- country already fully
determines country_set; (3) two clean, separately-interpretable fits are
easier to sanity-check than a further-inflated single model.

country_set is not present in master_results.csv (it is a personas.csv
design-file column, not part of the model output) and is joined in here
via the country column.

Countries do not overlap between the two subsets, so profession
coefficients ARE directly rank-correlated between fits (both fits use the
same 30 professions), but country coefficients are NOT (each subset has
only its own 10 countries) -- country comparison instead reports each
subset's own ranking and the spread (max-min) of country effects, to see
whether geographic effects are similarly sized regardless of which 10
countries are examined.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from _common import load_master, cast_formula_dtypes, TABLES_DIR, ROOT_DIR

POOLED_MODELS = ["llama", "gemma", "qwen", "ministral"]
POOLED_FACTORS = ["topic", "profession", "country", "gender", "age"]


def load_country_set_map():
    personas = pd.read_csv(f"{ROOT_DIR}/data/personas.csv")
    mapping = personas[["country", "country_set"]].drop_duplicates().set_index("country")["country_set"]
    print(f"country_set mapping ({mapping.nunique()} sets): "
          f"{mapping.value_counts().to_dict()}")
    return mapping


def fit_pooled_subset(sub, label):
    dropped = [f for f in POOLED_FACTORS if sub[f].nunique() <= 1]
    factors = [f for f in POOLED_FACTORS if f not in dropped]
    if dropped:
        print(f"  [{label}] dropped zero-variance factor(s): {dropped}")

    model_term = 'C(model, Treatment(reference="llama"))'
    main_terms = " + ".join(f"C({f})" for f in factors) + f" + {model_term}"
    interaction_terms = " + ".join(f"C({f}):{model_term}" for f in factors)
    formula = f"rating_numeric ~ {main_terms} + {interaction_terms}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_hc3 = smf.ols(formula, data=sub).fit(cov_type="HC3")
        for w in caught:
            print(f"  WARNING (HC3, {label}): {w.message}")
    res_cluster = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})

    print(f"  [{label}] n={len(sub):,}  n_params={len(res_hc3.params)}  R2={res_hc3.rsquared:.4f}  "
          f"n_countries={sub['country'].nunique()}")

    ci = res_cluster.conf_int()
    ct = pd.DataFrame({
        "subset": label, "term": res_hc3.params.index, "coef": res_hc3.params.values,
        "se_hc3": res_hc3.bse.values, "p_hc3": res_hc3.pvalues.values,
        "se_cluster": res_cluster.bse.values, "p_cluster": res_cluster.pvalues.values,
    })
    return ct, res_hc3.rsquared, len(sub)


def main():
    df = cast_formula_dtypes(load_master())
    country_set = load_country_set_map()

    sub = df[(df["response_condition"] == "A_forced") & df["strict_is_valid"] & df["rating_numeric"].notnull()]
    sub = sub[sub["model"].isin(POOLED_MODELS)].copy()
    sub["country_set"] = sub["country"].map(country_set)
    assert sub["country_set"].notnull().all(), "some countries failed to map to a country_set"

    print(f"\nTotal pooled Condition-A rows (4 models): {len(sub):,}")
    print(sub.groupby("country_set", observed=True).size().to_string())

    results = {}
    for label in ["original_10", "added_10"]:
        subset = sub[sub["country_set"] == label]
        print(f"\n{'='*78}\n{label.upper()}\n{'='*78}")
        ct, r2, n = fit_pooled_subset(subset, label)
        results[label] = ct

    combined = pd.concat(results.values(), ignore_index=True)
    out_path = f"{TABLES_DIR}/country_set_robustness.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    print(f"\n{'='*78}\nPROFESSION RANKING COMPARISON: original_10 vs added_10\n{'='*78}")
    prof_o = results["original_10"][results["original_10"]["term"].str.match(r"^C\(profession\)\[T\.[^]]+\]$")].copy()
    prof_a = results["added_10"][results["added_10"]["term"].str.match(r"^C\(profession\)\[T\.[^]]+\]$")].copy()
    prof_o["profession"] = prof_o["term"].str.extract(r"\[T\.([^]]+)\]")
    prof_a["profession"] = prof_a["term"].str.extract(r"\[T\.([^]]+)\]")
    merged = prof_o[["profession", "coef"]].rename(columns={"coef": "coef_original10"}).merge(
        prof_a[["profession", "coef"]].rename(columns={"coef": "coef_added10"}), on="profession")
    rho, p = stats.spearmanr(merged["coef_original10"], merged["coef_added10"])
    print(f"Spearman rho (n={len(merged)} professions): {rho:.4f}  (p={p:.2e})")
    print(f"\nTop 5 professions, original_10 vs added_10:")
    print(f"  original_10: {merged.sort_values('coef_original10', ascending=False).head(5)['profession'].tolist()}")
    print(f"  added_10:    {merged.sort_values('coef_added10', ascending=False).head(5)['profession'].tolist()}")
    print(f"Bottom 5 professions, original_10 vs added_10:")
    print(f"  original_10: {merged.sort_values('coef_original10').head(5)['profession'].tolist()}")
    print(f"  added_10:    {merged.sort_values('coef_added10').head(5)['profession'].tolist()}")
    merged.to_csv(f"{TABLES_DIR}/country_set_profession_comparison.csv", index=False)

    print(f"\n{'='*78}\nCOUNTRY EFFECTS: each subset's own ranking (10 countries each, no overlap --")
    print(f"reported separately, spread compared, not rank-correlated)\n{'='*78}")
    for label in ["original_10", "added_10"]:
        ctry = results[label][results[label]["term"].str.match(r"^C\(country\)\[T\.[^]]+\]$")].copy()
        ctry["country"] = ctry["term"].str.extract(r"\[T\.([^]]+)\]")
        ctry = ctry.sort_values("coef", ascending=False)
        spread = ctry["coef"].max() - ctry["coef"].min()
        print(f"\n  {label} (n={len(ctry)+1} countries incl. reference, spread={spread:.4f}):")
        print(ctry[["country", "coef", "p_cluster"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n{'='*78}\nMAIN-EFFECT TERMS (topic, gender, age, model): direction/significance comparison\n{'='*78}")
    for term_prefix in ["C(topic)", "C(gender)", "C(age)", 'C(model, Treatment(reference="llama"))']:
        o = results["original_10"][results["original_10"]["term"].str.startswith(term_prefix) &
                                     ~results["original_10"]["term"].str.contains(":")]
        a = results["added_10"][results["added_10"]["term"].str.startswith(term_prefix) &
                                  ~results["added_10"]["term"].str.contains(":")]
        m = o[["term", "coef", "p_cluster"]].rename(columns={"coef": "coef_o", "p_cluster": "p_o"}).merge(
            a[["term", "coef", "p_cluster"]].rename(columns={"coef": "coef_a", "p_cluster": "p_a"}), on="term")
        m["sign_agrees"] = np.sign(m["coef_o"]) == np.sign(m["coef_a"])
        print(f"\n  {term_prefix}: {m['sign_agrees'].sum()}/{len(m)} terms agree in sign between subsets")
        print(m.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
