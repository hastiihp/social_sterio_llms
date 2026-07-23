"""Fix 7, final attempt at the proportional-odds check: within each topic.

Two prior attempts failed for all 4 models:
  1. Full formula (gender+country+profession+age+condition+topic, ~60 params)
     -- every cutpoint-specific binary logit hit convergence failure or a
     singular matrix (too many dummies relative to sparse cells at extreme
     cutpoints).
  2. Reduced formula (gender+condition+topic only, ~9 params) -- still
     failed. Traced to topic itself: topic's effect is so strong and
     concentrated (75-98% of the ordinal model's pseudo-R^2 in step
     05b_ordinal_robustness.py) that some topics have a deterministic (0%
     or 100%) response rate at specific cutpoints, so the topic dummy
     perfectly predicts the binary outcome there -- the same separation
     pattern already seen in steps 6 and 7b.

This final attempt removes topic from the formula entirely by stratifying:
fit the cutpoint-specific binary logits WITHIN each topic separately
(formula: gender + condition only, ~3 params), for each of the 4 main
models. If a given (model, topic) cell still fails to converge, it is
reported as untestable for that cell specifically -- not silently dropped,
not assumed to satisfy the assumption. No further formula or method changes
are attempted after this per instruction.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from _common import load_master, cast_formula_dtypes, TABLES_DIR

INCLUDED_MODELS = ["llama", "gemma", "qwen", "ministral"]
CUTPOINTS = [1, 2, 3, 4]


def valid_subset(df, model):
    sub = df[(df["model"] == model) & df["strict_is_valid"] & df["rating_numeric"].notnull()].copy()
    sub["rating_cat"] = sub["rating_numeric"].astype(int)
    return sub


def test_one_topic(topic_sub, model_label, topic):
    formula = "answered_le_c ~ C(gender) + C(response_condition)"
    cutpoint_results = {}
    detail = []
    for c in CUTPOINTS:
        cp_sub = topic_sub.copy()
        cp_sub["answered_le_c"] = (cp_sub["rating_cat"] <= c).astype(int)
        rate = cp_sub["answered_le_c"].mean()
        if rate <= 0.001 or rate >= 0.999:
            detail.append(f"c<={c}:extreme({rate:.3f})")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_c = smf.logit(formula, data=cp_sub).fit(disp=0)
            if not res_c.mle_retvals.get("converged", True):
                detail.append(f"c<={c}:no-converge")
                continue
            cutpoint_results[c] = res_c
            detail.append(f"c<={c}:ok({rate:.3f})")
        except Exception as e:
            detail.append(f"c<={c}:FAILED({type(e).__name__})")

    if len(cutpoint_results) < 2:
        return [{"model": model_label, "topic": topic, "term": None, "testable": False,
                 "n_cutpoints_usable": len(cutpoint_results), "detail": "; ".join(detail),
                 "Q_stat": np.nan, "df": np.nan, "p_value": np.nan, "violated_at_0.05": None}]

    all_terms = set.intersection(*[set(r.params.index) for r in cutpoint_results.values()])
    rows = []
    for term in sorted(all_terms):
        betas = np.array([cutpoint_results[c].params[term] for c in cutpoint_results])
        ses = np.array([cutpoint_results[c].bse[term] for c in cutpoint_results])
        if np.any(ses <= 0) or not np.all(np.isfinite(ses)) or not np.all(np.isfinite(betas)):
            rows.append({"model": model_label, "topic": topic, "term": term, "testable": False,
                         "n_cutpoints_usable": len(cutpoint_results), "detail": "non-finite SE/coef",
                         "Q_stat": np.nan, "df": np.nan, "p_value": np.nan, "violated_at_0.05": None})
            continue
        weights = 1.0 / (ses ** 2)
        beta_bar = np.sum(betas * weights) / np.sum(weights)
        Q = np.sum(weights * (betas - beta_bar) ** 2)
        dof = len(cutpoint_results) - 1
        p = 1 - stats.chi2.cdf(Q, dof) if dof > 0 else np.nan
        rows.append({"model": model_label, "topic": topic, "term": term, "testable": True,
                     "n_cutpoints_usable": len(cutpoint_results), "detail": "; ".join(detail),
                     "Q_stat": Q, "df": dof, "p_value": p, "violated_at_0.05": bool(p < 0.05) if dof > 0 else None})
    return rows


def main():
    df = cast_formula_dtypes(load_master())
    all_rows = []
    for model in INCLUDED_MODELS:
        sub = valid_subset(df, model)
        print(f"\n{'='*78}\nMODEL: {model}\n{'='*78}")
        for topic in sorted(sub["topic"].unique()):
            topic_sub = sub[sub["topic"] == topic]
            rows = test_one_topic(topic_sub, model, topic)
            all_rows.extend(rows)
            for r in rows:
                if not r["testable"]:
                    print(f"  {topic:28s} {str(r['term']):25s} UNTESTABLE  ({r['detail']})")
                else:
                    flag = "VIOLATED" if r["violated_at_0.05"] else "holds"
                    print(f"  {topic:28s} {r['term']:25s} Q={r['Q_stat']:.3f} df={r['df']} p={r['p_value']:.4g} -> {flag}")

    result = pd.DataFrame(all_rows)
    out_path = f"{TABLES_DIR}/ordinal_proportional_odds_by_topic.csv"
    result.to_csv(out_path, index=False)

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    n_total_cells = len(result[result["term"].notna()]) + result[result["term"].isna()].shape[0]
    testable = result[result["testable"] == True]
    untestable = result[result["testable"] == False]
    print(f"Total (model, topic, term) rows: {len(result)}")
    print(f"Testable: {len(testable)}   Untestable: {len(untestable)}")
    if len(testable):
        n_violated = testable["violated_at_0.05"].sum()
        print(f"Of testable rows: {n_violated}/{len(testable)} show significant heterogeneity (p<0.05) "
              f"-- proportional odds violated for those.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
