"""Step 5: H1 variance-decomposition (partial R^2) comparison across all five
prompt types -- original, health, neutral, positive, negative_minor.

Extends analysis/08_variance_ranking.py's primary, preregistered H1 test
(Condition A only; partial R^2 per factor via nested-model SSE reduction;
persona-clustered joint Wald test for significance, alongside the point
estimate) from "original only" to all five prompt types, using the identical
formula and method throughout: rating_numeric ~ C(topic) + C(profession) +
C(country) + C(gender) + C(age).

No new inference and no refit of "original": its numbers are read directly
from tables/variance_ranking.csv (scope == "primary_conditionA"), the
already-audited values that survived clean-environment regeneration in
audit_full_project (Sections 14, 17, 22 -- GO). health/neutral/positive/
negative_minor are freshly fit here from results/ (see _common.py's
RESULTS_PATH_TEMPLATE), reusing
analysis_context/_common.py's cast_object/load_model_frame/
filter_pilot_condA_valid helpers (the same audited loading/filtering path
every other analysis_context/ script uses) and 08's own partial-R^2 method,
copied here rather than imported since 08 does not expose it as a reusable
function.

SCOPE NOTE -- do not conflate the two dataset sizes:
- original: the FULL 5,400-persona dataset (its own long-established scope,
  unchanged by this script).
- health / neutral / positive / negative_minor: the 180-persona pilot subset
  (4 countries x 5 professions x 3 genders x 3 ages) already established for
  every other analysis_context/ comparison (Step 0 of the cross-context
  task) -- not a matched 5,400-persona comparison. This is the existing
  data-availability scope for the four new contexts, not a new choice made
  by this script. See the "dataset_scope" column in the output CSV and the
  chart subtitles.

DeepSeek is excluded throughout: near-total non-compliance in every one of
the five conditions (see analysis_context/03 and analysis/09) leaves too few
valid Condition-A rows to fit this formula meaningfully in any context.

topic is included in the fitted formula as a control, matching 08 exactly,
but is NOT one of the four factors reported/plotted below -- consistent with
08's own h1_check() function, which likewise compares only
profession/country/gender/age against each other, not against topic.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import ROOT, load_model_frame, filter_pilot_condA_valid, filter_condA_valid, cast_object

sys.path.insert(0, os.path.join(ROOT, "analysis"))
from _style import (  # noqa: E402
    apply_base_style, FACTOR_COLOR, INK_PRIMARY, INK_SECONDARY, INK_MUTED,
    CLUSTER_NEVER_ABSTAIN_BG, CLUSTER_TOPIC_GATED_BG,
)

apply_base_style()

OUT_DIR = f"{ROOT}/analysis_context/output"

PROMPT_TYPES = ["original", "health", "neutral", "positive", "negative_minor"]
CONTEXT_PROMPT_TYPES = ["health", "neutral", "positive", "negative_minor"]
RANKED_MODELS = ["llama", "gemma", "qwen", "ministral"]  # deepseek excluded throughout, per task scope
FULL_FACTORS = ["topic", "profession", "country", "gender", "age"]
H1_FACTORS = ["profession", "country", "gender", "age"]  # topic is a fitted control, not an H1 candidate
DATASET_SCOPE = {
    "original": "full_5400_persona",
    "health": "pilot_180_persona", "neutral": "pilot_180_persona",
    "positive": "pilot_180_persona", "negative_minor": "pilot_180_persona",
}


def fit_variance_ranking(sub, model, prompt_type, dataset_scope):
    """Same method as analysis/08_variance_ranking.py's fit_variance_ranking:
    nested-model SSE-reduction partial R^2 (a point estimate, unaffected by
    clustering) plus a persona-clustered joint Wald test per factor
    (cov_type='cluster', cov_kwds={'groups': persona_id}) for significance."""
    import statsmodels.formula.api as smf

    n = len(sub)
    factors = [f for f in FULL_FACTORS if sub[f].nunique() > 1]
    dropped = [f for f in FULL_FACTORS if f not in factors]
    if dropped:
        print(f"    WARNING [{model}/{prompt_type}]: dropped zero-variance factor(s): {dropped}")

    formula = "rating_numeric ~ " + " + ".join(f"C({f})" for f in factors)
    sub = cast_object(sub, factors + ["persona_id"])
    full_res = smf.ols(formula, data=sub).fit()
    if full_res.df_resid <= 0:
        print(f"    FIT INVALID [{model}/{prompt_type}]: df_resid={full_res.df_resid:.0f}")
        return None

    cluster_res = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"]})
    try:
        cluster_wald = cluster_res.wald_test_terms(skip_single=False, scalar=True)
    except Exception as e:
        print(f"    Cluster-robust joint test FAILED [{model}/{prompt_type}]: {type(e).__name__}: {e}")
        cluster_wald = None

    sse_full = float((full_res.resid ** 2).sum())
    n_clusters = sub["persona_id"].nunique()
    rows = []
    for f in H1_FACTORS:
        if f not in factors:
            rows.append({"model": model, "prompt_type": prompt_type, "factor": f, "partial_r2": np.nan,
                         "chi2_cluster": np.nan, "p_cluster": np.nan, "n_rows": n,
                         "n_persona_clusters": n_clusters, "dataset_scope": dataset_scope,
                         "note": "zero variance -- not estimable"})
            continue
        reduced_factors = [x for x in factors if x != f]
        reduced_formula = "rating_numeric ~ " + " + ".join(f"C({x})" for x in reduced_factors) \
            if reduced_factors else "rating_numeric ~ 1"
        reduced_res = smf.ols(reduced_formula, data=sub).fit()
        sse_reduced = float((reduced_res.resid ** 2).sum())
        partial_r2 = (sse_reduced - sse_full) / sse_reduced if sse_reduced > 0 else np.nan

        chi2_cluster, p_cluster = np.nan, np.nan
        if cluster_wald is not None:
            term_key = f"C({f})"
            if term_key in cluster_wald.table.index:
                row = cluster_wald.table.loc[term_key]
                chi2_cluster, p_cluster = float(row["statistic"]), float(row["pvalue"])

        rows.append({"model": model, "prompt_type": prompt_type, "factor": f, "partial_r2": partial_r2,
                     "chi2_cluster": chi2_cluster, "p_cluster": p_cluster, "n_rows": n,
                     "n_persona_clusters": n_clusters, "dataset_scope": dataset_scope, "note": ""})
    return pd.DataFrame(rows)


def load_original_rows():
    """Reuse, don't refit: original's partial R^2 values come straight from the
    already-audited tables/variance_ranking.csv (primary_conditionA scope).
    n_rows/n_persona_clusters are recomputed here (cheap groupby, no regression)
    from analysis/master_results.csv purely so every prompt_type has those
    columns populated consistently -- the partial_r2/p_cluster values themselves
    are untouched, read-only."""
    vr = pd.read_csv(f"{ROOT}/tables/variance_ranking.csv")
    vr = vr[(vr["scope"] == "primary_conditionA") & (vr["model"].isin(RANKED_MODELS)) &
            (vr["factor"].isin(H1_FACTORS))].copy()
    vr["prompt_type"] = "original"
    vr["dataset_scope"] = DATASET_SCOPE["original"]
    vr = vr.rename(columns={"p_cluster": "p_cluster"})[
        ["model", "prompt_type", "factor", "partial_r2", "chi2_cluster", "p_cluster", "dataset_scope"]]

    master = pd.read_csv(f"{ROOT}/analysis/master_results.csv", keep_default_na=False, na_values=[""], low_memory=False)
    master["rating_numeric"] = pd.to_numeric(master["strict_parsed_rating"], errors="coerce")
    counts = {}
    for m in RANKED_MODELS:
        sub = master[(master["model"] == m) & (master["response_condition"] == "A_forced") &
                     master["strict_is_valid"] & master["rating_numeric"].notnull()]
        counts[m] = (len(sub), sub["persona_id"].nunique())
    vr["n_rows"] = vr["model"].map(lambda m: counts[m][0])
    vr["n_persona_clusters"] = vr["model"].map(lambda m: counts[m][1])
    vr["note"] = ""
    return vr


def run_context_prompt_type(prompt_type, scope="pilot"):
    """scope="pilot" (default, unchanged): 180-persona subset, matches the
    original behavior of this function exactly. scope="full": all 5,400
    personas, 20 countries x 30 professions -- the full-scale companion added
    per the full-scale extension task."""
    is_full = scope == "full"
    filt = filter_condA_valid if is_full else filter_pilot_condA_valid
    dataset_scope = "full_5400_persona" if is_full else "pilot_180_persona"
    scope_label = "full 5,400-persona" if is_full else "180-persona pilot"
    all_rows = []
    for model in RANKED_MODELS:
        df = load_model_frame(prompt_type, model)
        sub = filt(df)
        print(f"\n{'='*78}\n{prompt_type} / {model}   (n = {len(sub):,} strict-valid Condition-A {scope_label} rows)")
        res = fit_variance_ranking(sub, model, prompt_type, dataset_scope)
        if res is None:
            res = pd.DataFrame([
                {"model": model, "prompt_type": prompt_type, "factor": f, "partial_r2": np.nan,
                 "chi2_cluster": np.nan, "p_cluster": np.nan, "n_rows": len(sub), "n_persona_clusters": np.nan,
                 "dataset_scope": dataset_scope, "note": "fit invalid -- see log"}
                for f in H1_FACTORS
            ])
        else:
            print(res[["factor", "partial_r2", "p_cluster"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        all_rows.append(res)
    return pd.concat(all_rows, ignore_index=True)


def dominant_factor_summary(combined, scope="pilot"):
    """For each model, which of profession/country/gender/age has the highest
    partial R^2 under each prompt type, and does that verdict change across
    the five prompt types? scope selects which dataset_scope rows to use for
    the four context prompt types (original is always its own single
    full_5400_persona scope, unaffected by this parameter)."""
    dataset_scope = "full_5400_persona" if scope == "full" else "pilot_180_persona"
    records = []
    for model in RANKED_MODELS:
        dominants = {}
        for pt in PROMPT_TYPES:
            pt_scope = "full_5400_persona" if pt == "original" else dataset_scope
            sub = combined[(combined["model"] == model) & (combined["prompt_type"] == pt) &
                            (combined["dataset_scope"] == pt_scope)].dropna(subset=["partial_r2"])
            if sub.empty:
                dominants[pt] = None
                continue
            top = sub.loc[sub["partial_r2"].idxmax()]
            dominants[pt] = top["factor"]
        unique_dominants = set(v for v in dominants.values() if v is not None)
        if len(unique_dominants) == 1:
            factor = next(iter(unique_dominants))
            verdict = f"{model}: {factor} dominates in all 5 prompt types"
        else:
            counts = pd.Series(list(dominants.values())).value_counts()
            main_factor = counts.idxmax()
            main_n = counts.max()
            exceptions = [pt for pt, f in dominants.items() if f != main_factor]
            verdict = (f"{model}: {main_factor} dominates in {main_n} of 5 prompt types, "
                       f"but {', '.join(f'{dominants[pt]} takes over under {pt}' for pt in exceptions)}")
        records.append({"model": model, **{f"dominant_{pt}": dominants[pt] for pt in PROMPT_TYPES},
                        "changes_across_prompt_types": len(unique_dominants) > 1, "verdict": verdict})
        print(f"  {verdict}")
    return pd.DataFrame(records)


CLUSTER_BG = {"llama": CLUSTER_NEVER_ABSTAIN_BG, "gemma": CLUSTER_NEVER_ABSTAIN_BG,
              "qwen": CLUSTER_TOPIC_GATED_BG, "ministral": CLUSTER_TOPIC_GATED_BG}


def make_chart(combined, prompt_type, scope="pilot"):
    """scope only matters for the four context prompt types (original has a
    single fixed full-scale scope). scope="full" writes a _full5400-suffixed
    file instead of the un-suffixed pilot-scope one."""
    is_full = scope == "full" and prompt_type != "original"
    pt_scope = "full_5400_persona" if (prompt_type == "original" or scope == "full") else "pilot_180_persona"
    sub_pt = combined[(combined["prompt_type"] == prompt_type) & (combined["dataset_scope"] == pt_scope)]
    scope_label = "full 5,400-persona dataset" if pt_scope == "full_5400_persona" else "180-persona pilot subset"
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
    for ax, model in zip(axes, RANKED_MODELS):
        sub = sub_pt[sub_pt["model"] == model].set_index("factor").reindex(H1_FACTORS)
        vals = sub["partial_r2"]
        colors = [FACTOR_COLOR[f] for f in H1_FACTORS]
        ax.set_facecolor(CLUSTER_BG.get(model, "#fff"))
        y = np.arange(len(H1_FACTORS))
        plot_vals = vals.fillna(0).values
        ax.barh(y, plot_vals, color=colors, edgecolor="white", linewidth=0.5)
        for yi, (f, v) in enumerate(zip(H1_FACTORS, vals.values)):
            if pd.isna(v):
                ax.text(0.005, yi, "n/a", va="center", ha="left", fontsize=7, color=INK_MUTED, style="italic")
            else:
                ax.text(v + 0.005, yi, f"{v:.3f}", va="center", ha="left", fontsize=7, color=INK_SECONDARY)
        ax.set_yticks(y)
        ax.set_yticklabels(H1_FACTORS if model == "llama" else [], fontsize=8)
        ax.set_title(model, fontsize=10, color=INK_PRIMARY)
        xmax = combined["partial_r2"].max()
        ax.set_xlim(0, max(0.2, xmax * 1.15))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("partial R²", fontsize=8)
    fig.suptitle(f"Variance explained (partial R²) by factor, per model -- prompt type: {prompt_type}\n"
                 f"(Condition A / forced only; {scope_label}; deepseek excluded)",
                 fontsize=11, color=INK_PRIMARY, y=1.1)
    file_suffix = "_full5400" if is_full else ""
    path = f"{OUT_DIR}/variance_explained_{prompt_type}{file_suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("#" * 78)
    print("ORIGINAL -- reused from the already-audited tables/variance_ranking.csv, not refit")
    print("#" * 78)
    original_rows = load_original_rows()
    print(original_rows[["model", "factor", "partial_r2", "p_cluster"]].to_string(
        index=False, float_format=lambda x: f"{x:.4g}"))

    context_frames = []
    for prompt_type in CONTEXT_PROMPT_TYPES:
        print(f"\n{'#'*78}\nPROMPT TYPE: {prompt_type}  (180-persona PILOT subset, Condition A only)\n{'#'*78}")
        context_frames.append(run_context_prompt_type(prompt_type, scope="pilot"))

    print(f"\n{'#'*78}\nFULL-SCALE COMPANION: same 4 context prompt types, all 5,400 personas / 20 "
          f"countries / 30 professions\n{'#'*78}")
    for prompt_type in CONTEXT_PROMPT_TYPES:
        print(f"\n{'#'*78}\nPROMPT TYPE: {prompt_type}  (FULL 5,400-persona scope, Condition A only)\n{'#'*78}")
        context_frames.append(run_context_prompt_type(prompt_type, scope="full"))

    combined = pd.concat([original_rows] + context_frames, ignore_index=True)
    combined = combined[["model", "prompt_type", "factor", "partial_r2", "chi2_cluster", "p_cluster",
                          "n_rows", "n_persona_clusters", "dataset_scope", "note"]]
    out_csv = f"{OUT_DIR}/variance_ranking_all_prompt_types.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  (single file, both dataset_scope values present for every context prompt type)")

    print(f"\n{'='*78}\nCharts (one per prompt type per scope, same style as analysis/10_figures.py's fig7)\n{'='*78}")
    for prompt_type in PROMPT_TYPES:
        make_chart(combined, prompt_type, scope="pilot")
    for prompt_type in CONTEXT_PROMPT_TYPES:
        make_chart(combined, prompt_type, scope="full")

    print(f"\n{'='*78}\nDOMINANT-FACTOR SUMMARY (PILOT SCOPE): does the H1 'which factor dominates'\n"
          f"verdict change across the five prompt types?\n{'='*78}")
    summary = dominant_factor_summary(combined, scope="pilot")
    out_summary = f"{OUT_DIR}/dominant_factor_by_model.csv"
    summary.to_csv(out_summary, index=False)
    print(f"\nWrote {out_summary}")

    print(f"\n{'='*78}\nDOMINANT-FACTOR SUMMARY (FULL SCALE): same question, all 5,400 personas for the\n"
          f"four context prompt types\n{'='*78}")
    summary_full = dominant_factor_summary(combined, scope="full")
    out_summary_full = f"{OUT_DIR}/dominant_factor_by_model_full5400.csv"
    summary_full.to_csv(out_summary_full, index=False)
    print(f"\nWrote {out_summary_full}")


if __name__ == "__main__":
    main()
