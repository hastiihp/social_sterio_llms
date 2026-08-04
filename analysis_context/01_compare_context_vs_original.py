"""Exploratory: original (single-turn "friend_final" persona prompt) vs. each
of the four naturalistic multi-turn conversational framings (health, neutral,
positive, negative_minor), for the same 180-persona pilot subset used
throughout analysis_health/ (4 countries x 5 professions x 3 genders x 3
ages -- re-verified against data/personas.csv for this task, not redefined).

ONE parametrized script, not four ad hoc ones: run_context_comparison(context)
does the full comparison for a single context; main() loops over all four.
Reuses the audited logic from analysis_health/01, 02, and 04 via
analysis_context/_common.py -- persona-clustered inference (Fix H1), exact
permutation p-values for small-n rankings (Fix H4a), persona-bootstrap rank
uncertainty (Fix H4b), and canonical-key merge validation (Fix H6) -- rather
than re-deriving any of it.

Terminology (Step 0): health/neutral/positive/negative_minor are
CONVERSATIONAL FRAMINGS, not points on a valence scale -- they differ in
which dimension of the persona's situation is foregrounded (vulnerability,
domain-neutral small talk, competence/success, an impersonal external
event), not just in "how positive/negative" the framing is. See
_common.py's module docstring and CONTEXT_EXPERIMENT.md for the full note.

For each context, computes and saves four things (items 1-4 of Step 2):
  1. Matched-cell comparison, pooled (item 1a) and split by
     response_condition (item 1b): exact agreement, mean absolute
     difference, persona-clustered signed mean shift, Spearman correlation.
     Condition A = primary evidence (no valid-NA option, no selection
     mechanism). Condition B = descriptive only, with the sample-size
     caveat stated explicitly per model.
  2. Abstention comparison: rate shift per model (context - original,
     correct sign), full-dataset vs 180-persona-pilot x all-rows vs
     Condition-B-only 2x2, generalized from the health study's
     Ministral-only table to every model with a working NA option (qwen,
     ministral -- llama/gemma structurally cannot abstain in this prompt
     design, confirmed at ~0% in every condition; see 03_abstention_*.py
     for the full 5-condition picture).
  3. Topic-level breakdown of the Condition-B abstention shift, for the
     same models as (2).
  4. Ranking robustness (Condition A only): profession and country rank
     correlation, original vs. context, EXACT permutation p-values (not
     asymptotic -- invalid at n=4/5) and persona-bootstrap rank-position
     probabilities. Models with too few valid Condition-A pilot rows to
     fit the model (checked dynamically per model x context, not
     hardcoded) are skipped with an explicit warning rather than silently
     omitted.

Independent of analysis/ and analysis_health/ (does not import from or
write to either): reads results/results_{original,health,neutral,positive,
negative_minor}_{model}.csv directly (see _common.py's RESULTS_PATH_TEMPLATE).
Writes only to analysis_context/output/, never overwriting analysis_health/output/.
"""
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

ABSTENTION_CAPABLE_MODELS = ["qwen", "ministral"]  # confirmed in 03_abstention_stability_across_conditions.py:
                                                    # llama/gemma are ~0% abstention in every condition (no
                                                    # working NA option in practice); deepseek's near-total
                                                    # non-compliance is a separate phenomenon, not NA-abstention


def verify_pilot_subset():
    personas = pd.read_csv(f"{ROOT}/data/personas.csv")
    sub = personas[personas["country"].isin(PILOT_COUNTRIES) & personas["profession"].isin(PILOT_PROFESSIONS)]
    assert len(sub) == EXPECTED_PILOT_PERSONAS, (
        f"pilot subset size mismatch: got {len(sub)}, expected {EXPECTED_PILOT_PERSONAS}")
    return len(sub)


def section_matched_cells(orig_df, ctx_df, context, model_order=MODEL_ORDER, suffix=""):
    """Item 1a: pooled matched-cell comparison (both conditions together),
    generalizing analysis_health/01_compare_health_vs_original.py Step 3.
    suffix="_full5400" writes the full-5,400-persona companion output instead of
    overwriting the pilot-scope file -- caller controls scope entirely via which
    orig_df/ctx_df (pilot-filtered or not) it passes in."""
    print(f"\n{'='*78}\n[{context}] ITEM 1a: matched-cell comparison, pooled (both conditions)\n{'='*78}")
    merged_ind = merge_on_canonical_key(orig_df, ctx_df)
    counts = merged_ind["_merge"].value_counts()
    print(f"  post-merge indicator counts: {counts.to_dict()}")
    left_only, right_only = (merged_ind[merged_ind["_merge"] == s] for s in ["left_only", "right_only"])
    if len(left_only) or len(right_only):
        print(f"  WARNING: {len(left_only)} rows only in original, {len(right_only)} rows only in {context} -- "
              f"dropped from the inner-join comparison below.")
    else:
        print(f"  PASS: 0 unmatched rows on either side.")
    merged = merged_ind[merged_ind["_merge"] == "both"].drop(columns="_merge")
    print(f"  Total matched cells: {len(merged):,}")

    rows = []
    for m in model_order:
        sub = merged[merged["model"] == m]
        n_cells = len(sub)
        both_valid = sub[sub["strict_is_valid_orig"] & sub["strict_is_valid_ctx"] &
                          sub["rating_numeric_orig"].notnull() & sub["rating_numeric_ctx"].notnull()]
        n_both_valid = len(both_valid)
        row = {"context": context, "model": m, "n_matched_cells": n_cells, "n_both_valid_numeric": n_both_valid}

        if n_both_valid >= 2:
            diff = both_valid["rating_numeric_orig"] - both_valid["rating_numeric_ctx"]
            exact_agree = (diff == 0).mean()
            mad = diff.abs().mean()
            rho, rho_p = stats.spearmanr(both_valid["rating_numeric_orig"], both_valid["rating_numeric_ctx"])
            gd = clustered_diff_context_minus_orig(both_valid, "rating_numeric_ctx", "rating_numeric_orig", "persona_id")
            print(f"  {m:10s} n_cells={n_cells:5d} n_valid={n_both_valid:5d}  exact_agree={exact_agree:.4f}  "
                  f"MAD={mad:.4f}  rho={rho:.4f} (p={rho_p:.2e})  shift(ctx-orig)={gd['diff']:+.4f} "
                  f"p={gd['p_value']:.2e} (n_clusters={gd['n_clusters']})")
            row.update(exact_agreement_rate=exact_agree, mean_abs_diff=mad, spearman_r=rho, spearman_p=rho_p,
                       clustered_mean_shift_ctx_minus_orig=gd["diff"], clustered_shift_se=gd["se_cluster"],
                       clustered_shift_p=gd["p_value"], n_persona_clusters=gd["n_clusters"])
        else:
            print(f"  {m:10s} n_cells={n_cells:5d} n_valid={n_both_valid:5d}  (< 2 valid pairs, skipped)")
            row.update(exact_agreement_rate=np.nan, mean_abs_diff=np.nan, spearman_r=np.nan, spearman_p=np.nan,
                       clustered_mean_shift_ctx_minus_orig=np.nan, clustered_shift_se=np.nan,
                       clustered_shift_p=np.nan, n_persona_clusters=np.nan)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_DIR}/{context}_vs_original_summary{suffix}.csv", index=False)
    return summary, merged


def section_by_condition(orig_df, ctx_df, context, model_order=MODEL_ORDER, suffix=""):
    """Item 1b: split by response_condition, generalizing
    analysis_health/02_compare_by_condition.py. Condition A = PRIMARY
    (no valid-NA option -> no selection mechanism). Condition B =
    descriptive only, selected sample -- flagged explicitly per row when
    <50% of matched cells are valid in both framings."""
    print(f"\n{'='*78}\n[{context}] ITEM 1b: matched-cell comparison, split by condition\n{'='*78}")
    sample_type = {"A_forced": "primary_full_sample_no_valid_NA_option",
                   "B_optional": "descriptive_only_selected_sample"}
    merged_ind = merge_on_canonical_key(orig_df, ctx_df)
    merged = merged_ind[merged_ind["_merge"] == "both"].drop(columns="_merge")

    rows = []
    for m in model_order:
        for cond in CONDITIONS:
            sub = merged[(merged["model"] == m) & (merged["response_condition"] == cond)]
            n_cells = len(sub)
            both_valid = sub[sub["strict_is_valid_orig"] & sub["strict_is_valid_ctx"] &
                              sub["rating_numeric_orig"].notnull() & sub["rating_numeric_ctx"].notnull()]
            n_both_valid = len(both_valid)
            pct_valid = 100 * n_both_valid / n_cells if n_cells else float("nan")
            row = {"context": context, "model": m, "condition": cond, "sample_type": sample_type[cond],
                   "n_matched_cells": n_cells, "n_both_valid": n_both_valid, "pct_both_valid": pct_valid}
            flag = ""
            if cond == "B_optional" and n_cells and pct_valid < 50:
                flag = " ** <50% valid in both framings: descriptive only, self-selected sample **"

            if n_both_valid >= 2:
                diff = both_valid["rating_numeric_orig"] - both_valid["rating_numeric_ctx"]
                exact_agree = (diff == 0).mean()
                mad = diff.abs().mean()
                rho, rho_p = stats.spearmanr(both_valid["rating_numeric_orig"], both_valid["rating_numeric_ctx"])
                gd = clustered_diff_context_minus_orig(both_valid, "rating_numeric_ctx", "rating_numeric_orig", "persona_id")
                print(f"  {m:10s} {cond:10s} n_valid={n_both_valid:5d} ({pct_valid:5.1f}%)  shift(ctx-orig)="
                      f"{gd['diff']:+.4f} p={gd['p_value']:.2e} rho={rho:.4f}{flag}")
                row.update(exact_agreement_rate=exact_agree, mean_abs_diff=mad, spearman_r=rho, spearman_p=rho_p,
                           clustered_mean_shift_ctx_minus_orig=gd["diff"], clustered_shift_se=gd["se_cluster"],
                           clustered_shift_p=gd["p_value"], n_persona_clusters=gd["n_clusters"])
            else:
                print(f"  {m:10s} {cond:10s} n_valid={n_both_valid:5d}  (< 2 valid pairs, skipped){flag}")
                row.update(exact_agreement_rate=np.nan, mean_abs_diff=np.nan, spearman_r=np.nan, spearman_p=np.nan,
                           clustered_mean_shift_ctx_minus_orig=np.nan, clustered_shift_se=np.nan,
                           clustered_shift_p=np.nan, n_persona_clusters=np.nan)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/{context}_vs_original_by_condition{suffix}.csv", index=False)
    return df


def section_abstention_2x2(context, models=ABSTENTION_CAPABLE_MODELS):
    """Item 2: generalizes analysis_health/01's Fix-H2 2x2 table (sample:
    full 5,400 vs 180-pilot; denominator: all rows vs Condition-B-only) from
    Ministral-only to every abstention-capable model."""
    print(f"\n{'='*78}\n[{context}] ITEM 2: abstention rate shift, full 2x2 (sample x denominator)\n{'='*78}")

    def abst_rate(df, b_only):
        sub = df[df["response_condition"] == "B_optional"] if b_only else df
        return 100 * sub["is_abstention"].mean(), len(sub)

    all_rows = []
    for m in models:
        orig_full = load_model_frame("original", m)
        ctx_full = load_model_frame(context, m)
        orig_pilot, ctx_pilot = filter_pilot(orig_full), filter_pilot(ctx_full)
        for sample_label, o_df, c_df in [("full_5400_personas", orig_full, ctx_full),
                                          ("pilot_180_personas", orig_pilot, ctx_pilot)]:
            for denom_label, b_only in [("all_rows", False), ("condition_B_only", True)]:
                o_rate, o_n = abst_rate(o_df, b_only)
                c_rate, c_n = abst_rate(c_df, b_only)
                all_rows.append({"context": context, "model": m, "sample": sample_label, "denominator": denom_label,
                                  "n_orig": o_n, "n_ctx": c_n, "orig_rate_pct": o_rate, "ctx_rate_pct": c_rate,
                                  "shift_pp_ctx_minus_orig": c_rate - o_rate})
    table = pd.DataFrame(all_rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    table.to_csv(f"{OUT_DIR}/{context}_abstention_2x2.csv", index=False)
    return table


def section_abstention_significance(context, models=ABSTENTION_CAPABLE_MODELS, scope="pilot"):
    """Persona-clustered significance test for the abstention shift, Condition-B
    matched cells only (the one cell that is both a fair like-for-like comparison
    and matches the matched-cell merge used elsewhere in this script). scope="pilot"
    (default, 180-persona subset, matches the original item-2 test exactly) or
    "full" (all 5,400 personas, writes a separate _full5400-suffixed file rather
    than overwriting the pilot-scope one). Split out from section_abstention_2x2 so
    the (cheap) full 2x2 rate table -- which already reports both sample scopes as
    rows -- isn't needlessly recomputed twice."""
    scope_label = "full 5,400-persona" if scope == "full" else "pilot / Condition-B-only"
    filt = (lambda df: df) if scope == "full" else filter_pilot
    suffix = "_full5400" if scope == "full" else ""
    print(f"\n  [{context}] Persona-clustered abstention-shift significance ({scope_label}):")
    sig_rows = []
    for m in models:
        orig_s = filt(load_model_frame("original", m))
        ctx_s = filt(load_model_frame(context, m))
        merged_ind = merge_on_canonical_key(orig_s, ctx_s)
        merged = merged_ind[merged_ind["_merge"] == "both"].drop(columns="_merge")
        b_matched = merged[merged["response_condition"] == "B_optional"]
        gd = clustered_diff_context_minus_orig(b_matched, "is_abstention_ctx", "is_abstention_orig", "persona_id")
        print(f"    {m:10s} shift={100*gd['diff']:+.2f}pp  se={100*gd['se_cluster']:.2f}pp  "
              f"p={gd['p_value']:.2e}  (n_clusters={gd['n_clusters']})")
        sig_rows.append({"context": context, "model": m, "shift_pp": 100 * gd["diff"],
                          "se_pp": 100 * gd["se_cluster"], "p_value": gd["p_value"], "n_clusters": gd["n_clusters"]})
    df = pd.DataFrame(sig_rows)
    df.to_csv(f"{OUT_DIR}/{context}_abstention_significance{suffix}.csv", index=False)
    return df


def section_topic_breakdown(context, models=ABSTENTION_CAPABLE_MODELS, scope="pilot"):
    """Item 3: is the abstention shift concentrated in specific topics, or
    spread evenly? Generalizes analysis_health/01 Step 4 (Ministral-only)
    to every abstention-capable model, Condition B only. scope="pilot" (default,
    180-persona subset) or "full" (all 5,400 personas, writes a separate
    _full5400-suffixed file rather than overwriting the pilot-scope one)."""
    filt = (lambda df: df) if scope == "full" else filter_pilot
    suffix = "_full5400" if scope == "full" else ""
    scope_label = "full 5,400-persona dataset" if scope == "full" else "180-persona pilot subset"
    print(f"\n{'='*78}\n[{context}] ITEM 3: topic-level abstention shift breakdown ({scope_label})\n{'='*78}")
    all_rows = []
    for m in models:
        orig_s = filt(load_model_frame("original", m))
        ctx_s = filt(load_model_frame(context, m))
        merged_ind = merge_on_canonical_key(orig_s, ctx_s)
        merged = merged_ind[merged_ind["_merge"] == "both"].drop(columns="_merge")
        b_sub = merged[merged["response_condition"] == "B_optional"]
        print(f"\n  --- {m} ---")
        topic_shifts = []
        for topic in TOPIC_ORDER:
            t_sub = b_sub[b_sub["topic"] == topic]
            n = len(t_sub)
            if n == 0:
                continue
            orig_rate = t_sub["is_abstention_orig"].mean()
            ctx_rate = t_sub["is_abstention_ctx"].mean()
            shift = 100 * (ctx_rate - orig_rate)
            print(f"    {topic:28s} n={n:4d}  orig={100*orig_rate:6.2f}%  {context}={100*ctx_rate:6.2f}%  "
                  f"shift={shift:+7.2f}pp")
            topic_shifts.append(shift)
            all_rows.append({"context": context, "model": m, "topic": topic, "n": n,
                              "abstention_orig": orig_rate, "abstention_ctx": ctx_rate, "shift_pp": shift})
        if topic_shifts:
            shifts = pd.Series(topic_shifts)
            cv = shifts.std() / shifts.mean() if shifts.mean() else np.nan
            spread = "concentrated in specific topics" if shifts.std() > abs(shifts.mean()) * 0.5 else "spread roughly evenly"
            print(f"    -> {spread} (range=[{shifts.min():.2f}, {shifts.max():.2f}]pp, "
                  f"mean={shifts.mean():.2f}pp, CV={cv:.3f})")
    df = pd.DataFrame(all_rows)
    df.to_csv(f"{OUT_DIR}/{context}_abstention_by_topic{suffix}.csv", index=False)
    return df


def section_ranking_robustness(context, model_order=("llama", "gemma", "qwen", "ministral", "deepseek"), scope="pilot"):
    """Item 4: generalizes analysis_health/04_ranking_robustness.py.
    Condition A only (no selection mechanism). Skips any (model, context)
    with too few valid Condition-A rows to fit the full factor model, checked
    dynamically -- not a hardcoded model list -- and prints a warning rather
    than silently omitting.

    scope="pilot" (default): 180-persona subset, 4 countries / 5 professions,
    EXACT permutation p-values (enumerates all 24/120 permutations), N_BOOT=1000
    bootstrap resamples -- unchanged from before this function gained a scope
    parameter.

    scope="full": all 5,400 personas, 20 countries / 30 professions. Exact
    permutation enumeration is not a slow version of the same test here -- it is
    IMPOSSIBLE (20! / 30! permutations) -- so this uses monte_carlo_permutation_pvalue
    (200,000 draws, vectorized) instead, and N_BOOT_FULL=300 bootstrap resamples
    (down from 1,000; see _common.py's N_BOOT_FULL docstring for the compute-budget
    rationale). Every full-scale output is written to a separate _full5400-suffixed
    file -- never merged into or overwriting the pilot-scope exact-test output."""
    is_full = scope == "full"
    suffix = "_full5400" if is_full else ""
    scope_label = "full 5,400-persona dataset, 20 countries x 30 professions" if is_full \
        else "180-persona pilot subset, 4 countries x 5 professions"
    filt = filter_condA_valid if is_full else filter_pilot_condA_valid
    profession_levels = ALL_PROFESSIONS if is_full else sorted(PILOT_PROFESSIONS)
    country_levels = ALL_COUNTRIES if is_full else sorted(PILOT_COUNTRIES)
    perm_fn = monte_carlo_permutation_pvalue if is_full else exact_permutation_pvalue
    perm_method_label = f"monte_carlo_{N_MONTE_CARLO}" if is_full else "exact_enumeration"
    n_boot = N_BOOT_FULL if is_full else N_BOOT
    ref_prof, ref_ctry = profession_levels[0], country_levels[0]

    print(f"\n{'='*78}\n[{context}] ITEM 4: ranking robustness (profession, country), Condition A only "
          f"({scope_label})\n{'='*78}")

    prof_rows, ctry_rows, exact_rows, bootstrap_rows = [], [], [], []
    for m in model_order:
        orig = filt(load_model_frame("original", m))
        ctx = filt(load_model_frame(context, m))
        print(f"\n  --- {m} ---  original n={len(orig):,}   {context} n={len(ctx):,}")
        if len(orig) < MIN_VALID_FOR_RANKING or len(ctx) < MIN_VALID_FOR_RANKING:
            print(f"  SKIPPED: fewer than {MIN_VALID_FOR_RANKING} valid Condition-A rows in "
                  f"{'original' if len(orig) < MIN_VALID_FOR_RANKING else context} "
                  f"(orig={len(orig)}, {context}={len(ctx)}) -- cannot fit a stable ranking model.")
            continue

        res_orig, _ = fit_model(orig, f"{m}/original")
        res_ctx, _ = fit_model(ctx, f"{m}/{context}")

        for factor, ref, target_rows in [("profession", ref_prof, prof_rows), ("country", ref_ctry, ctry_rows)]:
            co = extract_coefs(res_orig, f"C({factor})[T.")
            cc = extract_coefs(res_ctx, f"C({factor})[T.")
            merged = co[["level", "coef"]].rename(columns={"coef": "coef_orig"}).merge(
                cc[["level", "coef"]].rename(columns={"coef": "coef_ctx"}), on="level", how="outer")
            if ref not in merged["level"].values:
                merged = pd.concat([merged, pd.DataFrame([{"level": ref, "coef_orig": 0.0, "coef_ctx": 0.0}])],
                                    ignore_index=True)
            merged["model"], merged["context"] = m, context
            target_rows.append(merged)

            rho, p, n_perm = perm_fn(merged["coef_orig"], merged["coef_ctx"])
            add_tie_aware_ranks(merged, "coef_orig", "orig")
            add_tie_aware_ranks(merged, "coef_ctx", "ctx")
            top_o, top_o_set = extreme_label(merged, "coef_orig", highest=True)
            bot_o, bot_o_set = extreme_label(merged, "coef_orig", highest=False)
            top_c, top_c_set = extreme_label(merged, "coef_ctx", highest=True)
            bot_c, bot_c_set = extreme_label(merged, "coef_ctx", highest=False)
            print(f"    {factor:11s} n={len(merged)}  rho={rho:.4f}  p={p:.4g} ({perm_method_label}, "
                  f"{n_perm} {'draws' if is_full else 'perms'})  "
                  f"TOP: {'MATCH' if top_o_set==top_c_set else 'DIFFERENT'} ('{top_o}' vs '{top_c}')  "
                  f"BOTTOM: {'MATCH' if bot_o_set==bot_c_set else 'DIFFERENT'} ('{bot_o}' vs '{bot_c}')")
            exact_rows.append({"context": context, "model": m, "factor": factor, "spearman_r": rho,
                                "permutation_p": p, "n_permutations": n_perm, "method": perm_method_label,
                                "top_orig": top_o, "top_ctx": top_c, "bottom_orig": bot_o, "bottom_ctx": bot_c})

        for framing, sub in [("original", orig), ("context", ctx)]:
            factors_used = [f for f in FACTORS if sub[f].nunique() > 1]
            boot = bootstrap_ranks(sub, {"profession": profession_levels, "country": country_levels},
                                    factors_used, {"profession": ref_prof, "country": ref_ctry}, n_boot=n_boot)
            boot["model"], boot["context"], boot["framing"] = m, context, framing
            bootstrap_rows.append(boot)

    for name, rows in [("ranking_robustness_profession", prof_rows), ("ranking_robustness_country", ctry_rows)]:
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(f"{OUT_DIR}/{context}_{name}{suffix}.csv", index=False)
    exact_df = pd.DataFrame(exact_rows)
    pvalues_name = "ranking_robustness_exact_pvalues" if not is_full else "ranking_robustness_pvalues"
    exact_df.to_csv(f"{OUT_DIR}/{context}_{pvalues_name}{suffix}.csv", index=False)
    if bootstrap_rows:
        pd.concat(bootstrap_rows, ignore_index=True).to_csv(f"{OUT_DIR}/{context}_ranking_robustness_bootstrap{suffix}.csv", index=False)
    if is_full:
        print(f"\n  NOTE: full scale (20 countries / 30 professions) uses Monte Carlo permutation")
        print(f"  ({N_MONTE_CARLO:,} draws), not exact enumeration -- 20!/30! permutations are computationally")
        print(f"  impossible to enumerate. p-value resolution is 1/{N_MONTE_CARLO+1:,} (~{1/(N_MONTE_CARLO+1):.1e}),")
        print(f"  so conventional significance (p<.05) is reachable here, unlike the pilot's n=4/5 ceiling.")
    else:
        print(f"\n  NOTE: at n=4 (country), minimum possible two-sided exact p is 2/24=0.0833 -- p<.05 is")
        print(f"  mathematically unreachable regardless of agreement strength. Not a negative finding.")
    return exact_df


def run_context_comparison(context):
    """Pilot scope (180 personas) -- unchanged from before this task, byte-for-byte
    (verified separately; see conversation). Every output file here keeps its
    original, un-suffixed name."""
    assert context in CONTEXTS, f"unknown context {context!r}, expected one of {CONTEXTS}"
    print(f"\n{'#'*78}\n# CONTEXT: {context}  (PILOT SCOPE: 180 personas, 4 countries x 5 professions)\n{'#'*78}")

    orig_frames, ctx_frames = [], []
    for m in MODEL_ORDER:
        orig_frames.append(filter_pilot(load_model_frame("original", m)))
        ctx_frames.append(filter_pilot(load_model_frame(context, m)))
    orig_df, ctx_df = pd.concat(orig_frames, ignore_index=True), pd.concat(ctx_frames, ignore_index=True)

    orig_personas, ctx_personas = set(orig_df["persona_id"]), set(ctx_df["persona_id"])
    print(f"  pilot personas: original={len(orig_personas)}  {context}={len(ctx_personas)}  "
          f"overlap={len(orig_personas & ctx_personas)}  expected={EXPECTED_PILOT_PERSONAS}")

    section_matched_cells(orig_df, ctx_df, context)
    section_by_condition(orig_df, ctx_df, context)
    section_abstention_2x2(context)
    section_abstention_significance(context, scope="pilot")
    section_topic_breakdown(context, scope="pilot")
    section_ranking_robustness(context, scope="pilot")


def run_context_comparison_full(context):
    """Full scope (5,400 personas, 20 countries x 30 professions) companion to
    run_context_comparison, added per the full-scale extension task. Every output
    file here is written with a _full5400 suffix -- it never overwrites the
    pilot-scope files above, so both are always available side by side."""
    assert context in CONTEXTS, f"unknown context {context!r}, expected one of {CONTEXTS}"
    print(f"\n{'#'*78}\n# CONTEXT: {context}  (FULL SCOPE: 5,400 personas, 20 countries x 30 professions)\n{'#'*78}")

    orig_frames, ctx_frames = [], []
    for m in MODEL_ORDER:
        orig_frames.append(load_model_frame("original", m))
        ctx_frames.append(load_model_frame(context, m))
    orig_df, ctx_df = pd.concat(orig_frames, ignore_index=True), pd.concat(ctx_frames, ignore_index=True)

    orig_personas, ctx_personas = set(orig_df["persona_id"]), set(ctx_df["persona_id"])
    print(f"  full personas: original={len(orig_personas)}  {context}={len(ctx_personas)}  "
          f"overlap={len(orig_personas & ctx_personas)}  expected=5400")

    section_matched_cells(orig_df, ctx_df, context, suffix="_full5400")
    section_by_condition(orig_df, ctx_df, context, suffix="_full5400")
    section_abstention_significance(context, scope="full")
    section_topic_breakdown(context, scope="full")
    section_ranking_robustness(context, scope="full")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_pilot = verify_pilot_subset()
    print(f"Verified 180-persona pilot subset against data/personas.csv: {n_pilot} personas "
          f"(4 countries x 5 professions x 3 genders x 3 ages).")

    requested = sys.argv[1:] if len(sys.argv) > 1 else CONTEXTS
    for context in requested:
        run_context_comparison(context)
    for context in requested:
        run_context_comparison_full(context)

    print(f"\n{'='*78}\nDONE. Outputs written to {OUT_DIR}/{{context}}_*.csv (pilot) and "
          f"{OUT_DIR}/{{context}}_*_full5400.csv (full) for context in {requested}\n{'='*78}")


if __name__ == "__main__":
    main()
