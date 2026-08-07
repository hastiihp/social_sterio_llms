"""Stage 3: Topic-Specific Country / Profession Rankings.

Synthesizes already-computed, already-audited per-topic regression
coefficients into two long-format tables: one row per (country, topic),
one row per (profession, topic). FULL-SCALE (5,400 personas, 20 countries,
30 professions) throughout -- the source (analysis/05c_topic_specific_models.py)
has no pilot/full distinction; it has always been full-scale, same as
Stage 1's compliance/cross-model-agreement sources.

No new model is fit here. The one derived step -- turning a raw
coefficient into a rank position -- is a mechanical sort over
already-audited numbers, using the project's own established tie-aware
ranking function (analysis_context._common.add_tie_aware_ranks, imported
directly rather than reimplemented, to guarantee identical tie-handling
behavior rather than risk a subtly different methodology). No new p-value,
significance test, or model fit is introduced anywhere in this script.

Source: tables/topic_specific_models.csv (analysis/05c_topic_specific_models.py,
Condition A only per analysis_plan.md Section 7, main study -- always
full-scale). Confirmed to cover exactly 4 models (llama/gemma/qwen/ministral;
deepseek already excluded upstream, same n=63 sparsity justification as
elsewhere) x 7 topics x (19 non-reference country terms + 29 non-reference
profession terms). The omitted reference level in each factor (Argentina
for country, accountant for profession -- confirmed by diffing the term
list against the full 20-country/30-profession set) is added back at
coefficient 0.0 before ranking, matching the same convention used in
Stage 2's source data (analysis_context/01_compare_context_vs_original.py's
ranking-robustness tables, which do the same reference-level reinsertion).

No bootstrap top/bottom probability or ranking-robustness p-value exists at
this (topic x country/profession) granularity anywhere in this project --
that level of robustness testing was never run per-topic. Only rank
position and the underlying coefficient are reported; this is stated
explicitly rather than a robustness column being silently omitted without
comment.

DEGENERATE CELLS (found by this script's own validation, not assumed
absent): two (model, topic) cells -- llama/economic_redistribution and
ministral/gender_equality -- have literally zero rating variance under
Condition A (every persona got the same rating, 4.0 = "Agree", for that
model x topic, regardless of country/profession/gender/age). This is
already documented and intentional in the source script
(analysis/05c_topic_specific_models.py's fit_topic_model(): it explicitly
checks for this before fitting and writes a single
term=="DEGENERATE_NO_VARIANCE" row instead of coefficients, rather than
reporting a meaningless fit). No country or profession coefficients exist
for these two cells at all -- not because a level is missing, but because
the whole model is undefined. This script's own validation asserts the set
of degenerate cells is EXACTLY these two (raising if a third,
previously-unknown one appeared) and leaves rank_{model}/coef_{model} as
NaN for exactly llama's economic_redistribution rows and ministral's
gender_equality rows in the output, rather than silently fabricating a
1-level "ranking" from the reference level alone.

Companion table: topic_r2_summary.csv, read directly from
tables/topic_specific_r2.csv (per model x topic, how much variance the
full topic-specific model explains overall) -- context for reading the
per-level ranks, not a per-level statistic itself.

RELIABILITY FLAG (added after initial review): r2_{model} and
reliability_{model} columns are embedded directly into every row of both
per-level output tables, not left only in the topic_r2_summary.csv
companion -- a reader working from the per-level CSV alone (e.g. pulling a
number for a chart) would otherwise have no way to notice a low-R^2 or
degenerate cell without separately opening a different file.
reliability_{model} is one of: "OK", "LOW_R2" (r2 < LOW_R2_THRESHOLD =
0.15 -- a design choice made in this script, not an audited project
threshold; see the constant's own comment for where 0.15 came from), or
"DEGENERATE_NO_VARIANCE" (matches the source's own marker; rank/coef/r2
are NaN for these rows).
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{ROOT}/analysis_taxonomy/output"
sys.path.insert(0, f"{ROOT}/analysis_context")
from _common import add_tie_aware_ranks  # noqa: E402 -- reuse the audited tie-aware ranker, don't reimplement

MODELS = ["llama", "gemma", "qwen", "ministral"]
REFERENCE = {"country": "Argentina", "profession": "accountant"}
KNOWN_DEGENERATE_CELLS = {("llama", "economic redistribution"), ("ministral", "gender equality")}

# Not an audited/established project threshold -- a design choice made here, stated
# explicitly rather than presented as derived. Chosen at the natural gap in the
# empirical R^2 distribution across all 26 non-degenerate (model, topic) cells: the
# three lowest values (0.037, 0.089, 0.111) sit well apart from the rest (next value
# 0.228, a jump of +0.117 vs. the +0.02-0.05 jumps within the low cluster itself).
LOW_R2_THRESHOLD = 0.15


def load_r2_lookup():
    df = pd.read_csv(f"{ROOT}/tables/topic_specific_r2.csv")
    df = df[df["model"].isin(MODELS)]
    return df.set_index(["model", "topic"])["r2"]


def reliability_flag(model, topic, r2_lookup):
    if (model, topic) in KNOWN_DEGENERATE_CELLS:
        return "DEGENERATE_NO_VARIANCE"
    r2 = r2_lookup.get((model, topic))
    if pd.isna(r2):
        return "UNKNOWN"
    return "LOW_R2" if r2 < LOW_R2_THRESHOLD else "OK"


def load_topic_coefs(factor):
    df = pd.read_csv(f"{ROOT}/tables/topic_specific_models.csv")

    degenerate = set(df.loc[df["term"] == "DEGENERATE_NO_VARIANCE", ["model", "topic"]].itertuples(index=False, name=None))
    assert degenerate == KNOWN_DEGENERATE_CELLS, (
        f"{factor}: degenerate-cell set changed since this script was written -- "
        f"found {degenerate}, expected exactly {KNOWN_DEGENERATE_CELLS}. Investigate before proceeding.")

    prefix = f"C({factor})[T."
    sub = df[df["term"].str.startswith(prefix)].copy()
    sub["level"] = sub["term"].str.slice(len(prefix), -1)
    sub = sub[["model", "topic", "level", "coef"]]

    # confirm expected scope before proceeding -- fail loudly, not silently, on a mismatch
    models_found = sorted(sub["model"].unique())
    assert models_found == sorted(MODELS), f"{factor}: expected models {MODELS}, found {models_found}"
    topics_found = sorted(df["topic"].unique())
    assert len(topics_found) == 7, f"{factor}: expected 7 topics, found {len(topics_found)}: {topics_found}"

    personas = pd.read_csv(f"{ROOT}/data/personas.csv")
    all_levels = sorted(personas[factor].unique()) if factor == "profession" else sorted(personas["country"].unique())
    expected_n = len(all_levels)

    # add the omitted reference level back at coefficient 0.0, once per (model, topic) --
    # except for the known-degenerate cells, which get no rows at all (no coefficients
    # exist for them, so no ranking can be derived -- left as a gap, not a fabricated rank)
    ref_rows = []
    for m in MODELS:
        for t in topics_found:
            if (m, t) in KNOWN_DEGENERATE_CELLS:
                continue
            ref_rows.append({"model": m, "topic": t, "level": REFERENCE[factor], "coef": 0.0})
    sub = pd.concat([sub, pd.DataFrame(ref_rows)], ignore_index=True)

    counts = sub.groupby(["model", "topic"])["level"].nunique()
    non_degenerate_counts = counts[~counts.index.isin(KNOWN_DEGENERATE_CELLS)]
    bad = non_degenerate_counts[non_degenerate_counts != expected_n]
    assert bad.empty, f"{factor}: expected {expected_n} levels per (model,topic), found mismatches:\n{bad}"
    assert not any(idx in counts.index for idx in KNOWN_DEGENERATE_CELLS), (
        f"{factor}: a known-degenerate cell unexpectedly has coefficient rows -- investigate")

    return sub


def rank_within_model_topic(df):
    out = []
    for (m, t), grp in df.groupby(["model", "topic"]):
        grp = grp.copy()
        add_tie_aware_ranks(grp, "coef", "topic")
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def pivot_wide(df, level_col):
    ranked = rank_within_model_topic(df)
    r2_lookup = load_r2_lookup()
    all_topics = sorted(set(t for _, t in KNOWN_DEGENERATE_CELLS) | set(df["topic"].unique()))

    frames = []
    for m in MODELS:
        sub = ranked[ranked["model"] == m][[level_col, "topic", "coef", "rank_topic", "rank_label_topic"]]
        sub = sub.rename(columns={"coef": f"coef_{m}", "rank_topic": f"rank_{m}", "rank_label_topic": f"rank_label_{m}"})
        frames.append(sub.set_index([level_col, "topic"]))
    wide = frames[0]
    for f in frames[1:]:
        wide = wide.join(f, how="outer")
    wide = wide.reset_index()

    # embed R^2 and a reliability flag directly into every row, per model, per topic --
    # not left in a separate companion table a reader of this CSV might not open.
    for m in MODELS:
        wide[f"r2_{m}"] = wide["topic"].map(lambda t: r2_lookup.get((m, t)))
        wide[f"reliability_{m}"] = wide["topic"].map(lambda t: reliability_flag(m, t, r2_lookup))

    return wide


def build_r2_summary():
    df = pd.read_csv(f"{ROOT}/tables/topic_specific_r2.csv")
    return df[df["model"].isin(MODELS)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    country_raw = load_topic_coefs("country")
    country_raw = country_raw.rename(columns={"level": "country"})
    country_df = pivot_wide(country_raw, "country")
    country_df = country_df.sort_values(["country", "topic"])
    country_path = f"{OUT_DIR}/country_topic_profiles.csv"
    country_df.to_csv(country_path, index=False)
    print(f"Wrote {country_path} ({len(country_df)} rows, expect 20 x 7 = 140)")
    assert len(country_df) == 140

    profession_raw = load_topic_coefs("profession")
    profession_raw = profession_raw.rename(columns={"level": "profession"})
    profession_df = pivot_wide(profession_raw, "profession")
    profession_df = profession_df.sort_values(["profession", "topic"])
    profession_path = f"{OUT_DIR}/profession_topic_profiles.csv"
    profession_df.to_csv(profession_path, index=False)
    print(f"Wrote {profession_path} ({len(profession_df)} rows, expect 30 x 7 = 210)")
    assert len(profession_df) == 210

    r2_df = build_r2_summary()
    r2_path = f"{OUT_DIR}/topic_r2_summary.csv"
    r2_df.to_csv(r2_path, index=False)
    print(f"Wrote {r2_path} ({len(r2_df)} rows, expect 4 x 7 = 28)")

    print()
    print("=== country_topic_profiles.csv preview (climate change) ===")
    print(country_df[country_df.topic == "climate change"][
        ["country", "rank_llama", "rank_gemma", "rank_qwen", "rank_ministral"]
    ].to_string(index=False))
    print()
    print("=== profession_topic_profiles.csv preview (climate change) ===")
    print(profession_df[profession_df.topic == "climate change"][
        ["profession", "rank_llama", "rank_gemma", "rank_qwen", "rank_ministral"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
