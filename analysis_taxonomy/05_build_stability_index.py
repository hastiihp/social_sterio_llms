"""Stage 5: Stability Index (per country / per profession).

Entity-level analog of Stage 4's model-level Context Sensitivity Index:
one row per country (20), one row per profession (30), each with 3
already-derivable stability signals per model, laid out side by side --
NOT collapsed into one score, same design choice as Stage 4. No new
statistical claims: every number is read directly from an existing,
already-audited output file, or is a trivial max-min range over
already-verified rank positions. Full-scale throughout. DeepSeek excluded
throughout, consistent with Stages 1-4.

Components:
  1. topic_rank_range_{model} -- range (max-min) of this level's rank
     position across the 7 topics, read from Stage 3's own output
     (analysis_taxonomy/output/{country,profession}_topic_profiles.csv).
     Degenerate topics (rank is NaN -- see Stage 3's own documentation)
     are dropped before computing the range, and topic_rank_n_valid_{model}
     records how many of the 7 topics actually contributed, so a range
     computed from fewer than 7 topics is never silently indistinguishable
     from one computed from all 7.
  2. framing_rank_range_{model} -- range (max-min) of this level's rank
     position across the original prompt plus the 4 conversational
     framings (5 positions total, matching this project's established
     "5 prompt types" framing used throughout Stages 1 and 4). Built from
     rank_orig + rank_ctx in the same four ranking-robustness source files
     Stage 2 used (analysis_context/output/{context}_ranking_robustness_
     {country,profession}_full5400.csv), pulling rank_ctx this time (Stage
     2's own output only kept rank_orig).
  3. bootstrap_top_pct_{model} / bootstrap_bottom_pct_{model} -- carried
     forward unchanged from Stage 2's own output
     (analysis_taxonomy/output/{country,profession}_profiles.csv) -- not
     recomputed, just included here so both a topic-based and a
     framing-based instability signal plus a resampling-robustness signal
     are visible in one place.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{ROOT}/analysis_taxonomy/output"

CONTEXTS = ["health", "neutral", "positive", "negative_minor"]
RANKED_MODELS = ["llama", "gemma", "qwen", "ministral"]
TOPICS = ["climate change", "economic redistribution", "gender equality", "immigration",
          "lgbtq rights", "religion and secularism", "trust in government"]


def component_1_topic_rank_range(factor, level_col):
    src = f"{OUT_DIR}/{factor}_topic_profiles.csv"
    df = pd.read_csv(src)
    assert len(df[level_col].unique()) * 7 == len(df), f"{factor}: expected {level_col} x 7 topics rows"

    out = {}
    for level, grp in df.groupby(level_col):
        row = {}
        for m in RANKED_MODELS:
            vals = grp[f"rank_{m}"].dropna()
            row[f"topic_rank_range_{m}"] = (vals.max() - vals.min()) if len(vals) >= 2 else float("nan")
            row[f"topic_rank_n_valid_{m}"] = len(vals)
        out[level] = row
    return out, src


def component_2_framing_rank_range(factor):
    frames = {}
    for c in CONTEXTS:
        src = f"{ROOT}/analysis_context/output/{c}_ranking_robustness_{factor}_full5400.csv"
        df = pd.read_csv(src)
        frames[c] = df[df["model"].isin(RANKED_MODELS)]

    # cross-check rank_orig identical across all 4 context files before use (same check as Stage 2)
    base = frames["health"].set_index(["level", "model"])["rank_orig"]
    for c in CONTEXTS[1:]:
        cmp = frames[c].set_index(["level", "model"])["rank_orig"]
        diff = (base - cmp).abs()
        assert (diff.fillna(0) < 1e-9).all(), f"{factor}: rank_orig mismatch between health and {c}"

    levels = sorted(frames["health"]["level"].unique())
    out = {level: {} for level in levels}
    for m in RANKED_MODELS:
        for level in levels:
            positions = [base.loc[(level, m)]]
            for c in CONTEXTS:
                rc = frames[c].set_index(["level", "model"])["rank_ctx"]
                positions.append(rc.loc[(level, m)])
            out[level][f"framing_rank_range_{m}"] = max(positions) - min(positions)
    return out, [f"{ROOT}/analysis_context/output/{c}_ranking_robustness_{factor}_full5400.csv" for c in CONTEXTS]


def component_3_bootstrap(factor, level_col):
    src = f"{OUT_DIR}/{factor}_profiles.csv"
    df = pd.read_csv(src)
    cols = [level_col] + [f"bootstrap_top_pct_{m}" for m in RANKED_MODELS] + \
           [f"bootstrap_bottom_pct_{m}" for m in RANKED_MODELS]
    return df[cols].set_index(level_col).to_dict("index"), src


def build(factor, level_col):
    t1, t1_src = component_1_topic_rank_range(factor, level_col)
    t2, t2_src = component_2_framing_rank_range(factor)
    t3, t3_src = component_3_bootstrap(factor, level_col)

    levels = sorted(t1.keys())
    assert set(t1.keys()) == set(t2.keys()) == set(t3.keys()), f"{factor}: level sets disagree across components"

    rows = []
    for level in levels:
        row = {level_col: level}
        row.update(t1[level])
        row.update(t2[level])
        row.update(t3[level])
        rows.append(row)
    return pd.DataFrame(rows), (t1_src, t2_src, t3_src)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    country_df, country_src = build("country", "country")
    country_path = f"{OUT_DIR}/country_stability_index.csv"
    country_df.to_csv(country_path, index=False)
    print(f"Wrote {country_path} ({len(country_df)} rows)")
    print(f"  sources: {country_src}")

    profession_df, profession_src = build("profession", "profession")
    profession_path = f"{OUT_DIR}/profession_stability_index.csv"
    profession_df.to_csv(profession_path, index=False)
    print(f"Wrote {profession_path} ({len(profession_df)} rows)")
    print(f"  sources: {profession_src}")

    print()
    print("=== country_stability_index.csv preview ===")
    preview_cols = ["country"] + [f"topic_rank_range_{m}" for m in RANKED_MODELS] + \
                    [f"framing_rank_range_{m}" for m in RANKED_MODELS]
    print(country_df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
