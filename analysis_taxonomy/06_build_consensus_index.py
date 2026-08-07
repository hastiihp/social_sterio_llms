"""Stage 6: Consensus Index.

Which persona x topic cells do the four compliant models (llama, gemma,
qwen, ministral -- deepseek excluded, same established exclusion pattern
as every prior stage) agree on, and which cause maximum disagreement?
Original prompt, Condition A, full-scale (5,400 personas x 7 topics =
37,800 cells). No new inference: every rating reused directly from
results/results_original_{model}.csv's already-audited strict_parsed_rating
column. All 4 models are 100% strict-valid under Condition A (reverified
directly here, not assumed from Stage 1's summary) -- every cell has
exactly 4 ratings, no missing-data handling needed.

Unlike Stages 1-5, this stage's task explicitly calls for new (but simple,
transparent, fully-disclosed) descriptive computation over already-audited
ratings -- a per-cell standard deviation, a rank, and a chi-square
goodness-of-fit check -- not a re-run of model inference. "No new
inference" is respected: no model is called again, nothing upstream of
strict_parsed_rating is touched.

sd_across_models uses pandas' default sample standard deviation (ddof=1,
n=4 ratings per cell) -- stated explicitly since no ddof convention is
established elsewhere in this project for a 4-observation SD.
"""
import os

import pandas as pd
from scipy.stats import chisquare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{ROOT}/analysis_taxonomy/output"

MODELS = ["llama", "gemma", "qwen", "ministral"]
HIGH_DISAGREEMENT_FRACTION = 0.10  # design choice, stated explicitly -- top/bottom 10% by SD


def load_model_condA(model):
    src = f"{ROOT}/results/results_original_{model}.csv"
    df = pd.read_csv(src, low_memory=False)
    condA = df[df["response_condition"] == "A_forced"].copy()
    assert len(condA) == 37800, f"{model}: expected 37800 Condition-A rows, got {len(condA)}"
    assert condA["strict_is_valid"].all(), f"{model}: not all Condition-A rows are strict-valid"
    assert condA[["persona_id", "topic"]].drop_duplicates().shape[0] == 37800, (
        f"{model}: persona_id x topic is not unique across Condition-A rows")
    condA["rating"] = pd.to_numeric(condA["strict_parsed_rating"], errors="raise")
    return condA[["persona_id", "country", "gender", "age", "profession", "topic", "rating"]], src


def build_wide_table():
    frames = {}
    sources = []
    for m in MODELS:
        df, src = load_model_condA(m)
        sources.append(src)
        frames[m] = df.set_index(["persona_id", "topic"])

    # cross-check persona attributes (country/gender/age/profession) agree across all
    # 4 models for the same persona_id -- they must, since it's the same canonical
    # persona grid, but verify rather than assume
    base_attrs = frames["llama"][["country", "gender", "age", "profession"]]
    for m in MODELS[1:]:
        cmp_attrs = frames[m][["country", "gender", "age", "profession"]]
        mismatch = (base_attrs != cmp_attrs.reindex(base_attrs.index)).any(axis=1)
        assert not mismatch.any(), f"{m}: persona attributes disagree with llama for {mismatch.sum()} rows"

    wide = base_attrs.copy()
    for m in MODELS:
        wide[f"rating_{m}"] = frames[m]["rating"]
    wide = wide.reset_index()

    rating_cols = [f"rating_{m}" for m in MODELS]
    assert wide[rating_cols].isna().sum().sum() == 0, "unexpected missing ratings after merge"

    wide["mean_rating"] = wide[rating_cols].mean(axis=1)
    wide["sd_across_models"] = wide[rating_cols].std(axis=1, ddof=1)
    wide = wide.sort_values("sd_across_models", ascending=True).reset_index(drop=True)
    wide["consensus_rank"] = wide.index + 1  # 1 = lowest SD = highest consensus

    return wide, sources


def chi_square_pattern_check(group_df, full_df, factor, label):
    """Goodness-of-fit: does `factor`'s distribution within group_df differ from
    its distribution in the full 37,800-cell population (base rate)? The study
    design is a full factorial grid, so the base rate is exactly proportional to
    each category's share of the full design -- not assumed uniform, computed
    directly from full_df each time."""
    base_counts = full_df[factor].value_counts()
    base_props = base_counts / base_counts.sum()
    group_counts = group_df[factor].value_counts().reindex(base_counts.index, fill_value=0)
    expected = base_props * len(group_df)
    stat, p = chisquare(f_obs=group_counts.values, f_exp=expected.values)
    print(f"  [{label}] {factor}: chi2={stat:.3f}, df={len(base_counts)-1}, p={p:.4g}")
    # per-category observed vs expected rate, for the ones most over/under-represented
    rate_ratio = (group_counts / expected).sort_values(ascending=False)
    print(f"    most over-represented: {rate_ratio.index[0]} ({rate_ratio.iloc[0]:.2f}x base rate)")
    print(f"    most under-represented: {rate_ratio.index[-1]} ({rate_ratio.iloc[-1]:.2f}x base rate)")
    return {"group": label, "factor": factor, "chi2": stat, "df": len(base_counts) - 1, "p_value": p}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    wide, sources = build_wide_table()
    print("Sources used (all 4 models' Condition-A rows, original prompt, full-scale):")
    for s in sources:
        print(f"  {s}")
    print()

    out_cols = ["persona_id", "country", "profession", "gender", "age", "topic",
                "mean_rating", "sd_across_models", "consensus_rank"] + [f"rating_{m}" for m in MODELS]
    full_out = wide[out_cols]
    full_path = f"{OUT_DIR}/consensus_index.csv"
    full_out.to_csv(full_path, index=False)
    print(f"Wrote {full_path} ({len(full_out)} rows)")

    top20_consensus = full_out.head(20)
    top20_disagreement = full_out.tail(20).sort_values("sd_across_models", ascending=False)
    top20_consensus.to_csv(f"{OUT_DIR}/consensus_top20_highest_consensus.csv", index=False)
    top20_disagreement.to_csv(f"{OUT_DIR}/consensus_top20_highest_disagreement.csv", index=False)
    print(f"Wrote consensus_top20_highest_consensus.csv and consensus_top20_highest_disagreement.csv")
    print()

    n = len(wide)
    k = int(round(n * HIGH_DISAGREEMENT_FRACTION))
    high_disagreement_group = wide.sort_values("sd_across_models", ascending=False).head(k)
    high_consensus_group = wide.sort_values("sd_across_models", ascending=True).head(k)
    print(f"High-disagreement / high-consensus group size: {k} cells each (top/bottom {HIGH_DISAGREEMENT_FRACTION*100:.0f}% of {n})")
    print()

    pattern_rows = []
    print("=== Chi-square goodness-of-fit: HIGH-DISAGREEMENT group vs. full-population base rate ===")
    for factor in ["topic", "country", "profession", "gender"]:
        pattern_rows.append(chi_square_pattern_check(high_disagreement_group, wide, factor, "high_disagreement"))
    print()
    print("=== Chi-square goodness-of-fit: HIGH-CONSENSUS group vs. full-population base rate ===")
    for factor in ["topic", "country", "profession", "gender"]:
        pattern_rows.append(chi_square_pattern_check(high_consensus_group, wide, factor, "high_consensus"))

    pattern_df = pd.DataFrame(pattern_rows)
    pattern_path = f"{OUT_DIR}/consensus_pattern_chisquare.csv"
    pattern_df.to_csv(pattern_path, index=False)
    print()
    print(f"Wrote {pattern_path}")

    print()
    print("=== SD distribution summary ===")
    print(wide["sd_across_models"].describe())


if __name__ == "__main__":
    main()
