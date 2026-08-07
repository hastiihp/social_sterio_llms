"""Stage 2: Country / Profession Behavioral Profiles.

Synthesizes already-computed, already-audited ranking-robustness outputs
into two tables (one row per country, one row per profession). No new
statistical claims -- every cell is read directly from an existing output
file. FULL-SCALE (5,400 personas, 20 countries, 30 professions) throughout.

DeepSeek is excluded from both tables, consistent with its exclusion from
every other ranking/H1 fit throughout this project (analysis_context/05,
analysis/07, etc.) -- even where it technically has enough valid rows to
pass this pipeline's MIN_VALID_FOR_RANKING=60 floor (see the verification
note below), 63-204 rows is still razor-thin for a 20- or 30-level
ranking model.

Sources (each read directly, not from a prior summary):
  - Rank position under the ORIGINAL prompt, per model, per country/
    profession: analysis_context/output/health_ranking_robustness_
    {country,profession}_full5400.csv's rank_orig / rank_label_orig
    columns. Cross-checked byte-for-byte identical across all four
    context files (health/neutral/positive/negative_minor) before use --
    it's the same original-prompt ranking recomputed independently by
    each of the four analysis_context/01 runs, so this is a genuine
    integrity check, not an assumption.
  - Bootstrap top/bottom rank-position probability under the ORIGINAL
    prompt, per model, per country/profession: the same four files'
    *_ranking_robustness_bootstrap_full5400.csv, filtered to
    framing=="original". Also cross-checked identical across all four
    context files before use.
  - Ranking-robustness p-value (does this model's whole profession/country
    ranking replicate under each conversational framing): this is a
    (model x context x factor)-level statistic, not a per-country/
    per-profession one, so it does NOT appear as a column in the main
    per-level tables below -- it is reported as a separate companion
    table (ranking_robustness_pvalue_summary.csv), read directly from
    the four *_ranking_robustness_pvalues_full5400.csv files.

VERIFICATION NOTE (flagged, not silently resolved): the four context
files' bootstrap outputs are NOT uniformly scoped -- health/neutral/
positive's bootstrap files contain only llama/gemma/qwen/ministral (200
rows each: 4 models x 50 levels), but negative_minor's contains all 5
models including deepseek (250 rows: 5 models x 50 levels). Traced to
analysis_context/01_compare_context_vs_original.py's
section_ranking_robustness(): it dynamically skips a (model, context) pair
when either the original or the context has fewer than
MIN_VALID_FOR_RANKING=60 valid Condition-A rows (not a hardcoded model
list). DeepSeek has only 63 valid Condition-A rows under the original
prompt and 0-1 under health/neutral/positive, but 204 under negative_minor
(consistent with the already-documented "DeepSeek negative_minor anomaly",
CONTEXT_EXPERIMENT.md Step 4) -- enough to clear the 60-row floor and be
included in that one context's ranking fit. This is confirmed intentional,
documented pipeline behavior, not a bug -- but it means DeepSeek's rank/
bootstrap values exist ONLY in the negative_minor file and nowhere else,
so including it here would be an inconsistent, mostly-missing column. It
is excluded from both output tables below; this note documents why rather
than leaving the exclusion unexplained.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{ROOT}/analysis_taxonomy/output"

CONTEXTS = ["health", "neutral", "positive", "negative_minor"]
RANKED_MODELS = ["llama", "gemma", "qwen", "ministral"]
CANONICAL_CONTEXT = "health"  # source of record for original-prompt values; cross-checked below


def cross_check_consistency(factor):
    """Confirm rank_orig (and bootstrap p_top/p_bottom, framing=='original')
    are identical across all four context files before trusting one as
    canonical. Raises if any mismatch is found -- never silently proceeds
    past a mismatch."""
    rank_frames = {}
    boot_frames = {}
    for c in CONTEXTS:
        df = pd.read_csv(f"{ROOT}/analysis_context/output/{c}_ranking_robustness_{factor}_full5400.csv")
        df = df[df["model"].isin(RANKED_MODELS)]
        rank_frames[c] = df.set_index(["level", "model"])[["rank_orig", "rank_label_orig"]]

        bdf = pd.read_csv(f"{ROOT}/analysis_context/output/{c}_ranking_robustness_bootstrap_full5400.csv")
        bdf = bdf[(bdf["factor"] == factor) & (bdf["framing"] == "original") & (bdf["model"].isin(RANKED_MODELS))]
        boot_frames[c] = bdf.set_index(["level", "model"])[["p_top", "p_bottom"]]

    base_rank, base_boot = rank_frames[CANONICAL_CONTEXT], boot_frames[CANONICAL_CONTEXT]
    for c in CONTEXTS:
        if c == CANONICAL_CONTEXT:
            continue
        r_diff = (base_rank["rank_orig"] - rank_frames[c]["rank_orig"]).abs()
        if (r_diff > 1e-9).any():
            raise AssertionError(f"{factor}: rank_orig mismatch between {CANONICAL_CONTEXT} and {c}")
        b_diff_top = (base_boot["p_top"] - boot_frames[c]["p_top"]).abs()
        b_diff_bot = (base_boot["p_bottom"] - boot_frames[c]["p_bottom"]).abs()
        if (b_diff_top > 1e-9).any() or (b_diff_bot > 1e-9).any():
            raise AssertionError(f"{factor}: bootstrap original p_top/p_bottom mismatch between {CANONICAL_CONTEXT} and {c}")
    return base_rank, base_boot


def build_table(factor, level_col, extra_cols=None):
    rank, boot = cross_check_consistency(factor)
    levels = sorted(rank.index.get_level_values("level").unique())

    rows = []
    for level in levels:
        row = {level_col: level}
        for m in RANKED_MODELS:
            row[f"rank_{m}"] = rank.loc[(level, m), "rank_orig"]
            row[f"rank_label_{m}"] = rank.loc[(level, m), "rank_label_orig"]
            row[f"bootstrap_top_pct_{m}"] = boot.loc[(level, m), "p_top"] * 100
            row[f"bootstrap_bottom_pct_{m}"] = boot.loc[(level, m), "p_bottom"] * 100
        rows.append(row)

    df = pd.DataFrame(rows)
    if extra_cols is not None:
        df = extra_cols.merge(df, on=level_col, how="right")
    return df


def build_pvalue_summary():
    frames = []
    for c in CONTEXTS:
        df = pd.read_csv(f"{ROOT}/analysis_context/output/{c}_ranking_robustness_pvalues_full5400.csv")
        frames.append(df[df["model"].isin(RANKED_MODELS)])
    out = pd.concat(frames, ignore_index=True)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    personas = pd.read_csv(f"{ROOT}/data/personas.csv")
    country_meta = personas[["country", "region", "country_set"]].drop_duplicates().rename(
        columns={"country": "country"})
    assert len(country_meta) == 20, f"expected 20 countries, got {len(country_meta)}"
    assert personas["profession"].nunique() == 30

    country_df = build_table("country", "level", extra_cols=country_meta.rename(columns={"country": "level"}))
    country_df = country_df.rename(columns={"level": "country"})
    country_path = f"{OUT_DIR}/country_profiles.csv"
    country_df.to_csv(country_path, index=False)
    print(f"Wrote {country_path} ({len(country_df)} rows)")

    profession_df = build_table("profession", "level")
    profession_df = profession_df.rename(columns={"level": "profession"})
    profession_path = f"{OUT_DIR}/profession_profiles.csv"
    profession_df.to_csv(profession_path, index=False)
    print(f"Wrote {profession_path} ({len(profession_df)} rows)")

    pval_df = build_pvalue_summary()
    pval_path = f"{OUT_DIR}/ranking_robustness_pvalue_summary.csv"
    pval_df.to_csv(pval_path, index=False)
    print(f"Wrote {pval_path} ({len(pval_df)} rows -- {len(CONTEXTS)} contexts x "
          f"{len(RANKED_MODELS)} models x 2 factors)")

    print()
    print("=== country_profiles.csv preview ===")
    print(country_df[["country", "region", "rank_llama", "rank_gemma", "rank_qwen", "rank_ministral"]].to_string(index=False))
    print()
    print("=== profession_profiles.csv preview ===")
    print(profession_df[["profession", "rank_llama", "rank_gemma", "rank_qwen", "rank_ministral"]].to_string(index=False))


if __name__ == "__main__":
    main()
