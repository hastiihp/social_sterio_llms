"""Exploratory: diagnose why DeepSeek went from 63/75,600 valid (original) to
0/75,600 valid (health-conversation variant) -- a complete drop, not just a
further decline. Uses only raw_text already saved in the two result files;
no new inference.

Fix H5: an earlier conversational summary of this diagnosis (not persisted
in this script -- it was prose, not code) characterized the cause as
"primarily a tokenizer/parser bug" and cited a "19-fold whitespace
artifact". Both claims are removed here: the causal claim overstates what a
raw_text pattern match can establish (no tokenizer/generation logs were
examined), and the fold-ratio does not reproduce cleanly -- it depends
entirely on which "no-space run" definition is used (one definition gives
~188-fold, a stricter one is undefined because the original-framing count
is exactly zero under it). Rather than pick a definition to make a clean
ratio, Section 5 below reports the one number that IS well-defined and
reproducible regardless of definition: how many health-condition responses
a conservative, explicitly-specified parser can recover a plausible rating
from, with the causal question (tokenizer? chat template? decoding?)
stated as unresolved from available logs, not asserted.

Independent of analysis/: reads results/full_results_deepseek.csv and
results_health/health_full_results_deepseek.csv directly (FULL datasets,
not the 180-persona pilot subset -- the 63->0 drop being diagnosed is a
full-dataset fact). Does not touch analysis/ or master_results.csv.
"""
import re

import numpy as np
import pandas as pd

ROOT = "/Users/hastihosseinpour/Desktop/social_sterio_llms"


def load_safe(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def main():
    orig = load_safe(f"{ROOT}/results/full_results_deepseek.csv")
    health = load_safe(f"{ROOT}/results_health/health_full_results_deepseek.csv")

    print("=" * 78)
    print("1. 15 RANDOM raw_text SAMPLES (health-conversation variant), full text")
    print("=" * 78)
    rng_sample = health.sample(15, random_state=42)
    for i, (_, r) in enumerate(rng_sample.iterrows(), 1):
        print(f"\n--- Sample {i}: persona={r['persona_id']} topic={r['topic']} "
              f"condition={r['response_condition']} parse_failure_reason={r['parse_failure_reason']} ---")
        print(f"raw_text: {r['raw_text']}")

    print(f"\n{'='*78}")
    print("2. IS DEEPSEEK RESPONDING TO THE RATING QUESTION, OR THE HEALTH SMALL-TALK?")
    print("   Descriptive keyword heuristic (NOT a validated semantic classification --")
    print("   flagged as such, same caveat as step 9's epistemic/values-based refusal check)")
    print("=" * 78)
    health_kw = ["sleep", "stress", "rest", "therapist", "doctor", "counsel", "talk to someone",
                 "self-care", "relax", "advice", "take care of himself", "take care of herself",
                 "seek help", "well-being", "wellbeing", "mental health", "support system"]
    rating_kw = ["agree", "disagree", "rating of", "scale of 1", "estimate", "1 to 5", "would rate"]

    def has_any(text, kws):
        t = text.lower()
        return any(k in t for k in kws)

    health["mentions_health_kw"] = health["raw_text"].apply(lambda t: has_any(t, health_kw))
    health["mentions_rating_kw"] = health["raw_text"].apply(lambda t: has_any(t, rating_kw))
    health["contains_digit_1_5"] = health["raw_text"].str.contains(r"\b[1-5]\b", regex=True, na=False)

    n = len(health)
    print(f"  n={n:,} total health-variant rows")
    print(f"  mentions health/small-talk keywords (sleep/stress/therapist/advice/etc.): "
          f"{health['mentions_health_kw'].sum():,} ({100*health['mentions_health_kw'].mean():.1f}%)")
    print(f"  mentions rating-task keywords (agree/disagree/scale/estimate/etc.):        "
          f"{health['mentions_rating_kw'].sum():,} ({100*health['mentions_rating_kw'].mean():.1f}%)")
    print(f"  contains a bare digit 1-5 anywhere in the text:                            "
          f"{health['contains_digit_1_5'].sum():,} ({100*health['contains_digit_1_5'].mean():.1f}%)")
    both = (health["mentions_health_kw"] & health["mentions_rating_kw"]).sum()
    health_only = (health["mentions_health_kw"] & ~health["mentions_rating_kw"]).sum()
    rating_only = (~health["mentions_health_kw"] & health["mentions_rating_kw"]).sum()
    neither = (~health["mentions_health_kw"] & ~health["mentions_rating_kw"]).sum()
    print(f"\n  breakdown: both keyword types={both:,} ({100*both/n:.1f}%)  "
          f"health-only={health_only:,} ({100*health_only/n:.1f}%)  "
          f"rating-only={rating_only:,} ({100*rating_only/n:.1f}%)  "
          f"neither={neither:,} ({100*neither/n:.1f}%)")

    print("\n  Sample of rows that mention health keywords but NOT rating keywords")
    print("  (candidate 'pulled into small-talk instead of rating' cases):")
    candidates = health[health["mentions_health_kw"] & ~health["mentions_rating_kw"]]
    if len(candidates):
        for i, (_, r) in enumerate(candidates.sample(min(5, len(candidates)), random_state=1).iterrows(), 1):
            print(f"\n  [{i}] persona={r['persona_id']} topic={r['topic']}: {r['raw_text'][:300]}")
    else:
        print("  (none found)")

    print(f"\n{'='*78}")
    print("3. raw_text CHARACTER LENGTH: health-conversation vs original")
    print("=" * 78)
    orig_len = orig["raw_text"].str.len()
    health_len = health["raw_text"].str.len()
    print(f"  original:  n={len(orig_len):,}  mean={orig_len.mean():.1f}  median={orig_len.median():.1f}  "
          f"std={orig_len.std():.1f}  min={orig_len.min()}  max={orig_len.max()}")
    print(f"  health:    n={len(health_len):,}  mean={health_len.mean():.1f}  median={health_len.median():.1f}  "
          f"std={health_len.std():.1f}  min={health_len.min()}  max={health_len.max()}")
    print(f"\n  Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    p{p}: original={np.percentile(orig_len, p):.0f}   health={np.percentile(health_len, p):.0f}")
    print(f"\n  -> health-variant responses are on average "
          f"{'LONGER' if health_len.mean() > orig_len.mean() else 'SHORTER'} than original "
          f"({health_len.mean():.1f} vs {orig_len.mean():.1f} chars, "
          f"{100*(health_len.mean()/orig_len.mean()-1):+.1f}%).")

    # length by category, since the mix of categories itself shifted (Section 4)
    print(f"\n  Length by parse_failure_reason category (health variant):")
    print(health.groupby("parse_failure_reason", observed=True)["raw_text"].apply(
        lambda s: s.str.len().mean()).rename("mean_length").to_string())
    print(f"\n  Length by parse_failure_reason category (original):")
    print(orig.groupby("parse_failure_reason", observed=True)["raw_text"].apply(
        lambda s: s.str.len().mean()).rename("mean_length").to_string())

    print(f"\n{'='*78}")
    print("4. parse_failure_reason DISTRIBUTION: did the TYPE of non-compliance change?")
    print("=" * 78)
    orig_counts = orig["parse_failure_reason"].value_counts()
    health_counts = health["parse_failure_reason"].value_counts()
    all_reasons = sorted(set(orig_counts.index) | set(health_counts.index))
    comp = pd.DataFrame({
        "original_n": [orig_counts.get(r, 0) for r in all_reasons],
        "original_pct": [100 * orig_counts.get(r, 0) / len(orig) for r in all_reasons],
        "health_n": [health_counts.get(r, 0) for r in all_reasons],
        "health_pct": [100 * health_counts.get(r, 0) / len(health) for r in all_reasons],
    }, index=all_reasons)
    comp["pct_point_shift"] = comp["health_pct"] - comp["original_pct"]
    print(comp.to_string(float_format=lambda x: f"{x:8.2f}"))

    print(f"\n  Notable: 'salvageable_numeric' (a rating embedded in prose, extractable via regex)")
    print(f"  was {comp.loc['salvageable_numeric','original_pct']:.1f}% originally, now "
          f"{comp.loc['salvageable_numeric','health_pct']:.1f}% -- "
          f"{'ELIMINATED' if comp.loc['salvageable_numeric','health_pct']==0 else 'reduced'}.")
    print(f"  'other_malformed' (no extractable rating at all) went from "
          f"{comp.loc['other_malformed','original_pct']:.1f}% to {comp.loc['other_malformed','health_pct']:.1f}%.")
    print(f"  'explicit_refusal' went from {comp.loc['explicit_refusal','original_pct']:.1f}% to "
          f"{comp.loc['explicit_refusal','health_pct']:.1f}%.")

    print(f"\n  Sample of 'other_malformed' health-variant responses (the now-dominant category):")
    malformed = health[health["parse_failure_reason"] == "other_malformed"]
    for i, (_, r) in enumerate(malformed.sample(5, random_state=2).iterrows(), 1):
        print(f"\n  [{i}] persona={r['persona_id']} topic={r['topic']}: {r['raw_text'][:300]}")

    print(f"\n  Sample of 'explicit_refusal' health-variant responses:")
    refusal = health[health["parse_failure_reason"] == "explicit_refusal"]
    for i, (_, r) in enumerate(refusal.sample(5, random_state=3).iterrows(), 1):
        print(f"\n  [{i}] persona={r['persona_id']} topic={r['topic']}: {r['raw_text'][:300]}")

    print(f"\n{'='*78}")
    print("5. FIX H5 -- AUDIT-SENSITIVITY OUTPUT: conservative compact-text parser")
    print("=" * 78)
    print("  Definition (exact, reproducible): search raw_text for the literal substring")
    print("  'respond' + optional whitespace + 'with' + optional whitespace + a single digit 1-5")
    print("  (case-insensitive; matches both the compact 'respondwith4' form and normally-spaced")
    print("  'respond with 4'). This is intentionally narrow -- it does not attempt to recover every")
    print("  possible phrasing, only the one pattern that recurs across the samples printed above.")
    compact_pattern = re.compile(r"respond\s*with\s*([1-5])", re.IGNORECASE)

    def extract_compact(text):
        m = compact_pattern.search(text)
        return int(m.group(1)) if m else None

    health["compact_parser_rating"] = health["raw_text"].apply(extract_compact)
    n_recovered = int(health["compact_parser_rating"].notnull().sum())
    pct_recovered = 100 * n_recovered / len(health)
    print(f"\n  A conservative compact-text parser recovers {pct_recovered:.2f}% ({n_recovered:,}/{len(health):,}) of")
    print(f"  health-condition DeepSeek responses as containing a plausible rating; the remainder show")
    print(f"  refusal or other noncompliance patterns. The cause of the compact-text formatting")
    print(f"  (tokenizer, chat template, or decoding) is not established from available logs.")
    print(f"\n  Recovered rating value distribution (all from the narrow pattern above):")
    print(health["compact_parser_rating"].value_counts().sort_index().to_string())

    compact_out = health[health["compact_parser_rating"].notnull()][
        ["persona_id", "topic", "response_condition", "raw_text", "parse_failure_reason", "compact_parser_rating"]]
    compact_out.to_csv(f"{ROOT}/analysis_health/output/deepseek_health_compact_parser_audit.csv", index=False)
    print(f"\n  Wrote {ROOT}/analysis_health/output/deepseek_health_compact_parser_audit.csv "
          f"({len(compact_out):,} rows)")
    print(f"\n  This is a supplementary audit-sensitivity output, not a replacement of the strict-format")
    print(f"  parse_failure_reason classification used everywhere else in this project (Section 4 above,")
    print(f"  and analysis/ throughout) -- per analysis_plan.md, salvaged/recovered ratings are diagnostic")
    print(f"  only and are never substituted into primary results.")


if __name__ == "__main__":
    main()
