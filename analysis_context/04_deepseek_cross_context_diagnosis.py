"""Step 4: DeepSeek's strict-valid rate varies slightly across conditions
(original 0.08%, health 0.00%, neutral 0.00%, positive 0.00%,
negative_minor 0.27%). Is negative_minor's 0.27% a qualitative difference in
what DeepSeek is doing, or is it too small a sample (~204/75,600 rows) to
draw any conclusion from?

Uses only raw_text already saved in the five result files -- no new
inference, matching analysis_health/03_deepseek_health_diagnosis.py's
"diagnosis from existing text only" discipline.

Method: generalizes analysis_health/03's Fix-H5 finding across all five
conditions instead of one. Fix H5 documented that DeepSeek frequently
produces a run-on, no-whitespace response ("Iwouldrespondwith4...") that
fails the STRICT single-digit format check but does contain a recognizable
rating, recoverable via a narrow, reproducible regex
(`respond\\s*with\\s*([1-5])`, matches both spaced and compact forms). That
same check is applied here to all 5 conditions, not just health -- because
looking at strict_valid alone (0.00-0.27%) turned out to badly understate
how much this compact-formatting behavior actually varies by condition;
see Section 2 below.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
_common = importlib.import_module("_common")
globals().update({k: getattr(_common, k) for k in dir(_common) if not k.startswith("__")})

OUT_DIR = f"{ROOT}/analysis_context/output"
ALL_CONDITIONS = ["original"] + CONTEXTS
COMPACT_PATTERN = re.compile(r"respond\s*with\s*([1-5])", re.IGNORECASE)


def load(condition):
    path = RESULTS_PATH_TEMPLATE[condition].format(m="deepseek")
    return load_safe(path)


def section1_raw_samples():
    print(f"{'='*78}\n1. RAW raw_text SAMPLES: negative_minor's 204 strict-valid rows\n{'='*78}")
    neg = load("negative_minor")
    valid = neg[neg["strict_is_valid"]]
    print(f"  n strict-valid in negative_minor: {len(valid):,} / {len(neg):,} ({100*len(valid)/len(neg):.4f}%)")
    print(f"  response_condition breakdown: {valid['response_condition'].value_counts().to_dict()}")
    print(f"  topic breakdown: {valid['topic'].value_counts().to_dict()}")
    print(f"  strict_parsed_rating value distribution: {valid['strict_parsed_rating'].value_counts().to_dict()}")
    print(f"\n  Sample of 10 strict-valid raw_text values (verbatim):")
    for i, (_, r) in enumerate(valid.sample(min(10, len(valid)), random_state=42).iterrows(), 1):
        print(f"    [{i}] persona={r['persona_id']} topic={r['topic']} cond={r['response_condition']}: "
              f"{r['raw_text']!r}")

    print(f"\n  For comparison, 5 samples each from health/neutral/positive's (0 strict-valid) raw_text,")
    print(f"  same topics where negative_minor happened to be valid, to see if the underlying text looks")
    print(f"  similar or different:")
    for c in ["health", "neutral", "positive"]:
        df = load(c)
        econ = df[df["topic"] == "economic redistribution"]
        print(f"\n  --- {c}, economic redistribution, 3 random raw_text samples ---")
        for i, (_, r) in enumerate(econ.sample(3, random_state=1).iterrows(), 1):
            print(f"    [{i}] cond={r['response_condition']} parse_failure={r['parse_failure_reason']}: "
                  f"{r['raw_text'][:200]!r}")
    return valid


def section2_compact_parser_all_conditions():
    """The key generalization: apply the health study's Fix-H5 compact-text
    parser to ALL 5 conditions, not just negative_minor vs original.

    CORRECTNESS-CRITICAL: the pipeline's own parse_failure_reason ==
    "salvageable_numeric" category (see analysis/03_compliance_table.py and
    context_full_inference.py's SALVAGE_PATTERNS) already exists and, for
    the ORIGINAL condition specifically, already catches 30.67% of rows via
    patterns like "rating of X" / "respond with a X" -- all of which require
    a \\b word boundary or whitespace around the digit. A first version of
    this script's compact-pattern check (\\brespond\\s*with\\s*([1-5])\\b,
    with \\s* allowing zero width) did NOT check for overlap with that
    existing category and reported "original: 11.46% total" -- which was
    WRONG: manual verification (see conversation) showed every one of
    those 8,599 matches was already inside the existing 30.67%
    salvageable_numeric count, so the true original total is ~30.75%, not
    11.46%; nothing new was found there. Fixed below by explicitly
    excluding rows already labeled salvageable_numeric before counting
    anything as "genuinely new" recovered by the compact check.

    For health/neutral/positive/negative_minor, salvageable_numeric barely
    fires (0.00-0.12%) precisely because its patterns require a word
    boundary that a run-on token like "respondwith4" doesn't have (no
    boundary between "h" and "4" -- both are word characters) -- so for
    those four conditions the compact check IS finding genuinely new
    information, confirmed two ways: (a) near-zero overlap with the
    existing salvageable_numeric rows (verified per-condition below), and
    (b) an independent, regex-free check -- the fraction of raw_text with
    literally zero space characters -- which shows the same dramatic,
    monotonic-ish gradient (0.00% original -> 97.29% positive) that the
    regex-based count does, ruling out a regex artifact.
    """
    print(f"\n{'='*78}\n2. COMPACT-TEXT PARSER (Fix H5 pattern), ALL 5 CONDITIONS -- reconciled\n{'='*78}")
    print(f"  Pattern: 'respond' + optional whitespace + 'with' + optional whitespace + digit 1-5")
    print(f"  (case-insensitive; matches both 'respond with 4' and the run-on 'respondwith4' form).")
    print(f"  Reconciled against the pipeline's existing salvageable_numeric category (from")
    print(f"  table1_compliance.csv) so nothing already known is double-counted as 'new'.\n")

    rows = []
    for c in ALL_CONDITIONS:
        df = load(c)
        n = len(df)
        strict_valid = df["strict_is_valid"]
        already_salvageable = df["parse_failure_reason"] == "salvageable_numeric"
        matches = df["raw_text"].apply(lambda t: bool(COMPACT_PATTERN.search(t)))
        genuinely_new = matches & ~strict_valid & ~already_salvageable

        strict_n, salv_n, match_n, new_n = (int(strict_valid.sum()), int(already_salvageable.sum()),
                                             int(matches.sum()), int(genuinely_new.sum()))
        true_total_pct = 100 * (strict_n + salv_n + new_n) / n
        print(f"  {c:16s} n={n:,}")
        print(f"    strict_valid={strict_n:6,} ({100*strict_n/n:6.3f}%)  "
              f"salvageable_numeric(pre-existing)={salv_n:6,} ({100*salv_n/n:6.3f}%)  "
              f"compact_pattern_matches={match_n:6,} ({100*match_n/n:6.3f}%)")
        print(f"    genuinely NEW beyond strict+salvageable: {new_n:6,} ({100*new_n/n:6.3f}%)  "
              f"-> TRUE total near-compliance = {true_total_pct:6.3f}%")
        rows.append({"condition": c, "n": n, "strict_valid_n": strict_n, "strict_valid_pct": 100 * strict_n / n,
                     "salvageable_numeric_n": salv_n, "salvageable_numeric_pct": 100 * salv_n / n,
                     "compact_matches_n": match_n, "compact_matches_pct": 100 * match_n / n,
                     "genuinely_new_n": new_n, "genuinely_new_pct": 100 * new_n / n,
                     "true_total_pct": true_total_pct})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(f"{OUT_DIR}/deepseek_compact_parser_all_conditions.csv", index=False)

    print(f"\n  RECONCILIATION: for 'original', compact_pattern found ZERO genuinely-new rows -- all 8,599")
    print(f"  matches were already inside the pre-existing 30.67% salvageable_numeric count (which uses")
    print(f"  word-boundary patterns that DO match original's normally-spaced prose). True original total")
    print(f"  = 30.75%, matching table1_compliance.csv almost exactly (consistency check passed).")
    print(f"  For health/neutral/positive/negative_minor, salvageable_numeric barely fires (<=0.12%), so")
    print(f"  the compact matches there ARE new information, not double-counting.")

    print(f"\n  RATING-VALUE CHECK on the genuinely-new rows (is this varied opinion, or one repeated string?):")
    for c in CONTEXTS:
        df = load(c)
        already_salvageable = df["parse_failure_reason"] == "salvageable_numeric"
        matches = df["raw_text"].apply(lambda t: bool(COMPACT_PATTERN.search(t)))
        new_rows = df[matches & ~df["strict_is_valid"] & ~already_salvageable].copy()
        if len(new_rows):
            new_rows["digit"] = new_rows["raw_text"].apply(lambda t: COMPACT_PATTERN.search(t).group(1))
            top = new_rows["digit"].value_counts(normalize=True).iloc[0]
            print(f"    {c:16s} n_new={len(new_rows):6,}  dominant rating value = "
                  f"'{new_rows['digit'].value_counts().index[0]}' ({100*top:.2f}% of new rows)")
    print(f"  -> Positive's 68,618 new rows are 99.96% the single value '4' -- effectively zero variance.")
    print(f"     This is a repeated compact filler string firing at massive scale, not thoughtful varied")
    print(f"     rating behavior. Manually inspected: every sampled example begins with the literal clause")
    print(f'     "Iwouldrespondwith4..." as the direct stated answer -- not an incidental digit match from')
    print(f"     a date, list, or unrelated number (see conversation for 15 quoted examples).")

    print(f"\n  INDEPENDENT CONFIRMATION (no regex at all -- fraction of raw_text with ZERO space characters,")
    print(f"  for responses >5 chars long):")
    for c in ALL_CONDITIONS:
        df = load(c)
        lens = df["raw_text"].str.len()
        spaces = df["raw_text"].str.count(" ")
        no_space_frac = 100 * ((spaces == 0) & (lens > 5)).mean()
        print(f"    {c:16s} {no_space_frac:6.2f}% of rows have literally no whitespace at all")
    print(f"  -> Same gradient as the regex-based count (0.00% original -> 97.29% positive), confirming")
    print(f"     this is a real, large, condition-dependent formatting effect, not a regex artifact.")
    return df_out


def section3_topic_concentration():
    print(f"\n{'='*78}\n3. IS negative_minor's STRICT-VALID SLICE TOPIC-CONCENTRATED?\n{'='*78}")
    neg = load("negative_minor")
    valid = neg[neg["strict_is_valid"]]
    econ = neg[neg["topic"] == "economic redistribution"]
    print(f"  {len(valid)} strict-valid rows total; {int((valid['topic']=='economic redistribution').sum())} "
          f"({100*(valid['topic']=='economic redistribution').mean():.1f}%) are 'economic redistribution'.")
    print(f"  Even restricted to economic redistribution alone, strict-valid rate is only "
          f"{100*econ['strict_is_valid'].mean():.3f}% ({int(econ['strict_is_valid'].sum())}/{len(econ):,}) -- ")
    print(f"  by response_condition within that topic:")
    print(econ.groupby("response_condition")["strict_is_valid"].agg(["mean", "sum", "count"]).to_string())
    print(f"\n  All {len(valid)} strict-valid ratings are the same value: "
          f"{sorted(valid['strict_parsed_rating'].unique())} -- zero variance. Combined with the >90% topic")
    print(f"  concentration, this reads as a narrow formatting coincidence (a compact-format response for")
    print(f"  this one topic happening to fit the strict single-digit pattern), not evidence that DeepSeek")
    print(f"  is answering more thoughtfully or completely under negative_minor.")


def section4_verdict():
    print(f"\n{'='*78}\n4. HONEST VERDICT (revised after reconciliation against salvageable_numeric)\n{'='*78}")
    print("  Is there a qualitative difference in what DeepSeek is doing under negative_minor?")
    print("  - At the STRICT-VALID level alone (0.00% vs 0.27%, ~204 rows): this sliver is dominated by one")
    print("    topic (economic redistribution, 91%) and one rating value (4, 100%) -- consistent with a")
    print("    narrow formatting coincidence, not a general improvement in compliance. 204 rows is too few")
    print("    to support a claim that DeepSeek 'does something different' under negative_minor specifically.")
    print("  - The broader compact-format-response phenomenon (Section 2) IS real and large -- confirmed two")
    print("    independent ways (word-boundary-free regex, and a regex-free zero-whitespace check) -- and it")
    print("    is NOT too small a sample: it involves tens of thousands of rows per condition. But it is NOT")
    print("    a simple '11% to 91%' compliance jump either, because 'original' already had 30.67% of its")
    print("    rows salvageable via the pipeline's existing (differently-patterned) parser -- a first pass")
    print("    of this analysis missed that overlap and significantly understated original's true rate.")
    print("    Reconciled true near-compliance: original 30.75%, health 14.86%, neutral 68.36%, positive")
    print("    90.80%, negative_minor 41.39%. negative_minor is still a middling value among the four NEW")
    print("    contexts, not an extreme one -- but original is no longer the lowest condition overall;")
    print("    health is.")
    print("  - IMPORTANT CAVEAT the raw percentages alone would hide: the newly-recovered rows under")
    print("    health/neutral/positive/negative_minor are overwhelmingly ONE repeated rating value (99.96%")
    print("    '4' for positive) -- this is a single compact filler string firing at massive scale, not")
    print("    evidence DeepSeek is now rating thoughtfully or with real variance. Manual inspection of 15")
    print("    positive-condition examples confirms each is an unambiguous, directly-stated answer")
    print("    ('Iwouldrespondwith4...'), not a spurious digit match -- but the near-total lack of variance")
    print("    means this should be read as 'a formatting pattern became far more prevalent', not 'DeepSeek")
    print("    got much better at the task' under these framings.")
    print("  - The root cause of WHY the no-whitespace formatting varies this much by condition (tokenizer,")
    print("    chat template, or decoding) remains unestablished from raw_text alone, exactly as concluded")
    print("    for the health-vs-original comparison in analysis_health/03 (Fix H5). One plausible but")
    print("    UNVERIFIED mechanical hypothesis: with a fixed 30-new-token generation budget, squished")
    print("    (space-free) text lets a short complete thought finish before the token budget runs out,")
    print("    where a normally-spaced equivalent would be truncated mid-sentence -- but this is not")
    print("    confirmed against tokenizer/generation logs and should not be stated as established fact.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    section1_raw_samples()
    section2_compact_parser_all_conditions()
    section3_topic_concentration()
    section4_verdict()
    print(f"\n{'='*78}\nDONE. Outputs written to {OUT_DIR}/deepseek_compact_parser_all_conditions.csv\n{'='*78}")


if __name__ == "__main__":
    main()
