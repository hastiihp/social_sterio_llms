"""Step 2: structural validation of the merged dataset.

Checks row counts, persona_id structural uniqueness (one row per
persona x topic x condition, not literal row-uniqueness -- each persona
appears 14x per model by design: 7 topics x 2 conditions), completeness of
every gender/country/profession/age/topic/condition combination, and full
per-model balance tables for each factor. Prints a clear pass/fail summary;
does not raise on failure, so all checks run and are reported together.
"""
import pandas as pd

from _common import (
    load_master,
    MODEL_ORDER,
    EXPECTED_ROWS_PER_MODEL,
    GENDERS,
    AGES,
    CONDITIONS,
)

EXPECTED_PERSONAS = 5400
EXPECTED_TOPICS = 7
EXPECTED_REPLICATES_PER_PERSONA = EXPECTED_TOPICS * len(CONDITIONS)  # 14


def check(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return passed


def main():
    df = load_master()
    results = []

    print("=" * 70)
    print("1. ROW COUNTS")
    print("=" * 70)
    for model in MODEL_ORDER:
        n = (df["model"] == model).sum()
        results.append(check(
            f"{model}: row count == {EXPECTED_ROWS_PER_MODEL:,}",
            n == EXPECTED_ROWS_PER_MODEL,
            f"got {n:,}",
        ))

    print()
    print("=" * 70)
    print("2. PERSONA STRUCTURE (5,400 personas x 7 topics x 2 conditions = 14 rows/persona)")
    print("=" * 70)
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        n_personas = sub["persona_id"].nunique()
        results.append(check(
            f"{model}: distinct persona_id count == {EXPECTED_PERSONAS:,}",
            n_personas == EXPECTED_PERSONAS,
            f"got {n_personas:,}",
        ))
        counts_per_persona = sub.groupby("persona_id").size()
        bad = counts_per_persona[counts_per_persona != EXPECTED_REPLICATES_PER_PERSONA]
        results.append(check(
            f"{model}: every persona has exactly {EXPECTED_REPLICATES_PER_PERSONA} rows (7 topics x 2 conditions)",
            len(bad) == 0,
            f"{len(bad)} persona(s) with wrong count" if len(bad) else "",
        ))
        dup = sub.duplicated(subset=["persona_id", "topic", "response_condition"]).sum()
        results.append(check(
            f"{model}: no duplicate persona_id x topic x condition rows",
            dup == 0,
            f"{dup} duplicates" if dup else "",
        ))

    persona_sets = {m: set(df.loc[df["model"] == m, "persona_id"]) for m in MODEL_ORDER}
    ref = persona_sets[MODEL_ORDER[0]]
    same_across_models = all(persona_sets[m] == ref for m in MODEL_ORDER)
    results.append(check(
        "same set of 5,400 persona_ids used identically across all 5 models",
        same_across_models,
    ))

    print()
    print("=" * 70)
    print("3. NO MISSING VALUES IN KEY DESIGN COLUMNS")
    print("=" * 70)
    design_cols = ["gender", "country", "profession", "age", "topic", "response_condition"]
    for col in design_cols:
        n_missing = (df[col].astype(str).str.strip() == "").sum() + df[col].isnull().sum()
        results.append(check(f"no missing values in '{col}'", n_missing == 0, f"{n_missing} missing" if n_missing else ""))

    print()
    print("=" * 70)
    print("4. FACTOR LEVEL SETS ARE THE EXPECTED CANONICAL SETS")
    print("=" * 70)
    results.append(check(f"gender levels == {GENDERS}", sorted(df["gender"].unique()) == sorted(GENDERS), f"got {sorted(df['gender'].unique())}"))
    results.append(check(f"age levels == {AGES}", sorted(df["age"].unique()) == sorted(AGES), f"got {sorted(df['age'].unique())}"))
    results.append(check(f"condition levels == {CONDITIONS}", sorted(df["response_condition"].unique()) == sorted(CONDITIONS), f"got {sorted(df['response_condition'].unique())}"))
    results.append(check(f"topic count == {EXPECTED_TOPICS}", df["topic"].nunique() == EXPECTED_TOPICS, f"got {df['topic'].nunique()}: {sorted(df['topic'].unique())}"))

    print()
    print("=" * 70)
    print("5. FULL DESIGN COMPLETENESS: every country x profession x gender x age")
    print("   combination present, with exactly 14 rows (7 topics x 2 conditions), per model")
    print("=" * 70)
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        combo_counts = sub.groupby(["country", "profession", "gender", "age"]).size()
        n_combos = len(combo_counts)
        expected_combos = sub["country"].nunique() * sub["profession"].nunique() * len(GENDERS) * len(AGES)
        results.append(check(
            f"{model}: distinct country x profession x gender x age combos == {expected_combos}",
            n_combos == expected_combos,
            f"got {n_combos}",
        ))
        bad_combo = combo_counts[combo_counts != EXPECTED_REPLICATES_PER_PERSONA]
        results.append(check(
            f"{model}: every combo has exactly {EXPECTED_REPLICATES_PER_PERSONA} rows",
            len(bad_combo) == 0,
            f"{len(bad_combo)} combo(s) with wrong count" if len(bad_combo) else "",
        ))

    print()
    print("=" * 70)
    print("6. PER-MODEL BALANCE TABLES")
    print("=" * 70)
    for factor in ["gender", "country", "profession", "age", "topic", "response_condition"]:
        print(f"\n--- balance by {factor} (rows per level, per model) ---")
        tab = pd.crosstab(df[factor], df["model"])
        tab = tab[MODEL_ORDER]
        print(tab.to_string())
        # balanced means every level x model cell is equal within each model's column
        balanced = tab.apply(lambda col: col.nunique() == 1, axis=0).all()
        results.append(check(f"{factor}: perfectly balanced across levels within every model", bool(balanced)))

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"ALL CHECKS PASSED ({n_pass}/{n_total})")
    else:
        print(f"{n_total - n_pass} CHECK(S) FAILED ({n_pass}/{n_total} passed) -- see [FAIL] lines above")


if __name__ == "__main__":
    main()
