"""Step 3: Table 1 -- strict-parsing compliance/format-adherence by model.

Per analysis_plan.md Section 3, the taxonomy observed in this dataset is:

  strict_is_valid=True,  is_abstention=False  -> valid numeric rating (1-5)
  strict_is_valid=True,  is_abstention=True   -> valid abstention ("NA", Condition B only)
  strict_is_valid=False, parse_failure_reason=explicit_refusal    -> model declined to speculate
  strict_is_valid=False, parse_failure_reason=salvageable_numeric -> format failed, but a rating
                                                                      could be extracted from prose
  strict_is_valid=False, parse_failure_reason=other_malformed     -> format failed, no salvageable
                                                                      rating extractable
  strict_is_valid=False, parse_failure_reason=technical_failure   -> infrastructure/inference error

"% strict valid" is the overall strict-format-compliance rate (numeric + NA
combined). "% NA (valid abstention)" is reported separately as a diagnostic
breakdown of that same strict-valid population, not an additional bucket --
so columns do not sum to 100% by simple addition; see the printed note.
Per Section 3, models exceeding 5% non-technical invalid rate are flagged
for manual review, not excluded (Section 10).
"""
import pandas as pd

from _common import load_master, MODEL_ORDER, TABLES_DIR, NON_TECHNICAL_INVALID_THRESHOLD, display_template


def pct(numer, denom):
    return 100.0 * numer / denom if denom else float("nan")


def main():
    df = load_master()

    rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        n = len(sub)

        n_strict_valid = sub["strict_is_valid"].sum()
        n_na = (sub["is_abstention"]).sum()
        n_salvageable = (sub["parse_failure_reason"] == "salvageable_numeric").sum()
        n_refusal = (sub["parse_failure_reason"] == "explicit_refusal").sum()
        n_other_malformed = (sub["parse_failure_reason"] == "other_malformed").sum()
        n_technical = (sub["parse_failure_reason"] == "technical_failure").sum()

        n_non_technical_invalid = n_salvageable + n_refusal + n_other_malformed

        rows.append({
            "model": model,
            "n_rows": n,
            "pct_strict_valid": pct(n_strict_valid, n),
            "pct_na_valid_abstention": pct(n_na, n),
            "pct_salvageable_numeric": pct(n_salvageable, n),
            "pct_explicit_refusal": pct(n_refusal, n),
            "pct_other_malformed": pct(n_other_malformed, n),
            "pct_technical_failure": pct(n_technical, n),
            "pct_non_technical_invalid": pct(n_non_technical_invalid, n),
        })

    table = pd.DataFrame(rows).set_index("model").loc[MODEL_ORDER].reset_index()
    out_path = f"{TABLES_DIR}/table1_compliance.csv"
    table.to_csv(out_path, index=False)

    template = display_template(df["prompt_template_version"].iloc[0])
    print(f"Prompt template: {template}")
    print()
    print("TABLE 1: strict-parsing compliance by model (%, over all rows per model)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    print()
    print("Note: 'pct_na_valid_abstention' is a diagnostic subset of 'pct_strict_valid'")
    print("(every valid NA is also strict-valid), not an additional bucket. The five")
    print("mutually-exclusive buckets that partition 100% of rows are:")
    print("  strict_valid_numeric + na_valid_abstention + salvageable_numeric")
    print("  + explicit_refusal + other_malformed + technical_failure")
    check = table.copy()
    check["pct_strict_valid_numeric"] = check["pct_strict_valid"] - check["pct_na_valid_abstention"]
    check["partition_sum"] = (
        check["pct_strict_valid_numeric"] + check["pct_na_valid_abstention"]
        + check["pct_salvageable_numeric"] + check["pct_explicit_refusal"]
        + check["pct_other_malformed"] + check["pct_technical_failure"]
    )
    print(check[["model", "pct_strict_valid_numeric", "pct_na_valid_abstention",
                 "pct_salvageable_numeric", "pct_explicit_refusal", "pct_other_malformed",
                 "pct_technical_failure", "partition_sum"]].to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    print()
    print("Section 3 flag -- non-technical invalid rate (salvageable_numeric + explicit_refusal")
    print(f"+ other_malformed) exceeding {NON_TECHNICAL_INVALID_THRESHOLD*100:.0f}%:")
    for _, r in table.iterrows():
        flag = "FLAGGED for manual review" if r["pct_non_technical_invalid"] > NON_TECHNICAL_INVALID_THRESHOLD * 100 else "ok"
        print(f"  {r['model']:10s} {r['pct_non_technical_invalid']:6.2f}%  -> {flag}")

    print()
    print("Section 3 note: technical_failure rows are 0% for all models in this dataset")
    print("-- consistent with Section 10 (technical failures are auto-rerun before the")
    print("dataset is finalized, so none should remain at this stage).")

    print()
    print("--- supplementary: per-model x per-condition breakdown (Section 3 also asks for this) ---")
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        for cond in ["A_forced", "B_optional"]:
            csub = sub[sub["response_condition"] == cond]
            n = len(csub)
            n_strict_valid = csub["strict_is_valid"].sum()
            n_non_tech_invalid = (csub["parse_failure_reason"].isin(["salvageable_numeric", "explicit_refusal", "other_malformed"])).sum()
            print(f"  {model:10s} {cond:12s} strict_valid={pct(n_strict_valid,n):6.2f}%  non_technical_invalid={pct(n_non_tech_invalid,n):6.2f}%")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
