"""Step 9: DeepSeek-specific report -- separate treatment per analysis_plan.md
Section 15 ("DeepSeek's non-compliance is documented as a per-model finding,
not treated as a pipeline defect").

Consolidates: strict validity / salvageable / refusal rates (per condition),
illustrative raw_text examples per category (to check the plan's claim of
"clear content engagement... rather than generic refusal or failure to
parse the prompt"), and an explicit summary of whether/how deepseek was
included in each inferential model fit in steps 5-8, given its very low
strict-valid rate (63/75,600 = 0.08%).
"""
import pandas as pd

from _common import load_master, TABLES_DIR

pd.set_option("display.max_colwidth", 100)


def main():
    df = load_master()
    d = df[df["model"] == "deepseek"]
    n = len(d)

    print("=" * 78)
    print("1. COMPLIANCE RATES (strict parsing), overall and by condition")
    print("=" * 78)
    reasons = ["none", "salvageable_numeric", "explicit_refusal", "other_malformed", "technical_failure"]
    rows = []
    for cond in ["overall", "A_forced", "B_optional"]:
        sub = d if cond == "overall" else d[d["response_condition"] == cond]
        cn = len(sub)
        row = {"scope": cond, "n": cn}
        for r in reasons:
            row[r] = 100.0 * (sub["parse_failure_reason"] == r).sum() / cn
        rows.append(row)
    rate_table = pd.DataFrame(rows)
    rate_table.to_csv(f"{TABLES_DIR}/deepseek_compliance_by_condition.csv", index=False)
    print(rate_table.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print()
    print(f"Strict validity rate: {100*(d['parse_failure_reason']=='none').sum()/n:.2f}%  "
          f"(n={ (d['parse_failure_reason']=='none').sum() }/{n})")
    print(f"Salvageable rate:     {100*(d['parse_failure_reason']=='salvageable_numeric').sum()/n:.2f}%")
    print(f"Refusal rate:         {100*(d['parse_failure_reason']=='explicit_refusal').sum()/n:.2f}%")
    print(f"Other malformed rate: {100*(d['parse_failure_reason']=='other_malformed').sum()/n:.2f}%")
    print(f"Technical failure:    {100*(d['parse_failure_reason']=='technical_failure').sum()/n:.2f}%")
    print()
    print("All 63 strictly-valid rows are Condition A (forced); DeepSeek produced ZERO strictly-valid")
    print("rows -- numeric or NA -- under Condition B. Its Condition-B answered rate, under the")
    print("corrected definition (Fix 5: answered = strict_is_valid & rating_numeric.notnull(), not the")
    print("earlier ~is_abstention definition that miscounted non-compliant text as 'answered'), is")
    print("correctly 0% -- i.e. 100% NON-RESPONSE under Condition B. Non-response here is NOT the same")
    print("as valid abstention: it is composed entirely of refusal/malformed/salvageable-but-noncompliant")
    print("text (see the breakdown below), never a clean, strictly-parsed 'NA'.")

    print()
    print("=" * 78)
    print("2. WHAT THE NON-COMPLIANT OUTPUT ACTUALLY LOOKS LIKE (checking analysis_plan.md Section 15's")
    print("   claim: 'clear content engagement... rather than generic refusal or failure to parse')")
    print("=" * 78)
    for reason, label in [("none", "strict-valid (n=63)"),
                           ("salvageable_numeric", "salvageable_numeric"),
                           ("explicit_refusal", "explicit_refusal"),
                           ("other_malformed", "other_malformed")]:
        sub = d[d["parse_failure_reason"] == reason]
        print(f"\n--- {label}, n={len(sub)} ---")
        for t in sub["raw_text"].head(2):
            print(f"  {t[:180]}")

    print()
    print("Closer check on 'explicit_refusal' (n=31,396): is this a semantic/values refusal")
    print("(objecting to the premise of demographic-based speculation) or an epistemic/format")
    print("refusal (declining to invent a specific number for a fictional person)?")
    refusal = d[d["parse_failure_reason"] == "explicit_refusal"]
    n_format_opener = refusal["raw_text"].str.startswith("I cannot provide").sum()
    n_appropriateness = refusal["raw_text"].str.contains("appropriate", case=False, na=False).sum()
    n_stereotype_lang = refusal["raw_text"].str.contains("stereotyp|generalization|unfair", case=False, na=False, regex=True).sum()
    print(f"  starts with 'I cannot provide...' (epistemic/format refusal opener): {n_format_opener}/{len(refusal)} ({100*n_format_opener/len(refusal):.1f}%)")
    print(f"  mentions 'appropriate' (possible values-based language):            {n_appropriateness}/{len(refusal)} ({100*n_appropriateness/len(refusal):.1f}%)")
    print(f"  mentions stereotype/generalization/unfair language:                  {n_stereotype_lang}/{len(refusal)} ({100*n_stereotype_lang/len(refusal):.1f}%)")
    print("  -> A substantial majority of refusals use insufficient-information language (e.g. 'I cannot")
    print("     provide an exact integer...') while still reasoning about demographic context in the same")
    print("     response, which is more consistent with an epistemic objection to false precision than a")
    print("     values-based refusal to engage -- though this is a descriptive heuristic based on keyword")
    print("     patterns (substring/opener matching on raw_text), not a validated semantic classification.")
    print("     A keyword count cannot rule out mixed or ambiguous cases within the 84%/1%/0% buckets above;")
    print("     treat this as suggestive framing for Section 15's claim, not a confirmed categorization.")

    print()
    print("=" * 78)
    print("3. WHETHER/HOW DEEPSEEK IS INCLUDED IN EACH INFERENTIAL MODEL (steps 5-8)")
    print("=" * 78)
    print("""
  Step 5 (hypothesis models):
    - Per-model OLS: INCLUDED, n=63. gender and response_condition dropped from its formula
      (zero variance: all 63 rows are Condition A, gender constant "neutral"). R^2=0.85,
      n_params=30 vs n=63 (df_resid=33) -- flagged as low observation-to-parameter ratio. Also
      fit with persona-clustered SEs and a mixed-effects (random-intercept) variant per Fix 2,
      reported alongside the original HC3 SEs in tables/hypothesis_model_deepseek.csv.
    - Pooled model (Condition A only, Section 4 spec): EXCLUDED per Fix 4. An earlier version of
      this pipeline included deepseek here with the C(model) reference level pinned to llama to
      contain numerical instability (deepseek-as-reference had caused ~1e10 coefficients that
      contaminated every OTHER model's estimates too). That workaround is superseded: deepseek's
      n=63, covering only 15/30 professions, 13/20 countries, and 1/3 genders, makes its
      interaction terms non-identifiable IN PRINCIPLE, not merely numerically unstable -- pinning
      the reference level only relocated where the instability showed up. The pooled model is now
      fit on llama/gemma/qwen/ministral only (236 terms, 0 unstable), and deepseek's own per-model
      regression above remains its separately reported, heavily-caveated result.

  Step 6 (abstention / optional-condition logistic regression):
    - EXCLUDED from the fitted regression: deepseek has zero variance in the "answered" outcome
      under Condition B (correctly 0% answered / 100% non-response under the Fix-5 definition --
      see section 1 above), which would cause perfect separation. Reported only in the
      descriptive answered-rate tables (with the non-compliance caveat attached), not modeled.

  Step 7 (cross-model agreement):
    - EXCLUDED from the primary 4-model Spearman/kappa matrix. Reported separately
      (deepseek x each other model, n=63 matched cells each) with an explicit sparsity flag.
      Notably showed a NEGATIVE correlation with every other model (strongest with llama,
      rho=-0.667) on this 63-cell, 2-topic slice -- flagged as intriguing but not
      interpretable given the sample.

  Step 8 (variance ranking / partial R^2):
    - INCLUDED but flagged exploratory/unreliable throughout, and explicitly EXCLUDED from the
      H1 support check (profession vs. country/age/gender), which was run for the other four
      models only. gender could not be ranked at all (zero variance). Step 8 now reports two
      scopes (Fix 3: primary Condition-A-only vs. exploratory A+B-pooled), but this distinction
      does not affect deepseek's own numbers -- all 63 of its valid rows are Condition A already,
      so its partial R^2 values are identical under both scopes.
""")

    print("=" * 78)
    print("4. CONTEXT FROM analysis_plan.md Section 15 (prompt-reinforcement comparison)")
    print("=" * 78)
    print("""
  A reinforced-instruction variant (friend_v2_strict) was tested pre-pilot specifically to
  raise deepseek's compliance. It improved deepseek's strict-compliance from 0/12 to 6/12 on
  the matched test set -- a real improvement, but below the predefined 9/12 adoption threshold.
  Independently, friend_v2_strict also changed rating values for 4/5 other models on identical
  persona-topic pairs and altered Falcon-H1's abstention decision outright, which was treated as
  the primary reason for rejection regardless of the deepseek threshold. The frozen template
  (friend_final in this codebase's output; friend_v2_explicit_gender in raw provenance) is the
  ORIGINAL, unreinforced wording. DeepSeek's low compliance under this template is therefore a
  documented, deliberate per-model finding -- not an artifact of an unusual or under-tested
  prompt -- consistent with the treatment above.
""")

    print("=" * 78)
    print("SUMMARY LINE FOR WRITE-UP")
    print("=" * 78)
    print(f"DeepSeek: {100*(d['parse_failure_reason']=='none').sum()/n:.2f}% strict-valid "
          f"({(d['parse_failure_reason']=='none').sum()}/{n} rows, all Condition A; 0% strict-valid, "
          f"i.e. 100% non-response, under Condition B), "
          f"{100*(d['parse_failure_reason']=='salvageable_numeric').sum()/n:.1f}% salvageable, "
          f"{100*(d['parse_failure_reason']=='explicit_refusal').sum()/n:.1f}% refusal (predominantly "
          f"insufficient-information language by keyword heuristic, not a validated semantic "
          f"classification -- see section 2), {100*(d['parse_failure_reason']=='other_malformed').sum()/n:.1f}% "
          f"other malformed, 0% technical failure. Included in the per-model OLS hypothesis models (step 5a) "
          f"with explicit low-power caveats and clustered/mixed-effects SEs (Fix 2); EXCLUDED from the pooled "
          f"model (step 5b, Fix 4 -- n=63 makes its interaction terms non-identifiable in principle, not just "
          f"unstable); excluded from the abstention logistic regression (step 6) and the primary cross-model "
          f"agreement matrix (step 7) due to structural zero-variance / sparsity; included but flagged "
          f"exploratory in variance ranking (step 8, both scopes per Fix 3). Never excluded from descriptive "
          f"reporting per analysis_plan.md Section 10.")


if __name__ == "__main__":
    main()
