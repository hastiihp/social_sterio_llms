# Context audit artifacts

- `reports/adversarial_validation_report.md`: executive verdict, evidence, qualifications, manuscript readiness, and final verdict.
- `outputs/issue_register.csv`: severity/file/line/evidence/fix register.
- `outputs/findings_reproduction_ledger.csv`: 128 reported-versus-reproduced findings claims.
- `outputs/prompt_render_samples.json`: verbatim independently selected prompt renders.
- `outputs/integrity_15_files.csv`: row/persona/key checks and hashes for all 15 result files.
- `outputs/ministral_*`: raw rates, clustered comparisons, topic and gender diagnostics.
- `outputs/deepseek_*`: reconciliation, digit distributions, raw samples, and gender diagnostics.
- `outputs/cross_context_*`: independently reproduced correlations.
- `outputs/ranking_*`: independently reproduced exact permutations and bootstrap comparison.
- `logs/audit_run.log`: execution record.
- `run_audit.py`, `run_supplemental.py`, `build_reproduction_ledger.py`: reproducible audit code; none launches inference.
