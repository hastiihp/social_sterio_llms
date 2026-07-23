"""Shared constants and helpers for the analysis pipeline.

See analysis_plan.md for the authoritative methodology this pipeline follows.
"""
import os

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ANALYSIS_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR = os.path.join(ROOT_DIR, "tables")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
MASTER_CSV = os.path.join(ANALYSIS_DIR, "master_results.csv")

MODEL_FILES = {
    "llama": "full_results_llama.csv",
    "gemma": "full_results_gemma.csv",
    "qwen": "full_results_qwen.csv",
    "ministral": "full_results_ministral.csv",
    "deepseek": "full_results_deepseek.csv",
}
MODEL_ORDER = ["llama", "gemma", "qwen", "ministral", "deepseek"]

EXPECTED_ROWS_PER_MODEL = 75600

# The frozen prompt template that produced this data is logged in provenance
# (prompt_template_version column) as "friend_v2_explicit_gender". Per
# analysis_plan.md's naming note, this is renamed to "friend_final" in all
# output/write-up text to avoid confusion with the separately-tested and
# rejected "friend_v2_strict" reinforcement variant. Raw provenance values
# are left untouched in the data itself; this mapping is applied only when
# producing human-facing labels/output.
TEMPLATE_RENAME = {
    "friend_v2_explicit_gender": "friend_final",
}


def display_template(raw_value: str) -> str:
    """Map a raw provenance prompt_template_version value to its write-up name."""
    return TEMPLATE_RENAME.get(raw_value, raw_value)


def load_master(**kwargs):
    """Load analysis/master_results.csv the one correct way.

    The literal string "NA" is a valid, meaningful value in raw_text /
    normalized_text / strict_parsed_rating (the model's exact compliant
    Condition-B abstention output), not a missing-value marker. A plain
    pd.read_csv() applies pandas' default NA sentinel list -- which
    includes "NA" -- and silently nulls those cells out. Every script that
    reads the master file MUST go through this loader instead of calling
    pd.read_csv(MASTER_CSV) directly, or the bug re-appears on load even
    though it was fixed at merge time.
    """
    import pandas as pd

    defaults = dict(keep_default_na=False, na_values=[""], low_memory=False)
    defaults.update(kwargs)
    return pd.read_csv(MASTER_CSV, **defaults)


GENDERS = ["female", "male", "neutral"]
AGES = [25, 45, 65]
CONDITIONS = ["A_forced", "B_optional"]
DEMOGRAPHIC_FACTORS = ["gender", "country", "profession", "age"]
NON_TECHNICAL_INVALID_THRESHOLD = 0.05  # Section 3: flag if non-technical invalid rate exceeds 5%


def clustered_proportion_ci(data, value_col, cluster_col):
    """Cluster-robust point estimate + 95% CI for a single proportion (0/1
    value_col), via an intercept-only linear-probability model with
    persona-clustered SEs. Always estimable given any within-group variance,
    unlike a logit -- the standard fallback used across this pipeline (07b,
    05c) whenever a comparison hits deterministic separation (rate exactly 0
    or 1) in one arm/group: report the deterministic side as a structural
    fact, and give this cluster-robust CI for whichever side still has
    variance, rather than forcing an uninterpretable model fit.
    """
    import statsmodels.formula.api as smf

    n_clusters = data[cluster_col].nunique()
    if n_clusters < 2 or len(data) < 2:
        return dict(prop=float("nan"), se_cluster=float("nan"), ci_low=float("nan"),
                    ci_high=float("nan"), n_clusters=n_clusters)
    res = smf.ols(f"{value_col} ~ 1", data=data).fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
    ci = res.conf_int().loc["Intercept"]
    return dict(prop=float(res.params["Intercept"]), se_cluster=float(res.bse["Intercept"]),
                ci_low=float(ci[0]), ci_high=float(ci[1]), n_clusters=n_clusters)

# Columns referenced inside C(...) formula terms or as statsmodels cov_kwds
# 'groups' across the regression scripts (05*, 06, 07b, 08).
FORMULA_DTYPE_COLUMNS = ["gender", "country", "profession", "age", "topic",
                          "response_condition", "model", "persona_id"]


def cast_formula_dtypes(df, columns=None):
    """Cast formula-facing columns off pandas' Arrow-backed StringDtype to plain
    numpy object dtype, in place on a copy.

    load_master() correctly returns pandas 3.0's default StringDtype for text
    columns (see its own docstring re: the "NA"-as-missing-value bug) -- that
    fix is untouched here. But older patsy/statsmodels formula parsing raises
    `TypeError: Cannot interpret '<StringDtype(na_value=nan)>' as a data type`
    when a C(column) term or a cov_kwds={'groups': df[col]} argument
    references a StringDtype column directly, since it does a raw
    np.issubdtype-style check that doesn't recognize the extension dtype.
    Call this on the frame returned by load_master(), at the point of use in
    each regression script, right before any smf.ols/smf.logit/OrderedModel/
    mixedlm call -- not inside load_master() itself, which should keep
    returning the data as originally loaded.
    """
    import pandas as pd

    if columns is None:
        columns = FORMULA_DTYPE_COLUMNS
    df = df.copy()
    for col in columns:
        if col in df.columns and isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype("object")
    return df
