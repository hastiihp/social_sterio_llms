#!/usr/bin/env python3
"""Read-only independent reproduction for the neutral/positive grammar-fix audit."""
from __future__ import annotations

import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_grammar_fix" / "outputs"
LOG = ROOT / "audit_grammar_fix" / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

MODELS = ["llama", "gemma", "qwen", "ministral", "deepseek"]
CONDITIONS = ["original", "health", "neutral", "positive", "negative_minor"]
CONTEXTS = ["health", "neutral", "positive", "negative_minor"]
PILOT_COUNTRIES = ["Germany", "Brazil", "Nigeria", "South Korea"]
PILOT_PROFESSIONS = ["lawyer", "registered nurse", "truck driver", "farmer", "computer programmer"]
PATHS = {
    "original": ROOT / "results" / "full_results_{m}.csv",
    "health": ROOT / "results_health" / "health_full_results_{m}.csv",
    "neutral": ROOT / "results_context" / "neutral_full_results_{m}.csv",
    "positive": ROOT / "results_context" / "positive_full_results_{m}.csv",
    "negative_minor": ROOT / "results_context" / "negative_minor_full_results_{m}.csv",
}


def load(condition: str, model: str) -> pd.DataFrame:
    d = pd.read_csv(str(PATHS[condition]).format(m=model), keep_default_na=False, na_values=[""], low_memory=False)
    d["rating_numeric"] = pd.to_numeric(d["strict_parsed_rating"], errors="coerce")
    return d


def pilot_a(condition: str, model: str) -> pd.DataFrame:
    d = load(condition, model)
    return d[d.country.isin(PILOT_COUNTRIES) & d.profession.isin(PILOT_PROFESSIONS)
             & d.response_condition.eq("A_forced") & d.strict_is_valid & d.rating_numeric.notna()].copy()


def clustered_shift(merged: pd.DataFrame) -> tuple[float, float]:
    d = merged.copy()
    d["diff"] = d.rating_numeric_ctx - d.rating_numeric_orig
    fit = smf.ols("diff ~ 1", d).fit(cov_type="cluster", cov_kwds={"groups": d.persona_id})
    return float(fit.params["Intercept"]), float(fit.pvalues["Intercept"])


def exact_perm(x, y):
    x, y = np.round(np.asarray(x, float), 9), np.round(np.asarray(y, float), 9)
    obs = stats.spearmanr(x, y).statistic
    if np.isnan(obs): obs = 0.0
    vals = []
    for p in itertools.permutations(range(len(y))):
        r = stats.spearmanr(x, y[list(p)]).statistic
        vals.append(0.0 if np.isnan(r) else r)
    return float(obs), sum(abs(v) >= abs(obs) - 1e-9 for v in vals) / len(vals)


def fit_coefs(d: pd.DataFrame, factor: str) -> pd.Series:
    for c in ["gender", "country", "profession", "age", "topic", "persona_id"]:
        d[c] = d[c].astype(object)
    fit = smf.ols("rating_numeric ~ C(gender)+C(country)+C(profession)+C(age)+C(topic)", d).fit()
    levels = sorted(PILOT_PROFESSIONS if factor == "profession" else PILOT_COUNTRIES)
    ref = levels[0]
    out = {ref: 0.0}
    prefix = f"C({factor})[T."
    for term, value in fit.params.items():
        if term.startswith(prefix):
            out[term[len(prefix):-1]] = float(value)
    return pd.Series(out).reindex(levels)


def prompt_checks():
    rows = []
    exhaustive = []
    for ctx in ["neutral", "positive"]:
        d = pd.read_csv(ROOT / "data_context" / f"{ctx}_prompts_full.csv")
        nb = d[d.gender.eq("neutral")].drop_duplicates("persona_id")
        # Deterministic diversity sample: first five rows with distinct country/profession pairs.
        sample = nb.drop_duplicates(["country", "profession"]).iloc[np.linspace(0, nb.drop_duplicates(["country", "profession"]).shape[0]-1, 5, dtype=int)]
        for _, r in sample.iterrows():
            text = json.loads(r.messages_json)[2]["content"]
            bad = bool(re.search(r"\b[Tt]hey\b[^.]{0,180}\b(?:has|is)\b", text))
            expected = " have " if ctx == "neutral" else " are "
            rows.append({"context": ctx, "persona_id": r.persona_id, "country": r.country,
                         "profession": r.profession, "turn2": text, "expected_present": expected in text,
                         "bad_they_has_is": bad})
        bad_count = 0
        for s in nb.messages_json:
            text = json.loads(s)[2]["content"]
            suffix = text.split(". ")[-1]
            bad_count += int((" has " in suffix) if ctx == "neutral" else (" is really excited" in suffix))
        exhaustive.append({"context": ctx, "neutral_personas": len(nb), "bad_suffixes": bad_count})
    pd.DataFrame(rows).to_csv(OUT / "nonbinary_prompt_samples.csv", index=False)
    pd.DataFrame(exhaustive).to_csv(OUT / "nonbinary_prompt_exhaustive.csv", index=False)


def timestamps_and_duplicates():
    rows = []
    for ctx in ["neutral", "positive"]:
        for m in MODELS:
            p = ROOT / "results_context" / f"{ctx}_full_results_{m}.csv"
            d = pd.read_csv(p, usecols=["timestamp"], low_memory=False)
            ts = pd.to_datetime(d.timestamp, utc=True)
            rows.append({"context": ctx, "model": m, "path": str(p.relative_to(ROOT)),
                         "mtime_epoch": p.stat().st_mtime, "mtime_local": pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").tz_convert("Europe/Paris"),
                         "min_embedded_utc": ts.min(), "max_embedded_utc": ts.max(), "rows": len(d)})
    pd.DataFrame(rows).to_csv(OUT / "result_provenance.csv", index=False)
    matches = []
    wanted = re.compile(r"^(neutral|positive)_full_results_(llama|gemma|qwen|ministral|deepseek)\.csv$")
    for base, dirs, files in os.walk(ROOT):
        if ".git" in Path(base).parts: continue
        for f in files:
            if wanted.match(f): matches.append(str((Path(base) / f).relative_to(ROOT)))
    pd.DataFrame({"path": sorted(matches)}).to_csv(OUT / "result_duplicate_inventory.csv", index=False)


def abstention():
    rows = []
    for c in CONDITIONS:
        d = load(c, "ministral")
        b = d[d.response_condition.eq("B_optional")]
        rows.append({"condition": c, "n": len(b), "abstentions": int(b.is_abstention.sum()),
                     "rate_pct": 100 * b.is_abstention.mean()})
    pd.DataFrame(rows).to_csv(OUT / "ministral_abstention.csv", index=False)


def rating_shifts():
    key = ["persona_id", "country", "profession", "gender", "age", "topic"]
    rows = []
    for c in CONTEXTS:
        for m in MODELS[:-1]:
            o, x = pilot_a("original", m), pilot_a(c, m)
            z = o[key + ["rating_numeric"]].merge(x[key + ["rating_numeric"]], on=key,
                                                    suffixes=("_orig", "_ctx"), validate="one_to_one")
            shift, p = clustered_shift(z)
            rows.append({"context": c, "model": m, "n": len(z), "shift": shift, "p": p,
                         "significant_0_05": p < .05})
    pd.DataFrame(rows).to_csv(OUT / "rating_shifts_current_raw.csv", index=False)


def clustering():
    key = ["persona_id", "country", "profession", "gender", "age", "topic"]
    rows = []
    frames = {(c,m): pilot_a(c,m) for c in CONDITIONS for m in MODELS[:-1]}
    for m in MODELS[:-1]:
        for c1, c2 in itertools.combinations(CONDITIONS, 2):
            a, b = frames[c1,m], frames[c2,m]
            z = a[key+["rating_numeric"]].merge(b[key+["rating_numeric"]], on=key, suffixes=("_1","_2"), validate="one_to_one")
            rows.append({"model":m,"condition_1":c1,"condition_2":c2,"n":len(z),
                         "rho":stats.spearmanr(z.rating_numeric_1,z.rating_numeric_2).statistic,
                         "context_context":c1 != "original" and c2 != "original"})
    df=pd.DataFrame(rows); df.to_csv(OUT/"cross_context_raw.csv",index=False)
    df.groupby("context_context").rho.agg(["mean","count"]).reset_index().to_csv(OUT/"cross_context_summary.csv",index=False)
    df[df.context_context].groupby(["condition_1","condition_2"]).rho.mean().sort_values(ascending=False).reset_index().to_csv(OUT/"context_pair_order.csv",index=False)


def llama_positive_ranks():
    o, p = fit_coefs(pilot_a("original","llama"),"profession"), fit_coefs(pilot_a("positive","llama"),"profession")
    rho, pv = exact_perm(o,p)
    d=pd.DataFrame({"profession":o.index,"coef_original":o.values,"coef_positive":p.values})
    d["rank_original"]=d.coef_original.rank(method="min",ascending=False).astype(int)
    d["rank_positive"]=d.coef_positive.rank(method="min",ascending=False).astype(int)
    d.to_csv(OUT/"llama_positive_profession_ranks_current.csv",index=False)
    pd.DataFrame([{"rho":rho,"exact_p":pv}]).to_csv(OUT/"llama_positive_profession_current_summary.csv",index=False)


def output_freshness():
    cutoff=max((ROOT/"results_context"/f"{c}_full_results_{m}.csv").stat().st_mtime for c in ["neutral","positive"] for m in MODELS)
    rows=[]
    for p in sorted((ROOT/"analysis_context"/"output").glob("*.csv")):
        # These outputs directly or jointly involve neutral/positive. Health-only and negative-only files do not.
        involved = p.name.startswith(("neutral_","positive_","cross_context_","abstention_stability_","deepseek_"))
        if involved: rows.append({"path":str(p.relative_to(ROOT)),"mtime_epoch":p.stat().st_mtime,"after_latest_result":p.stat().st_mtime>cutoff,"delta_seconds":p.stat().st_mtime-cutoff})
    pd.DataFrame(rows).to_csv(OUT/"analysis_output_freshness.csv",index=False)


if __name__ == "__main__":
    prompt_checks(); timestamps_and_duplicates(); abstention(); rating_shifts(); clustering(); llama_positive_ranks(); output_freshness()
    print("audit reproduction complete")
