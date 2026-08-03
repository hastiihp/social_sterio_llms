#!/usr/bin/env python3
"""Independent, read-only adversarial audit of analysis_context.

Reads existing prompts/results. Writes only audit_context/outputs and logs.
No inference or experiment is launched.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_context" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CONTEXTS = ["health", "neutral", "positive", "negative_minor"]
NEW_CONTEXTS = ["neutral", "positive", "negative_minor"]
MODELS = ["llama", "gemma", "qwen", "ministral", "deepseek"]
RANK_MODELS = MODELS[:-1]
PILOT_COUNTRIES = ["Germany", "Brazil", "Nigeria", "South Korea"]
PILOT_PROFESSIONS = ["lawyer", "registered nurse", "truck driver", "farmer", "computer programmer"]
KEY = ["persona_id", "country", "profession", "gender", "age", "topic", "response_condition"]
COMPACT = re.compile(r"respond\s*with\s*([1-5])", re.I)


def path_for(condition: str, model: str) -> Path:
    if condition == "original":
        return ROOT / "results" / f"full_results_{model}.csv"
    if condition == "health":
        return ROOT / "results_health" / f"health_full_results_{model}.csv"
    return ROOT / "results_context" / f"{condition}_full_results_{model}.csv"


def load(condition: str, model: str, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(path_for(condition, model), keep_default_na=False,
                     na_values=[""], low_memory=False, usecols=usecols)
    if "strict_parsed_rating" in df:
        df["rating_numeric"] = pd.to_numeric(df["strict_parsed_rating"], errors="coerce")
    return df


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def integrity_audit():
    canonical = pd.read_csv(ROOT / "data" / "personas.csv")
    canonical_ids = set(canonical.persona_id.astype(str))
    rows = []
    for c in NEW_CONTEXTS:
        prompt = pd.read_csv(ROOT / "data_context" / f"{c}_prompts_full.csv",
                             usecols=KEY + ["prompt_template_version"])
        for m in MODELS:
            p = path_for(c, m)
            df = pd.read_csv(p, usecols=KEY + ["prompt_template_version"], low_memory=False)
            ids = set(df.persona_id.astype(str))
            dup = int(df.duplicated(KEY).sum())
            prompt_merge = prompt.merge(df, on=KEY, how="outer", indicator=True,
                                        validate="one_to_one", suffixes=("_prompt", "_result"))
            rows.append({
                "context": c, "model": m, "path": str(p.relative_to(ROOT)),
                "rows": len(df), "unique_personas": df.persona_id.nunique(),
                "missing_canonical_ids": len(canonical_ids - ids),
                "extra_persona_ids": len(ids - canonical_ids),
                "duplicate_canonical_keys": dup,
                "prompt_left_only": int((prompt_merge._merge == "left_only").sum()),
                "result_right_only": int((prompt_merge._merge == "right_only").sum()),
                "sha256": sha256(p),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "integrity_15_files.csv", index=False)
    return out


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assigned_string(path: Path, variable: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(variable)


def prompt_audit():
    health = module(ROOT / "data_health" / "health_render_prompts_full.py", "health_prompt")
    context = module(ROOT / "data_context" / "context_render_prompts_full.py", "context_prompt")
    original_a = assigned_string(ROOT / "data" / "render_prompts.py", "CONDITION_A")
    original_b = assigned_string(ROOT / "data" / "render_prompts.py", "CONDITION_B")
    tail_rows = []
    for cond, oc, hc, cc in [("A_forced", original_a, health.CONDITION_A_TAIL, context.CONDITION_A_TAIL),
                              ("B_optional", original_b, health.CONDITION_B_TAIL, context.CONDITION_B_TAIL)]:
        tail_rows.append({"condition": cond, "original_eq_health": oc == hc,
                          "original_eq_context": oc == cc, "bytes": len(oc.encode()),
                          "sha256": hashlib.sha256(oc.encode()).hexdigest()})

    personas = pd.read_csv(ROOT / "data" / "personas.csv")
    topics = pd.read_csv(ROOT / "data" / "topics.csv")
    topic_statement = topics.iloc[0].topic_statement
    samples = []
    checks = []
    for c in NEW_CONTEXTS:
        stored = pd.read_csv(ROOT / "data_context" / f"{c}_prompts_full.csv",
                             keep_default_na=False)
        for gender in ["male", "female", "neutral"]:
            chosen = personas[personas.gender == gender].iloc[[0, 1]]
            for _, prow in chosen.iterrows():
                row = next(pd.DataFrame([prow]).itertuples(index=False))
                expected = context.build_conversation(row, topic_statement, "A_forced", context.CONTEXTS[c])
                gotrow = stored[(stored.persona_id == prow.persona_id) &
                                (stored.topic == topics.iloc[0].topic) &
                                (stored.response_condition == "A_forced")].iloc[0]
                got = json.loads(gotrow.messages_json)
                pn, poss = prow.subject_pronoun, prow.possessive_pronoun
                grammar = f"{pn.capitalize()} {prow.be_verb} " in got[2]["content"]
                if c == "negative_minor":
                    grammar = grammar and f"one of {poss} bags" in got[2]["content"]
                elif c == "neutral":
                    grammar = grammar and f"and {context.HAVE_VERB[gender]} been arranging" in got[2]["content"]
                elif c == "positive":
                    grammar = grammar and f"and {prow.be_verb} really excited" in got[2]["content"]
                checks.append({"context": c, "gender": gender, "persona_id": prow.persona_id,
                               "stored_equals_fresh_render": got == expected,
                               "pronoun_grammar_check": grammar,
                               "turn3_ends_exact_A_tail": got[-1]["content"].endswith(original_a)})
                samples.append({"context": c, "gender": gender, "persona_id": prow.persona_id,
                                "messages": got})

    # Health: same breadth, including the historically error-prone have/has agreement.
    hp = pd.read_csv(ROOT / "data_health" / "health_prompts_full.csv", keep_default_na=False)
    for gender in ["male", "female", "neutral"]:
        for _, prow in personas[personas.gender == gender].iloc[[0, 1]].iterrows():
            row = next(pd.DataFrame([prow]).itertuples(index=False))
            expected = health.build_conversation(row, topic_statement, "A_forced")
            gotrow = hp[(hp.persona_id == prow.persona_id) & (hp.topic == topics.iloc[0].topic) &
                        (hp.response_condition == "A_forced")].iloc[0]
            got = json.loads(gotrow.messages_json)
            hv = "have" if gender == "neutral" else "has"
            nhv = "haven't" if gender == "neutral" else "hasn't"
            grammar = (f" {nhv} been" in got[0]["content"] and
                       f"{prow.subject_pronoun.capitalize()} {hv} been" in got[2]["content"] and
                       f"{hv.capitalize()} {prow.subject_pronoun} been" in got[3]["content"])
            checks.append({"context": "health", "gender": gender, "persona_id": prow.persona_id,
                           "stored_equals_fresh_render": got == expected,
                           "pronoun_grammar_check": grammar,
                           "turn3_ends_exact_A_tail": got[-1]["content"].endswith(original_a)})
            samples.append({"context": "health", "gender": gender, "persona_id": prow.persona_id,
                            "messages": got})
    pd.DataFrame(tail_rows).to_csv(OUT / "prompt_tail_byte_identity.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "prompt_render_checks.csv", index=False)
    (OUT / "prompt_render_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))


def cluster_intercept(diff: pd.Series, clusters: pd.Series):
    x = pd.DataFrame({"d": diff.astype(float), "cluster": clusters.astype(str)})
    fit = smf.ols("d ~ 1", x).fit(cov_type="cluster", cov_kwds={"groups": x.cluster})
    return float(fit.params.Intercept), float(fit.bse.Intercept), float(fit.pvalues.Intercept)


def ministral_audit():
    frames = {c: load(c, "ministral", KEY + ["is_abstention"])
              for c in ["original"] + CONTEXTS}
    rates, tests, topics = [], [], []
    for c, df in frames.items():
        for cond in ["A_forced", "B_optional"]:
            sub = df[df.response_condition == cond]
            rates.append({"context": c, "condition": cond, "n": len(sub),
                          "abstentions": int(sub.is_abstention.sum()),
                          "rate_pct": 100 * sub.is_abstention.mean()})
    orig = frames["original"]
    for c in CONTEXTS:
        ctx = frames[c]
        merged = orig.merge(ctx, on=KEY, suffixes=("_orig", "_ctx"), validate="one_to_one")
        b = merged[merged.response_condition == "B_optional"]
        d, se, p = cluster_intercept(b.is_abstention_ctx.astype(int) - b.is_abstention_orig.astype(int), b.persona_id)
        tests.append({"context": c, "n": len(b), "clusters": b.persona_id.nunique(),
                      "shift_pp": 100*d, "cluster_se_pp": 100*se, "cluster_p": p})
        for topic, g in b.groupby("topic", sort=True):
            td, tse, tp = cluster_intercept(g.is_abstention_ctx.astype(int) - g.is_abstention_orig.astype(int), g.persona_id)
            topics.append({"context": c, "topic": topic, "n": len(g),
                           "original_pct": 100*g.is_abstention_orig.mean(),
                           "context_pct": 100*g.is_abstention_ctx.mean(),
                           "shift_pp": 100*td, "cluster_se_pp": 100*tse, "cluster_p": tp})
    pd.DataFrame(rates).to_csv(OUT / "ministral_rates_raw.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "ministral_clustered_context_vs_original.csv", index=False)
    pd.DataFrame(topics).to_csv(OUT / "ministral_topic_clustered_full.csv", index=False)


def deepseek_audit():
    rows, dist, examples = [], [], []
    for c in ["original"] + CONTEXTS:
        df = load(c, "deepseek", ["persona_id", "topic", "response_condition", "raw_text",
                                   "strict_is_valid", "strict_parsed_rating", "parse_failure_reason"])
        text = df.raw_text.fillna("").astype(str)
        matches = text.str.extract(COMPACT, expand=False)
        salv = df.parse_failure_reason.eq("salvageable_numeric")
        new = matches.notna() & ~df.strict_is_valid & ~salv
        total = df.strict_is_valid | salv | new
        rows.append({"condition": c, "n": len(df), "strict_n": int(df.strict_is_valid.sum()),
                     "strict_pct": 100*df.strict_is_valid.mean(), "salvage_n": int(salv.sum()),
                     "salvage_pct": 100*salv.mean(), "compact_match_n": int(matches.notna().sum()),
                     "genuinely_new_n": int(new.sum()), "genuinely_new_pct": 100*new.mean(),
                     "true_total_n": int(total.sum()), "true_total_pct": 100*total.mean(),
                     "zero_space_gt5_pct": 100*((text.str.count(" ") == 0) & (text.str.len() > 5)).mean()})
        for digit, n in matches[new].value_counts().sort_index().items():
            dist.append({"condition": c, "digit": digit, "n": int(n), "pct_of_new": 100*n/new.sum()})
        # Independently selected deterministic samples; include both requested raw context files.
        if c in ["positive", "negative_minor"]:
            candidates = df[new].copy()
            candidates["digit"] = matches[new]
            for _, r in candidates.sample(min(20, len(candidates)), random_state=20260802).iterrows():
                examples.append({"condition": c, "persona_id": r.persona_id, "topic": r.topic,
                                 "response_condition": r.response_condition, "digit": r.digit,
                                 "raw_text": r.raw_text})
    pd.DataFrame(rows).to_csv(OUT / "deepseek_reconciliation_raw.csv", index=False)
    pd.DataFrame(dist).to_csv(OUT / "deepseek_new_digit_distribution.csv", index=False)
    pd.DataFrame(examples).to_csv(OUT / "deepseek_raw_samples_independent.csv", index=False)


def pilot_valid(condition, model):
    df = load(condition, model, KEY + ["strict_is_valid", "strict_parsed_rating"])
    return df[df.country.isin(PILOT_COUNTRIES) & df.profession.isin(PILOT_PROFESSIONS) &
              (df.response_condition == "A_forced") & df.strict_is_valid & df.rating_numeric.notna()].copy()


def correlations_audit():
    rows = []
    conditions = ["original"] + CONTEXTS
    for m in RANK_MODELS:
        frames = {c: pilot_valid(c, m) for c in conditions}
        for c1, c2 in itertools.combinations(conditions, 2):
            x = frames[c1][KEY[:-1] + ["rating_numeric"]].merge(
                frames[c2][KEY[:-1] + ["rating_numeric"]], on=KEY[:-1],
                suffixes=("_1", "_2"), validate="one_to_one")
            rho = stats.spearmanr(x.rating_numeric_1, x.rating_numeric_2).statistic
            rows.append({"model": m, "condition_1": c1, "condition_2": c2,
                         "n": len(x), "spearman_r": rho,
                         "is_context_pair": c1 != "original" and c2 != "original"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cross_context_correlations_raw.csv", index=False)
    summary = out.groupby("is_context_pair").spearman_r.agg(["mean", "min", "max", "count"]).reset_index()
    summary.to_csv(OUT / "cross_context_correlation_summary.csv", index=False)
    pair = out[out.is_context_pair].groupby(["condition_1", "condition_2"]).spearman_r.mean().reset_index()
    pair.to_csv(OUT / "cross_context_pair_means.csv", index=False)


def exact_p(x, y):
    x, y = np.round(np.asarray(x, float), 9), np.round(np.asarray(y, float), 9)
    obs = stats.spearmanr(x, y).statistic
    obs = 0.0 if np.isnan(obs) else obs
    vals = []
    for perm in itertools.permutations(range(len(y))):
        r = stats.spearmanr(x, y[list(perm)]).statistic
        vals.append(0.0 if np.isnan(r) else r)
    return obs, sum(abs(r) >= abs(obs)-1e-9 for r in vals)/len(vals), len(vals)


def fit_rank(df):
    factors = ["gender", "country", "profession", "age", "topic"]
    for c in factors:
        df[c] = df[c].astype(object)
    return smf.ols("rating_numeric ~ " + " + ".join(f"C({x})" for x in factors), df).fit()


def coefs(res, factor, levels):
    ref = sorted(levels)[0]
    out = {ref: 0.0}
    prefix = f"C({factor})[T."
    for k, v in res.params.items():
        if k.startswith(prefix):
            out[k[len(prefix):-1]] = float(v)
    return pd.Series(out).reindex(sorted(levels))


def bootstrap_extremes(df, model, context, framing, B=1000):
    rng = np.random.default_rng(0)
    ids = df.persona_id.unique()
    groups = {pid:g for pid, g in df.groupby("persona_id", observed=True)}
    levels = {"profession": sorted(PILOT_PROFESSIONS), "country": sorted(PILOT_COUNTRIES)}
    count = {(f,l): [0,0] for f in levels for l in levels[f]}
    success = 0
    for _ in range(B):
        sample = rng.choice(ids, len(ids), replace=True)
        boot = pd.concat([groups[x] for x in sample], ignore_index=True)
        try:
            res = fit_rank(boot)
        except Exception:
            continue
        success += 1
        for f, lvls in levels.items():
            v = coefs(res, f, lvls)
            count[(f, v.idxmax())][0] += 1
            count[(f, v.idxmin())][1] += 1
    return [{"context": context, "model": model, "framing": framing, "factor": f, "level": l,
             "p_top": count[(f,l)][0]/success, "p_bottom": count[(f,l)][1]/success,
             "bootstrap_B": success} for f in levels for l in levels[f]]


def ranking_audit():
    exact_rows, boot_rows = [], []
    for c in CONTEXTS:
        prior_exact = pd.read_csv(ROOT / "analysis_context" / "output" / f"{c}_ranking_robustness_exact_pvalues.csv")
        prior_boot = pd.read_csv(ROOT / "analysis_context" / "output" / f"{c}_ranking_robustness_bootstrap.csv")
        for m in RANK_MODELS:
            o, x = pilot_valid("original", m), pilot_valid(c, m)
            ro, rx = fit_rank(o), fit_rank(x)
            for factor, levels in [("profession", PILOT_PROFESSIONS), ("country", PILOT_COUNTRIES)]:
                rho, p, n = exact_p(coefs(ro, factor, levels), coefs(rx, factor, levels))
                old = prior_exact[(prior_exact.model == m) & (prior_exact.factor == factor)].iloc[0]
                exact_rows.append({"context": c, "model": m, "factor": factor,
                                   "audit_rho": rho, "reported_rho": old.spearman_r,
                                   "audit_exact_p": p, "reported_exact_p": old.exact_permutation_p,
                                   "audit_n_permutations": n, "reported_n_permutations": old.n_permutations})
            boot_rows += bootstrap_extremes(o, m, c, "original")
            boot_rows += bootstrap_extremes(x, m, c, c)
        # comparison is done after all contexts, preserving raw audit values.
    exact_df = pd.DataFrame(exact_rows)
    exact_df.to_csv(OUT / "ranking_exact_independent.csv", index=False)
    boot = pd.DataFrame(boot_rows)
    reported = pd.concat([pd.read_csv(ROOT / "analysis_context" / "output" /
                                     f"{c}_ranking_robustness_bootstrap.csv") for c in CONTEXTS], ignore_index=True)
    merged = boot.merge(reported, on=["context","model","framing","factor","level"],
                        suffixes=("_audit","_reported"), validate="one_to_one")
    merged["p_top_absdiff"] = (merged.p_top_audit - merged.p_top_reported).abs()
    merged["p_bottom_absdiff"] = (merged.p_bottom_audit - merged.p_bottom_reported).abs()
    merged.to_csv(OUT / "ranking_bootstrap_independent_vs_reported.csv", index=False)


def inherited_code_audit():
    # AST-normalized source comparisons document reuse/drift without trusting comments.
    ctx = ast.parse((ROOT / "analysis_context" / "_common.py").read_text())
    h1 = ast.parse((ROOT / "analysis_health" / "01_compare_health_vs_original.py").read_text())
    h4 = ast.parse((ROOT / "analysis_health" / "04_ranking_robustness.py").read_text())
    wanted = ["exact_permutation_pvalue", "bootstrap_ranks", "merge_on_canonical_key",
              "clustered_diff_context_minus_orig"]
    def funcs(tree): return {n.name: ast.unparse(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    cf, hf1, hf4 = funcs(ctx), funcs(h1), funcs(h4)
    rows = []
    for name in wanted:
        source = hf4.get(name) or hf1.get(name) or ""
        rows.append({"helper": name, "present_context": name in cf,
                     "audited_analogue_present": bool(source),
                     "context_ast_sha256": hashlib.sha256(cf.get(name, "").encode()).hexdigest(),
                     "audited_ast_sha256": hashlib.sha256(source.encode()).hexdigest() if source else "",
                     "note": "generic renaming/type cast may prevent byte identity; manually diff AST/source"})
    pd.DataFrame(rows).to_csv(OUT / "inherited_helper_trace.csv", index=False)


def main():
    integrity_audit()
    prompt_audit()
    inherited_code_audit()
    ministral_audit()
    deepseek_audit()
    correlations_audit()
    ranking_audit()
    print("audit complete")


if __name__ == "__main__":
    main()
