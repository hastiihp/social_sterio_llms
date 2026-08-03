#!/usr/bin/env python3
"""Supplemental raw-data checks for the context audit; no inference."""
import itertools
import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_audit as a

OUT = a.OUT


def rating_findings():
    rows = []
    for c in a.CONTEXTS:
        for m in a.RANK_MODELS:
            o, x = a.pilot_valid("original", m), a.pilot_valid(c, m)
            z = o[a.KEY[:-1] + ["rating_numeric"]].merge(
                x[a.KEY[:-1] + ["rating_numeric"]], on=a.KEY[:-1],
                suffixes=("_orig", "_ctx"), validate="one_to_one")
            d, se, p = a.cluster_intercept(z.rating_numeric_ctx-z.rating_numeric_orig, z.persona_id)
            rho = stats.spearmanr(z.rating_numeric_orig, z.rating_numeric_ctx).statistic
            rows.append({"context": c, "model": m, "n_pairs": len(z),
                         "clusters": z.persona_id.nunique(), "shift": d, "cluster_se": se,
                         "cluster_p": p, "exact_agreement": (z.rating_numeric_orig == z.rating_numeric_ctx).mean(),
                         "spearman_r": rho})
    pd.DataFrame(rows).to_csv(OUT / "rating_findings_raw.csv", index=False)


def all_model_rates():
    rows = []
    for m in a.MODELS:
        for c in ["original"] + a.CONTEXTS:
            df = a.load(c, m, ["response_condition","is_abstention","strict_is_valid"])
            b = df[df.response_condition == "B_optional"]
            rows.append({"model":m,"context":c,"n":len(df),
                         "abstention_all_pct":100*df.is_abstention.mean(),
                         "abstention_B_pct":100*b.is_abstention.mean(),
                         "strict_valid_all_pct":100*df.strict_is_valid.mean()})
    pd.DataFrame(rows).to_csv(OUT / "all_model_context_rates_raw.csv", index=False)


def ministral_order_and_gender():
    frames = {c:a.load(c,"ministral",a.KEY+["is_abstention"])
              for c in ["original"]+a.CONTEXTS}
    order = ["original","health","neutral","positive","negative_minor"]
    adjacent=[]
    for c1,c2 in zip(order,order[1:]):
        z=frames[c1].merge(frames[c2],on=a.KEY,suffixes=("_1","_2"),validate="one_to_one")
        z=z[z.response_condition=="B_optional"]
        d,se,p=a.cluster_intercept(z.is_abstention_2.astype(int)-z.is_abstention_1.astype(int),z.persona_id)
        adjacent.append({"condition_1":c1,"condition_2":c2,"shift_2_minus_1_pp":100*d,
                         "cluster_se_pp":100*se,"cluster_p":p,"clusters":z.persona_id.nunique()})
    pd.DataFrame(adjacent).to_csv(OUT/"ministral_adjacent_monotonic_tests.csv",index=False)
    gender=[]
    for c,df in frames.items():
        b=df[df.response_condition=="B_optional"]
        for g,x in b.groupby("gender"):
            gender.append({"context":c,"gender":g,"n":len(x),"rate_pct":100*x.is_abstention.mean()})
    pd.DataFrame(gender).to_csv(OUT/"ministral_rates_by_gender.csv",index=False)


def deepseek_gender_and_strict_slice():
    pattern=re.compile(r"respond\s*with\s*([1-5])",re.I)
    rows=[]
    for c in ["positive","negative_minor"]:
        df=a.load(c,"deepseek",["gender","topic","raw_text","strict_is_valid","strict_parsed_rating","parse_failure_reason"])
        match=df.raw_text.fillna("").str.extract(pattern,expand=False)
        salv=df.parse_failure_reason.eq("salvageable_numeric")
        new=match.notna() & ~df.strict_is_valid & ~salv
        for g,x in df.groupby("gender"):
            ix=x.index
            total=x.strict_is_valid | salv.loc[ix] | new.loc[ix]
            rows.append({"context":c,"gender":g,"n":len(x),"true_total_pct":100*total.mean(),
                         "new_pct":100*new.loc[ix].mean(),
                         "digit4_pct_of_new":100*(match.loc[ix][new.loc[ix]]=="4").mean()})
    pd.DataFrame(rows).to_csv(OUT/"deepseek_reconciliation_by_gender.csv",index=False)
    neg=a.load("negative_minor","deepseek",["topic","strict_is_valid","strict_parsed_rating"])
    v=neg[neg.strict_is_valid]
    numeric=pd.to_numeric(v.strict_parsed_rating,errors="coerce")
    pd.DataFrame([{"strict_n":len(v),"economic_n":int((v.topic=="economic redistribution").sum()),
                   "economic_pct":100*(v.topic=="economic redistribution").mean(),
                   "digit4_n":int((numeric==4).sum()),
                   "digit4_pct":100*(numeric==4).mean()}]).to_csv(
                       OUT/"deepseek_negative_strict_slice.csv",index=False)


def neutral_degenerate():
    o=a.pilot_valid("original","ministral")
    n=a.pilot_valid("neutral","ministral")
    # Condition B requires a load without the Cond-A filter.
    use=a.KEY+["strict_is_valid","strict_parsed_rating"]
    o=a.load("original","ministral",use); n=a.load("neutral","ministral",use)
    o=o[o.country.isin(a.PILOT_COUNTRIES)&o.profession.isin(a.PILOT_PROFESSIONS)&(o.response_condition=="B_optional")]
    n=n[n.country.isin(a.PILOT_COUNTRIES)&n.profession.isin(a.PILOT_PROFESSIONS)&(n.response_condition=="B_optional")]
    z=o.merge(n,on=a.KEY,suffixes=("_orig","_neutral"),validate="one_to_one")
    z["num_orig"]=pd.to_numeric(z.strict_parsed_rating_orig,errors="coerce")
    z["num_neutral"]=pd.to_numeric(z.strict_parsed_rating_neutral,errors="coerce")
    both=z[z.strict_is_valid_orig & z.strict_is_valid_neutral & z.num_orig.notna() & z.num_neutral.notna()]
    pd.DataFrame([{"both_valid_n":len(both),"all_original_4":bool((both.num_orig==4).all()),
                   "all_neutral_4":bool((both.num_neutral==4).all()),
                   "original_unique":repr(sorted(both.num_orig.unique())),
                   "neutral_unique":repr(sorted(both.num_neutral.unique()))}]).to_csv(
                       OUT/"ministral_neutral_degenerate_raw.csv",index=False)


def main():
    rating_findings(); all_model_rates(); ministral_order_and_gender(); deepseek_gender_and_strict_slice(); neutral_degenerate()
    print("supplemental complete")


if __name__ == "__main__": main()
