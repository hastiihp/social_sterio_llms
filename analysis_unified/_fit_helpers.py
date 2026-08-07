"""Mixed-model fitting helpers for the unified variance-decomposition model.

fit_mixedlm_with_fallback is adapted directly from analysis/05_hypothesis_models.py's
function of the same name (same optimizer fallback order: default -> lbfgs -> powell;
same explicit res.converged check rather than treating "no exception" as success; same
policy of returning None rather than ever merging a non-converged fit). Extended here
with a `reml` parameter, since the LRT comparison in this script requires ML (not REML)
fits for both the full and reduced model, while the headline coefficient table uses REML
(the standard preference for unbiased variance-component estimation) -- the original
function hardcoded reml=True throughout, since 05_hypothesis_models.py never needed ML.
"""
import warnings

import statsmodels.formula.api as smf


def fit_mixedlm_with_fallback(formula, data, groups, label, reml=True, maxiter=200):
    attempts = [("default", {}), ("lbfgs", {"method": "lbfgs"}), ("powell", {"method": "powell"})]
    for name, kwargs in attempts:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                md = smf.mixedlm(formula, data=data, groups=groups)
                res = md.fit(reml=reml, maxiter=maxiter, **kwargs)
                for w in caught:
                    print(f"  WARNING (MixedLM fit, {label}, attempt={name}, reml={reml}): {w.message}", flush=True)
            if res.converged:
                print(f"  MixedLM ({label}, reml={reml}): CONVERGED on attempt '{name}' (llf={res.llf:.4f}).", flush=True)
                return res, f"converged ({name})"
            else:
                print(f"  MixedLM ({label}, reml={reml}): attempt '{name}' did NOT converge (res.converged=False).", flush=True)
        except Exception as e:
            print(f"  MixedLM ({label}, reml={reml}): attempt '{name}' FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"  MixedLM ({label}, reml={reml}): ALL optimizer attempts failed to converge.", flush=True)
    return None, "MIXED_MODEL_NONCONVERGED_EXCLUDED"
