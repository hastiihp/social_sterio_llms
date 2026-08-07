"""Step 10: figures 2-7.

All model colors are the validated palette from _style.py (checked with the
dataviz skill's validator in MODEL_ORDER sequence). The two behavioral
clusters found in steps 6-8 (llama/gemma/deepseek: never abstain, vs.
qwen/ministral: topic-gated abstention the majority of the time) are
conveyed via panel layout, background tinting, and direct annotation
rather than color hue-family, because no reordering of hue-family-grouped
colors survived the CVD validator (orange-red and orange-yellow adjacent
pairs both failed) -- color stays tied to model identity; clustering is a
separate, explicit visual channel.

Condition A (forced) is used wherever a rating value is plotted (Fig 2, 4,
5), consistent with step 7's precedent: it is the only condition where all
four main models are 100% strict-valid, giving an unconfounded, fully
matched sample. Fig 3 is specifically about Condition B (that IS the
abstention-eligible condition).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from _common import load_master, MODEL_ORDER, FIGURES_DIR
from _style import (
    apply_base_style, MODEL_COLOR, FACTOR_COLOR, INK_PRIMARY, INK_SECONDARY,
    INK_MUTED, GRIDLINE, BASELINE, DIVERGING_POS, DIVERGING_NEG,
    CLUSTER_NEVER_ABSTAIN_BG, CLUSTER_TOPIC_GATED_BG, CLUSTER_FLAGGED_BG,
    DEEPSEEK_HATCH,
)

apply_base_style()

CLUSTER_BG = {
    "llama": CLUSTER_NEVER_ABSTAIN_BG, "gemma": CLUSTER_NEVER_ABSTAIN_BG,
    "qwen": CLUSTER_TOPIC_GATED_BG, "ministral": CLUSTER_TOPIC_GATED_BG,
    "deepseek": CLUSTER_FLAGGED_BG,
}
TOPIC_ORDER = ["climate change", "economic redistribution", "gender equality",
               "immigration", "lgbtq rights", "religion and secularism", "trust in government"]


def savefig(fig, name):
    path = f"{FIGURES_DIR}/{name}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def fig2_rating_distributions(df):
    a = df[(df["response_condition"] == "A_forced") & df["strict_is_valid"] & df["rating_numeric"].notnull()]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        sub = a[a["model"] == model]
        n = len(sub)
        counts = sub["rating_numeric"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        pct = 100 * counts / n if n else counts * 0
        color = MODEL_COLOR[model]
        hatch = DEEPSEEK_HATCH if model == "deepseek" else None
        ax.set_facecolor(CLUSTER_BG[model])
        bars = ax.bar([1, 2, 3, 4, 5], pct.values, color=color, hatch=hatch, edgecolor="white", linewidth=0.5, width=0.7)
        for x, v in zip([1, 2, 3, 4, 5], pct.values):
            if v > 0:
                ax.text(x, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
        title = f"{model}\n(n={n:,})" if model != "deepseek" else f"{model}\n(n={n:,}, EXPLORATORY)"
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("rating (1-5)", fontsize=8)
        ax.set_ylim(0, max(65, pct.max() * 1.25 if n else 65))
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(left=False)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of strict-valid responses", fontsize=9)
    fig.suptitle("Fig 2. Rating distributions by model (Condition A / forced, strict-valid only)",
                 fontsize=11, color=INK_PRIMARY, y=1.06)
    fig.text(0.5, -0.06,
              "DeepSeek panel uses only 63 rows (0.08% strict-valid rate) covering 2/7 topics -- hatched and flagged as exploratory, not comparable in precision to the other four.",
              ha="center", fontsize=8, color=INK_MUTED, wrap=True)
    savefig(fig, "fig2_rating_distributions.png")


def fig3_abstention_by_topic(df):
    b = df[df["response_condition"] == "B_optional"].copy()
    # Fix 5: answered = strict_is_valid & rating_numeric.notnull() (a real valid rating), NOT
    # ~is_abstention (which counted deepseek's malformed/refusal text as "answered"). This flips
    # deepseek from a misleading 0% abstention to the accurate ~100% (it has zero valid Condition-B
    # ratings at all -- see step 3/9/6). llama/gemma/qwen/ministral are unaffected (verified
    # identical under both definitions in step 6).
    b["answered"] = b["strict_is_valid"] & b["rating_numeric"].notnull()
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), sharey=False,
                              gridspec_kw={"width_ratios": [1.9, 1, 1, 1, 1]})
    for ax, model in zip(axes, MODEL_ORDER):
        sub = b[b["model"] == model]
        rates = sub.groupby("topic", observed=True)["answered"].mean().reindex(TOPIC_ORDER) * 100
        abstain_rates = 100 - rates
        color = MODEL_COLOR[model]
        hatch = DEEPSEEK_HATCH if model == "deepseek" else None
        ax.set_facecolor(CLUSTER_BG[model])
        bars = ax.barh(range(len(TOPIC_ORDER)), abstain_rates.values, color=color, hatch=hatch, edgecolor="white", linewidth=0.5)
        for y, v in enumerate(abstain_rates.values):
            ax.text(v + 2 if v < 90 else v - 2, y, f"{v:.0f}%", va="center",
                    ha="left" if v < 90 else "right", fontsize=7,
                    color=INK_SECONDARY if v < 90 else "white")
        ax.set_ylim(-0.6, len(TOPIC_ORDER) - 0.4)
        ax.set_yticks(range(len(TOPIC_ORDER)))
        if model == "llama":
            ax.set_yticklabels(TOPIC_ORDER, fontsize=9)
        else:
            ax.set_yticklabels([])
        overall = 100 * (1 - sub["answered"].mean())
        title = f"{model}\n(overall: {overall:.1f}%)" if model != "deepseek" else f"{model}\n(overall: {overall:.1f}%, EXPLORATORY)"
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% not answered", fontsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Fig 3. Non-response rate by topic, Condition B (optional)", fontsize=11, color=INK_PRIMARY, y=1.08)
    fig.subplots_adjust(bottom=0.32)
    fig.text(0.5, -0.20,
              "llama, gemma: 0% non-response (genuinely answer every time). qwen, ministral: topic-gated abstention --\n"
              "near-total non-response on immigration/trust in government, substantial engagement on gender equality.\n"
              "deepseek: ~100% non-response under the corrected definition (answered = a real valid numeric rating) --\n"
              "it produced zero strictly-valid Condition-B ratings; its bars are not abstention in llama/gemma's sense,\n"
              "they reflect near-total format non-compliance (see step 3/9).",
              ha="center", fontsize=8, color=INK_MUTED)
    savefig(fig, "fig3_abstention_by_topic.png")


def fig4_country_topic_heatmap(df):
    a = df[(df["response_condition"] == "A_forced") & df["strict_is_valid"] & df["rating_numeric"].notnull()]
    countries = sorted(a["country"].unique())
    fig, axes = plt.subplots(1, 5, figsize=(24, 7.5), sharey=False,
                              gridspec_kw={"width_ratios": [1.6, 1, 1, 1, 1]})
    pivots = {}
    for model in MODEL_ORDER:
        sub = a[a["model"] == model]
        piv = sub.pivot_table(index="country", columns="topic", values="rating_numeric", aggfunc="mean")
        piv = piv.reindex(index=countries, columns=TOPIC_ORDER)
        pivots[model] = piv

    # Fixed 1-5 scale (the true rating scale) used identically for every panel, including
    # deepseek -- so deepseek's low values (many cells = 1.0, see fig 2) render correctly
    # instead of being clipped by a tighter scale fit only to the other four models.
    vmin, vmax = 1.0, 5.0
    cmap = plt.get_cmap("Blues")

    for ax, model in zip(axes, MODEL_ORDER):
        piv = pivots[model]
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(TOPIC_ORDER)))
        ax.set_xticklabels([t[:10] for t in TOPIC_ORDER], rotation=90, fontsize=8)
        ax.set_yticks(range(len(countries)))
        if model == "llama":
            ax.set_yticklabels(countries, fontsize=9)
        else:
            ax.set_yticklabels([])
        n_valid_cells = int(piv.notnull().sum().sum())
        title = f"{model}" if model != "deepseek" else f"{model}\n(EXPLORATORY, {n_valid_cells}/140 cells)"
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if model == "deepseek":
            # so few populated cells that direct numeric labels are clearer than color alone
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if not np.isnan(v):
                        txt_color = "white" if v > 3 else INK_PRIMARY
                        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=txt_color)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.6])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                 cax=cbar_ax, label="mean rating (1-5)")
    fig.subplots_adjust(bottom=0.22, wspace=0.08)
    fig.suptitle("Fig 4. Country x topic mean rating, Condition A (forced), strict-valid only",
                 fontsize=12, color=INK_PRIMARY, y=0.98)
    fig.text(0.5, 0.02,
              "Same 1-5 color scale on every panel. llama/gemma/qwen/ministral are fully populated (140/140 cells each).\n"
              "deepseek's panel has valid data in only 18/140 cells (2/7 topics, 13/20 countries) -- cell values are printed directly\n"
              "since a 18-cell heatmap is not readable by color alone; exploratory only, not comparable to the other four.",
              ha="center", fontsize=8.5, color=INK_MUTED)
    savefig(fig, "fig4_country_topic_heatmap.png")


def fig5_pooled_coefficients():
    # Fix 6: this plot shows LLAMA's profession contrasts specifically, not an average pooled
    # across all models. The pooled model fits profession:model interactions with llama as the
    # reference level for C(model) (see step 5) -- so the "C(profession)[T.x]" MAIN EFFECT terms
    # are, by construction of treatment coding, llama's own profession contrasts. Every other
    # model's profession effect is llama's contrast PLUS that model's own profession:model
    # interaction term (not plotted here). Fix 4 also removed deepseek from this pooled model
    # entirely (its interaction terms were not identifiable given its n=63 sparse coverage).
    ct = pd.read_csv(f"{FIGURES_DIR}/../tables/hypothesis_model_pooled.csv")
    prof = ct[ct["term"].str.match(r"^C\(profession\)\[T\.[^]]+\]$")].copy()
    prof["profession"] = prof["term"].str.extract(r"\[T\.([^]]+)\]")
    prof = prof.sort_values("coef_hc3")

    fig, ax = plt.subplots(figsize=(7, 8))
    colors = [DIVERGING_POS if c >= 0 else DIVERGING_NEG for c in prof["coef_hc3"]]
    y = np.arange(len(prof))
    ax.errorbar(prof["coef_hc3"], y, xerr=1.96 * prof["se_cluster"], fmt="none",
                ecolor=BASELINE, elinewidth=1, capsize=2, zorder=1)
    ax.scatter(prof["coef_hc3"], y, color=colors, s=28, zorder=2, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(prof["profession"], fontsize=8)
    ax.set_xlabel("coefficient (rating scale, relative to 'accountant')", fontsize=9)
    ax.set_title("Fig 5. Profession contrasts relative to accountant, LLAMA (reference model)\n"
                 "(pooled model, Condition A, llama/gemma/qwen/ministral; 95% CI, persona-clustered SE)",
                 fontsize=11, color=INK_PRIMARY)
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DIVERGING_POS, markersize=7, label="coef > 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DIVERGING_NEG, markersize=7, label="coef < 0"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, frameon=False)
    fig.text(0.5, -0.02,
              "These are LLAMA's profession contrasts specifically (llama is the reference level for 'model' in this\n"
              "interaction model), NOT an average across models. Gemma/qwen/ministral's profession effects differ --\n"
              "visible in their own profession:model interaction terms, not shown in this single plot -- see\n"
              "tables/hypothesis_model_pooled.csv for the full table. DeepSeek is excluded from this pooled model entirely\n"
              "(Fix 4: n=63 makes its interactions non-identifiable, not just unstable) -- see its own regression in step 9.",
              ha="center", fontsize=8, color=INK_MUTED)
    savefig(fig, "fig5_pooled_coefficients.png")


def fig6_agreement_matrix():
    spearman = pd.read_csv(f"{FIGURES_DIR}/../tables/cross_model_spearman_matrix.csv", index_col=0)
    kappa = pd.read_csv(f"{FIGURES_DIR}/../tables/cross_model_kappa_matrix.csv", index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, label in zip(axes, [spearman, kappa], ["Spearman correlation", "Weighted (quadratic) Cohen's kappa"]):
        vals = mat.values.astype(float)
        im = ax.imshow(vals, cmap="Blues", vmin=0.3, vmax=1.0)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=9)
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                v = vals[i, j]
                txt_color = "white" if v > 0.72 else INK_PRIMARY
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=txt_color)
        ax.set_title(label, fontsize=10, color=INK_PRIMARY)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Fig 6. Cross-model agreement matrix (llama, gemma, qwen, ministral only)\nCondition A, matched persona-topic cells, n=37,800 per pair",
                 fontsize=11, color=INK_PRIMARY, y=1.1)
    fig.subplots_adjust(bottom=0.28)
    fig.text(0.5, -0.12,
              "deepseek excluded: only n<=63 matched cells per pair (its own strict-valid Condition-A count) -- too sparse for a\n"
              "meaningful correlation. Its pairwise numbers are reported separately in tables/cross_model_agreement_deepseek_pairs.csv\n"
              "(notably negative with every model, e.g. rho=-0.67 vs llama, on this tiny 2-topic slice).",
              ha="center", fontsize=8, color=INK_MUTED)
    savefig(fig, "fig6_agreement_matrix.png")


def fig7_variance_explained():
    # Fix 3: variance_ranking.csv now contains both the primary (Condition A only, the actual
    # pre-specified H1 test) and exploratory (A+B pooled, the original scope) rows, distinguished
    # by a "scope" column. This figure uses the primary scope only.
    vr_all = pd.read_csv(f"{FIGURES_DIR}/../tables/variance_ranking.csv")
    vr = vr_all[vr_all["scope"] == "primary_conditionA"]
    factors_order = ["topic", "profession", "country", "gender", "age"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharex=True)
    for ax, model in zip(axes, MODEL_ORDER):
        sub = vr[vr["model"] == model].set_index("factor").reindex(factors_order)
        vals = sub["partial_r2"]
        colors = [FACTOR_COLOR[f] for f in factors_order]
        hatch = DEEPSEEK_HATCH if model == "deepseek" else None
        ax.set_facecolor(CLUSTER_BG.get(model, "#fff") if model != "deepseek" else CLUSTER_FLAGGED_BG)
        y = np.arange(len(factors_order))
        plot_vals = vals.fillna(0).values
        ax.barh(y, plot_vals, color=colors, hatch=hatch, edgecolor="white", linewidth=0.5)
        for yi, (f, v) in enumerate(zip(factors_order, vals.values)):
            if np.isnan(v):
                ax.text(0.01, yi, "n/a (no variance)", va="center", ha="left", fontsize=7, color=INK_MUTED, style="italic")
            else:
                ax.text(v + 0.01, yi, f"{v:.3f}", va="center", ha="left", fontsize=7, color=INK_SECONDARY)
        ax.set_yticks(y)
        ax.set_yticklabels(factors_order if model == "llama" else [], fontsize=8)
        title = f"{model}" if model != "deepseek" else f"{model}\n(EXPLORATORY, n=63)"
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlim(0, max(0.85, vr["partial_r2"].max() * 1.05))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("partial R²", fontsize=8)
    fig.suptitle("Fig 7. Variance explained (partial R²) by factor, per model\n(Condition A / forced only -- the pre-specified H1 scope, per Fix 3)",
                 fontsize=11, color=INK_PRIMARY, y=1.08)
    fig.text(0.5, -0.08,
              "H1 (profession > country/age/gender) holds for qwen/ministral but not llama/gemma, where gender explains more\n"
              "variance than profession -- same verdict as the original A+B-pooled version (see tables/variance_ranking.csv,\n"
              "'exploratory_pooled_AB' rows, for comparison). deepseek's ranking (profession, country dominate) is likely an\n"
              "artifact of only 15/30 professions and 13/20 countries being represented in its 63 valid rows -- not comparable.",
              ha="center", fontsize=8, color=INK_MUTED)
    savefig(fig, "fig7_variance_explained.png")


def main():
    df = load_master()
    print("Generating figures...")
    fig2_rating_distributions(df)
    fig3_abstention_by_topic(df)
    fig4_country_topic_heatmap(df)
    fig5_pooled_coefficients()
    fig6_agreement_matrix()
    fig7_variance_explained()
    print("Done.")


if __name__ == "__main__":
    main()
