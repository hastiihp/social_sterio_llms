"""Generates every figure used in FULL_PROJECT_REPORT.md.

Every figure's exact source file/column is cited in its own function's
docstring/comment. Nothing here is estimated -- each function reads
directly from the project's already-audited output CSVs. Run from the
project root or from analysis_report/ (paths are resolved relative to
this file).

Palette: fixed categorical hue order taken from the dataviz skill's
reference palette (references/palette.md) -- blue/orange/aqua/yellow/
magenta, assigned by entity identity and held constant across every
figure (llama=blue, gemma=orange, qwen=aqua, ministral=yellow,
deepseek=magenta; the same order is reused for the 4 H1 candidate
factors -- gender=blue, profession=orange, country=aqua, age=yellow --
in the figures that plot factors instead of models). Sequential magnitude
charts (single series) use one hue (blue, step 450). The cross-model
correlation heatmap uses the diverging blue<->red pair since correlation
is a polarity measure, centered at its neutral point.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{ROOT}/figures/report"

# ---- palette (references/palette.md, categorical light-mode steps) ----
MODEL_COLORS = {
    "llama": "#2a78d6",
    "gemma": "#eb6834",
    "qwen": "#1baf7a",
    "ministral": "#eda100",
    "deepseek": "#e87ba4",
}
FACTOR_COLORS = {
    "gender": "#2a78d6",
    "profession": "#eb6834",
    "country": "#1baf7a",
    "age": "#eda100",
}
MODEL_ORDER = ["llama", "gemma", "qwen", "ministral"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titleweight": "bold",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    })


def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{name}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------
# FIG 1: Condition-B abstention rate by model, original prompt.
# Source: tables/paired_comparison_summary.csv (n_b_abstained / n_matched)
# ---------------------------------------------------------------------
def fig1_abstention_by_model():
    df = pd.read_csv(f"{ROOT}/tables/paired_comparison_summary.csv")
    df = df[df["model"].isin(MODEL_ORDER)].set_index("model").loc[MODEL_ORDER]
    rate = 100 * df["n_b_abstained"] / df["n_matched"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    bars = ax.bar(MODEL_ORDER, rate.values, color=colors, width=0.6)
    for b, v in zip(bars, rate.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, color=INK_PRIMARY)
    ax.set_ylabel("Condition-B abstention rate (%)")
    ax.set_title("The two-family behavioral split\n(Condition-B abstention rate, original prompt)")
    ax.set_ylim(0, 100)
    savefig(fig, "fig1_abstention_by_model.png")


# ---------------------------------------------------------------------
# FIG 2: H1 partial R^2 by factor, grouped by model.
# Source: tables/variance_ranking.csv, scope=="primary_conditionA"
# ---------------------------------------------------------------------
def fig2_h1_partial_r2_by_model():
    df = pd.read_csv(f"{ROOT}/tables/variance_ranking.csv")
    df = df[(df["scope"] == "primary_conditionA") & (df["model"].isin(MODEL_ORDER)) &
            (df["factor"].isin(FACTOR_COLORS.keys()))]

    factors = ["gender", "profession", "country", "age"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, factor in enumerate(factors):
        vals = [df[(df.model == m) & (df.factor == factor)]["partial_r2"].iloc[0] for m in MODEL_ORDER]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=factor, color=FACTOR_COLORS[factor])
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Partial R²")
    ax.set_title("H1: which demographic factor dominates, per model?\n(original prompt, Condition A)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    savefig(fig, "fig2_h1_partial_r2_by_model.png")


# ---------------------------------------------------------------------
# FIG 3: covert-midpoint pattern -- % of forced-Condition-A ratings equal to
# the scale midpoint (3), split by whether Condition B would have abstained.
# Source: tables/paired_comparison_summary.csv
#   (pct_A_eq3_given_B_abstained, pct_A_eq3_given_B_answered), qwen/ministral only
# ---------------------------------------------------------------------
def fig3_covert_midpoint():
    df = pd.read_csv(f"{ROOT}/tables/paired_comparison_summary.csv")
    models = ["qwen", "ministral"]
    df = df[df["model"].isin(models)].set_index("model").loc[models]

    x = np.arange(len(models))
    width = 0.32
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, 100 * df["pct_A_eq3_given_B_abstained"], width=width,
           label="B would have abstained", color="#e34948")
    ax.bar(x + width / 2, 100 * df["pct_A_eq3_given_B_answered"], width=width,
           label="B actually answered", color="#2a78d6")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("% of forced Condition-A ratings = 3 (midpoint)")
    ax.set_title("The covert-midpoint pattern\n(forced rating = scale midpoint, by what B would have done)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_ylim(0, 100)
    savefig(fig, "fig3_covert_midpoint.png")


# ---------------------------------------------------------------------
# FIG 4: cross-model rating agreement (Spearman rho), original prompt.
# Source: tables/cross_model_spearman_matrix.csv
# ---------------------------------------------------------------------
def fig4_cross_model_agreement_heatmap():
    df = pd.read_csv(f"{ROOT}/tables/cross_model_spearman_matrix.csv", index_col=0)
    df = df.loc[MODEL_ORDER, MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_yticklabels(MODEL_ORDER)
    for i in range(len(MODEL_ORDER)):
        for j in range(len(MODEL_ORDER)):
            v = df.values[i, j]
            txt_color = "white" if abs(v) > 0.6 else INK_PRIMARY
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_color, fontsize=10)
    ax.set_title("Cross-model rating agreement\n(Spearman ρ, original prompt, Condition A)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.grid(False)
    savefig(fig, "fig4_cross_model_agreement_heatmap.png")


# ---------------------------------------------------------------------
# FIG 5: health-study Condition-A rating shift, per model.
# Source: analysis_health/output/health_vs_original_by_condition.csv,
#   condition=="A_forced" row, clustered_mean_shift_health_minus_orig
#   (NOT health_vs_original_summary.csv -- that file pools Condition A+B,
#   confirmed while building this report; see FULL_PROJECT_REPORT.md's
#   methodology note)
# ---------------------------------------------------------------------
def fig5_health_rating_shift():
    df = pd.read_csv(f"{ROOT}/analysis_health/output/health_vs_original_by_condition.csv")
    df = df[(df["condition"] == "A_forced") & (df["model"].isin(MODEL_ORDER))].set_index("model").loc[MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    vals = df["clustered_mean_shift_health_minus_orig"]
    bars = ax.bar(MODEL_ORDER, vals.values, color=colors, width=0.6)
    for b, v in zip(bars, vals.values):
        ax.text(b.get_x() + b.get_width() / 2, -0.008, f"{v:.3f}", ha="center", va="top", fontsize=10, color=INK_PRIMARY)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_ylabel("Mean rating shift (health − original)")
    ax.set_title("Health-conversation framing shifts ratings downward\n(180-persona pilot, Condition A, all p<10⁻²¹)")
    ax.margins(y=0.15)
    savefig(fig, "fig5_health_rating_shift.png")


# ---------------------------------------------------------------------
# FIG 6: Ministral's Condition-B abstention rate across all 5 prompt types.
# Source: analysis_context/output/abstention_stability_rate_table.csv,
#   model=="ministral", abstention_rate_condB_pct
# ---------------------------------------------------------------------
def fig6_ministral_abstention_decline():
    df = pd.read_csv(f"{ROOT}/analysis_context/output/abstention_stability_rate_table.csv")
    order = ["original", "health", "neutral", "positive", "negative_minor"]
    df = df[df["model"] == "ministral"].set_index("condition").loc[order]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(order, df["abstention_rate_condB_pct"], color=MODEL_COLORS["ministral"], width=0.6)
    for b, v in zip(bars, df["abstention_rate_condB_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, color=INK_PRIMARY)
    ax.set_ylabel("Condition-B abstention rate (%)")
    ax.set_title("Ministral's abstention rate falls with every conversational framing\n(full scale, 5,400 personas)")
    ax.set_ylim(0, 100)
    savefig(fig, "fig6_ministral_abstention_decline.png")


# ---------------------------------------------------------------------
# FIG 7: cross-context structural clustering.
# Source: analysis_context/output/cross_context_clustering_summary_full5400.csv
#   (2-group means) and cross_context_pairwise_similarity_ranked_full5400.csv
#   (6 underlying pairwise values, right panel)
# ---------------------------------------------------------------------
def fig7_cross_context_clustering():
    summ = pd.read_csv(f"{ROOT}/analysis_context/output/cross_context_clustering_summary_full5400.csv")
    pairs = pd.read_csv(f"{ROOT}/analysis_context/output/cross_context_pairwise_similarity_ranked_full5400.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1, 1.8], "wspace": 0.55})

    ax = axes[0]
    labels = ["context vs.\noriginal", "context vs.\ncontext"]
    vals = [summ.loc[summ.comparison == "context_vs_original", "mean_spearman_r"].iloc[0],
            summ.loc[summ.comparison == "context_vs_context", "mean_spearman_r"].iloc[0]]
    bars = ax.bar(labels, vals, color=["#2a78d6", "#eb6834"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_ylim(0, 1)
    ax.set_title("Group means")

    ax = axes[1]
    pairs = pairs.sort_values("mean_spearman_r")
    pair_labels = pairs["condition_1"] + " / " + pairs["condition_2"]
    bars = ax.barh(pair_labels, pairs["mean_spearman_r"], color="#eb6834")
    ax.tick_params(axis="y", labelsize=9.5)
    for b, v in zip(bars, pairs["mean_spearman_r"]):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Mean Spearman ρ")
    ax.set_xlim(0, 1)
    ax.set_title("All 6 context-context pairs")

    fig.suptitle("The four conversational framings resemble each other more than the original prompt", y=1.03, fontweight="bold")
    savefig(fig, "fig7_cross_context_clustering.png")


# ---------------------------------------------------------------------
# FIG 8: H1 dominant factor, per model, across all 5 prompt types.
# Source: analysis_context/output/dominant_factor_by_model_full5400.csv
# ---------------------------------------------------------------------
def fig8_h1_dominant_factor_grid():
    df = pd.read_csv(f"{ROOT}/analysis_context/output/dominant_factor_by_model_full5400.csv")
    prompt_types = ["original", "health", "neutral", "positive", "negative_minor"]
    df = df.set_index("model").loc[MODEL_ORDER]
    grid = df[[f"dominant_{pt}" for pt in prompt_types]]

    factor_list = ["gender", "profession", "country", "age"]
    factor_idx = {f: i for i, f in enumerate(factor_list)}
    z = np.array([[factor_idx[grid.loc[m, f"dominant_{pt}"]] for pt in prompt_types] for m in MODEL_ORDER])

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([FACTOR_COLORS[f] for f in factor_list])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(z, cmap=cmap, vmin=-0.5, vmax=len(factor_list) - 0.5, aspect="auto")
    ax.set_xticks(range(len(prompt_types)))
    ax.set_xticklabels(prompt_types, rotation=20, ha="right")
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    for i, m in enumerate(MODEL_ORDER):
        for j, pt in enumerate(prompt_types):
            label = grid.loc[m, f"dominant_{pt}"]
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5,
                     color="white" if label in ("gender", "profession") else INK_PRIMARY)
    ax.set_title("Which factor dominates attributed opinion, per model x prompt type?\n(full scale, 5,400 personas)")
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FACTOR_COLORS[f]) for f in factor_list]
    ax.legend(handles, factor_list, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    savefig(fig, "fig8_h1_dominant_factor_grid.png")


# ---------------------------------------------------------------------
# FIG 9: Context Sensitivity Index, 4 components, small multiples.
# Source: analysis_taxonomy/output/context_sensitivity_index.csv
# ---------------------------------------------------------------------
def fig9_context_sensitivity_index():
    df = pd.read_csv(f"{ROOT}/analysis_taxonomy/output/context_sensitivity_index.csv")
    df = df[df["model"].isin(MODEL_ORDER)].set_index("model").loc[MODEL_ORDER]

    components = [
        ("abstention_range_pct", "Abstention range (pp)", None),
        ("avg_abs_rating_shift", "Avg. rating shift (magnitude)", None),
        ("avg_ranking_rho_country", "Avg ranking ρ (country)", (0, 1)),
        ("avg_ranking_rho_profession", "Avg ranking ρ (profession)", (0, 1)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (col, title, ylim) in zip(axes, components):
        colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
        ax.bar(MODEL_ORDER, df[col], color=colors, width=0.6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle("Context Sensitivity Index: four components, not collapsed into one score", y=1.06, fontweight="bold")
    savefig(fig, "fig9_context_sensitivity_index.png")


# ---------------------------------------------------------------------
# FIG 10: Consensus Index -- distribution of cross-model rating SD.
# Source: analysis_taxonomy/output/consensus_index.csv, sd_across_models
# ---------------------------------------------------------------------
def fig10_consensus_sd_histogram():
    df = pd.read_csv(f"{ROOT}/analysis_taxonomy/output/consensus_index.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["sd_across_models"], bins=40, color="#2a78d6", edgecolor=SURFACE, linewidth=0.5)
    pct_zero = 100 * (df["sd_across_models"] == 0).mean()
    ax.axvline(0, color="#e34948", linewidth=1.5, linestyle="--")
    ax.text(0.05, ax.get_ylim()[1] * 0.9, f"{pct_zero:.1f}% of all 37,800 cells\nare exact 4-way agreement (SD=0)",
            fontsize=9.5, color=INK_PRIMARY)
    ax.set_xlabel("Standard deviation across 4 models' ratings (1-5 scale)")
    ax.set_ylabel("Number of persona x topic cells")
    ax.set_title("Consensus Index: cross-model rating agreement across all 37,800 cells\n(original prompt, Condition A)")
    savefig(fig, "fig10_consensus_sd_histogram.png")


# ---------------------------------------------------------------------
# FIG 11: unified model -- partial R^2 per term.
# Source: analysis_unified/output/variance_decomposition_model.csv
# ---------------------------------------------------------------------
def fig11_unified_variance_decomposition():
    df = pd.read_csv(f"{ROOT}/analysis_unified/output/variance_decomposition_model.csv")
    df = df.sort_values("partial_r2_ols", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.barh(df["term"], df["partial_r2_ols"], color="#2a78d6")
    for b, v in zip(bars, df["partial_r2_ols"]):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9.5)
    ax.set_xlabel("Partial R² (OLS SSE-reduction)")
    ax.set_title("Unified mixed-effects model: term-specific partial R²\nAll 5 prompt types x 4 models pooled, Condition A, full scale")
    ax.set_xlim(0, df["partial_r2_ols"].max() * 1.2)
    savefig(fig, "fig11_unified_variance_decomposition.png")


def main():
    apply_style()
    print("Generating report figures...")
    fig1_abstention_by_model()
    fig2_h1_partial_r2_by_model()
    fig3_covert_midpoint()
    fig4_cross_model_agreement_heatmap()
    fig5_health_rating_shift()
    fig6_ministral_abstention_decline()
    fig7_cross_context_clustering()
    fig8_h1_dominant_factor_grid()
    fig9_context_sensitivity_index()
    fig10_consensus_sd_histogram()
    fig11_unified_variance_decomposition()
    print("Done.")


if __name__ == "__main__":
    main()
