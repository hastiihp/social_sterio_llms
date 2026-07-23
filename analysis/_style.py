"""Shared chart styling -- validated palette per the dataviz skill.

Model colors validated with scripts/validate_palette.js in MODEL_ORDER
sequence (llama, gemma, qwen, ministral, deepseek): all hard gates (CVD
adjacent separation, normal-vision floor, lightness band, chroma floor)
pass in both light and dark mode. Sub-3:1 contrast on 3 of 5 colors is
flagged WARN by the validator -- mitigated here by always pairing color
with direct labels/legends, never rendering text in these hues.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_COLOR = {
    "llama": "#2a78d6",
    "gemma": "#eb6834",
    "qwen": "#1baf7a",
    "ministral": "#eda100",
    "deepseek": "#e87ba4",
}

FACTOR_COLOR = {
    "topic": "#2a78d6",
    "profession": "#eb6834",
    "country": "#1baf7a",
    "gender": "#eda100",
    "age": "#4a3aa7",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
DIVERGING_POS = "#2a78d6"
DIVERGING_NEG = "#e34948"
NEUTRAL_GRAY = "#f0efec"

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

CLUSTER_NEVER_ABSTAIN_BG = "#eef4fc"   # faint blue tint: llama, gemma
CLUSTER_TOPIC_GATED_BG = "#fdf1e8"     # faint orange tint: qwen, ministral
CLUSTER_FLAGGED_BG = "#f5f0f3"         # faint neutral tint: deepseek

DEEPSEEK_HATCH = "///"


def apply_base_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.size": 10,
    })
