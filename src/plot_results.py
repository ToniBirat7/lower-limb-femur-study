"""
Summary figures for the test set, drawn from the CSVs in results/.

    python src/plot_results.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import FIGURES_DIR, result_path

METHODS = [  # bottom to top
    ("classical", "Threshold baseline", "0.5"),
    ("totalseg", "TotalSegmentator", "C2"),
    ("monai", "MONAI U-Net", "C1"),
    ("nnunet", "nnU-Net", "C0"),
]
ARROW = {"arrowstyle": "->", "color": "0.35", "lw": 0.8}


def note(ax, text, xy, offset):
    ax.annotate(text, xy=xy, xytext=offset, textcoords="offset points", fontsize=8, arrowprops=ARROW)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", alpha=0.5)


def segmentation_figure(out_png):
    frames = {k: pd.read_csv(result_path("segmentation", f"{k}_test.csv")).set_index("subject_id") for k, _, _ in METHODS}
    rng = np.random.default_rng(0)
    jitter = {k: dict(zip(frames[k].index, rng.uniform(-0.15, 0.15, len(frames[k])))) for k, _, _ in METHODS}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, label in [(axes[0], "Dice", "Dice"), (axes[1], "HD95_mm", "HD95 (mm, log scale)")]:
        for i, (k, _, color) in enumerate(METHODS):
            vals = frames[k][col]
            ax.boxplot(vals, positions=[i], widths=0.5, orientation="horizontal", showfliers=False,
                       patch_artist=True, boxprops={"facecolor": color, "alpha": 0.2, "edgecolor": color},
                       medianprops={"color": "black"})
            ax.scatter(vals, i + np.array([jitter[k][s] for s in vals.index]), s=18, color=color, zorder=3)
        ax.set_xlabel(label)
        clean(ax)
    axes[1].set_xscale("log")
    axes[0].set_yticks(range(len(METHODS)), [name for _, name, _ in METHODS])

    def point(k, col, sid):
        i = [m[0] for m in METHODS].index(k)
        return frames[k].loc[sid, col], i + jitter[k][sid]

    note(axes[1], "Pat027: stray region\nnear the midline", point("nnunet", "HD95_mm", "Pat027"), (-95, -28))
    note(axes[0], "Pat084: misses\nthe distal femur", point("totalseg", "Dice", "Pat084"), (15, 22))
    note(axes[1], "Pat084", point("totalseg", "HD95_mm", "Pat084"), (-45, 22))

    fig.suptitle("Segmentation on the 20 test patients")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def agreement_figure(out_png):
    ref = pd.read_csv(result_path("planning", "reference_test.csv"))
    agree = pd.read_csv(result_path("planning", "agreement_test.csv")).set_index(["method", "measure"])
    merged = {m: ref.merge(pd.read_csv(result_path("planning", f"{m}_test.csv")), on="subject_id", suffixes=("_ref", "_pred"))
              for m in ["nnunet", "monai"]}
    panels = [
        ("head_diameter_mm", "Head diameter (mm)"),
        ("nsa_deg", "Neck-shaft angle (degrees)"),
        ("anteversion_deg", "Anteversion (degrees)"),
        ("ama_deg", "Mechanical-anatomical angle (degrees)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    for ax, (col, title) in zip(axes.flat, panels):
        ax.axhline(0, color="0.4", lw=0.8)
        row = agree.loc[("nnunet", col)]
        ax.axhline(row["bias"], color="C0", lw=1, ls="--")
        for limit in (row["loa_low"], row["loa_high"]):
            ax.axhline(limit, color="C0", lw=0.8, ls=":")
        for m, marker, color in [("monai", "^", "C1"), ("nnunet", "o", "C0")]:
            d = merged[m]
            ax.scatter(d[f"{col}_ref"], d[f"{col}_pred"] - d[f"{col}_ref"], marker=marker, s=26, color=color, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("reference")
        ax.set_ylabel("predicted minus reference")
        clean(ax)

    p58 = merged["nnunet"].set_index("subject_id").loc["Pat058"]
    note(axes[0, 1], "Pat058", (p58["nsa_deg_ref"], p58["nsa_deg_pred"] - p58["nsa_deg_ref"]), (-55, 10))
    note(axes[1, 0], "Pat058", (p58["anteversion_deg_ref"], p58["anteversion_deg_pred"] - p58["anteversion_deg_ref"]), (25, 5))

    handles = [
        Line2D([], [], marker="o", ls="", color="C0", label="nnU-Net"),
        Line2D([], [], marker="^", ls="", color="C1", label="MONAI U-Net"),
        Line2D([], [], ls="--", color="C0", label="nnU-Net mean difference"),
        Line2D([], [], ls=":", color="C0", label="nnU-Net 95% limits of agreement"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Planning measurements on predicted masks against the reference surfaces (20 test femurs)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    segmentation_figure(FIGURES_DIR / "segmentation_test.png")
    agreement_figure(FIGURES_DIR / "planning_agreement.png")


if __name__ == "__main__":
    main()
