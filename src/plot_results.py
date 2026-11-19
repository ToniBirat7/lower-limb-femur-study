"""
Generate summary figures from the saved test-set CSV files.

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

from src.config import FIGURES_DIR, RESULTS_DIR


def plot_segmentation_test(out_png):
    """Box and strip plots of Dice and HD95 across the four segmentation methods."""
    methods = [
        ("classical", "Threshold baseline", "0.55"),
        ("totalseg", "TotalSegmentator", "C2"),
        ("monai", "MONAI 3D U-Net", "C1"),
        ("nnunet", "nnU-Net v2", "C0"),
    ]
    frames = {key: pd.read_csv(RESULTS_DIR / f"segmentation_{key}_test.csv") for key, _, _ in methods}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    rng = np.random.default_rng(7)
    y_pos = np.arange(len(methods))
    point_coords = {}

    for ax, col, xlabel, log_scale in [
        (axes[0], "Dice", "Dice similarity coefficient", False),
        (axes[1], "HD95_mm", "95th percentile Hausdorff distance (mm)", True),
    ]:
        data = [frames[k][col].values for k, _, _ in methods]
        bp = ax.boxplot(
            data,
            positions=y_pos,
            orientation="horizontal",
            widths=0.45,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "0.4", "linewidth": 1.0},
            capprops={"color": "0.4", "linewidth": 1.0},
            boxprops={"linewidth": 1.0},
        )
        for patch, (_, _, color) in zip(bp["boxes"], methods):
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
            patch.set_edgecolor(color)

        for idx, (key, _, color) in enumerate(methods):
            df_m = frames[key]
            vals = df_m[col].values
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ys = np.full_like(vals, idx, dtype=float) + jitter
            point_coords[(col, key)] = dict(zip(df_m["subject_id"], zip(vals, ys)))
            ax.scatter(vals, ys, s=22, color=color, alpha=0.85, zorder=3)

        if log_scale:
            ax.set_xscale("log")
            ax.set_xlim(0.4, 300.0)
        else:
            ax.set_xlim(0.50, 1.02)

        ax.set_xlabel(xlabel)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([label for _, label, _ in methods])

    # Annotate key failure cases observed on the 20 test scans
    x_27, y_27 = point_coords[("HD95_mm", "nnunet")]["Pat027"]
    axes[1].annotate(
        f"Pat027 ({x_27:.1f} mm\nstray 22 ml region)",
        xy=(x_27, y_27),
        xytext=(38.0, 2.38),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9},
    )

    x_84d, y_84d = point_coords[("Dice", "totalseg")]["Pat084"]
    axes[0].annotate(
        f"Pat084 ({x_84d:.2f})",
        xy=(x_84d, y_84d),
        xytext=(0.54, 1.38),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9},
    )
    x_84h, y_84h = point_coords[("HD95_mm", "totalseg")]["Pat084"]
    axes[1].annotate(
        f"Pat084 ({x_84h:.1f} mm\nshort distally)",
        xy=(x_84h, y_84h),
        xytext=(38.0, 1.45),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9},
    )

    fig.suptitle("Held-out test set segmentation accuracy (n = 20 patients)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_planning_agreement(out_png):
    """Difference plots (predicted minus reference vs reference) for nnU-Net and MONAI."""
    ref = pd.read_csv(RESULTS_DIR / "planning_reference_test.csv")
    nn = pd.read_csv(RESULTS_DIR / "planning_nnunet_test.csv")
    mo = pd.read_csv(RESULTS_DIR / "planning_monai_test.csv")
    ag = pd.read_csv(RESULTS_DIR / "planning_agreement_test.csv").set_index(["method", "measure"])

    m_nn = ref.merge(nn, on="subject_id", suffixes=("_ref", "_pred"))
    m_mo = ref.merge(mo, on="subject_id", suffixes=("_ref", "_pred"))

    panels = [
        ("head_diameter_mm", "Femoral head diameter (mm)", "Reference head diameter (mm)", "mm", (0.03, 0.04, "bottom")),
        ("nsa_deg", "Neck-shaft angle (deg)", "Reference neck-shaft angle (deg)", "deg", (0.03, 0.04, "bottom")),
        ("anteversion_deg", "Femoral anteversion (deg)", "Reference anteversion (deg)", "deg", (0.97, 0.04, "bottom")),
        ("ama_deg", "Mechanical-anatomical angle (deg)", "Reference mechanical-anatomical angle (deg)", "deg", (0.03, 0.96, "top")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0))
    for ax, (col, title, xlabel, unit, (tx, ty, va)) in zip(axes.flat, panels):
        x_nn = m_nn[f"{col}_ref"]
        y_nn = m_nn[f"{col}_pred"] - x_nn
        x_mo = m_mo[f"{col}_ref"]
        y_mo = m_mo[f"{col}_pred"] - x_mo

        row_nn = ag.loc[("nnunet", col)]
        row_mo = ag.loc[("monai", col)]

        ax.axhline(0.0, color="0.4", lw=0.9)
        ax.axhline(row_nn["bias"], color="C0", lw=1.0, ls="--", alpha=0.8)
        ax.axhline(row_nn["loa_low"], color="C0", lw=0.8, ls=":", alpha=0.75)
        ax.axhline(row_nn["loa_high"], color="C0", lw=0.8, ls=":", alpha=0.75)

        h_mo = ax.scatter(x_mo, y_mo, marker="^", s=34, color="C1", alpha=0.8, label="MONAI 3D U-Net")
        h_nn = ax.scatter(x_nn, y_nn, marker="o", s=34, color="C0", alpha=0.85, label="nnU-Net v2")

        summary_txt = (
            f"nnU-Net MAD: {row_nn['mean_abs_diff']:.2f} {unit} (bias {row_nn['bias']:+.2f})\n"
            f"MONAI MAD: {row_mo['mean_abs_diff']:.2f} {unit} (bias {row_mo['bias']:+.2f})"
        )
        ha = "right" if tx > 0.5 else "left"
        ax.text(
            tx,
            ty,
            summary_txt,
            transform=ax.transAxes,
            fontsize=8,
            va=va,
            ha=ha,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.90, "edgecolor": "0.8"},
        )

        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Predicted - reference ({unit})")
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Annotate notable outliers discussed in the README
    p58_nn = m_nn.set_index("subject_id").loc["Pat058"]
    axes[0, 1].annotate(
        f"Pat058 ({p58_nn['nsa_deg_pred'] - p58_nn['nsa_deg_ref']:+.1f} deg)",
        xy=(p58_nn["nsa_deg_ref"], p58_nn["nsa_deg_pred"] - p58_nn["nsa_deg_ref"]),
        xytext=(p58_nn["nsa_deg_ref"] - 14.5, -10.5),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9},
    )
    axes[1, 0].annotate(
        f"Pat058 ({p58_nn['anteversion_deg_pred'] - p58_nn['anteversion_deg_ref']:+.1f} deg)",
        xy=(p58_nn["anteversion_deg_ref"], p58_nn["anteversion_deg_pred"] - p58_nn["anteversion_deg_ref"]),
        xytext=(p58_nn["anteversion_deg_ref"] + 4.5, -30.5),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9},
    )
    p28_mo = m_mo.set_index("subject_id").loc["Pat028"]
    axes[0, 0].annotate(
        f"Pat028 MONAI ({p28_mo['head_diameter_mm_pred'] - p28_mo['head_diameter_mm_ref']:+.1f} mm)",
        xy=(p28_mo["head_diameter_mm_ref"], p28_mo["head_diameter_mm_pred"] - p28_mo["head_diameter_mm_ref"]),
        xytext=(48.0, -3.5),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9},
    )

    fig.legend(
        [h_nn, h_mo],
        ["nnU-Net v2 (dashed/dotted: mean bias and 95% LoA)", "MONAI 3D U-Net"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Planning measurements on predicted masks vs reference surfaces (n = 20 test femurs)", fontsize=11.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_segmentation_test(FIGURES_DIR / "segmentation_test.png")
    plot_planning_agreement(FIGURES_DIR / "planning_agreement.png")
    print("Saved figures/segmentation_test.png and figures/planning_agreement.png")


if __name__ == "__main__":
    main()
