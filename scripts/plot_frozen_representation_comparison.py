"""Render the frozen seven-leaf representation comparison figure from its source CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = (
    ("macro_auc", "ROC-AUC"),
    ("macro_average_precision", "PR-AUC / Average Precision"),
    ("macro_f1", "F1 (threshold = 0.5)"),
    ("macro_balanced_accuracy", "Balanced Accuracy"),
)


def render(package_dir: Path) -> tuple[Path, Path]:
    source = package_dir / "final_macro_metrics.csv"
    data = pd.read_csv(source)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=True)
    colors = {"Top-n SAE": "#0072B2", "PCA-n": "#D55E00", "Random SAE-n": "#999999"}
    markers = {"Top-n SAE": "o", "PCA-n": "s", "Random SAE-n": "^"}
    baselines = {
        "Hidden State": ("#009E73", "-"),
        "Full SAE": ("#CC79A7", "--"),
        "Stable Core SAE": ("#000000", ":"),
    }
    for ax, (column, title) in zip(axes.ravel(), METRICS):
        for representation in ("Top-n SAE", "PCA-n", "Random SAE-n"):
            group = data[data["representation"].eq(representation)].sort_values("top_n")
            ax.plot(
                group["top_n"], group[column], label=representation,
                color=colors[representation], marker=markers[representation], linewidth=2.0,
            )
            if representation == "Random SAE-n":
                std_column = f"{column}_seed_std"
                if std_column in group.columns:
                    y = pd.to_numeric(group[column], errors="coerce")
                    spread = pd.to_numeric(group[std_column], errors="coerce").fillna(0.0)
                    ax.fill_between(group["top_n"], y - spread, y + spread, color=colors[representation], alpha=0.16)
        for representation, (color, linestyle) in baselines.items():
            row = data[data["representation"].eq(representation)]
            if not row.empty:
                label = representation + (" (exploratory)" if representation == "Stable Core SAE" else "")
                ax.axhline(float(row[column].iloc[0]), color=color, linestyle=linestyle, linewidth=1.7, label=label)
        ax.set_title(title)
        ax.set_ylabel("Seven-leaf macro score")
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Number of retained dimensions (n)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=3, frameon=False)
    fig.suptitle("Frozen representation comparison (7 MISC leaf labels)", fontsize=14)
    fig.text(0.5, 0.086, "Stable Core SAE is exploratory because selection was not nested in the outer CV.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    figure_dir = package_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "representation_comparison_leaf7.png"
    pdf = figure_dir / "representation_comparison_leaf7.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    png, pdf = render(args.package_dir)
    print(f"[done] {png}")
    print(f"[done] {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
