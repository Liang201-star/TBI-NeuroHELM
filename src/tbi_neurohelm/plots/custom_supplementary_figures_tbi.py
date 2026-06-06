from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tbi_neurohelm.common import read_csv, save_figure, model_order


def plot_supplementary_fig1_jury_robustness(input_dir: Path, output_base: Path) -> None:
    """Create a MedHELM-style supplementary robustness figure.

    Reference Supplementary Fig. 1 reports stability of LLM-based filtering with mean scores
    and standard deviations. For TBI-NeuroHELM, the analogous stability question is whether
    model ranking/performance is stable across LLM-jury subsets and scoring dimensions.
    """
    tendency = read_csv(input_dir, "19_jury_model_scoring_tendency_summary.csv")
    robust = read_csv(input_dir, "20b_jury_robustness_summary.csv").copy()
    combo = read_csv(input_dir, "20_jury_combination_robustness.csv").copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel a: scoring tendency by judge and dimension.
    dims = ["mean_accuracy", "mean_completeness", "mean_clarity", "mean_safety", "mean_overall"]
    x = np.arange(len(dims))
    width = 0.22
    for k, (_, row) in enumerate(tendency.iterrows()):
        axes[0].bar(x + (k - 1) * width, [row[d] for d in dims], width=width, label=row["judge_model"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Accuracy", "Completeness", "Clarity", "Safety", "Overall"], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylim(1, 5.2)
    axes[0].set_ylabel("Mean jury score")
    axes[0].set_title("a", loc="left", fontweight="bold", fontsize=11)
    axes[0].grid(axis="y", color="#b0b0b0", alpha=0.4)
    axes[0].legend(fontsize=7, frameon=True)

    # Panel b: robustness correlation to the 3-jury reference.
    robust["subset_label"] = robust["jury_subset"].astype(str).str.replace("+", "+\n", regex=False)
    robust = robust.sort_values(["subset_size", "spearman_rank_corr_with_reference", "pearson_macro_corr_with_reference"], ascending=[True, False, False])
    x = np.arange(len(robust))
    axes[1].plot(x, robust["spearman_rank_corr_with_reference"], marker="o", label="Spearman rank")
    axes[1].plot(x, robust["pearson_macro_corr_with_reference"], marker="s", label="Pearson macro")
    axes[1].axhline(0.9, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(robust["subset_label"], rotation=45, ha="right", fontsize=7)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Correlation with 3-jury reference")
    axes[1].set_title("b", loc="left", fontweight="bold", fontsize=11)
    axes[1].grid(axis="y", color="#b0b0b0", alpha=0.4)
    axes[1].legend(fontsize=7, frameon=True)

    # Panel c: rank stability range across jury subsets.
    order = model_order(input_dir)
    rank_stats = combo.groupby("model_name")["rank"].agg(["min", "max", "mean"]).reindex(order)
    y = np.arange(len(rank_stats))
    xerr = np.vstack([rank_stats["mean"] - rank_stats["min"], rank_stats["max"] - rank_stats["mean"]])
    axes[2].errorbar(rank_stats["mean"], y, xerr=xerr, fmt="o", capsize=3)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(rank_stats.index, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].invert_xaxis()
    axes[2].set_xlabel("Rank across jury subsets")
    axes[2].set_title("c", loc="left", fontweight="bold", fontsize=11)
    axes[2].grid(axis="x", color="#b0b0b0", alpha=0.4)

    plt.tight_layout(w_pad=2.0)
    save_figure(output_base)

    # Source data, one file per panel.
    tendency.to_csv(output_base.with_name(output_base.name + "_panel_a_source.csv"), index=False)
    robust.to_csv(output_base.with_name(output_base.name + "_panel_b_source.csv"), index=False)
    rank_stats.reset_index().to_csv(output_base.with_name(output_base.name + "_panel_c_source.csv"), index=False)
