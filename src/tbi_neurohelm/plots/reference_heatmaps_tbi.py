from __future__ import annotations

from pathlib import Path
import pandas as pd

from tbi_neurohelm.common import (
    CATEGORY_ORDER,
    CATEGORY_SHORT,
    annotated_heatmap,
    model_order,
    read_csv,
)


def benchmark_matrix(input_dir: Path) -> pd.DataFrame:
    scores = read_csv(input_dir, "06_benchmark_scores_long.csv")
    manifest = read_csv(input_dir, "00_benchmark_manifest.csv")
    order = manifest["benchmark_id"].astype(str).tolist()
    mat = scores.pivot_table(index="model_name", columns="benchmark_id", values="benchmark_score_norm", aggfunc="mean")
    return mat.reindex(index=model_order(input_dir), columns=order)


def category_matrix(input_dir: Path) -> pd.DataFrame:
    df = read_csv(input_dir, "13_category_scores.csv")
    value_col = "category_score_item_weighted" if "category_score_item_weighted" in df.columns else "category_score"
    mat = df.pivot_table(index="model_name", columns="category", values=value_col, aggfunc="mean")
    cols = [c for c in CATEGORY_ORDER if c in mat.columns]
    mat = mat.reindex(index=model_order(input_dir), columns=cols)
    mat = mat.rename(columns={c: CATEGORY_SHORT.get(c, c) for c in mat.columns})
    return mat


def subcategory_matrix(input_dir: Path) -> pd.DataFrame:
    df = read_csv(input_dir, "14_subcategory_scores.csv")
    manifest_order = read_csv(input_dir, "24_instance_distribution_summary.csv")
    order = (
        manifest_order[["category", "subcategory"]]
        .drop_duplicates()
        .sort_values(["category", "subcategory"])["subcategory"]
        .astype(str)
        .tolist()
    )
    mat = df.pivot_table(index="model_name", columns="subcategory", values="subcategory_score", aggfunc="mean")
    cols = [c for c in order if c in mat.columns]
    return mat.reindex(index=model_order(input_dir), columns=cols)


def make_reference_heatmaps(input_dir: Path, figures_dir: Path) -> None:
    annotated_heatmap(
        benchmark_matrix(input_dir),
        figures_dir / "fig3_benchmark_heatmap",
        "",
        figsize=(13, 5.8),
        x_label_width=12,
    )
    annotated_heatmap(
        category_matrix(input_dir),
        figures_dir / "fig4_category_heatmap",
        "",
        figsize=(8.5, 5.5),
        x_label_width=13,
    )
    annotated_heatmap(
        subcategory_matrix(input_dir),
        figures_dir / "extended_data_fig1_subcategory_heatmap",
        "",
        figsize=(16, 6),
        x_label_width=18,
    )
