from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from tbi_neurohelm.common import read_csv, write_table_outputs


def extended_data_table1_benchmark_suite(input_dir: Path, out: Path) -> pd.DataFrame:
    """Benchmark-suite composition table, analogous to MedHELM extended benchmark inventory tables."""
    manifest = read_csv(input_dir, "00_benchmark_manifest.csv")
    df = manifest.copy()
    df = df.rename(columns={
        "benchmark_id": "Benchmark",
        "benchmark_display_name": "Benchmark name",
        "category": "Category",
        "subcategories": "Subcategories",
        "total_instances": "Total instances",
        "closed_instances": "Closed QA instances",
        "open_instances": "Open-task instances",
        "basic_instances": "Basic",
        "intermediate_instances": "Intermediate",
        "advanced_instances": "Advanced",
    })
    df["Primary scoring method"] = df.apply(
        lambda r: "Exact match; LLM-jury" if r.get("Open-task instances", 0) > 0 else "Exact match",
        axis=1,
    )
    cols = [
        "Benchmark", "Benchmark name", "Category", "Subcategories", "Total instances",
        "Closed QA instances", "Open-task instances", "Basic", "Intermediate", "Advanced",
        "Primary scoring method",
    ]
    df = df[cols]
    write_table_outputs(df, out)
    return df


def extended_data_table2_cost_performance(input_dir: Path, out: Path) -> pd.DataFrame:
    """Model-level token use, cost, and performance table."""
    costs = read_csv(input_dir, "08_costs.csv")
    lb = read_csv(input_dir, "07_leaderboard.csv")
    grouped = costs.groupby(["Model", "Provider"], as_index=False).agg(
        **{
            "Benchmark input tokens": ("benchmark_input_tokens", "sum"),
            "Benchmark output tokens": ("benchmark_output_tokens", "sum"),
            "Benchmark cost (USD)": ("benchmark_cost", "sum"),
            "Jury input tokens": ("jury_input_tokens", "sum"),
            "Jury output tokens": ("jury_output_tokens", "sum"),
            "Jury cost (USD)": ("jury_cost", "sum"),
            "Total cost (USD)": ("Cost Input Output Tokens", "sum"),
        }
    )
    merged = grouped.merge(lb[["Model", "Mean win rate", "Macro-average"]], on="Model", how="left")
    merged = merged.sort_values("Mean win rate", ascending=False)
    write_table_outputs(merged, out)
    return merged


def supplementary_table1_instance_distribution(input_dir: Path, out: Path) -> pd.DataFrame:
    """Detailed distribution of TBI-NeuroHELM instances."""
    df = read_csv(input_dir, "24_instance_distribution_summary.csv").copy()
    df = df.rename(columns={
        "category": "Category",
        "subcategory": "Subcategory",
        "benchmark_id": "Benchmark",
        "benchmark_display_name": "Benchmark name",
        "question_type": "Question type",
        "complexity": "Complexity",
        "n_items": "Number of instances",
    })
    df = df.sort_values(["Category", "Subcategory", "Benchmark", "Question type", "Complexity"])
    write_table_outputs(df, out)
    return df


def supplementary_table2_model_registry_and_run_config(input_dir: Path, out: Path) -> pd.DataFrame:
    """Model registry and evaluation configuration table."""
    registry = read_csv(input_dir, "10_model_registry.csv").copy()
    manifest_path = input_dir / "09_run_manifest.csv"
    config_path = input_dir / "00_evaluation_config.json"
    closed = open_t = jury = temperature = workers = None
    if manifest_path.exists():
        m = pd.read_csv(manifest_path).iloc[0]
        closed = m.get("closed_max_tokens")
        open_t = m.get("open_max_tokens")
        jury = m.get("jury_max_tokens")
        temperature = m.get("temperature")
        workers = m.get("workers")
    elif config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        closed = cfg.get("closed_max_tokens")
        open_t = cfg.get("open_max_tokens")
        jury = cfg.get("jury_max_tokens")
        temperature = cfg.get("temperature")
        workers = cfg.get("workers")

    keep = [
        "model_name", "provider", "api_model_id_requested", "observed_api_response_models_test",
        "used_as_test_model", "used_as_judge_model", "input_price_per_1m", "output_price_per_1m",
    ]
    keep = [c for c in keep if c in registry.columns]
    df = registry[keep].copy().rename(columns={
        "model_name": "Model",
        "provider": "Provider",
        "api_model_id_requested": "Requested API model ID",
        "observed_api_response_models_test": "Observed response model ID",
        "used_as_test_model": "Used as test model",
        "used_as_judge_model": "Used as jury model",
        "input_price_per_1m": "Input price per 1M tokens",
        "output_price_per_1m": "Output price per 1M tokens",
    })
    df["Temperature"] = temperature
    df["Closed max tokens"] = closed
    df["Open max tokens"] = open_t
    df["Jury max tokens"] = jury
    df["Workers"] = workers
    write_table_outputs(df, out)
    return df


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    mask = values.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def supplementary_table3_open_task_dimensions(input_dir: Path, out: Path) -> pd.DataFrame:
    """Open-task LLM-jury scoring dimensions by model, weighted by the number of open-task items."""
    raw = read_csv(input_dir, "18_open_score_dimensions.csv").copy()
    rows: list[dict[str, object]] = []
    for (model, provider), g in raw.groupby(["model_name", "provider"], dropna=False):
        w = g["n_open_items"] if "n_open_items" in g.columns else pd.Series([1] * len(g), index=g.index)
        rows.append({
            "Model": model,
            "Provider": provider,
            "Number of open-task responses": int(pd.to_numeric(w, errors="coerce").fillna(0).sum()),
            "Accuracy": _weighted_mean(g["mean_accuracy_mean"], w),
            "Completeness": _weighted_mean(g["mean_completeness_mean"], w),
            "Clarity": _weighted_mean(g["mean_clarity_mean"], w),
            "Safety": _weighted_mean(g["mean_safety_mean"], w),
            "Overall": _weighted_mean(g["mean_overall_raw_mean"], w),
            "Normalized open-task score": _weighted_mean(g["score_norm_mean"], w),
            "Mean jury SD": _weighted_mean(g["jury_sd_mean"], w),
            "Jury parse success rate": _weighted_mean(g["jury_parse_success_rate_mean"], w),
        })
    df = pd.DataFrame(rows).sort_values("Normalized open-task score", ascending=False)
    write_table_outputs(df, out)
    return df


def supplementary_table4_jury_combination_robustness(input_dir: Path, out: Path) -> pd.DataFrame:
    """Raw jury-robustness table, packaged as Supplementary Table 6 in the current Heliyon numbering.

    This replaces the earlier 4a--4d split and avoids a multi-worksheet supplementary table.
    Quality-control/readiness reports are retained as internal artifacts rather than formal manuscript tables.
    """
    summary = read_csv(input_dir, "20b_jury_robustness_summary.csv").copy()
    detailed = read_csv(input_dir, "20_jury_combination_robustness.csv").copy()

    if "reference_subset" in summary.columns and summary["reference_subset"].notna().any():
        reference_subset = str(summary["reference_subset"].dropna().iloc[0])
    else:
        max_size = detailed["subset_size"].max()
        reference_subset = str(detailed.loc[detailed["subset_size"] == max_size, "jury_subset"].iloc[0])

    ref = detailed[detailed["jury_subset"].astype(str) == reference_subset][["model_name", "rank"]].rename(columns={"rank": "reference_rank"})
    shifts: list[dict[str, object]] = []
    for subset, g in detailed.groupby("jury_subset", dropna=False):
        joined = g[["model_name", "rank"]].merge(ref, on="model_name", how="left")
        abs_shift = (pd.to_numeric(joined["rank"], errors="coerce") - pd.to_numeric(joined["reference_rank"], errors="coerce")).abs()
        shifts.append({
            "jury_subset": subset,
            "Maximum absolute rank shift": float(abs_shift.max()),
            "Mean absolute rank shift": float(abs_shift.mean()),
        })
    shift_df = pd.DataFrame(shifts)

    df = summary.merge(shift_df, on="jury_subset", how="left")
    rename = {
        "jury_subset": "Jury subset",
        "subset_size": "Number of jury models",
        "reference_subset": "Full-jury reference subset",
        "n_models": "Number of evaluated models",
        "spearman_rank_corr_with_reference": "Spearman rank correlation with full jury",
        "pearson_macro_corr_with_reference": "Pearson macro-average correlation with full jury",
        "top1_model": "Top-1 model",
        "top1_matches_reference": "Top-1 matches full jury",
        "top3_overlap_with_reference": "Top-3 overlap with full jury",
        "mean_macro_average": "Mean macro-average",
        "mean_win_rate": "Mean win rate",
    }
    df = df.rename(columns=rename)
    cols = [
        "Jury subset", "Number of jury models", "Full-jury reference subset",
        "Number of evaluated models", "Spearman rank correlation with full jury",
        "Pearson macro-average correlation with full jury", "Top-1 model",
        "Top-1 matches full jury", "Top-3 overlap with full jury",
        "Maximum absolute rank shift", "Mean absolute rank shift",
        "Mean macro-average", "Mean win rate",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["Number of jury models", "Spearman rank correlation with full jury", "Pearson macro-average correlation with full jury"], ascending=[True, False, False])
    write_table_outputs(df, out)
    return df


def write_internal_qc_artifacts(input_dir: Path, output_dir: Path) -> None:
    """Retain QC/readiness files for internal checks without numbering them as supplementary tables."""
    internal_dir = output_dir / "internal_qc"
    internal_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["12_quality_control_report.csv", "27_paper_readiness_report.csv"]:
        src = input_dir / filename
        if src.exists():
            pd.read_csv(src).to_csv(internal_dir / filename, index=False)
