from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True

CATEGORY_ORDER = [
    "Clinical Decision Support",
    "Clinical Documentation Generation",
    "Patient Communication and Education",
    "Medical Research Assistance",
]

CATEGORY_SHORT = {
    "Clinical Decision Support": "Clinical\ndecision\nsupport",
    "Clinical Documentation Generation": "Clinical\ndocumentation\ngeneration",
    "Patient Communication and Education": "Patient\ncommunication\nand education",
    "Medical Research Assistance": "Medical\nresearch\nassistance",
}

QUESTION_TYPE_LABELS = {"closed_qa": "Closed QA", "open_task": "Open task"}
MARKER_STYLES = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x", "d"]

REQUIRED_CORE = [
    "00_benchmark_manifest.csv",
    "06_benchmark_scores_long.csv",
    "07_leaderboard.csv",
    "08_costs.csv",
    "10_model_registry.csv",
    "13_category_scores.csv",
    "14_subcategory_scores.csv",
    "15_question_type_scores.csv",
    "18_open_score_dimensions.csv",
    "19_jury_model_scoring_tendency_summary.csv",
    "20_jury_combination_robustness.csv",
    "20b_jury_robustness_summary.csv",
    "24_instance_distribution_summary.csv",
]

REQUIRED_QC = [
    "01_model_outputs_raw.csv",
    "02_closed_item_scores.csv",
    "03_open_jury_scores_raw.csv",
    "12_quality_control_report.csv",
    "27_paper_readiness_report.csv",
]


def read_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    source = output_dir / "source_data"
    for p in (figures, tables, source):
        p.mkdir(parents=True, exist_ok=True)
    return figures, tables, source


def save_figure(base_path: Path, dpi: int = 500) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(base_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close()


def wrap_label(label: object, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))


def model_order(input_dir: Path) -> list[str]:
    lb = read_csv(input_dir, "07_leaderboard.csv")
    return lb.sort_values("Mean win rate", ascending=False)["Model"].astype(str).tolist()


def provider_map(input_dir: Path) -> dict[str, str]:
    reg_path = input_dir / "10_model_registry.csv"
    if reg_path.exists():
        reg = pd.read_csv(reg_path)
        if {"model_name", "provider"}.issubset(reg.columns):
            return dict(zip(reg["model_name"].astype(str), reg["provider"].astype(str)))
    lb = read_csv(input_dir, "07_leaderboard.csv")
    if {"Model", "Provider"}.issubset(lb.columns):
        return dict(zip(lb["Model"].astype(str), lb["Provider"].astype(str)))
    return {}


def write_table_outputs(df: pd.DataFrame, base_path: Path, float_format: str = "%.3f") -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    tex = df.to_latex(index=False, escape=False, float_format=float_format)
    base_path.with_suffix(".tex").write_text(tex, encoding="utf-8")


def annotated_heatmap(matrix: pd.DataFrame, output_base: Path, title: str, figsize: tuple[float, float], x_label_width: int = 16) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="viridis", norm=Normalize(vmin=0, vmax=1))

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([wrap_label(c, x_label_width) for c in matrix.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    if title:
        ax.set_title(title, fontsize=14, pad=12)

    # MedHELM-style visible cell grid with compact annotations.
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if val < 0.45 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    save_figure(output_base)
    matrix.to_csv(output_base.with_suffix(".source.csv"))


def validate_final_qc(input_dir: Path, strict: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fn in REQUIRED_CORE + REQUIRED_QC:
        rows.append({"check": f"required_file::{fn}", "status": "PASS" if (input_dir / fn).exists() else "FAIL", "value": str(input_dir / fn)})

    raw_path = input_dir / "01_model_outputs_raw.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        rows.append({"check": "raw_row_count_2160", "status": "PASS" if len(raw) == 2160 else "FAIL", "value": len(raw)})
        if "question_type" in raw.columns:
            n_closed = int((raw["question_type"].astype(str) == "closed_qa").sum())
            n_open = int((raw["question_type"].astype(str) == "open_task").sum())
            rows.append({"check": "raw_closed_row_count_1242", "status": "PASS" if n_closed == 1242 else "FAIL", "value": n_closed})
            rows.append({"check": "raw_open_row_count_918", "status": "PASS" if n_open == 918 else "FAIL", "value": n_open})
        if "finish_reason" in raw.columns:
            length_count = int((raw["finish_reason"].fillna("").astype(str) == "length").sum())
            rows.append({"check": "raw_finish_reason_length_zero", "status": "PASS" if length_count == 0 else "FAIL", "value": length_count})
        if "error_message" in raw.columns:
            err_count = int(raw["error_message"].fillna("").astype(str).str.len().gt(0).sum())
            rows.append({"check": "raw_api_error_zero", "status": "PASS" if err_count == 0 else "FAIL", "value": err_count})
        if "parse_success" in raw.columns:
            parse_fail = int((~raw["parse_success"].fillna(False).astype(bool)).sum())
            rows.append({"check": "raw_parse_fail_zero", "status": "PASS" if parse_fail == 0 else "FAIL", "value": parse_fail})

    jury_path = input_dir / "03_open_jury_scores_raw.csv"
    if jury_path.exists():
        jury = pd.read_csv(jury_path)
        rows.append({"check": "jury_row_count_2754", "status": "PASS" if len(jury) == 2754 else "FAIL", "value": len(jury)})
        if "judge_finish_reason" in jury.columns:
            j_length = int((jury["judge_finish_reason"].fillna("").astype(str) == "length").sum())
            rows.append({"check": "jury_finish_reason_length_zero", "status": "PASS" if j_length == 0 else "FAIL", "value": j_length})
        if "judge_parse_success" in jury.columns:
            j_parse_fail = int((~jury["judge_parse_success"].fillna(False).astype(bool)).sum())
            rows.append({"check": "jury_parse_fail_zero", "status": "PASS" if j_parse_fail == 0 else "FAIL", "value": j_parse_fail})

    readiness_path = input_dir / "27_paper_readiness_report.csv"
    if readiness_path.exists():
        r = pd.read_csv(readiness_path)
        if "status" in r.columns:
            fail_count = int((r["status"].fillna("").astype(str).str.upper() != "PASS").sum())
            rows.append({"check": "paper_readiness_all_pass", "status": "PASS" if fail_count == 0 else "FAIL", "value": fail_count})

    qc = pd.DataFrame(rows)
    if strict and (qc["status"] == "FAIL").any():
        failed = qc[qc["status"] == "FAIL"].to_string(index=False)
        raise RuntimeError("Strict QC failed; final figures/tables were not generated.\n" + failed)
    return qc
