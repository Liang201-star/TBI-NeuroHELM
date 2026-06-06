#!/usr/bin/env python3
"""Package raw figure/table outputs into the final Heliyon-oriented artifact names.

The plotting script keeps several historical raw names (for example,
``extended_data_fig1_subcategory_heatmap``). This script converts those raw
outputs into the current final manuscript/submission structure:

- main_figures/Figure_3--5
- main_tables/Table_1
- supplementary_figures/Supplementary_Figure_1--2
- supplementary_tables/Supplementary_Table_1--4 and 6--7

Figure 1, Figure 2, Supplementary Tables 5, 8 and 9 are manuscript-specific
assets. If you want them copied into the package, provide ``--assets_dir`` with
files already using the final submission names.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import pandas as pd

RENAME_MAP = {
    # Main figures generated from final results
    "figures/fig3_benchmark_heatmap.png": "main_figures/Figure_3_benchmark_heatmap.png",
    "figures/fig3_benchmark_heatmap.pdf": "main_figures/Figure_3_benchmark_heatmap.pdf",
    "figures/fig3_benchmark_heatmap.source.csv": "main_figures/Figure_3_benchmark_heatmap.source.csv",
    "figures/fig4_category_heatmap.png": "main_figures/Figure_4_category_heatmap.png",
    "figures/fig4_category_heatmap.pdf": "main_figures/Figure_4_category_heatmap.pdf",
    "figures/fig4_category_heatmap.source.csv": "main_figures/Figure_4_category_heatmap.source.csv",
    "figures/fig5_cost_vs_mean_win_rate.png": "main_figures/Figure_5_cost_vs_mean_win_rate.png",
    "figures/fig5_cost_vs_mean_win_rate.pdf": "main_figures/Figure_5_cost_vs_mean_win_rate.pdf",
    "figures/fig5_cost_vs_mean_win_rate.source.csv": "main_figures/Figure_5_cost_vs_mean_win_rate.source.csv",

    # Main table source
    "tables/table1_win_rate_leaderboard.csv": "main_tables/Table_1_TBI_NeuroHELM_model_performance_leaderboard.source.csv",
    "tables/table1_win_rate_leaderboard.tex": "main_tables/Table_1_TBI_NeuroHELM_model_performance_leaderboard.tex",

    # Supplementary figures in current manuscript order
    "figures/extended_data_fig1_subcategory_heatmap.png": "supplementary_figures/Supplementary_Figure_1_subcategory_heatmap.png",
    "figures/extended_data_fig1_subcategory_heatmap.pdf": "supplementary_figures/Supplementary_Figure_1_subcategory_heatmap.pdf",
    "figures/extended_data_fig1_subcategory_heatmap.source.csv": "supplementary_figures/Supplementary_Figure_1_subcategory_heatmap.source.csv",
    "figures/supplementary_fig1_jury_robustness.png": "supplementary_figures/Supplementary_Figure_2_jury_robustness.png",
    "figures/supplementary_fig1_jury_robustness.pdf": "supplementary_figures/Supplementary_Figure_2_jury_robustness.pdf",
    "figures/supplementary_fig1_jury_robustness_panel_a_source.csv": "supplementary_figures/Supplementary_Figure_2_jury_robustness_panel_a_source.csv",
    "figures/supplementary_fig1_jury_robustness_panel_b_source.csv": "supplementary_figures/Supplementary_Figure_2_jury_robustness_panel_b_source.csv",
    "figures/supplementary_fig1_jury_robustness_panel_c_source.csv": "supplementary_figures/Supplementary_Figure_2_jury_robustness_panel_c_source.csv",

    # Supplementary tables in current manuscript order
    "tables/extended_data_table1_benchmark_suite.csv": "supplementary_tables/Supplementary_Table_1_benchmark_suite.source.csv",
    "tables/extended_data_table1_benchmark_suite.tex": "supplementary_tables/Supplementary_Table_1_benchmark_suite.tex",
    "tables/supplementary_table1_instance_distribution.csv": "supplementary_tables/Supplementary_Table_2_instance_distribution.source.csv",
    "tables/supplementary_table1_instance_distribution.tex": "supplementary_tables/Supplementary_Table_2_instance_distribution.tex",
    "tables/supplementary_table2_model_registry_and_run_config.csv": "supplementary_tables/Supplementary_Table_3_model_registry_and_run_config.source.csv",
    "tables/supplementary_table2_model_registry_and_run_config.tex": "supplementary_tables/Supplementary_Table_3_model_registry_and_run_config.tex",
    "tables/supplementary_table3_open_task_dimensions.csv": "supplementary_tables/Supplementary_Table_4_open_task_dimensions.source.csv",
    "tables/supplementary_table3_open_task_dimensions.tex": "supplementary_tables/Supplementary_Table_4_open_task_dimensions.tex",
    "tables/supplementary_table4_jury_combination_robustness.csv": "supplementary_tables/Supplementary_Table_6_jury_combination_robustness.source.csv",
    "tables/supplementary_table4_jury_combination_robustness.tex": "supplementary_tables/Supplementary_Table_6_jury_combination_robustness.tex",
    "tables/extended_data_table2_cost_performance.csv": "supplementary_tables/Supplementary_Table_7_cost_performance.source.csv",
    "tables/extended_data_table2_cost_performance.tex": "supplementary_tables/Supplementary_Table_7_cost_performance.tex",
}

FINAL_ASSET_NAMES = [
    "main_figures/Figure_1_TBI_NeuroHELM_framework.png",
    "main_figures/Figure_1_TBI_NeuroHELM_framework.pdf",
    "main_figures/Figure_2_TBI_NeuroHELM_taxonomy.png",
    "main_figures/Figure_2_TBI_NeuroHELM_taxonomy.pdf",
    "supplementary_tables/Supplementary_Table_5_TBI_NeuroHELM_model_responses_and_jury_scores_final.xlsx",
    "supplementary_tables/Supplementary_Table_8_TBI_NeuroHELM_senior_neurologist_ratings_reference_style_final.xlsx",
    "supplementary_tables/Supplementary_Table_9_TBI_NeuroHELM_senior_neurologist_validation_reference_style_final.xlsx",
]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_index(output_dir: Path) -> None:
    rows = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_file() and p.name != "artifact_index.csv":
            rows.append({"relative_path": p.relative_to(output_dir).as_posix(), "suffix": p.suffix.lower(), "size_bytes": p.stat().st_size})
    pd.DataFrame(rows).to_csv(output_dir / "artifact_index.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", required=True, type=Path, help="Raw output folder produced by scripts/make_figures_tables.py")
    parser.add_argument("--output_dir", required=True, type=Path, help="Final Heliyon-oriented artifact folder")
    parser.add_argument("--assets_dir", type=Path, default=None, help="Optional folder containing final Figure 1/2 and Supplementary Table 5/8/9 files")
    args = parser.parse_args()

    generated_dir = args.generated_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crosswalk_rows = []
    missing = []
    for raw_rel, final_rel in RENAME_MAP.items():
        ok = copy_if_exists(generated_dir / raw_rel, output_dir / final_rel)
        crosswalk_rows.append({"raw_output": raw_rel, "final_submission_path": final_rel, "copied": ok})
        if not ok:
            missing.append(raw_rel)

    if args.assets_dir is not None:
        assets_dir = args.assets_dir.resolve()
        for rel in FINAL_ASSET_NAMES:
            ok = copy_if_exists(assets_dir / rel, output_dir / rel)
            crosswalk_rows.append({"raw_output": f"assets_dir/{rel}", "final_submission_path": rel, "copied": ok})

    pd.DataFrame(crosswalk_rows).to_csv(output_dir / "Heliyon_figure_table_crosswalk.csv", index=False)
    write_index(output_dir)

    if missing:
        print("Warning: some raw outputs were not found. See Heliyon_figure_table_crosswalk.csv")
    print(f"Packaged artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
