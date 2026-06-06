# Heliyon artifact naming crosswalk

The raw figure/table generator keeps several historical raw output names inherited from the earlier MedHELM-style and Extended Data working structure. The current Heliyon-oriented submission materials use the final main/supplementary numbering used in the manuscript and submission package.

Run:

```bash
python scripts/package_heliyon_artifacts.py \
  --generated_dir path/to/raw_artifacts \
  --output_dir heliyon_artifacts
```

Important final-name conversions include:

- `figures/fig3_benchmark_heatmap.*` -> `main_figures/Figure_3_benchmark_heatmap.*`
- `figures/fig4_category_heatmap.*` -> `main_figures/Figure_4_category_heatmap.*`
- `figures/fig5_cost_vs_mean_win_rate.*` -> `main_figures/Figure_5_cost_vs_mean_win_rate.*`
- `tables/table1_win_rate_leaderboard.*` -> `main_tables/Table_1_TBI_NeuroHELM_model_performance_leaderboard.*`
- `figures/extended_data_fig1_subcategory_heatmap.*` -> `supplementary_figures/Supplementary_Figure_1_subcategory_heatmap.*`
- `figures/supplementary_fig1_jury_robustness.*` -> `supplementary_figures/Supplementary_Figure_2_jury_robustness.*`
- `tables/extended_data_table1_benchmark_suite.*` -> `supplementary_tables/Supplementary_Table_1_benchmark_suite.*`
- `tables/supplementary_table1_instance_distribution.*` -> `supplementary_tables/Supplementary_Table_2_instance_distribution.*`
- `tables/supplementary_table2_model_registry_and_run_config.*` -> `supplementary_tables/Supplementary_Table_3_model_registry_and_run_config.*`
- `tables/supplementary_table3_open_task_dimensions.*` -> `supplementary_tables/Supplementary_Table_4_open_task_dimensions.*`
- `tables/supplementary_table4_jury_combination_robustness.*` -> `supplementary_tables/Supplementary_Table_6_jury_combination_robustness.*`
- `tables/extended_data_table2_cost_performance.*` -> `supplementary_tables/Supplementary_Table_7_cost_performance.*`

The full CSV crosswalk is provided as `docs/Heliyon_figure_table_crosswalk.csv`. Table captions and notes used in the submission package are summarized in `docs/Table_caption_and_note_index.csv`.
