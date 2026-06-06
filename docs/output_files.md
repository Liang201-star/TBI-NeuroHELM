# Output files

The evaluation workflow writes intermediate and final CSV files into the output directory. Key files include:

- `01_model_outputs_raw.csv`: raw model responses. This file may contain full contexts/prompts and is not included in this public repository.
- `02_closed_item_scores.csv`: exact-match scores for closed-ended items.
- `03_open_jury_scores_raw.csv`: raw LLM-jury ratings. This file may contain model outputs and is not included in this public repository.
- `04_open_item_scores.csv`: aggregated open-task item scores.
- `07_leaderboard.csv`: model-level benchmark leaderboard source.
- `08_costs.csv`: estimated model cost source.
- `13_category_scores.csv`: category-level scores.
- `14_subcategory_scores.csv`: subcategory-level scores.
- `20_jury_combination_robustness.csv`: jury robustness source.

Use `scripts/make_figures_tables.py` to generate raw figure/table artifacts and `scripts/package_heliyon_artifacts.py` to convert raw artifact names to the final Heliyon-oriented file names.


## Current manuscript table/figure numbering

The current Heliyon submission package uses the renumbered supplementary structure documented in `docs/Heliyon_figure_table_crosswalk.csv`. Public aggregate source files in `figure_source_data/` and `table_source_data/` use the same final numbering. Supplementary Tables 5, 8, and 9 are not included as public source-data files because they contain full model responses, senior neurologist ratings, or validation workbooks.
