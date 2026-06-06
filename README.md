# TBI-NeuroHELM

This repository contains code accompanying the manuscript **TBI-NeuroHELM**, a MedHELM-inspired evaluation of large language models for traumatic brain injury (TBI) neurological assessment tasks.

The repository includes code for:

- model inference through an OpenAI-compatible API endpoint;
- closed-ended exact-match scoring;
- open-ended LLM-jury scoring;
- aggregation of benchmark, category, subcategory and cost-performance results;
- generation of manuscript figures and tables;
- packaging raw figure/table outputs into the final Heliyon-oriented artifact names.

## Repository layout

```text
TBI-NeuroHELM/
├── scripts/
│   ├── run_tbi_neurohelm_eval.py
│   ├── make_figures_tables.py
│   ├── package_heliyon_artifacts.py
│   └── run_example_workflow.py
├── src/tbi_neurohelm/
│   ├── plots/
│   └── tables/
├── prompts/
├── examples/
├── figure_source_data/
├── table_source_data/
├── configs/
└── docs/
```

## What is public in this repository?

This repository contains two public manuscript example instances selected from the 240-instance TBI-NeuroHELM benchmark:

- `TBI-CDS-B01-I002`: a closed-ended TBI severity-grading example scored by exact match.
- `TBI-DOC-B02-I018`: an open-ended progress-note generation example scored by the LLM-jury workflow.

These examples are provided only to illustrate the input format and scoring workflow. They are not intended to replace the full benchmark.

The full 240-instance benchmark, full raw model outputs, and full raw LLM-jury outputs are not included in this public repository to reduce benchmark contamination risk and preserve the integrity of future evaluations. They may be made available from the corresponding author upon reasonable request.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the public example preflight

This checks the input schema and configuration without calling any LLM API:

```bash
python scripts/run_example_workflow.py
```

Equivalent explicit command:

```bash
python scripts/run_tbi_neurohelm_eval.py \
  --input_csv examples/manuscript_example_input_instances.csv \
  --mapping_xlsx examples/manuscript_example_case_id_mapping.xlsx \
  --benchmark_xlsx examples/manuscript_example_benchmark_master.xlsx \
  --prompt_txt prompts/system_prompt.txt \
  --out_dir example_outputs_preflight \
  --mode preflight
```

## Running model evaluation

To run the full private benchmark, provide the full private input files and set API keys by environment variables. Do not place real API keys in the source code.

```bash
export APIYI_API_KEY="your_api_key"
python scripts/run_tbi_neurohelm_eval.py \
  --input_csv path/to/full_private_input_instances.csv \
  --mapping_xlsx path/to/full_private_case_id_mapping.xlsx \
  --benchmark_xlsx path/to/full_private_benchmark_master.xlsx \
  --prompt_txt prompts/system_prompt.txt \
  --out_dir tbi_neurohelm_outputs \
  --mode all \
  --workers 5
```

For per-model keys, see `.env.example`.

## Generate figures and tables

After a full evaluation run has produced `tbi_neurohelm_outputs/`, generate raw manuscript artifacts:

```bash
python scripts/make_figures_tables.py \
  --input_dir tbi_neurohelm_outputs \
  --output_dir raw_artifacts \
  --strict_qc
```

Then convert raw artifact names into the final Heliyon-oriented structure:

```bash
python scripts/package_heliyon_artifacts.py \
  --generated_dir raw_artifacts \
  --output_dir heliyon_artifacts
```

If you have finalized Figure 1, Figure 2, Supplementary Table 5, Supplementary Table 8 and Supplementary Table 9 as manuscript-specific assets, place them in a folder using final submission paths and pass it with `--assets_dir`. In the current manuscript numbering, Supplementary Figure 1 is the subcategory heatmap, Supplementary Figure 2 is the LLM-jury robustness figure, Supplementary Table 1 is the benchmark suite table, and Supplementary Table 7 is the cost-performance table.

## Figure and table source data

Aggregated figure source data and formal table source files are provided in:

- `figure_source_data/`
- `table_source_data/`

These source-data filenames follow the current Heliyon submission numbering. Full response/rating workbooks corresponding to Supplementary Tables 5, 8, and 9 are not included in this public repository.

These files are derived from the final analysis outputs and do not include the full private benchmark instances.

## Code availability statement template

The source code used for model evaluation, LLM-jury scoring, aggregation of benchmark results, cost analysis, and generation of manuscript figures and tables is available at https://github.com/Liang201-star/TBI-NeuroHELM. The repository includes prompt templates, two public manuscript example instances, aggregate figure/table source data, and documentation for reproducing the analysis workflow. API keys, full benchmark instances, full raw model responses, and full raw LLM-jury outputs are not included.

## Data availability statement template

The full set of 240 TBI-NeuroHELM benchmark instances is not publicly released to reduce benchmark contamination risk and preserve the integrity of future evaluations. De-identified benchmark instances may be made available from the corresponding author upon reasonable request. Aggregated source data required to reproduce the manuscript figures and tables are provided in this repository and the supplementary materials.
