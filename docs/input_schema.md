# Input schema

The evaluation workflow expects three input files plus a system prompt.

## Input instances CSV
Required columns:

- `runtime_id`: anonymized case/item identifier used during model calls.
- `question_type`: either `closed_qa` or `open_task`.
- `context`: clinical/research context provided to the model.
- `prompt`: task-specific prompt.

## Case mapping workbook
Required columns:

- `runtime_id`
- `instance_id`

## Benchmark master workbook
The full workflow expects the sheets used in `examples/manuscript_example_benchmark_master.xlsx`:

- `instances`
- `benchmark_groups`
- `subcategory_mix`
- `scoring_rules`
- `open_scoring_anchors`

The public repository contains two manuscript examples only. The full 240-instance benchmark is available from the corresponding author upon reasonable request.
