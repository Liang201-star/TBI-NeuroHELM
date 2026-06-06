# Reproducibility notes

The public repository supports two levels of reproducibility:

1. **Code-flow demonstration.** The two public manuscript examples in `examples/` allow users to validate the input schema and run the preflight workflow without API keys.
2. **Figure/table reproduction from aggregate source data.** Aggregated figure and table source files are provided in `figure_source_data/` and `table_source_data/`.

The full 240-instance benchmark, full model responses, and full raw LLM-jury outputs are not included in the public repository to reduce benchmark contamination risk and preserve the integrity of future evaluations. These materials may be made available from the corresponding author upon reasonable request.

LLM API results can vary with model snapshots, provider routing, and API platform behavior. Reproducing the exact full run therefore requires the same private benchmark files, model snapshots, prompt templates, scoring rules, and API configuration used in the manuscript.
