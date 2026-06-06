#!/usr/bin/env python3
"""Run the public manuscript examples through the preflight workflow.

This script intentionally uses --mode preflight, so it does not call any LLM API
and does not require API keys. It verifies that the two public example instances
have the same schema expected by the full private benchmark workflow.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "example_outputs_preflight"

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "run_tbi_neurohelm_eval.py"),
    "--input_csv", str(ROOT / "examples" / "manuscript_example_input_instances.csv"),
    "--mapping_xlsx", str(ROOT / "examples" / "manuscript_example_case_id_mapping.xlsx"),
    "--benchmark_xlsx", str(ROOT / "examples" / "manuscript_example_benchmark_master.xlsx"),
    "--prompt_txt", str(ROOT / "prompts" / "system_prompt.txt"),
    "--out_dir", str(OUT),
    "--mode", "preflight",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"Preflight outputs written to {OUT}")
