from __future__ import annotations

from pathlib import Path
import pandas as pd

from tbi_neurohelm.common import read_csv, write_table_outputs


def build_table(input_dir: Path) -> pd.DataFrame:
    lb = read_csv(input_dir, "07_leaderboard.csv")
    cols = [
        "Model",
        "Provider",
        "Mean win rate",
        "Win SD",
        "Mean win rate (tie=0.5)",
        "Mean win rate (tie=win)",
        "Macro-average",
        "SD",
    ]
    cols = [c for c in cols if c in lb.columns]
    df = lb[cols].copy().sort_values("Mean win rate", ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def write_table(input_dir: Path, output_base: Path) -> pd.DataFrame:
    df = build_table(input_dir)
    write_table_outputs(df, output_base)
    return df
