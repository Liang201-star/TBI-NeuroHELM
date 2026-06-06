from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tbi_neurohelm.common import MARKER_STYLES, provider_map, read_csv, save_figure


def build_cost_data(input_dir: Path) -> pd.DataFrame:
    costs = read_csv(input_dir, "08_costs.csv")
    lb = read_csv(input_dir, "07_leaderboard.csv")
    grouped = costs.groupby(["Model", "Provider"], as_index=False).agg(
        total_cost=("Cost Input Output Tokens", "sum"),
        benchmark_cost=("benchmark_cost", "sum"),
        jury_cost=("jury_cost", "sum"),
        benchmark_input_tokens=("benchmark_input_tokens", "sum"),
        benchmark_output_tokens=("benchmark_output_tokens", "sum"),
        jury_input_tokens=("jury_input_tokens", "sum"),
        jury_output_tokens=("jury_output_tokens", "sum"),
    )
    merged = grouped.merge(lb[["Model", "Mean win rate", "Macro-average", "Win SD", "SD"]], on="Model", how="left")
    return merged.sort_values("Mean win rate", ascending=False)


def plot_cost_vs_win_rate(input_dir: Path, output_base: Path) -> pd.DataFrame:
    df = build_cost_data(input_dir)
    providers = sorted(df["Provider"].dropna().astype(str).unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(providers), 1)))
    provider_color = dict(zip(providers, colors))

    # Marker assignment follows the MedHELM reference logic: color by provider, marker by model within provider.
    provider_counts: dict[str, int] = {p: 0 for p in providers}
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for _, row in df.iterrows():
        provider = str(row["Provider"])
        marker = MARKER_STYLES[provider_counts.get(provider, 0) % len(MARKER_STYLES)]
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        label = f"{row['Model']} ({provider})"
        ax.scatter(
            row["total_cost"],
            row["Mean win rate"],
            s=90,
            marker=marker,
            color=provider_color.get(provider),
            edgecolor="black",
            linewidth=0.5,
            label=label,
            zorder=3,
        )

    ax.set_xlabel("Cost (USD)", fontsize=11)
    ax.set_ylabel("Mean win rate", fontsize=11)
    ax.grid(True, color="#b0b0b0", alpha=0.75, linewidth=0.7)
    y_min = max(0, float(df["Mean win rate"].min()) - 0.08)
    y_max = min(1, float(df["Mean win rate"].max()) + 0.08)
    ax.set_ylim(y_min, y_max)
    x_pad = (float(df["total_cost"].max()) - float(df["total_cost"].min())) * 0.08 or 1.0
    ax.set_xlim(max(0, float(df["total_cost"].min()) - x_pad), float(df["total_cost"].max()) + x_pad)
    ax.legend(title="Model (provider)", loc="upper left", fontsize=8, title_fontsize=9, frameon=True)
    plt.tight_layout()
    save_figure(output_base)
    df.to_csv(output_base.with_suffix(".source.csv"), index=False)
    return df
