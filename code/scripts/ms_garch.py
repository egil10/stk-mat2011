"""
ms_garch.py — Plotly visualisation for MS-GARCH results.

The heavy lifting (MLE fitting) is done in R via  code/R/ms_garch.R
which outputs CSV files to  code/plots/models/.  This script reads
those CSVs and generates interactive Plotly HTML plots.

Run order:
  1. source("code/R/ms_garch.R")  in RStudio   → produces PDFs + CSVs
  2. python code/scripts/ms_garch.py            → produces HTML plots

Outputs:
  code/plots/plotly/ms_garch_simulated.html
  code/plots/plotly/ms_garch_{pair}_202601.html
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MODELS_DIR = PROJECT_ROOT / "code" / "plots" / "models"
PLOTLY_DIR = PROJECT_ROOT / "code" / "plots" / "plotly"
PLOTLY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_simulated(csv_path: Path) -> None:
    """Plot simulated MS-GARCH results from R output."""
    df = pd.read_csv(csv_path)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Simulated Returns (high-vol injected days 400–600)",
            "MS-GARCH Conditional Volatility",
        ],
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    fig.add_trace(
        go.Scatter(
            x=df["day"], y=df["returns"],
            mode="lines",
            line=dict(color="steelblue", width=0.7),
            name="Returns",
        ),
        row=1, col=1,
    )

    # Highlight injected regime
    fig.add_vrect(
        x0=400, x1=600,
        fillcolor="rgba(255,0,0,0.08)",
        line_width=0,
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["day"], y=df["cond_vol"],
            mode="lines",
            line=dict(color="darkorange", width=1.2),
            name="Cond. Vol",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600, template="plotly_white",
        title_text="MS-GARCH — Simulated Returns with Injected High-Vol Regime",
    )
    fig.update_yaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", row=2, col=1)
    fig.update_xaxes(title_text="Day", row=2, col=1)

    html_path = PLOTLY_DIR / "ms_garch_simulated.html"
    fig.write_html(str(html_path))
    print(f"  saved {html_path.name}")


def plot_pair(pair: str, csv_path: Path) -> None:
    """Plot tick-data MS-GARCH results from R output."""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"{pair} — Log-returns (bps)",
            f"{pair} — MS-GARCH Conditional Volatility",
        ],
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["log_ret"],
            mode="lines",
            line=dict(color="steelblue", width=0.5),
            name="Log-return (bps)",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["cond_vol"],
            mode="lines",
            line=dict(color="darkorange", width=1),
            name="Cond. Vol",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600, template="plotly_white",
        title_text=f"MS-GARCH (2-regime sGARCH) — {pair} Jan 2026",
    )
    fig.update_yaxes(title_text="bps", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", row=2, col=1)

    pair_lc = pair.lower()
    html_path = PLOTLY_DIR / f"ms_garch_{pair_lc}_202601.html"
    fig.write_html(str(html_path))
    print(f"  saved {html_path.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== MS-GARCH Plotly visualisation ===")
    print(f"Reading R outputs from {MODELS_DIR.relative_to(PROJECT_ROOT)}\n")

    # Simulated
    sim_csv = MODELS_DIR / "ms_garch_simulated.csv"
    if sim_csv.exists():
        plot_simulated(sim_csv)
    else:
        print(f"  SKIP: {sim_csv.name} not found — run code/R/ms_garch.R first")

    # Tick data pairs
    for pair in ("EURUSD", "USDZAR", "XAUUSD"):
        pair_csv = MODELS_DIR / f"ms_garch_{pair.lower()}_202601.csv"
        if pair_csv.exists():
            plot_pair(pair, pair_csv)
        else:
            print(f"  SKIP: {pair_csv.name} not found — run code/R/ms_garch.R first")

    print(f"\nHTML plots saved to {PLOTLY_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
