"""
plots_jan.py — EDA and pre-averaging plots for EURUSD, USDZAR, XAUUSD (Jan 2026).

Outputs:
  code/plots/jan/summary.pdf          — per-pair tick counts, gaps, data range
  code/plots/jan/{pair}_ticks.pdf     — hourly tick counts + gap detection
  code/plots/jan/{pair}_micro.pdf     — inter-tick durations & tick-size distributions
  code/plots/jan/{pair}_preavg.pdf    — raw vs pre-averaged mid for a sample day
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_jan import TickData, load_three_pairs

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PLOTS_DIR = PROJECT_ROOT / "code" / "plots" / "jan"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PreMode = Literal["time", "ticks"]


@dataclass
class PreParams:
    mode: PreMode
    window_ms: Optional[int] = None
    n_ticks: Optional[int] = None


# ---------------------------------------------------------------------------
# Pre-averaging
# ---------------------------------------------------------------------------
def preavg(df: pd.DataFrame, params: PreParams) -> pd.DataFrame:
    """Pre-average mid prices by time window (ms) or by fixed tick count."""
    if params.mode == "time":
        if not params.window_ms:
            raise ValueError("window_ms required for time mode.")
        rule = f"{params.window_ms}ms"
        s = df.set_index("datetime")["mid"]
        grouped = s.resample(rule)
        out = pd.DataFrame(
            {
                "datetime": grouped.mean().index,
                "mid": grouped.mean().values,
                "n": grouped.count().values,
            }
        )
        out = out.dropna(subset=["mid"])
    elif params.mode == "ticks":
        if not params.n_ticks:
            raise ValueError("n_ticks required for tick mode.")
        n = params.n_ticks
        idx = np.arange(len(df)) // n
        grouped = df.groupby(idx, as_index=False)
        out = grouped.agg(
            datetime=("datetime", "first"),
            mid=("mid", "mean"),
            n=("mid", "size"),
        )
    else:
        raise ValueError(f"Unknown mode: {params.mode}")

    out["pair"] = df["pair"].iloc[0]
    return out


def _downsample(df: pd.DataFrame, max_pts: int = 5000) -> pd.DataFrame:
    if len(df) <= max_pts:
        return df
    step = max(len(df) // max_pts, 1)
    return df.iloc[::step].copy()


# ---------------------------------------------------------------------------
# 1. Summary table (console + HTML)
# ---------------------------------------------------------------------------
def print_summary(data: dict[str, TickData]) -> pd.DataFrame:
    rows = []
    for pair, td in data.items():
        df = td.df
        dt = df["datetime"]
        dt_diff = dt.diff().dt.total_seconds().dropna()
        gaps_gt60 = (dt_diff > 60).sum()
        rows.append(
            {
                "pair": pair,
                "ticks": len(df),
                "start": dt.min(),
                "end": dt.max(),
                "mean_mid": df["mid"].mean(),
                "std_mid": df["mid"].std(),
                "median_dt_ms": dt_diff.median() * 1000,
                "mean_dt_ms": dt_diff.mean() * 1000,
                "gaps_gt_60s": gaps_gt60,
            }
        )
    summary = pd.DataFrame(rows)
    print("\n=== January 2026 tick-data summary ===")
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# 2. Tick count by hour + gap detection
# ---------------------------------------------------------------------------
def plot_ticks_and_gaps(td: TickData) -> None:
    pair = td.pair.lower()
    df = td.df.copy()

    # Hourly tick count
    df["hour"] = df["datetime"].dt.floor("h")
    counts = df.groupby("hour").size().reset_index(name="ticks")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"{td.pair} — Tick count per hour",
            f"{td.pair} — Gaps > 60 s (red dots)",
        ],
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(x=counts["hour"], y=counts["ticks"], name="Ticks/hour",
               marker_color="steelblue"),
        row=1, col=1,
    )

    # Gaps > 60 s
    dt_sec = df["datetime"].diff().dt.total_seconds()
    big = df[dt_sec > 60].copy()
    big["gap_s"] = dt_sec[dt_sec > 60].values

    fig.add_trace(
        go.Scatter(
            x=big["datetime"], y=big["gap_s"],
            mode="markers", name="Gap (s)",
            marker=dict(color="red", size=5),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=700, template="plotly_white",
        title_text=f"{td.pair} {td.month} — tick activity & gaps",
        showlegend=False,
    )
    fig.update_yaxes(title_text="Ticks", row=1, col=1)
    fig.update_yaxes(title_text="Gap (s)", row=2, col=1)

    out = PLOTS_DIR / f"{pair}_{td.month}_ticks.pdf"
    fig.write_image(str(out))
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# 3. Microstructure: inter-tick durations + tick-size distributions
# ---------------------------------------------------------------------------
def plot_microstructure(td: TickData) -> None:
    pair = td.pair.lower()
    df = td.df.sort_values("datetime").copy()

    dt_sec = df["datetime"].diff().dt.total_seconds().dropna()
    dmid = np.diff(df["mid"].to_numpy())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Inter-tick duration (< 10 s)",
            "Duration CDF (< 5 s)",
            "Tick-size (Δ mid)",
            "Tick-size CDF",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # 3a — inter-tick histogram
    dt_clip = dt_sec[dt_sec < 10]
    fig.add_trace(
        go.Histogram(x=dt_clip, nbinsx=120, marker_color="steelblue",
                     name="Δt"),
        row=1, col=1,
    )

    # 3b — CDF
    sorted_dt = np.sort(dt_sec[dt_sec < 5].values)
    n = min(len(sorted_dt), 2000)
    idx = np.linspace(0, len(sorted_dt) - 1, n).astype(int)
    fig.add_trace(
        go.Scatter(x=sorted_dt[idx], y=np.linspace(0, 1, n),
                   mode="lines", line=dict(color="steelblue"), name="CDF Δt"),
        row=1, col=2,
    )

    # 3c — tick-size histogram
    clip = dmid[(np.abs(dmid) < np.percentile(np.abs(dmid), 99.5))]
    fig.add_trace(
        go.Histogram(x=clip, nbinsx=120, marker_color="darkorange",
                     name="Δmid"),
        row=2, col=1,
    )

    # 3d — tick-size CDF
    sorted_dm = np.sort(dmid)
    n2 = min(len(sorted_dm), 2000)
    idx2 = np.linspace(0, len(sorted_dm) - 1, n2).astype(int)
    fig.add_trace(
        go.Scatter(x=sorted_dm[idx2], y=np.linspace(0, 1, n2),
                   mode="lines", line=dict(color="darkorange"), name="CDF Δmid"),
        row=2, col=2,
    )

    fig.update_layout(
        height=700, template="plotly_white",
        title_text=f"{td.pair} {td.month} — microstructure",
        showlegend=False,
    )
    out = PLOTS_DIR / f"{pair}_{td.month}_micro.pdf"
    fig.write_image(str(out))
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# 4. Price plot: raw mid vs pre-averaged (one day)
# ---------------------------------------------------------------------------
def plot_preavg_day(td: TickData, date: str, params: PreParams) -> None:
    df = td.df.copy()
    day = pd.to_datetime(date).date()
    df = df[df["datetime"].dt.date == day]
    if df.empty:
        print(f"  {td.pair}: no data on {date}, skipping")
        return

    pre = preavg(df, params)
    raw_ds = _downsample(df)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=raw_ds["datetime"], y=raw_ds["mid"],
            mode="lines", name="mid (raw)",
            line=dict(color="lightgray", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pre["datetime"], y=pre["mid"],
            mode="lines", name=f"preavg ({params.mode})",
            line=dict(color="blue", width=2),
        )
    )

    tag = (
        f"time_{params.window_ms}ms" if params.mode == "time"
        else f"ticks_{params.n_ticks}"
    )
    fig.update_layout(
        title=f"{td.pair} — {date} mid vs preavg ({tag})",
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        template="plotly_white",
    )

    pair = td.pair.lower()
    out = PLOTS_DIR / f"{pair}_{td.month}_{date}_{tag}.pdf"
    fig.write_image(str(out))
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# 5. Multi-day price overview (downsampled full month)
# ---------------------------------------------------------------------------
def plot_month_overview(td: TickData) -> None:
    df = _downsample(td.df, max_pts=8000)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["mid"],
            mode="lines", name="mid",
            line=dict(color="steelblue", width=1),
        )
    )
    fig.update_layout(
        title=f"{td.pair} {td.month} — mid price (full month, downsampled)",
        xaxis_title="Date",
        yaxis_title="Mid price",
        template="plotly_white",
    )
    pair = td.pair.lower()
    out = PLOTS_DIR / f"{pair}_{td.month}_month.pdf"
    fig.write_image(str(out))
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading data...")
    data = load_three_pairs()

    summary = print_summary(data)

    # Pre-avg parameter sets
    time_params = PreParams(mode="time", window_ms=100)
    tick_params = PreParams(mode="ticks", n_ticks=10)

    # Sample days (Mon-Fri in Jan 2026)
    sample_days = ["2026-01-06", "2026-01-15", "2026-01-22"]

    for td in data.values():
        print(f"\n--- {td.pair} ---")
        plot_month_overview(td)
        plot_ticks_and_gaps(td)
        plot_microstructure(td)
        for day in sample_days:
            plot_preavg_day(td, day, time_params)
            plot_preavg_day(td, day, tick_params)

    print(f"\nAll plots saved to {PLOTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
