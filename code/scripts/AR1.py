"""
AR1.py — Rolling and non-overlapping AR(1) estimation on tick data.

Fits AR(1) on rolling / non-overlapping windows through a trading day,
plots estimated phi (autoregressive coefficient) and sigma (residual std)
across the day. Compare across pairs, days, and pre-averaging settings.

Outputs:  code/plots/models/ar1_*.pdf
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_jan import TickData, load_three_pairs
from plots_jan import preavg, PreParams

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MODELS_DIR = PROJECT_ROOT / "code" / "plots" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core: fit AR(1) via OLS on a single 1-D array
# ---------------------------------------------------------------------------
def fit_ar1_ols(y: np.ndarray) -> dict:
    """Fit AR(1) y_t = phi * y_{t-1} + eps  via OLS.

    Returns dict with keys: phi, sigma, n, mu (intercept).
    Returns None values if not enough data.
    """
    if len(y) < 5:
        return {"phi": np.nan, "sigma": np.nan, "mu": np.nan, "n": len(y)}

    Y = y[1:]
    X = y[:-1]
    n = len(Y)
    X_mat = np.column_stack([np.ones(n), X])

    # OLS: beta = (X'X)^{-1} X'Y
    try:
        beta = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"phi": np.nan, "sigma": np.nan, "mu": np.nan, "n": n}

    mu, phi = beta[0], beta[1]
    resid = Y - X_mat @ beta
    sigma = np.std(resid, ddof=2)
    return {"phi": phi, "sigma": sigma, "mu": mu, "n": n}


# ---------------------------------------------------------------------------
# Rolling / non-overlapping windowed estimation
# ---------------------------------------------------------------------------
WindowMode = Literal["rolling", "non_overlapping"]


def ar1_windows(
    mid: np.ndarray,
    times: np.ndarray,
    window_size: int,
    mode: WindowMode = "rolling",
    step: Optional[int] = None,
) -> pd.DataFrame:
    """Estimate AR(1) on windows through a series.

    Parameters
    ----------
    mid : price series (mid or pre-averaged)
    times : corresponding timestamps
    window_size : number of observations per window
    mode : 'rolling' (overlapping) or 'non_overlapping'
    step : step size for rolling mode (default = window_size // 2)
    """
    if step is None:
        step = max(window_size // 2, 1)

    results = []
    n = len(mid)

    if mode == "non_overlapping":
        starts = range(0, n - window_size + 1, window_size)
    else:
        starts = range(0, n - window_size + 1, step)

    for i in starts:
        chunk = mid[i : i + window_size]
        t_start = times[i]
        t_end = times[min(i + window_size - 1, n - 1)]
        res = fit_ar1_ols(chunk)
        res["t_start"] = t_start
        res["t_end"] = t_end
        res["t_mid"] = t_start + (t_end - t_start) / 2
        results.append(res)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plot AR(1) params through one day
# ---------------------------------------------------------------------------
def plot_ar1_day(
    td: TickData,
    date: str,
    window_size: int = 500,
    mode: WindowMode = "rolling",
    pre_params: Optional[PreParams] = None,
) -> Optional[Path]:
    """Fit AR(1) across one trading day, plot phi and sigma."""
    df = td.df.copy()
    day = pd.to_datetime(date).date()
    df = df[df["datetime"].dt.date == day]
    if len(df) < window_size * 2:
        print(f"  {td.pair} {date}: too few ticks ({len(df)}), skipping")
        return None

    # Optionally pre-average first
    if pre_params is not None:
        df = preavg(df, pre_params)

    mid = df["mid"].to_numpy()
    times = pd.to_datetime(df["datetime"]).to_numpy()

    res = ar1_windows(mid, times, window_size, mode)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["phi (AR coeff)", "sigma (resid std)"],
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=res["t_mid"], y=res["phi"],
            mode="lines+markers", marker=dict(size=3),
            line=dict(width=1.5, color="steelblue"),
            name="phi",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=res["t_mid"], y=res["sigma"],
            mode="lines+markers", marker=dict(size=3),
            line=dict(width=1.5, color="darkorange"),
            name="sigma",
        ),
        row=2, col=1,
    )

    pre_tag = ""
    if pre_params:
        if pre_params.mode == "time":
            pre_tag = f" | preavg {pre_params.window_ms}ms"
        else:
            pre_tag = f" | preavg {pre_params.n_ticks}ticks"

    fig.update_layout(
        height=600, template="plotly_white",
        title_text=(
            f"{td.pair} — AR(1) {mode} w={window_size} on {date}{pre_tag}"
        ),
        showlegend=False,
    )

    pair = td.pair.lower()
    tag = f"w{window_size}_{mode}"
    if pre_params:
        if pre_params.mode == "time":
            tag += f"_pre{pre_params.window_ms}ms"
        else:
            tag += f"_pre{pre_params.n_ticks}t"
    out = MODELS_DIR / f"ar1_{pair}_{date}_{tag}.pdf"
    fig.write_image(str(out))
    print(f"  saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# Compare phi across multiple days for one pair
# ---------------------------------------------------------------------------
def plot_ar1_multiday(
    td: TickData,
    dates: list[str],
    window_size: int = 500,
    mode: WindowMode = "rolling",
    pre_params: Optional[PreParams] = None,
) -> Optional[Path]:
    """Overlay AR(1) phi from several days to look for intraday patterns."""
    fig = go.Figure()

    colors = ["steelblue", "darkorange", "green", "red", "purple"]

    for i, date in enumerate(dates):
        df = td.df.copy()
        day = pd.to_datetime(date).date()
        df = df[df["datetime"].dt.date == day]
        if len(df) < window_size * 2:
            continue

        if pre_params is not None:
            df = preavg(df, pre_params)

        mid = df["mid"].to_numpy()
        times = pd.to_datetime(df["datetime"]).to_numpy()
        res = ar1_windows(mid, times, window_size, mode)

        # Convert to intra-day time (hours since midnight)
        t0 = np.datetime64(f"{date}T00:00:00")
        t_mid = pd.to_datetime(res["t_mid"]).dt.tz_localize(None).to_numpy().astype("datetime64[ms]")
        hours = (t_mid - t0) / np.timedelta64(1, "h")

        fig.add_trace(
            go.Scatter(
                x=hours, y=res["phi"],
                mode="lines", name=date,
                line=dict(width=1.5, color=colors[i % len(colors)]),
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    pre_tag = ""
    if pre_params:
        if pre_params.mode == "time":
            pre_tag = f" | preavg {pre_params.window_ms}ms"
        else:
            pre_tag = f" | preavg {pre_params.n_ticks}ticks"

    fig.update_layout(
        template="plotly_white",
        title=f"{td.pair} — AR(1) phi across days (w={window_size}){pre_tag}",
        xaxis_title="Hour of day (UTC)",
        yaxis_title="phi",
        height=450,
    )

    pair = td.pair.lower()
    tag = f"w{window_size}_{mode}"
    if pre_params:
        if pre_params.mode == "time":
            tag += f"_pre{pre_params.window_ms}ms"
        else:
            tag += f"_pre{pre_params.n_ticks}t"
    out = MODELS_DIR / f"ar1_{pair}_multiday_{tag}.pdf"
    fig.write_image(str(out))
    print(f"  saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("Loading data...")
    data = load_three_pairs()

    sample_days = ["2026-01-06", "2026-01-08", "2026-01-15", "2026-01-22"]
    window_sizes = [200, 500, 1000]
    pre_time = PreParams(mode="time", window_ms=100)
    pre_tick = PreParams(mode="ticks", n_ticks=10)

    # Build task list
    tasks = []
    for td in data.values():
        # A) Raw — single day, different window sizes
        for ws in window_sizes:
            tasks.append((plot_ar1_day, (td, "2026-01-15"), dict(window_size=ws, mode="rolling")))

        # B) With pre-averaging — 100ms time window
        tasks.append((plot_ar1_day, (td, "2026-01-15"), dict(window_size=200, pre_params=pre_time)))

        # C) With pre-averaging — 10 ticks
        tasks.append((plot_ar1_day, (td, "2026-01-15"), dict(window_size=200, pre_params=pre_tick)))

        # D) Multi-day overlay
        tasks.append((plot_ar1_multiday, (td, sample_days), dict(window_size=500)))
        tasks.append((plot_ar1_multiday, (td, sample_days), dict(window_size=500, pre_params=pre_time)))

    print(f"\nGenerating {len(tasks)} AR(1) plots in parallel...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fn, *args, **kw) for fn, args, kw in tasks]
        for f in as_completed(futures):
            f.result()

    print(f"\nAll AR(1) plots saved to {MODELS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
