"""
ms_ar.py — Markov-Switching Autoregression (MS-AR) for regime detection.

Demonstrates MS-AR on the US macro dataset (GNP growth) using statsmodels,
then applies the same logic to tick-level mid-price returns for EURUSD,
USDZAR, and XAUUSD (January 2026).

Math:  y_t = mu_{S_t} + phi * (y_{t-1} - mu_{S_{t-1}}) + eps_t

Outputs:
  code/plots/models/ms_ar_gnp_macro.pdf        — macro GNP recession regimes
  code/plots/plotly/ms_ar_gnp_macro.html        — interactive version
  code/plots/models/ms_ar_{pair}_202601.pdf     — tick-data regime detection
  code/plots/plotly/ms_ar_{pair}_202601.html     — interactive version
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression,
)

from data_jan import TickData, load_three_pairs
from plots_jan import preavg, PreParams

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MODELS_DIR = PROJECT_ROOT / "code" / "plots" / "models"
PLOTLY_DIR = PROJECT_ROOT / "code" / "plots" / "plotly"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTLY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Part 1: Macro GNP dataset (textbook example)
# ---------------------------------------------------------------------------
def fit_macro_msar() -> None:
    """Fit MS-AR(4) on US GNP growth — classic recession-detection demo."""
    print("\n=== Part 1: Macro GNP — MS-AR(4) ===")

    # Load macroeconomic dataset
    macro = sm.datasets.macrodata.load_pandas().data
    gnp_growth = 100 * macro["realgdp"].pct_change().dropna()
    gnp_growth.name = "gnp_growth"

    # Fit Markov-Switching Autoregression
    model = MarkovAutoregression(
        gnp_growth,
        k_regimes=2,
        order=4,
        switching_ar=False,
        switching_variance=True,
    )
    result = model.fit(search_reps=20, disp=False)

    print(result.summary())

    # Smoothed recession probability (Regime 0 = low growth / recession)
    recession_prob = result.smoothed_marginal_probabilities[0]
    index = gnp_growth.index

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "US Real GNP Growth (annualised quarterly %)",
            "Recession Probability (Smoothed — Regime 0)",
        ],
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    fig.add_trace(
        go.Scatter(
            x=list(index),
            y=gnp_growth.values,
            mode="lines",
            line=dict(color="steelblue", width=1.2),
            name="GNP growth",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=list(index),
            y=recession_prob,
            mode="lines",
            fill="tozeroy",
            line=dict(color="crimson", width=1),
            fillcolor="rgba(220,20,60,0.25)",
            name="P(Recession)",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        height=600, template="plotly_white",
        title_text="Markov-Switching AR(4) — US Real GNP Growth",
    )
    fig.update_yaxes(title_text="Growth %", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1)

    # Save PDF + interactive HTML
    pdf_path = MODELS_DIR / "ms_ar_gnp_macro.pdf"
    html_path = PLOTLY_DIR / "ms_ar_gnp_macro.html"
    fig.write_image(str(pdf_path))
    fig.write_html(str(html_path))
    print(f"  saved {pdf_path.name}")
    print(f"  saved {html_path.name}")


# ---------------------------------------------------------------------------
# Part 2: Tick-data MS-AR on mid-price returns
# ---------------------------------------------------------------------------
def fit_tick_msar(
    td: TickData,
    pre_params: Optional[PreParams] = None,
    k_regimes: int = 2,
    order: int = 1,
) -> Optional[Path]:
    """Fit MS-AR on pre-averaged log-returns for all of Jan 2026."""
    pair = td.pair
    df = td.df.copy()

    if pre_params is not None:
        df = preavg(df, pre_params)

    mid = df["mid"].to_numpy()
    times = pd.to_datetime(df["datetime"]).to_numpy()

    # Log returns (in bps for numerical stability)
    log_ret = np.diff(np.log(mid)) * 1e4
    log_ret = log_ret[np.isfinite(log_ret)]
    times = times[1 : len(log_ret) + 1]

    # Subsample: MS-AR MLE is expensive; use ~10 000 points
    max_pts = 10_000
    if len(log_ret) > max_pts:
        step = len(log_ret) // max_pts
        log_ret = log_ret[::step]
        times = times[::step]

    ret_series = pd.Series(log_ret, name=f"{pair}_ret")

    print(f"  {pair}: fitting MS-AR({order}) on {len(ret_series):,} points ...")
    try:
        model = MarkovAutoregression(
            ret_series,
            k_regimes=k_regimes,
            order=order,
            switching_ar=False,
            switching_variance=True,
        )
        try:
            result = model.fit(search_reps=20, disp=False)
        except Exception:
            # Retry with more search reps (exotic pairs may need it)
            print(f"  {pair}: retrying with search_reps=50 ...")
            result = model.fit(search_reps=50, disp=False)
    except Exception as e:
        print(f"  {pair}: MS-AR fitting failed — {e}")
        return None

    # Smoothed probabilities: high-vol regime
    high_vol_prob = result.smoothed_marginal_probabilities[k_regimes - 1]

    # Strip timezone for plotting
    times_clean = pd.to_datetime(times).tz_localize(None) if hasattr(
        pd.to_datetime(times), "tz_localize"
    ) else pd.to_datetime(times)

    pre_tag = ""
    if pre_params:
        if pre_params.mode == "time":
            pre_tag = f" | preavg {pre_params.window_ms}ms"
        else:
            pre_tag = f" | preavg {pre_params.n_ticks}ticks"

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Log-returns (bps)",
            f"P(High-volatility regime) — Smoothed",
        ],
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    fig.add_trace(
        go.Scatter(
            x=times_clean,
            y=log_ret,
            mode="lines",
            line=dict(color="steelblue", width=0.5),
            name="Log-return (bps)",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=times_clean,
            y=high_vol_prob,
            mode="lines",
            fill="tozeroy",
            line=dict(color="darkorange", width=1),
            fillcolor="rgba(255,140,0,0.25)",
            name="P(High vol)",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        height=600, template="plotly_white",
        title_text=f"MS-AR({order}) — {pair} Jan 2026{pre_tag}",
    )
    fig.update_yaxes(title_text="bps", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1)

    # File names
    pair_lc = pair.lower()
    pdf_out = MODELS_DIR / f"ms_ar_{pair_lc}_202601.pdf"
    html_out = PLOTLY_DIR / f"ms_ar_{pair_lc}_202601.html"
    fig.write_image(str(pdf_out))
    fig.write_html(str(html_out))
    print(f"  saved {pdf_out.name}")
    print(f"  saved {html_out.name}")
    return pdf_out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    # Part 1: Macro demo
    fit_macro_msar()

    # Part 2: Tick data
    print("\n=== Part 2: Tick Data — MS-AR(1) ===")
    print("Loading data...")
    data = load_three_pairs()

    pre_time = PreParams(mode="time", window_ms=100)

    for td in data.values():
        fit_tick_msar(td, pre_params=pre_time)

    print(f"\nAll MS-AR plots saved to:")
    print(f"  PDF:  {MODELS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  HTML: {PLOTLY_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
