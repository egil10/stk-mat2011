"""
ms_garch.py — Markov-Switching GARCH (MS-GARCH) for volatility clustering.

Uses rpy2 to bridge Python to R's MSGARCH package, since no pure-Python
MS-GARCH implementation exists.

Prerequisites:
  - R installed and in system PATH
  - install.packages("MSGARCH") run inside R
  - pip install rpy2

Part 1: Demo on simulated returns with injected high-vol regime.
Part 2: Applied to tick-level mid-price returns for EURUSD, USDZAR, XAUUSD.

Outputs:
  code/plots/models/ms_garch_simulated.pdf      — simulated data demo
  code/plots/plotly/ms_garch_simulated.html      — interactive version
  code/plots/models/ms_garch_{pair}_202601.pdf   — tick-data MS-GARCH
  code/plots/plotly/ms_garch_{pair}_202601.html   — interactive version
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MODELS_DIR = PROJECT_ROOT / "code" / "plots" / "models"
PLOTLY_DIR = PROJECT_ROOT / "code" / "plots" / "plotly"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTLY_DIR.mkdir(parents=True, exist_ok=True)


def _init_rpy2():
    """Import and activate rpy2 → MSGARCH bridge."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    msgarch = importr("MSGARCH")
    return ro, msgarch


# ---------------------------------------------------------------------------
# Part 1: Simulated data demo
# ---------------------------------------------------------------------------
def fit_simulated_msgarch() -> None:
    """Generate returns with injected high-vol regime, fit MS-GARCH."""
    print("\n=== Part 1: Simulated Returns — MS-GARCH ===")

    ro, msgarch = _init_rpy2()

    # Simulate 1000 days of returns
    np.random.seed(42)
    n = 1000
    returns = np.random.normal(loc=0, scale=1, size=n)
    # Inject high volatility regime between days 400-600
    returns[400:600] = np.random.normal(loc=0, scale=3, size=200)

    # Create MS-GARCH specification: 2-regime sGARCH with normal innovations
    spec = msgarch.CreateSpec(
        variance_spec=ro.ListVector({"spec": ro.StrVector(["sGARCH", "sGARCH"])}),
        distribution_spec=ro.ListVector(
            {"spec": ro.StrVector(["norm", "norm"])}
        ),
        switch_spec=ro.ListVector({"do": ro.BoolVector([True])}),
    )

    # Fit via Maximum Likelihood
    r_returns = ro.FloatVector(returns)
    print("  Fitting MS-GARCH on simulated data (this may take a moment)...")
    fit = msgarch.FitML(spec=spec, data=r_returns)

    # Extract conditional volatility
    vol_r = fit.rx2("vol")
    cond_vol = np.array(vol_r).flatten()

    # Plot: 2-panel comparison
    days = np.arange(n)

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
            x=days,
            y=returns,
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
            x=days[: len(cond_vol)],
            y=cond_vol,
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

    pdf_path = MODELS_DIR / "ms_garch_simulated.pdf"
    html_path = PLOTLY_DIR / "ms_garch_simulated.html"
    fig.write_image(str(pdf_path))
    fig.write_html(str(html_path))
    print(f"  saved {pdf_path.name}")
    print(f"  saved {html_path.name}")


# ---------------------------------------------------------------------------
# Part 2: Tick-data MS-GARCH
# ---------------------------------------------------------------------------
def fit_tick_msgarch(
    pair: str,
    df: pd.DataFrame,
    pre_params=None,
) -> Optional[Path]:
    """Fit MS-GARCH on pre-averaged mid-price log-returns."""
    ro, msgarch = _init_rpy2()

    # Optional pre-averaging
    if pre_params is not None:
        from plots_jan import preavg

        df = preavg(df, pre_params)

    mid = df["mid"].to_numpy()
    times = pd.to_datetime(df["datetime"]).to_numpy()

    # Log returns
    log_ret = np.diff(np.log(mid)) * 1e4  # in bps
    log_ret = log_ret[np.isfinite(log_ret)]
    times = times[1 : len(log_ret) + 1]

    # Subsample: MS-GARCH MLE is very expensive; cap at ~5 000 points
    max_pts = 5_000
    if len(log_ret) > max_pts:
        step = len(log_ret) // max_pts
        log_ret = log_ret[::step]
        times = times[::step]

    print(f"  {pair}: fitting MS-GARCH on {len(log_ret):,} points ...")

    try:
        spec = msgarch.CreateSpec(
            variance_spec=ro.ListVector(
                {"spec": ro.StrVector(["sGARCH", "sGARCH"])}
            ),
            distribution_spec=ro.ListVector(
                {"spec": ro.StrVector(["norm", "norm"])}
            ),
            switch_spec=ro.ListVector({"do": ro.BoolVector([True])}),
        )
        r_returns = ro.FloatVector(log_ret.tolist())
        fit = msgarch.FitML(spec=spec, data=r_returns)
        vol_r = fit.rx2("vol")
        cond_vol = np.array(vol_r).flatten()
    except Exception as e:
        print(f"  {pair}: MS-GARCH fitting failed — {e}")
        return None

    # Strip timezone
    times_clean = pd.to_datetime(times).tz_localize(None) if hasattr(
        pd.to_datetime(times), "tz_localize"
    ) else pd.to_datetime(times)

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
            x=times_clean[: len(cond_vol)],
            y=cond_vol,
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
    pdf_out = MODELS_DIR / f"ms_garch_{pair_lc}_202601.pdf"
    html_out = PLOTLY_DIR / f"ms_garch_{pair_lc}_202601.html"
    fig.write_image(str(pdf_out))
    fig.write_html(str(html_out))
    print(f"  saved {pdf_out.name}")
    print(f"  saved {html_out.name}")
    return pdf_out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    # Part 1: Simulated demo
    fit_simulated_msgarch()

    # Part 2: Tick data
    print("\n=== Part 2: Tick Data — MS-GARCH ===")
    print("Loading data...")
    from data_jan import load_three_pairs
    from plots_jan import PreParams

    data = load_three_pairs()
    pre_time = PreParams(mode="time", window_ms=100)

    for td in data.values():
        fit_tick_msgarch(td.pair, td.df.copy(), pre_params=pre_time)

    print(f"\nAll MS-GARCH plots saved to:")
    print(f"  PDF:  {MODELS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  HTML: {PLOTLY_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
