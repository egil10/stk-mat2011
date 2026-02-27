"""
hmm.py — Hidden Markov Model regime detection for tick data.

Fits a Gaussian HMM on log-returns (or pre-averaged returns) to detect
latent volatility/trend regimes.  Plots state probabilities, regime-colored
prices, and transition matrix.

Outputs:  code/plots/models/hmm_*.pdf
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn.hmm import GaussianHMM

from data_jan import TickData, load_three_pairs
from plots_jan import preavg, PreParams

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MODELS_DIR = PROJECT_ROOT / "code" / "plots" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def make_features(
    df: pd.DataFrame,
    pre_params: Optional[PreParams] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build HMM features from mid prices.

    Returns (features, mid_prices, timestamps).
    features columns: [log_return, |log_return|]
    """
    if pre_params is not None:
        df = preavg(df, pre_params)

    mid = df["mid"].to_numpy()
    times = pd.to_datetime(df["datetime"]).to_numpy()

    # Log returns
    log_ret = np.diff(np.log(mid))

    # Features: return + absolute return (volatility proxy)
    features = np.column_stack([
        log_ret,
        np.abs(log_ret),
    ])

    return features, mid[1:], times[1:]


# ---------------------------------------------------------------------------
# Fit HMM
# ---------------------------------------------------------------------------
def fit_hmm(
    features: np.ndarray,
    n_states: int = 2,
    n_iter: int = 100,
    random_state: int = 42,
) -> GaussianHMM:
    """Fit Gaussian HMM and return the model."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(features)
    return model


# ---------------------------------------------------------------------------
# Plot one day of HMM results
# ---------------------------------------------------------------------------
REGIME_COLORS = ["steelblue", "darkorange", "green", "red", "purple"]


def plot_hmm_day(
    td: TickData,
    date: str,
    n_states: int = 2,
    pre_params: Optional[PreParams] = None,
) -> Optional[Path]:
    """Fit HMM on one trading day, plot regimes and probabilities."""
    df = td.df.copy()
    day = pd.to_datetime(date).date()
    df = df[df["datetime"].dt.date == day]
    if len(df) < 500:
        print(f"  {td.pair} {date}: too few ticks ({len(df)}), skipping")
        return None

    features, mid, times = make_features(df, pre_params)
    if len(features) < 100:
        print(f"  {td.pair} {date}: too few features, skipping")
        return None

    model = fit_hmm(features, n_states=n_states)
    states = model.predict(features)
    probs = model.predict_proba(features)

    # Sort states by mean volatility (state 0 = low vol)
    vol_by_state = [np.abs(features[states == s, 0]).mean() for s in range(n_states)]
    order = np.argsort(vol_by_state)
    state_map = {old: new for new, old in enumerate(order)}
    states = np.array([state_map[s] for s in states])
    probs = probs[:, order]

    # Strip timezone for plotting
    times_clean = pd.to_datetime(times).tz_localize(None) if hasattr(pd.to_datetime(times), 'tz_localize') else times

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Price colored by regime", "State probabilities"],
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )

    # Price colored by regime
    for s in range(n_states):
        mask = states == s
        label = f"State {s} ({'low' if s == 0 else 'high'} vol)"
        fig.add_trace(
            go.Scatter(
                x=times_clean[mask] if isinstance(times_clean, np.ndarray) else times[mask],
                y=mid[mask],
                mode="markers",
                marker=dict(
                    color=REGIME_COLORS[s % len(REGIME_COLORS)],
                    size=2,
                ),
                name=label,
            ),
            row=1, col=1,
        )

    # State probabilities (high-vol state)
    high_state = n_states - 1
    fig.add_trace(
        go.Scatter(
            x=times_clean if isinstance(times_clean, np.ndarray) else times,
            y=probs[:, high_state],
            mode="lines",
            line=dict(color="darkorange", width=1),
            name=f"P(state {high_state})",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)

    pre_tag = ""
    if pre_params:
        if pre_params.mode == "time":
            pre_tag = f" | preavg {pre_params.window_ms}ms"
        else:
            pre_tag = f" | preavg {pre_params.n_ticks}ticks"

    fig.update_layout(
        height=650, template="plotly_white",
        title_text=f"{td.pair} — HMM {n_states}-state on {date}{pre_tag}",
    )
    fig.update_yaxes(title_text="Mid price", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1)

    pair = td.pair.lower()
    tag = f"{n_states}s"
    if pre_params:
        if pre_params.mode == "time":
            tag += f"_pre{pre_params.window_ms}ms"
        else:
            tag += f"_pre{pre_params.n_ticks}t"
    out = MODELS_DIR / f"hmm_{pair}_{date}_{tag}.pdf"
    fig.write_image(str(out))
    print(f"  saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# Print transition matrix
# ---------------------------------------------------------------------------
def print_transition_matrix(model: GaussianHMM, pair: str) -> None:
    """Print the fitted transition matrix."""
    print(f"\n  {pair} transition matrix:")
    tm = model.transmat_
    for i in range(tm.shape[0]):
        row = "    " + "  ".join(f"{tm[i, j]:.4f}" for j in range(tm.shape[1]))
        print(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("Loading data...")
    data = load_three_pairs()

    sample_days = ["2026-01-06", "2026-01-15", "2026-01-22"]
    pre_time = PreParams(mode="time", window_ms=100)

    # Build task list
    tasks = []
    for td in data.values():
        for day in sample_days:
            tasks.append((plot_hmm_day, (td, day), dict(n_states=2)))
            tasks.append((plot_hmm_day, (td, day), dict(n_states=2, pre_params=pre_time)))
            tasks.append((plot_hmm_day, (td, day), dict(n_states=3, pre_params=pre_time)))

    print(f"\nGenerating {len(tasks)} HMM plots in parallel...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fn, *args, **kw) for fn, args, kw in tasks]
        for f in as_completed(futures):
            f.result()

    # Print transition matrices (sequential, quick)
    for td in data.values():
        df_day = td.df[td.df["datetime"].dt.date == pd.to_datetime("2026-01-15").date()]
        if len(df_day) > 500:
            features, _, _ = make_features(df_day, pre_time)
            model = fit_hmm(features, n_states=2)
            print_transition_matrix(model, td.pair)

    print(f"\nAll HMM plots saved to {MODELS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

