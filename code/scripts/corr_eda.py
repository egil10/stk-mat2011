"""
corr_eda.py — Currency Pair Correlation Explorer (Jan 2026)

Generates a 10-page PDF where each page analyses one "pair of pairs"
believed to be structurally correlated.  Each page contains:

  1. Indexed price overlay   (both series rebased to 100)
  2. Relative-value ratio    (series A / series B)
  3. 24h rolling correlation (Pearson on hourly log-returns)

Output:
  code/plots/eda/correlation_pairs.pdf

Data source: code/data/processed/*_last_202601.parquet
All pairs use the same HistData "last" feed for Jan 2026 so the
comparison is apples-to-apples.

Usage:
    python code/scripts/corr_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pathlib import Path
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "code" / "data" / "processed"
EDA_DIR = PROJECT_ROOT / "code" / "plots" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 10 hand-picked pairs that are expected to be highly correlated
# Each entry: (pair_a, pair_b, short_rationale)
# ---------------------------------------------------------------------------
CORR_PAIRS = [
    ("eurusd", "gbpusd",  "Both European majors vs USD — classic co-movement"),
    ("audusd", "nzdusd",  "Oceania commodity currencies — strongest FX correlation"),
    ("eurjpy", "gbpjpy",  "European currencies vs JPY — risk-on/off twins"),
    ("audjpy", "nzdjpy",  "Oceania × JPY — double correlated: AUD≈NZD & base JPY"),
    ("eurusd", "usdchf",  "Classic inverse pair — CHF & EUR geo-linked"),
    ("euraud", "eurnzd",  "EUR base, AUD/NZD quote — spread = AUDNZD cross"),
    ("eurcad", "gbpusd",  "EUR & GBP share European dynamics, CAD ≈ inverse USD"),
    ("usdcad", "usdchf",  "USD base, commodity vs safe-haven quote"),
    ("eurnok", "eursek",  "Scandinavian crosses — structurally linked economies"),
    ("xauusd", "usdchf",  "Gold & CHF both safe-haven proxies vs USD"),
]

# Formatting for display labels
PAIR_LABELS = {
    "eurusd": "EUR/USD", "gbpusd": "GBP/USD", "audusd": "AUD/USD",
    "nzdusd": "NZD/USD", "usdcad": "USD/CAD", "usdchf": "USD/CHF",
    "eurjpy": "EUR/JPY", "gbpjpy": "GBP/JPY", "audjpy": "AUD/JPY",
    "nzdjpy": "NZD/JPY", "eurgbp": "EUR/GBP", "euraud": "EUR/AUD",
    "eurnzd": "EUR/NZD", "eurcad": "EUR/CAD", "eurchf": "EUR/CHF",
    "audcad": "AUD/CAD", "audchf": "AUD/CHF", "audnzd": "AUD/NZD",
    "cadchf": "CAD/CHF", "cadjpy": "CAD/JPY", "chfjpy": "CHF/JPY",
    "xauusd": "XAU/USD", "bcousd": "BCO/USD", "eurnok": "EUR/NOK",
    "eursek": "EUR/SEK",
}

# ---------------------------------------------------------------------------
# Colour palette — one pair per colour family
# ---------------------------------------------------------------------------
COLORS_A = "#2563eb"   # vivid blue
COLORS_B = "#dc2626"   # vivid red
COLOR_RATIO = "#7c3aed" # purple
COLOR_CORR = "#059669"  # emerald green
COLOR_FILL = "#d1fae5"  # light emerald fill


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_hourly(pair: str) -> pd.Series:
    """Load tick parquet → resample to hourly last price → return Series."""
    path = DATA_DIR / f"{pair}_last_202601.parquet"
    df = pd.read_parquet(path, columns=["datetime", "price"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    hourly = (
        df.set_index("datetime")["price"]
        .resample("1h")
        .last()
        .dropna()
    )
    hourly.name = pair
    return hourly


def align_pair(a: pd.Series, b: pd.Series):
    """Inner-join on the hourly index so both series share identical timestamps."""
    merged = pd.concat([a, b], axis=1, join="inner").dropna()
    return merged.iloc[:, 0], merged.iloc[:, 1]


def indexed(s: pd.Series) -> pd.Series:
    """Rebase to 100 at the first observation."""
    return s / s.iloc[0] * 100


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 24) -> pd.Series:
    """Rolling Pearson correlation on log-returns."""
    ret_a = np.log(a / a.shift(1)).dropna()
    ret_b = np.log(b / b.shift(1)).dropna()
    # Align after differencing
    idx = ret_a.index.intersection(ret_b.index)
    return ret_a.loc[idx].rolling(window).corr(ret_b.loc[idx])


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------
def render_page(pdf, sa: pd.Series, sb: pd.Series, label_a: str,
                label_b: str, rationale: str, page_num: int):
    """Draw a single PDF page with three vertically-stacked charts."""

    fig = plt.figure(figsize=(11.69, 16.54))  # A3 portrait-ish for detail
    gs = GridSpec(4, 1, figure=fig, height_ratios=[3, 2, 2, 0.4],
                  hspace=0.35, top=0.93, bottom=0.04, left=0.09, right=0.95)

    # ── Suptitle ──────────────────────────────────────────────────────────
    overall_corr = sa.corr(sb)
    fig.suptitle(
        f"Page {page_num}/10  ·  {label_a}  vs  {label_b}",
        fontsize=18, fontweight="bold", y=0.97,
    )

    # ── Panel 1: Indexed price overlay ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ia, ib = indexed(sa), indexed(sb)
    ax1.plot(ia.index, ia.values, color=COLORS_A, linewidth=1.0,
             label=label_a, alpha=0.9)
    ax1.plot(ib.index, ib.values, color=COLORS_B, linewidth=1.0,
             label=label_b, alpha=0.9)
    ax1.set_title("Indexed Price (base = 100)", fontsize=13, pad=10)
    ax1.set_ylabel("Indexed level")
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.85)
    ax1.grid(alpha=0.18, linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax1.tick_params(axis="x", rotation=0)

    # Annotate overall Pearson r on the chart
    ax1.text(
        0.99, 0.04,
        f"Pearson r (levels) = {overall_corr:.4f}",
        transform=ax1.transAxes, fontsize=10, ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d1d5db", alpha=0.9),
    )

    # ── Panel 2: Relative-value ratio ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ratio = sa / sb
    ratio_mean = ratio.mean()
    ax2.plot(ratio.index, ratio.values, color=COLOR_RATIO, linewidth=0.9,
             alpha=0.85)
    ax2.axhline(ratio_mean, color=COLOR_RATIO, linewidth=0.7,
                linestyle="--", alpha=0.5)
    ax2.fill_between(
        ratio.index,
        ratio_mean - ratio.std(),
        ratio_mean + ratio.std(),
        color=COLOR_RATIO, alpha=0.07,
        label=f"±1σ band  (μ={ratio_mean:.5f})",
    )
    ax2.set_title(f"Relative Value  ({label_a} / {label_b})", fontsize=13, pad=10)
    ax2.set_ylabel("Ratio")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax2.grid(alpha=0.18, linewidth=0.5)

    # ── Panel 3: Rolling correlation ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    rc = rolling_corr(sa, sb, window=24)
    ax3.fill_between(rc.index, 0, rc.values, where=(rc >= 0),
                     color=COLOR_CORR, alpha=0.25)
    ax3.fill_between(rc.index, 0, rc.values, where=(rc < 0),
                     color="#dc2626", alpha=0.20)
    ax3.plot(rc.index, rc.values, color=COLOR_CORR, linewidth=0.9, alpha=0.85)
    ax3.axhline(0, color="black", linewidth=0.5)

    # Longer-window reference line
    rc_72 = rolling_corr(sa, sb, window=72)
    ax3.plot(rc_72.index, rc_72.values, color="#f59e0b", linewidth=1.2,
             alpha=0.7, label="72h rolling corr")

    mean_corr = rc.dropna().mean()
    ax3.axhline(mean_corr, color=COLOR_CORR, linewidth=0.7,
                linestyle=":", alpha=0.6)
    ax3.set_title("Rolling Correlation of Hourly Log-Returns", fontsize=13, pad=10)
    ax3.set_ylabel("Pearson r")
    ax3.set_ylim(-1.05, 1.05)
    ax3.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax3.grid(alpha=0.18, linewidth=0.5)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Annotate mean rolling corr
    ax3.text(
        0.99, 0.06,
        f"Mean 24h corr = {mean_corr:.4f}",
        transform=ax3.transAxes, fontsize=10, ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d1d5db", alpha=0.9),
    )

    # ── Footer: rationale text ────────────────────────────────────────────
    ax_footer = fig.add_subplot(gs[3])
    ax_footer.axis("off")
    ax_footer.text(
        0.5, 0.5, f"Rationale:  {rationale}",
        transform=ax_footer.transAxes, fontsize=10, ha="center", va="center",
        style="italic", color="#6b7280",
    )

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("  Currency Pair Correlation EDA  (Jan 2026)")
    print("=" * 64)

    # Pre-load all unique pairs we need
    needed = set()
    for a, b, _ in CORR_PAIRS:
        needed.add(a)
        needed.add(b)

    cache: dict[str, pd.Series] = {}
    for pair in sorted(needed):
        print(f"  Loading {PAIR_LABELS.get(pair, pair):>8s} ...", end="")
        cache[pair] = load_hourly(pair)
        print(f"  {len(cache[pair]):>5,} hourly bars")

    pdf_path = EDA_DIR / "correlation_pairs.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        for i, (pair_a, pair_b, rationale) in enumerate(CORR_PAIRS, 1):
            label_a = PAIR_LABELS.get(pair_a, pair_a.upper())
            label_b = PAIR_LABELS.get(pair_b, pair_b.upper())
            print(f"\n  Page {i:>2}/10: {label_a} vs {label_b}")

            sa, sb = align_pair(cache[pair_a], cache[pair_b])
            print(f"           {len(sa):,} aligned hourly observations")
            render_page(pdf, sa, sb, label_a, label_b, rationale, i)

    print(f"\n  ✓ PDF saved → {pdf_path.relative_to(PROJECT_ROOT)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
