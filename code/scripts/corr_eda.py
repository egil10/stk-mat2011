"""
corr_eda.py — Currency Pair Correlation Explorer (Jan 2026)

Generates two 10-page PDFs where each page analyses one "pair of pairs"
believed to be structurally correlated.

  1. Indexed price overlay   (both series rebased to 100)
  2. Relative-value ratio    (series A / series B)
  3. Rolling correlation (Pearson on log-returns)

Outputs:
  code/plots/eda/correlation_pairs_monthly.pdf
  code/plots/eda/correlation_pairs_morning.pdf

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

# Colours
COLORS_A = "#2563eb"   # vivid blue
COLORS_B = "#dc2626"   # vivid red
COLOR_RATIO = "#7c3aed"# purple
COLOR_CORR = "#059669" # emerald green
COLOR_FILL = "#d1fae5" # light emerald fill

# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
def load_hourly(pair: str) -> pd.Series:
    """Load tick parquet → resample to hourly last price."""
    path = DATA_DIR / f"{pair}_last_202601.parquet"
    df = pd.read_parquet(path, columns=["datetime", "price"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    hourly = df.set_index("datetime")["price"].resample("1h").last().dropna()
    hourly.name = pair
    return hourly

def load_morning(pair: str, date_str="2026-01-15") -> pd.Series:
    """Load tick parquet → filter to 07:00-12:00 UTC on a specific day."""
    path = DATA_DIR / f"{pair}_last_202601.parquet"
    df = pd.read_parquet(path, columns=["datetime", "price"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    mask = (df["datetime"].dt.date.astype(str) == date_str) & \
           (df["datetime"].dt.hour >= 7) & \
           (df["datetime"].dt.hour < 12)
    morning = df[mask].set_index("datetime")["price"]
    # Drop exact timestamp duplicates (keeping last)
    morning = morning[~morning.index.duplicated(keep="last")]
    morning.name = pair
    return morning

# ---------------------------------------------------------------------------
# Compute Helpers
# ---------------------------------------------------------------------------
def indexed(s: pd.Series) -> pd.Series:
    """Rebase to 100 at the first observation."""
    if len(s) == 0: return s
    return s / s.iloc[0] * 100

def rolling_corr_hourly(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation on aligned hourly log-returns."""
    merged = pd.concat([a, b], axis=1, join="inner").dropna()
    ma, mb = merged.iloc[:, 0], merged.iloc[:, 1]
    ret_a = np.log(ma / ma.shift(1)).dropna()
    ret_b = np.log(mb / mb.shift(1)).dropna()
    return ret_a.rolling(window).corr(ret_b)

def rolling_corr_time(a: pd.Series, b: pd.Series, rs="10s", window=60) -> pd.Series:
    """
    Rolling Pearson correlation on tick data by first resampling to a fixed grid 
    (e.g., 10s) and computing returns. This avoids microstructure noise from tick counts
    and uses a pure time window.
    """
    a_grid = a.resample(rs).last().ffill()
    b_grid = b.resample(rs).last().ffill()
    # align them
    merged = pd.concat([a_grid, b_grid], axis=1, join="inner").dropna()
    ma, mb = merged.iloc[:, 0], merged.iloc[:, 1]
    
    ret_a = np.log(ma / ma.shift(1)).dropna()
    ret_b = np.log(mb / mb.shift(1)).dropna()
    return ret_a.rolling(window).corr(ret_b)

def combine_ticks(a: pd.Series, b: pd.Series):
    """Combine tick series onto a dense timeline with forward-fill for step-like level plots."""
    combined_idx = a.index.union(b.index).sort_values()
    a_ff = a.reindex(combined_idx).ffill().bfill()
    b_ff = b.reindex(combined_idx).ffill().bfill()
    return a_ff, b_ff


# ---------------------------------------------------------------------------
# Render Function
# ---------------------------------------------------------------------------
def render_page(pdf, sa: pd.Series, sb: pd.Series, label_a: str,
                label_b: str, rationale: str, page_num: int, mode: str):
    
    fig = plt.figure(figsize=(11.69, 16.54))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[3, 2, 2, 0.4],
                  hspace=0.35, top=0.93, bottom=0.04, left=0.09, right=0.95)

    # Mode logic
    if mode == "monthly":
        title_suffix = "Monthly (Hourly bars)"
        time_fmt = mdates.DateFormatter("%b %d")
        # Align
        m_a, m_b = pd.concat([sa, sb], axis=1, join="inner").dropna().iloc[:, 0], \
                   pd.concat([sa, sb], axis=1, join="inner").dropna().iloc[:, 1]
        overall_corr = m_a.corr(m_b)
        ratio = m_a / m_b
        rc = rolling_corr_hourly(sa, sb, window=24)
        rc_slow = rolling_corr_hourly(sa, sb, window=72)
        rc_label = "24h"
        rc_slow_label = "72h rolling corr"
        line_w = 1.0
    else:
        title_suffix = "Morning Session (Jan 15, 07:00-12:00 UTC) – Tick Level"
        time_fmt = mdates.DateFormatter("%H:%M")
        # Align ticks densely via forward fill
        m_a, m_b = combine_ticks(sa, sb)
        overall_corr = m_a.corr(m_b)
        ratio = m_a / m_b
        # 10s grid returns: 60 periods = 10m, 180 periods = 30m
        rc = rolling_corr_time(sa, sb, rs="10s", window=60)
        rc_slow = rolling_corr_time(sa, sb, rs="10s", window=180)
        rc_label = "10m"
        rc_slow_label = "30m rolling corr (10s grid)"
        line_w = 0.5

    # ── Suptitle
    fig.suptitle(
        f"Page {page_num}/10  ·  {label_a}  vs  {label_b}  ·  {title_suffix}",
        fontsize=16, fontweight="bold", y=0.97,
    )

    # ── Panel 1: Indexed price
    ax1 = fig.add_subplot(gs[0])
    ia, ib = indexed(m_a), indexed(m_b)
    # Using 'post' step for morning tick data to explicitly show no-trades
    plot_style = "post" if mode == "morning" else "default"
    
    if plot_style == "post":
        ax1.step(ia.index, ia.values, color=COLORS_A, linewidth=line_w, label=label_a, alpha=0.9, where='post')
        ax1.step(ib.index, ib.values, color=COLORS_B, linewidth=line_w, label=label_b, alpha=0.9, where='post')
    else:
        ax1.plot(ia.index, ia.values, color=COLORS_A, linewidth=line_w, label=label_a, alpha=0.9)
        ax1.plot(ib.index, ib.values, color=COLORS_B, linewidth=line_w, label=label_b, alpha=0.9)
        
    ax1.set_title("Indexed Price (base = 100)", fontsize=13, pad=10)
    ax1.set_ylabel("Indexed level")
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.85)
    ax1.grid(alpha=0.18, linewidth=0.5)
    ax1.xaxis.set_major_formatter(time_fmt)

    ax1.text(0.99, 0.04, f"Pearson r (levels) = {overall_corr:.4f}",
             transform=ax1.transAxes, fontsize=10, ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d1d5db", alpha=0.9))

    # ── Panel 2: Ratio
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ratio_mean = ratio.mean()
    
    if plot_style == "post":
        ax2.step(ratio.index, ratio.values, color=COLOR_RATIO, linewidth=line_w, alpha=0.85, where='post')
    else:
        ax2.plot(ratio.index, ratio.values, color=COLOR_RATIO, linewidth=line_w, alpha=0.85)
        
    ax2.axhline(ratio_mean, color=COLOR_RATIO, linewidth=0.7, linestyle="--", alpha=0.5)
    ax2.fill_between(ratio.index, ratio_mean - ratio.std(), ratio_mean + ratio.std(),
                     color=COLOR_RATIO, alpha=0.07, label=f"±1σ band  (μ={ratio_mean:.5f})")
    ax2.set_title(f"Relative Value  ({label_a} / {label_b})", fontsize=13, pad=10)
    ax2.set_ylabel("Ratio")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax2.grid(alpha=0.18, linewidth=0.5)

    # ── Panel 3: Correlation
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(rc.index, 0, rc.values, where=(rc >= 0), color=COLOR_CORR, alpha=0.25)
    ax3.fill_between(rc.index, 0, rc.values, where=(rc < 0), color="#dc2626", alpha=0.20)
    ax3.plot(rc.index, rc.values, color=COLOR_CORR, linewidth=0.9, alpha=0.85)
    ax3.axhline(0, color="black", linewidth=0.5)

    # Slow ref
    ax3.plot(rc_slow.index, rc_slow.values, color="#f59e0b", linewidth=1.2, alpha=0.7, label=rc_slow_label)

    mean_corr = rc.dropna().mean()
    ax3.axhline(mean_corr, color=COLOR_CORR, linewidth=0.7, linestyle=":", alpha=0.6)
    title_corr_base = "Rolling Correlation of Hourly Log-Returns" if mode == "monthly" else "Rolling Correlation of 10s Log-Returns"
    ax3.set_title(title_corr_base, fontsize=13, pad=10)
    ax3.set_ylabel("Pearson r")
    
    # Safely avoid setting ylim if limits are NaN
    curr_min, curr_max = ax3.get_ylim()
    ax3.set_ylim(min(-1.05, curr_min), max(1.05, curr_max))

    ax3.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax3.grid(alpha=0.18, linewidth=0.5)
    ax3.xaxis.set_major_formatter(time_fmt)

    ax3.text(0.99, 0.06, f"Mean {rc_label} corr = {mean_corr:.4f}",
             transform=ax3.transAxes, fontsize=10, ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d1d5db", alpha=0.9))

    # ── Footer
    ax_footer = fig.add_subplot(gs[3])
    ax_footer.axis("off")
    ax_footer.text(0.5, 0.5, f"Rationale:  {rationale}",
                   transform=ax_footer.transAxes, fontsize=10, ha="center", va="center",
                   style="italic", color="#6b7280")

    pdf.savefig(fig, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def process_report(mode: str, needed_pairs: set):
    print(f"\n[{mode.upper()} REPORT]")
    cache = {}
    for pair in sorted(needed_pairs):
        if mode == "monthly":
            cache[pair] = load_hourly(pair)
        else:
            cache[pair] = load_morning(pair)
            
    pdf_path = EDA_DIR / f"correlation_pairs_{mode}.pdf"
    
    with PdfPages(str(pdf_path)) as pdf:
        for i, (pair_a, pair_b, rationale) in enumerate(CORR_PAIRS, 1):
            label_a = PAIR_LABELS.get(pair_a, pair_a.upper())
            label_b = PAIR_LABELS.get(pair_b, pair_b.upper())
            print(f"  Page {i:>2}/10: {label_a} vs {label_b} ...", end="", flush=True)
            render_page(pdf, cache[pair_a], cache[pair_b], label_a, label_b, rationale, i, mode)
            print(" done.")
            
    print(f"  ✓ PDF saved → {pdf_path.relative_to(PROJECT_ROOT)}")

def main():
    print("=" * 64)
    print("  Currency Pair Correlation EDA  (Two timeframes)")
    print("=" * 64)

    needed = set()
    for a, b, _ in CORR_PAIRS:
        needed.add(a)
        needed.add(b)

    process_report("monthly", needed)
    process_report("morning", needed)

    print("\n================================================================")
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
