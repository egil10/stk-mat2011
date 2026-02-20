"""
EUR/USD Tick-Level Microstructure Autocorrelation Analysis

True tick-by-tick analysis for market microstructure research:
- Tick returns autocorrelation (not aggregated)
- Price change direction autocorrelation
- Inter-tick time intervals
- Bid-ask bounce effects
- Spread dynamics

Run from project root: python code/scripts/tick_microstructure_acf.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pathlib import Path

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "raw"
PLOTS_DIR = SCRIPT_DIR.parent / "plots"

ASCII_FILE = DATA_DIR / "DAT_ASCII_EURUSD_T_202601.csv"

# For tick-level analysis, we use a smaller sample to keep computation tractable
# but still have enough data for meaningful ACF estimation
TICK_SAMPLE_SIZE = 500_000  # Number of consecutive ticks to analyze

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'grid.alpha': 0.5,
    'grid.color': '#dddddd',
    'figure.figsize': (11, 8),
})

COLORS = {
    'primary': '#1a73e8',
    'secondary': '#ea4335',
    'tertiary': '#34a853',
    'accent': '#9c27b0',
    'neutral': '#5f6368',
    'bid': '#1a73e8',
    'ask': '#ea4335',
}


# ==============================================================================
# Data Loading - NO AGGREGATION
# ==============================================================================

def load_tick_data(filepath: Path, n_ticks: int = None) -> pd.DataFrame:
    """Load raw tick data without any aggregation."""
    print(f"Loading tick data from {filepath.name}...")
    
    df = pd.read_csv(
        filepath,
        header=None,
        names=['datetime', 'bid', 'ask', 'volume'],
        dtype={'bid': float, 'ask': float, 'volume': int},
        nrows=n_ticks
    )
    
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S%f')
    df.set_index('datetime', inplace=True)
    
    # Derived tick-level features
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['spread'] = (df['ask'] - df['bid']) * 10000  # pips
    
    # Tick-to-tick changes (in pips, 1 pip = 0.0001)
    df['mid_change'] = df['mid'].diff() * 10000
    df['bid_change'] = df['bid'].diff() * 10000
    df['ask_change'] = df['ask'].diff() * 10000
    df['spread_change'] = df['spread'].diff()
    
    # Price change direction (-1, 0, +1)
    df['direction'] = np.sign(df['mid_change'])
    
    # Inter-tick time (in milliseconds)
    df['inter_tick_ms'] = df.index.to_series().diff().dt.total_seconds() * 1000
    
    # Log returns at tick level
    df['log_return'] = np.log(df['mid'] / df['mid'].shift(1)) * 10000  # basis points
    
    df = df.dropna()
    
    print(f"  Loaded {len(df):,} ticks")
    print(f"  Time range: {df.index.min()} to {df.index.max()}")
    print(f"  Avg inter-tick time: {df['inter_tick_ms'].mean():.1f} ms")
    
    return df


# ==============================================================================
# Tick-Level ACF Plotting Functions
# ==============================================================================

def plot_tick_returns_acf(df: pd.DataFrame, lags: int = 100) -> plt.Figure:
    """Page 1: ACF of tick-by-tick log returns."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    tick_returns = df['log_return'].values
    acf_values, confint = acf(tick_returns, nlags=lags, alpha=0.05)
    
    # Bar plot for cleaner visualization at tick level
    colors = [COLORS['secondary'] if abs(v) > 1.96/np.sqrt(len(tick_returns)) 
              else COLORS['primary'] for v in acf_values]
    
    ax.bar(range(len(acf_values)), acf_values, color=colors, alpha=0.7, width=0.8)
    
    ci = 1.96 / np.sqrt(len(tick_returns))
    ax.axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='95% CI')
    ax.axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xlabel('Lag (ticks)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Tick-Level Log Returns ACF (No Aggregation)')
    ax.set_xlim(-1, lags + 1)
    ax.legend(loc='upper right')
    
    # Key microstructure insight
    ax.annotate(f'Lag 1 ACF: {acf_values[1]:.4f}\n'
                f'(Negative = bid-ask bounce effect)',
                xy=(0.70, 0.85), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.9))
    
    plt.tight_layout()
    return fig


def plot_tick_returns_pacf(df: pd.DataFrame, lags: int = 50) -> plt.Figure:
    """Page 2: PACF of tick returns."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    tick_returns = df['log_return'].values
    pacf_values, confint = pacf(tick_returns, nlags=lags, alpha=0.05)
    
    colors = [COLORS['secondary'] if abs(v) > 1.96/np.sqrt(len(tick_returns)) 
              else COLORS['tertiary'] for v in pacf_values]
    
    ax.bar(range(len(pacf_values)), pacf_values, color=colors, alpha=0.7, width=0.8)
    
    ci = 1.96 / np.sqrt(len(tick_returns))
    ax.axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5)
    ax.axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xlabel('Lag (ticks)')
    ax.set_ylabel('Partial Autocorrelation')
    ax.set_title('Tick-Level Log Returns PACF')
    ax.set_xlim(-1, lags + 1)
    
    plt.tight_layout()
    return fig


def plot_direction_acf(df: pd.DataFrame, lags: int = 100) -> plt.Figure:
    """Page 3: ACF of price change direction (+1, 0, -1)."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Filter out zero moves for cleaner signal
    direction = df['direction'].values
    direction_nonzero = direction[direction != 0]
    
    acf_values, _ = acf(direction_nonzero, nlags=lags, alpha=0.05)
    
    colors = [COLORS['secondary'] if abs(v) > 1.96/np.sqrt(len(direction_nonzero)) 
              else COLORS['accent'] for v in acf_values]
    
    ax.bar(range(len(acf_values)), acf_values, color=colors, alpha=0.7, width=0.8)
    
    ci = 1.96 / np.sqrt(len(direction_nonzero))
    ax.axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='95% CI')
    ax.axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xlabel('Lag (ticks)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('ACF of Tick Direction (Sign of Price Change)')
    ax.set_xlim(-1, lags + 1)
    ax.legend(loc='upper right')
    
    # Interpretation
    if acf_values[1] < -ci:
        interpretation = 'Negative lag-1: Mean reversion / Bid-ask bounce'
    elif acf_values[1] > ci:
        interpretation = 'Positive lag-1: Momentum / Order flow clustering'
    else:
        interpretation = 'No significant lag-1: Random walk'
    
    ax.annotate(interpretation, xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.9))
    
    plt.tight_layout()
    return fig


def plot_squared_tick_returns_acf(df: pd.DataFrame, lags: int = 200) -> plt.Figure:
    """Page 4: ACF of squared tick returns - volatility clustering at tick level."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    squared_returns = (df['log_return'] ** 2).values
    acf_values, _ = acf(squared_returns, nlags=lags, alpha=0.05)
    
    ax.bar(range(len(acf_values)), acf_values, color=COLORS['secondary'], alpha=0.6, width=0.9)
    
    ci = 1.96 / np.sqrt(len(squared_returns))
    ax.axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='95% CI')
    ax.axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xlabel('Lag (ticks)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('ACF of Squared Tick Returns (Tick-Level Volatility Clustering)')
    ax.set_xlim(-1, lags + 1)
    ax.legend(loc='upper right')
    
    # Highlight persistence
    sig_lags = np.sum(acf_values[1:] > ci)
    ax.annotate(f'Significant lags: {sig_lags}/{lags}\n'
                f'Strong volatility clustering at tick level',
                xy=(0.65, 0.85), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.9))
    
    plt.tight_layout()
    return fig


def plot_spread_acf(df: pd.DataFrame, lags: int = 100) -> plt.Figure:
    """Page 5: ACF of bid-ask spread - spread dynamics."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    
    spread = df['spread'].values
    spread_change = df['spread_change'].values
    
    # ACF of spread level
    acf_level, _ = acf(spread, nlags=lags, alpha=0.05)
    axes[0].bar(range(len(acf_level)), acf_level, color=COLORS['tertiary'], alpha=0.7, width=0.9)
    ci = 1.96 / np.sqrt(len(spread))
    axes[0].axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[0].axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_ylabel('ACF')
    axes[0].set_title('ACF of Spread Level (Spread Persistence)')
    axes[0].set_xlim(-1, lags + 1)
    
    # ACF of spread changes
    acf_change, _ = acf(spread_change, nlags=lags, alpha=0.05)
    axes[1].bar(range(len(acf_change)), acf_change, color=COLORS['primary'], alpha=0.7, width=0.9)
    axes[1].axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[1].axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Lag (ticks)')
    axes[1].set_ylabel('ACF')
    axes[1].set_title('ACF of Spread Changes')
    axes[1].set_xlim(-1, lags + 1)
    
    fig.suptitle('Bid-Ask Spread Dynamics at Tick Level', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


def plot_inter_tick_time_acf(df: pd.DataFrame, lags: int = 100) -> plt.Figure:
    """Page 6: ACF of inter-tick durations - trade intensity clustering."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Remove outlier durations (likely market closures)
    durations = df['inter_tick_ms'].values
    q99 = np.percentile(durations, 99)
    durations_clean = durations[durations < q99]
    
    acf_values, _ = acf(durations_clean, nlags=lags, alpha=0.05)
    
    ax.bar(range(len(acf_values)), acf_values, color=COLORS['accent'], alpha=0.7, width=0.9)
    
    ci = 1.96 / np.sqrt(len(durations_clean))
    ax.axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='95% CI')
    ax.axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xlabel('Lag (ticks)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('ACF of Inter-Tick Durations (Trade Intensity Clustering)')
    ax.set_xlim(-1, lags + 1)
    ax.legend(loc='upper right')
    
    ax.annotate(f'Mean duration: {np.mean(durations_clean):.1f} ms\n'
                f'Positive ACF = clustering of fast/slow periods',
                xy=(0.65, 0.85), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#f3e5f5', alpha=0.9))
    
    plt.tight_layout()
    return fig


def plot_bid_ask_cross_acf(df: pd.DataFrame, lags: int = 50) -> plt.Figure:
    """Page 7: Cross-correlation between bid and ask changes."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    
    bid_change = df['bid_change'].values
    ask_change = df['ask_change'].values
    
    # ACF of bid changes
    acf_bid, _ = acf(bid_change, nlags=lags, alpha=0.05)
    axes[0].bar(range(len(acf_bid)), acf_bid, color=COLORS['bid'], alpha=0.7, width=0.9)
    ci = 1.96 / np.sqrt(len(bid_change))
    axes[0].axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[0].axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_ylabel('ACF')
    axes[0].set_title('ACF of Bid Price Changes')
    axes[0].set_xlim(-1, lags + 1)
    
    # ACF of ask changes
    acf_ask, _ = acf(ask_change, nlags=lags, alpha=0.05)
    axes[1].bar(range(len(acf_ask)), acf_ask, color=COLORS['ask'], alpha=0.7, width=0.9)
    axes[1].axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[1].axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Lag (ticks)')
    axes[1].set_ylabel('ACF')
    axes[1].set_title('ACF of Ask Price Changes')
    axes[1].set_xlim(-1, lags + 1)
    
    fig.suptitle('Bid vs Ask Price Change Autocorrelation', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


def plot_first_lags_detail(df: pd.DataFrame) -> plt.Figure:
    """Page 8: Detailed view of first 20 lags - microstructure effects."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    tick_returns = df['log_return'].values
    direction = df['direction'].values
    direction_nz = direction[direction != 0]
    squared_returns = tick_returns ** 2
    abs_returns = np.abs(tick_returns)
    
    lags = 20
    ci = 1.96 / np.sqrt(len(tick_returns))
    
    # Returns
    acf_ret, _ = acf(tick_returns, nlags=lags, alpha=0.05)
    axes[0, 0].bar(range(lags+1), acf_ret, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    axes[0, 0].axhline(y=ci, color=COLORS['secondary'], linestyle='--')
    axes[0, 0].axhline(y=-ci, color=COLORS['secondary'], linestyle='--')
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].set_title('Tick Returns')
    axes[0, 0].set_ylabel('ACF')
    for i, v in enumerate(acf_ret[:6]):
        axes[0, 0].annotate(f'{v:.3f}', xy=(i, v), ha='center', 
                           va='bottom' if v > 0 else 'top', fontsize=8)
    
    # Direction
    acf_dir, _ = acf(direction_nz, nlags=lags, alpha=0.05)
    axes[0, 1].bar(range(lags+1), acf_dir, color=COLORS['accent'], alpha=0.7, edgecolor='white')
    axes[0, 1].axhline(y=ci, color=COLORS['secondary'], linestyle='--')
    axes[0, 1].axhline(y=-ci, color=COLORS['secondary'], linestyle='--')
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].set_title('Direction')
    for i, v in enumerate(acf_dir[:6]):
        axes[0, 1].annotate(f'{v:.3f}', xy=(i, v), ha='center',
                           va='bottom' if v > 0 else 'top', fontsize=8)
    
    # Squared returns
    acf_sq, _ = acf(squared_returns, nlags=lags, alpha=0.05)
    axes[1, 0].bar(range(lags+1), acf_sq, color=COLORS['secondary'], alpha=0.7, edgecolor='white')
    axes[1, 0].axhline(y=ci, color=COLORS['neutral'], linestyle='--')
    axes[1, 0].axhline(y=-ci, color=COLORS['neutral'], linestyle='--')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_title('Squared Returns')
    axes[1, 0].set_xlabel('Lag (ticks)')
    axes[1, 0].set_ylabel('ACF')
    for i, v in enumerate(acf_sq[:6]):
        axes[1, 0].annotate(f'{v:.3f}', xy=(i, v), ha='center',
                           va='bottom' if v > 0 else 'top', fontsize=8)
    
    # Absolute returns
    acf_abs, _ = acf(abs_returns, nlags=lags, alpha=0.05)
    axes[1, 1].bar(range(lags+1), acf_abs, color=COLORS['tertiary'], alpha=0.7, edgecolor='white')
    axes[1, 1].axhline(y=ci, color=COLORS['neutral'], linestyle='--')
    axes[1, 1].axhline(y=-ci, color=COLORS['neutral'], linestyle='--')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].set_title('Absolute Returns')
    axes[1, 1].set_xlabel('Lag (ticks)')
    for i, v in enumerate(acf_abs[:6]):
        axes[1, 1].annotate(f'{v:.3f}', xy=(i, v), ha='center',
                           va='bottom' if v > 0 else 'top', fontsize=8)
    
    fig.suptitle('First 20 Tick Lags - Detailed View', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


def plot_ljung_box_tick(df: pd.DataFrame, max_lag: int = 20) -> plt.Figure:
    """Page 9: Ljung-Box test at tick level."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    tick_returns = df['log_return'].values
    squared_returns = tick_returns ** 2
    direction = df['direction'].values
    direction_nz = direction[direction != 0]
    durations = df['inter_tick_ms'].values
    durations = durations[durations < np.percentile(durations, 99)]
    
    series_list = [
        (tick_returns, 'Tick Returns', COLORS['primary']),
        (squared_returns, 'Squared Returns', COLORS['secondary']),
        (direction_nz, 'Direction', COLORS['accent']),
        (durations, 'Inter-tick Duration', COLORS['tertiary']),
    ]
    
    for ax, (series, title, color) in zip(axes.flat, series_list):
        lb = acorr_ljungbox(series, lags=range(1, max_lag+1), return_df=True)
        
        colors = [COLORS['tertiary'] if p > 0.05 else COLORS['secondary'] 
                  for p in lb['lb_pvalue']]
        ax.bar(lb.index, lb['lb_pvalue'], color=colors, alpha=0.7, width=0.8)
        ax.axhline(y=0.05, color='black', linestyle='--', linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Lag')
        ax.set_ylabel('p-value')
        ax.set_title(title)
        
        reject = (lb['lb_pvalue'] < 0.05).sum()
        ax.annotate(f'{reject}/{max_lag} significant', xy=(0.95, 0.95), 
                   xycoords='axes fraction', ha='right', va='top', fontsize=9)
    
    fig.suptitle('Ljung-Box Tests at Tick Level (H0: No Autocorrelation)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


def plot_microstructure_summary(df: pd.DataFrame) -> plt.Figure:
    """Page 10: Summary of tick-level microstructure findings."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis('off')
    
    tick_returns = df['log_return'].values
    acf_ret, _ = acf(tick_returns, nlags=10, alpha=0.05)
    acf_sq, _ = acf(tick_returns ** 2, nlags=10, alpha=0.05)
    
    direction = df['direction'].values
    direction_nz = direction[direction != 0]
    acf_dir, _ = acf(direction_nz, nlags=10, alpha=0.05)
    
    acf_dur, _ = acf(df['inter_tick_ms'].values[:100000], nlags=10, alpha=0.05)
    
    ci = 1.96 / np.sqrt(len(tick_returns))
    
    summary_text = f"""
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║               TICK-LEVEL MICROSTRUCTURE ANALYSIS SUMMARY                       ║
    ║                    EUR/USD High-Frequency Data - January 2026                  ║
    ╠════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                ║
    ║  DATA SUMMARY                                                                  ║
    ║  ───────────────────────────────────────────────────────────────────────────   ║
    ║    Total ticks analyzed: {len(df):,}                                       ║
    ║    Average inter-tick time: {df['inter_tick_ms'].mean():.1f} ms                               ║
    ║    Average spread: {df['spread'].mean():.3f} pips                                         ║
    ║    Tick return std: {np.std(tick_returns):.4f} bps                                        ║
    ║                                                                                ║
    ║  KEY AUTOCORRELATION FINDINGS (First 5 lags)                                   ║
    ║  ───────────────────────────────────────────────────────────────────────────   ║
    ║    Tick Returns:     [{acf_ret[1]:+.4f}, {acf_ret[2]:+.4f}, {acf_ret[3]:+.4f}, {acf_ret[4]:+.4f}, {acf_ret[5]:+.4f}]        ║
    ║    Direction:        [{acf_dir[1]:+.4f}, {acf_dir[2]:+.4f}, {acf_dir[3]:+.4f}, {acf_dir[4]:+.4f}, {acf_dir[5]:+.4f}]        ║
    ║    Squared Returns:  [{acf_sq[1]:+.4f}, {acf_sq[2]:+.4f}, {acf_sq[3]:+.4f}, {acf_sq[4]:+.4f}, {acf_sq[5]:+.4f}]        ║
    ║    Inter-tick Time:  [{acf_dur[1]:+.4f}, {acf_dur[2]:+.4f}, {acf_dur[3]:+.4f}, {acf_dur[4]:+.4f}, {acf_dur[5]:+.4f}]        ║
    ║    95% CI threshold: +/- {ci:.4f}                                               ║
    ║                                                                                ║
    ║  MICROSTRUCTURE EFFECTS DETECTED                                               ║
    ║  ───────────────────────────────────────────────────────────────────────────   ║
    ║    [{'X' if acf_ret[1] < -ci else ' '}] Bid-Ask Bounce: Negative lag-1 return autocorrelation              ║
    ║    [{'X' if acf_sq[1] > ci else ' '}] Volatility Clustering: Positive squared return ACF                  ║
    ║    [{'X' if acf_dur[1] > ci else ' '}] Trade Intensity Clustering: Positive duration ACF                  ║
    ║    [{'X' if acf_dir[1] < -ci else ' '}] Mean Reversion: Negative direction autocorrelation                ║
    ║                                                                                ║
    ║  IMPLICATIONS FOR MODELING                                                     ║
    ║  ───────────────────────────────────────────────────────────────────────────   ║
    ║    • Bid-ask bounce invalidates naive trend-following at tick level            ║
    ║    • GARCH/ACD models appropriate for volatility & duration dynamics           ║
    ║    • Roll model or similar needed to separate noise from true price moves      ║
    ║    • Hawkes processes may capture trade clustering                             ║
    ║                                                                                ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=9.5, fontfamily='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#333333'))
    
    return fig


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Generate tick-level microstructure ACF analysis as multi-page PDF."""
    print("\n" + "=" * 60)
    print("TICK-LEVEL MICROSTRUCTURE AUTOCORRELATION ANALYSIS")
    print("=" * 60 + "\n")
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tick data - NO aggregation
    df = load_tick_data(ASCII_FILE, n_ticks=TICK_SAMPLE_SIZE)
    
    print(f"\nTick-level statistics:")
    print(f"  Tick return mean: {df['log_return'].mean():.6f} bps")
    print(f"  Tick return std:  {df['log_return'].std():.4f} bps")
    print(f"  Up ticks:   {(df['direction'] > 0).sum():,}")
    print(f"  Down ticks: {(df['direction'] < 0).sum():,}")
    print(f"  No change:  {(df['direction'] == 0).sum():,}")
    
    output_path = PLOTS_DIR / "tick_microstructure_acf.pdf"
    
    print("\n" + "-" * 40)
    print("Generating tick-level plots...")
    print("-" * 40)
    
    with PdfPages(output_path) as pdf:
        print("  [1/9] Tick returns ACF")
        fig = plot_tick_returns_acf(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print("  [2/9] Tick returns PACF")
        fig = plot_tick_returns_pacf(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print("  [3/9] Direction ACF")
        fig = plot_direction_acf(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print("  [4/9] Squared tick returns ACF")
        fig = plot_squared_tick_returns_acf(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print("  [5/9] Spread dynamics ACF")
        fig = plot_spread_acf(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print("  [6/9] Inter-tick duration ACF")
        fig = plot_inter_tick_time_acf(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print("  [7/9] Bid/Ask changes ACF")
        fig = plot_bid_ask_cross_acf(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print("  [8/9] First lags detail")
        fig = plot_first_lags_detail(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Ljung-Box test skipped - too slow with 500k ticks
        
        print("  [9/9] Microstructure summary")
        fig = plot_microstructure_summary(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n{'=' * 60}")
    print(f"Saved: {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
