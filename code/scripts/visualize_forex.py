"""
EUR/USD Tick Data Visualization

Loads high-frequency forex tick data and generates exploratory visualizations.
Run from project root: python code/scripts/visualize_forex.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ==============================================================================
# Configuration
# ==============================================================================

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "raw"
PLOTS_DIR = SCRIPT_DIR.parent / "plots"

# Data files
ASCII_FILE = DATA_DIR / "DAT_ASCII_EURUSD_T_202601.csv"
ASK_FILE = DATA_DIR / "DAT_NT_EURUSD_T_ASK_202601.csv"
BID_FILE = DATA_DIR / "DAT_NT_EURUSD_T_BID_202601.csv"

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'grid.alpha': 0.6,
    'grid.color': '#cccccc',
})

# Color palette
COLORS = {
    'bid': '#1a73e8',      # Google Blue
    'ask': '#ea4335',      # Google Red
    'spread': '#34a853',   # Google Green
    'volume': '#fbbc04',   # Google Yellow
    'mid': '#9c27b0',      # Purple
}


# ==============================================================================
# Data Loading
# ==============================================================================

def load_ascii_data(filepath: Path, sample_frac: float = 0.1) -> pd.DataFrame:
    """Load the combined bid/ask tick data from ASCII format."""
    print(f"Loading ASCII data from {filepath.name}...")
    
    # Read CSV - format: datetime,bid,ask,volume
    df = pd.read_csv(
        filepath,
        header=None,
        names=['datetime', 'bid', 'ask', 'volume'],
        dtype={'bid': float, 'ask': float, 'volume': int}
    )
    
    # Parse datetime (format: YYYYMMDD HHMMSSmmm)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S%f')
    df.set_index('datetime', inplace=True)
    
    # Calculate derived features
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['spread'] = (df['ask'] - df['bid']) * 10000  # Convert to pips
    
    # Sample for faster processing
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).sort_index()
        print(f"  Sampled {len(df):,} ticks ({sample_frac*100:.0f}%)")
    
    print(f"  Loaded {len(df):,} ticks from {df.index.min()} to {df.index.max()}")
    return df


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_price_overview(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot 1: Price overview with bid, ask, and mid prices."""
    # Resample to reduce noise for overview
    hourly = df.resample('1h').agg({'bid': 'mean', 'ask': 'mean', 'mid': 'mean'}).dropna()
    
    ax.fill_between(hourly.index, hourly['bid'], hourly['ask'], 
                    alpha=0.2, color=COLORS['spread'], label='Bid-Ask Spread')
    ax.plot(hourly.index, hourly['mid'], color=COLORS['mid'], 
            linewidth=1.2, label='Mid Price', alpha=0.9)
    ax.plot(hourly.index, hourly['bid'], color=COLORS['bid'], 
            linewidth=0.6, alpha=0.6, label='Bid')
    ax.plot(hourly.index, hourly['ask'], color=COLORS['ask'], 
            linewidth=0.6, alpha=0.6, label='Ask')
    
    ax.set_ylabel('EUR/USD Rate')
    ax.set_title('EUR/USD Price Overview (January 2026)')
    ax.legend(loc='upper right', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))


def plot_spread_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot 2: Distribution of bid-ask spread."""
    spread_data = df['spread']
    
    # Remove outliers for better visualization
    q99 = spread_data.quantile(0.99)
    spread_filtered = spread_data[spread_data <= q99]
    
    ax.hist(spread_filtered, bins=50, color=COLORS['spread'], 
            alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvline(spread_filtered.median(), color=COLORS['mid'], 
               linestyle='--', linewidth=2, label=f'Median: {spread_filtered.median():.2f} pips')
    ax.axvline(spread_filtered.mean(), color=COLORS['ask'], 
               linestyle=':', linewidth=2, label=f'Mean: {spread_filtered.mean():.2f} pips')
    
    ax.set_xlabel('Spread (pips)')
    ax.set_ylabel('Frequency')
    ax.set_title('Bid-Ask Spread Distribution')
    ax.legend(loc='upper right', fontsize=9)


def plot_intraday_spread(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot 3: Intraday spread pattern by hour."""
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    hourly_spread = df_copy.groupby('hour')['spread'].agg(['mean', 'std'])
    
    hours = hourly_spread.index
    means = hourly_spread['mean']
    stds = hourly_spread['std']
    
    ax.bar(hours, means, color=COLORS['spread'], alpha=0.7, 
           edgecolor='white', linewidth=0.5, label='Mean Spread')
    ax.errorbar(hours, means, yerr=stds, fmt='none', 
                color=COLORS['mid'], capsize=3, capthick=1, alpha=0.7)
    
    # Mark trading sessions
    ax.axvspan(8, 16, alpha=0.1, color=COLORS['bid'], label='London Session')
    ax.axvspan(13, 21, alpha=0.1, color=COLORS['ask'], label='New York Session')
    
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Spread (pips)')
    ax.set_title('Average Spread by Hour of Day')
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc='upper right', fontsize=8)


def plot_returns_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot 4: Distribution of mid-price returns."""
    # Calculate log returns (resampled to reduce noise)
    minutely = df['mid'].resample('1min').last().dropna()
    returns = np.log(minutely / minutely.shift(1)).dropna() * 10000  # In basis points
    
    # Remove extreme outliers
    q01, q99 = returns.quantile(0.01), returns.quantile(0.99)
    returns_filtered = returns[(returns >= q01) & (returns <= q99)]
    
    ax.hist(returns_filtered, bins=100, color=COLORS['bid'], 
            alpha=0.7, edgecolor='white', linewidth=0.3, density=True)
    
    # Overlay normal distribution for comparison
    mu, sigma = returns_filtered.mean(), returns_filtered.std()
    x = np.linspace(returns_filtered.min(), returns_filtered.max(), 100)
    normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal_pdf, color=COLORS['ask'], linewidth=2, 
            label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    
    ax.set_xlabel('Returns (basis points)')
    ax.set_ylabel('Density')
    ax.set_title('1-Minute Log Returns Distribution')
    ax.legend(loc='upper right', fontsize=9)


def plot_tick_frequency(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot 5: Tick frequency over time (trading activity)."""
    tick_counts = df.resample('1h').size()
    
    ax.fill_between(tick_counts.index, tick_counts.values, 
                    alpha=0.5, color=COLORS['volume'])
    ax.plot(tick_counts.index, tick_counts.values, 
            color=COLORS['volume'], linewidth=1, alpha=0.8)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Ticks per Hour')
    ax.set_title('Trading Activity (Tick Frequency)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))


def plot_volatility_by_day(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot 6: Daily volatility (realized variance)."""
    # Calculate 5-minute returns and daily realized variance
    minutely_5 = df['mid'].resample('5min').last().dropna()
    returns = np.log(minutely_5 / minutely_5.shift(1)).dropna()
    
    daily_vol = returns.groupby(returns.index.date).std() * np.sqrt(288) * 100  # Annualized %
    
    colors = [COLORS['bid'] if v < daily_vol.median() else COLORS['ask'] for v in daily_vol.values]
    
    bars = ax.bar(range(len(daily_vol)), daily_vol.values, color=colors, 
                  alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axhline(daily_vol.median(), color=COLORS['mid'], linestyle='--', 
               linewidth=2, label=f'Median: {daily_vol.median():.2f}%')
    
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Volatility (annualized %)')
    ax.set_title('Daily Realized Volatility')
    ax.legend(loc='upper right', fontsize=9)
    
    # Set x-tick labels (every 5 days)
    tick_positions = range(0, len(daily_vol), 5)
    tick_labels = [str(daily_vol.index[i]) for i in tick_positions if i < len(daily_vol)]
    ax.set_xticks(list(tick_positions)[:len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Generate all visualizations and export to PDF."""
    print("\n" + "=" * 60)
    print("EUR/USD TICK DATA VISUALIZATION")
    print("=" * 60 + "\n")
    
    # Ensure plots directory exists
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data (10% sample for faster processing)
    df = load_ascii_data(ASCII_FILE, sample_frac=0.1)
    
    print("\n" + "-" * 40)
    print("Data Summary:")
    print("-" * 40)
    print(f"  Time range: {df.index.min()} to {df.index.max()}")
    print(f"  Total ticks: {len(df):,}")
    print(f"  Bid range: {df['bid'].min():.5f} - {df['bid'].max():.5f}")
    print(f"  Ask range: {df['ask'].min():.5f} - {df['ask'].max():.5f}")
    print(f"  Avg spread: {df['spread'].mean():.3f} pips")
    
    # Create figure with 6 subplots (3x2 grid)
    print("\n" + "-" * 40)
    print("Generating visualizations...")
    print("-" * 40)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('EUR/USD High-Frequency Tick Data Analysis\nJanuary 2026', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Generate each plot
    plot_price_overview(df, axes[0, 0])
    print("  [1/6] Price overview")
    
    plot_spread_distribution(df, axes[0, 1])
    print("  [2/6] Spread distribution")
    
    plot_intraday_spread(df, axes[1, 0])
    print("  [3/6] Intraday spread pattern")
    
    plot_returns_distribution(df, axes[1, 1])
    print("  [4/6] Returns distribution")
    
    plot_tick_frequency(df, axes[2, 0])
    print("  [5/6] Tick frequency")
    
    plot_volatility_by_day(df, axes[2, 1])
    print("  [6/6] Daily volatility")
    
    # Finalize layout
    plt.tight_layout()
    
    # Save to PDF
    output_path = PLOTS_DIR / "eurusd_tick_analysis.pdf"
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\n{'=' * 60}")
    print(f"Saved: {output_path}")
    print(f"{'=' * 60}\n")
    
    plt.close(fig)
    
    return df


if __name__ == "__main__":
    df = main()
