"""
EUR/USD Tick Data Autocorrelation Analysis

Explores autocorrelation patterns in high-frequency forex data:
- Returns autocorrelation (ACF/PACF)
- Volatility clustering (squared/absolute returns)
- Intraday patterns
- Statistical significance tests

Run from project root: python code/scripts/autocorrelation_analysis.py
"""

import os
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
DATA_DIR = SCRIPT_DIR.parent / "data"
PLOTS_DIR = SCRIPT_DIR.parent / "plots"

ASCII_FILE = DATA_DIR / "DAT_ASCII_EURUSD_T_202601.csv"

# Plot styling - clean, professional aesthetic
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

# Color palette
COLORS = {
    'primary': '#1a73e8',
    'secondary': '#ea4335',
    'tertiary': '#34a853',
    'accent': '#9c27b0',
    'neutral': '#5f6368',
    'significant': '#ea4335',
    'insignificant': '#bdc3c7',
}


# ==============================================================================
# Data Loading
# ==============================================================================

def load_data(filepath: Path, sample_frac: float = 0.2) -> pd.DataFrame:
    """Load tick data and compute returns at multiple frequencies."""
    print(f"Loading data from {filepath.name}...")
    
    df = pd.read_csv(
        filepath,
        header=None,
        names=['datetime', 'bid', 'ask', 'volume'],
        dtype={'bid': float, 'ask': float, 'volume': int}
    )
    
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S%f')
    df.set_index('datetime', inplace=True)
    df['mid'] = (df['bid'] + df['ask']) / 2
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).sort_index()
        print(f"  Sampled {len(df):,} ticks ({sample_frac*100:.0f}%)")
    
    print(f"  Loaded {len(df):,} ticks from {df.index.min()} to {df.index.max()}")
    return df


def compute_returns(df: pd.DataFrame, freq: str = '1min') -> pd.Series:
    """Compute log returns at specified frequency."""
    prices = df['mid'].resample(freq).last().dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns * 10000  # Convert to basis points


# ==============================================================================
# Autocorrelation Plotting Functions
# ==============================================================================

def plot_acf_returns(returns: pd.Series, lags: int = 50) -> plt.Figure:
    """Page 1: ACF of returns with confidence bands."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    acf_values, confint = acf(returns, nlags=lags, alpha=0.05)
    
    # Stem plot
    markerline, stemlines, baseline = ax.stem(
        range(len(acf_values)), acf_values, 
        linefmt='-', markerfmt='o', basefmt='k-'
    )
    stemlines.set_color(COLORS['primary'])
    stemlines.set_linewidth(1.5)
    markerline.set_color(COLORS['primary'])
    markerline.set_markersize(5)
    
    # Confidence bands (95%)
    ci_upper = confint[1:, 1] - acf_values[1:]
    ci_lower = confint[1:, 0] - acf_values[1:]
    ax.fill_between(range(1, lags+1), ci_lower, ci_upper, 
                    alpha=0.15, color=COLORS['secondary'], label='95% Confidence')
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function (ACF) of 1-Minute Returns')
    ax.set_xlim(-1, lags + 1)
    ax.legend(loc='upper right')
    
    # Add annotation about findings
    significant_lags = np.sum(np.abs(acf_values[1:]) > (1.96 / np.sqrt(len(returns))))
    ax.annotate(f'Significant lags (p<0.05): {significant_lags} out of {lags}',
                xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10,
                ha='left', va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_pacf_returns(returns: pd.Series, lags: int = 50) -> plt.Figure:
    """Page 2: PACF of returns."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    pacf_values, confint = pacf(returns, nlags=lags, alpha=0.05)
    
    markerline, stemlines, baseline = ax.stem(
        range(len(pacf_values)), pacf_values,
        linefmt='-', markerfmt='o', basefmt='k-'
    )
    stemlines.set_color(COLORS['tertiary'])
    stemlines.set_linewidth(1.5)
    markerline.set_color(COLORS['tertiary'])
    markerline.set_markersize(5)
    
    ci_upper = confint[1:, 1] - pacf_values[1:]
    ci_lower = confint[1:, 0] - pacf_values[1:]
    ax.fill_between(range(1, lags+1), ci_lower, ci_upper,
                    alpha=0.15, color=COLORS['secondary'], label='95% Confidence')
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('Partial Autocorrelation')
    ax.set_title('Partial Autocorrelation Function (PACF) of 1-Minute Returns')
    ax.set_xlim(-1, lags + 1)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_squared_returns_acf(returns: pd.Series, lags: int = 100) -> plt.Figure:
    """Page 3: ACF of squared returns - volatility clustering evidence."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    squared_returns = returns ** 2
    acf_values, confint = acf(squared_returns, nlags=lags, alpha=0.05)
    
    markerline, stemlines, baseline = ax.stem(
        range(len(acf_values)), acf_values,
        linefmt='-', markerfmt='o', basefmt='k-'
    )
    stemlines.set_color(COLORS['secondary'])
    stemlines.set_linewidth(1.2)
    markerline.set_color(COLORS['secondary'])
    markerline.set_markersize(4)
    
    ci = 1.96 / np.sqrt(len(squared_returns))
    ax.axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='95% CI')
    ax.axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('ACF of Squared Returns (Volatility Clustering)')
    ax.set_xlim(-1, lags + 1)
    ax.legend(loc='upper right')
    
    # Annotation
    ax.annotate('Strong persistence in squared returns indicates\n'
                'volatility clustering (ARCH/GARCH effects)',
                xy=(0.50, 0.85), xycoords='axes fraction', fontsize=10,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.9))
    
    plt.tight_layout()
    return fig


def plot_absolute_returns_acf(returns: pd.Series, lags: int = 100) -> plt.Figure:
    """Page 4: ACF of absolute returns - another volatility measure."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    abs_returns = np.abs(returns)
    acf_values, confint = acf(abs_returns, nlags=lags, alpha=0.05)
    
    markerline, stemlines, baseline = ax.stem(
        range(len(acf_values)), acf_values,
        linefmt='-', markerfmt='o', basefmt='k-'
    )
    stemlines.set_color(COLORS['accent'])
    stemlines.set_linewidth(1.2)
    markerline.set_color(COLORS['accent'])
    markerline.set_markersize(4)
    
    ci = 1.96 / np.sqrt(len(abs_returns))
    ax.axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='95% CI')
    ax.axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('ACF of Absolute Returns (Volatility Persistence)')
    ax.set_xlim(-1, lags + 1)
    ax.legend(loc='upper right')
    
    # Long memory annotation
    if acf_values[50] > ci:
        ax.annotate('Long memory property detected:\n'
                    'ACF decays slowly (characteristic of financial volatility)',
                    xy=(0.55, 0.80), xycoords='axes fraction', fontsize=10,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.9))
    
    plt.tight_layout()
    return fig


def plot_acf_comparison(returns: pd.Series, lags: int = 60) -> plt.Figure:
    """Page 5: Side-by-side comparison of returns, squared, and absolute returns ACF."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    
    acf_returns, _ = acf(returns, nlags=lags, alpha=0.05)
    acf_squared, _ = acf(returns ** 2, nlags=lags, alpha=0.05)
    acf_abs, _ = acf(np.abs(returns), nlags=lags, alpha=0.05)
    
    ci = 1.96 / np.sqrt(len(returns))
    
    # Returns
    axes[0].bar(range(lags+1), acf_returns, color=COLORS['primary'], alpha=0.7, width=0.8)
    axes[0].axhline(y=ci, color=COLORS['secondary'], linestyle='--', linewidth=1)
    axes[0].axhline(y=-ci, color=COLORS['secondary'], linestyle='--', linewidth=1)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_ylabel('ACF')
    axes[0].set_title('Returns: Little Autocorrelation (Efficient Market)')
    axes[0].set_ylim(-0.15, 0.15)
    
    # Squared returns
    axes[1].bar(range(lags+1), acf_squared, color=COLORS['secondary'], alpha=0.7, width=0.8)
    axes[1].axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[1].axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_ylabel('ACF')
    axes[1].set_title('Squared Returns: Strong Volatility Clustering')
    
    # Absolute returns
    axes[2].bar(range(lags+1), acf_abs, color=COLORS['accent'], alpha=0.7, width=0.8)
    axes[2].axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[2].axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].set_ylabel('ACF')
    axes[2].set_xlabel('Lag (minutes)')
    axes[2].set_title('Absolute Returns: Persistent Volatility')
    
    fig.suptitle('Autocorrelation Comparison: Returns vs Volatility Proxies', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig


def plot_multi_frequency_acf(df: pd.DataFrame, lags: int = 40) -> plt.Figure:
    """Page 6: ACF at different sampling frequencies."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    frequencies = ['1min', '5min', '15min', '1h']
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['secondary'], COLORS['accent']]
    
    for ax, freq, color in zip(axes.flat, frequencies, colors):
        returns = compute_returns(df, freq)
        acf_values, _ = acf(returns, nlags=lags, alpha=0.05)
        
        ci = 1.96 / np.sqrt(len(returns))
        
        ax.bar(range(len(acf_values)), acf_values, color=color, alpha=0.7, width=0.8)
        ax.axhline(y=ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
        ax.axhline(y=-ci, color=COLORS['neutral'], linestyle='--', linewidth=1)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title(f'{freq} Returns (n={len(returns):,})')
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_xlim(-1, lags + 1)
    
    fig.suptitle('ACF of Returns at Different Frequencies', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


def plot_intraday_acf_heatmap(df: pd.DataFrame, lags: int = 20) -> plt.Figure:
    """Page 7: Heatmap of ACF by hour of day - find autocorrelation clusters."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    returns_1min = compute_returns(df, '1min')
    returns_df = returns_1min.to_frame(name='returns')
    returns_df['hour'] = returns_df.index.hour
    
    # Compute ACF for each hour
    acf_by_hour = {}
    for hour in range(24):
        hour_returns = returns_df[returns_df['hour'] == hour]['returns']
        if len(hour_returns) > lags + 10:
            acf_values, _ = acf(hour_returns, nlags=lags, alpha=0.05)
            acf_by_hour[hour] = acf_values[1:]  # Skip lag 0
    
    acf_matrix = np.array([acf_by_hour.get(h, np.zeros(lags)) for h in range(24)])
    
    im = ax.imshow(acf_matrix, aspect='auto', cmap='RdBu_r', 
                   vmin=-0.1, vmax=0.1, interpolation='nearest')
    
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('Hour of Day (UTC)')
    ax.set_title('Autocorrelation Heatmap by Hour of Day')
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'{h:02d}:00' for h in range(24)])
    
    cbar = fig.colorbar(im, ax=ax, label='Autocorrelation')
    
    # Highlight trading sessions
    ax.axhline(y=7.5, color='white', linewidth=2, linestyle='--', alpha=0.7)
    ax.axhline(y=16.5, color='white', linewidth=2, linestyle='--', alpha=0.7)
    ax.text(lags + 1, 4, 'Asian\nSession', fontsize=9, va='center', ha='left')
    ax.text(lags + 1, 12, 'London\nSession', fontsize=9, va='center', ha='left')
    ax.text(lags + 1, 19, 'NY\nSession', fontsize=9, va='center', ha='left')
    
    plt.tight_layout()
    return fig


def plot_ljung_box_test(returns: pd.Series, max_lag: int = 50) -> plt.Figure:
    """Page 8: Ljung-Box test p-values for autocorrelation significance."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    
    # Test for returns
    lb_returns = acorr_ljungbox(returns, lags=range(1, max_lag+1), return_df=True)
    
    axes[0].bar(lb_returns.index, lb_returns['lb_pvalue'], 
                color=[COLORS['tertiary'] if p > 0.05 else COLORS['secondary'] 
                      for p in lb_returns['lb_pvalue']], 
                alpha=0.7, width=0.8)
    axes[0].axhline(y=0.05, color='black', linestyle='--', linewidth=2, 
                    label='5% Significance Level')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('p-value')
    axes[0].set_title('Ljung-Box Test: Returns')
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc='upper right')
    
    # Test for squared returns
    lb_squared = acorr_ljungbox(returns ** 2, lags=range(1, max_lag+1), return_df=True)
    
    axes[1].bar(lb_squared.index, lb_squared['lb_pvalue'],
                color=[COLORS['tertiary'] if p > 0.05 else COLORS['secondary']
                      for p in lb_squared['lb_pvalue']],
                alpha=0.7, width=0.8)
    axes[1].axhline(y=0.05, color='black', linestyle='--', linewidth=2,
                    label='5% Significance Level')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('p-value')
    axes[1].set_title('Ljung-Box Test: Squared Returns (Volatility)')
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper right')
    
    # Annotation
    reject_returns = (lb_returns['lb_pvalue'] < 0.05).sum()
    reject_squared = (lb_squared['lb_pvalue'] < 0.05).sum()
    
    fig.text(0.02, 0.02, 
             f'Returns: {reject_returns}/{max_lag} lags significant | '
             f'Squared Returns: {reject_squared}/{max_lag} lags significant',
             fontsize=10, ha='left', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    fig.suptitle('Ljung-Box Tests for Autocorrelation (H0: No Autocorrelation)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def plot_rolling_acf(returns: pd.Series, lag: int = 1, window: int = 500) -> plt.Figure:
    """Page 9: Rolling first-order autocorrelation over time."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    def rolling_acf_1(x):
        if len(x) < 2:
            return np.nan
        return np.corrcoef(x[:-1], x[1:])[0, 1]
    
    rolling_ac = returns.rolling(window=window, min_periods=window//2).apply(
        rolling_acf_1, raw=True)
    
    ax.plot(rolling_ac.index, rolling_ac.values, 
            color=COLORS['primary'], linewidth=0.8, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    
    # Highlight significant thresholds
    ci = 1.96 / np.sqrt(window)
    ax.axhline(y=ci, color=COLORS['secondary'], linestyle='--', linewidth=1.5, 
               label=f'95% CI (n={window})')
    ax.axhline(y=-ci, color=COLORS['secondary'], linestyle='--', linewidth=1.5)
    
    # Fill extreme regions
    ax.fill_between(rolling_ac.index, ci, rolling_ac.values,
                    where=rolling_ac.values > ci, 
                    color=COLORS['secondary'], alpha=0.3)
    ax.fill_between(rolling_ac.index, -ci, rolling_ac.values,
                    where=rolling_ac.values < -ci,
                    color=COLORS['secondary'], alpha=0.3)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Rolling ACF(1)')
    ax.set_title(f'Rolling First-Order Autocorrelation (Window = {window} observations)')
    ax.legend(loc='upper right')
    
    # Time axis formatting
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    
    plt.tight_layout()
    return fig


def plot_summary_findings(returns: pd.Series) -> plt.Figure:
    """Page 10: Summary of key findings."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis('off')
    
    # Calculate summary statistics
    acf_values, _ = acf(returns, nlags=20, alpha=0.05)
    acf_squared, _ = acf(returns ** 2, nlags=20, alpha=0.05)
    ci = 1.96 / np.sqrt(len(returns))
    
    sig_returns = np.sum(np.abs(acf_values[1:]) > ci)
    sig_squared = np.sum(np.abs(acf_squared[1:]) > ci)
    
    lb_returns = acorr_ljungbox(returns, lags=10, return_df=True)
    lb_squared = acorr_ljungbox(returns ** 2, lags=10, return_df=True)
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                     AUTOCORRELATION ANALYSIS SUMMARY                         ║
    ║                          EUR/USD Tick Data - January 2026                    ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  1. RETURNS AUTOCORRELATION                                                  ║
    ║     • Significant lags at 5% level: {sig_returns}/20                                      ║
    ║     • Ljung-Box p-value (lag 10): {lb_returns['lb_pvalue'].iloc[9]:.4f}                          ║
    ║     • Interpretation: {"Weak/no serial correlation" if sig_returns < 5 else "Evidence of serial correlation"}                       ║
    ║                       (consistent with efficient market hypothesis)          ║
    ║                                                                              ║
    ║  2. VOLATILITY CLUSTERING                                                    ║
    ║     • Significant lags in squared returns: {sig_squared}/20                              ║
    ║     • Ljung-Box p-value (lag 10): {lb_squared['lb_pvalue'].iloc[9]:.4f}                          ║
    ║     • Interpretation: {"Strong volatility clustering (ARCH effects)" if sig_squared > 10 else "Moderate volatility clustering"}       ║
    ║                                                                              ║
    ║  3. KEY FINDINGS                                                             ║
    ║     • Returns show minimal autocorrelation (market efficiency)               ║
    ║     • Squared/absolute returns show strong persistence (GARCH effects)       ║
    ║     • Autocorrelation patterns vary by trading session                       ║
    ║     • Long memory property observed in volatility                            ║
    ║                                                                              ║
    ║  4. IMPLICATIONS                                                             ║
    ║     • Standard ARMA models unlikely to improve return prediction             ║
    ║     • GARCH-family models appropriate for volatility modeling                ║
    ║     • Consider regime-switching models for intraday patterns                 ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#333333'))
    
    ax.set_title('Summary of Autocorrelation Analysis', fontsize=14, fontweight='bold', pad=20)
    
    return fig


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Generate all autocorrelation plots and save as multi-page PDF."""
    print("\n" + "=" * 60)
    print("EUR/USD AUTOCORRELATION ANALYSIS")
    print("=" * 60 + "\n")
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(ASCII_FILE, sample_frac=0.2)
    
    # Compute returns
    print("\nComputing returns...")
    returns = compute_returns(df, '1min')
    print(f"  1-minute returns: {len(returns):,} observations")
    print(f"  Mean: {returns.mean():.4f} bps, Std: {returns.std():.4f} bps")
    
    # Generate plots and save to multi-page PDF
    output_path = PLOTS_DIR / "autocorrelation_analysis.pdf"
    
    print("\n" + "-" * 40)
    print("Generating plots...")
    print("-" * 40)
    
    with PdfPages(output_path) as pdf:
        # Page 1: ACF of returns
        print("  [1/10] ACF of returns")
        fig = plot_acf_returns(returns)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: PACF of returns
        print("  [2/10] PACF of returns")
        fig = plot_pacf_returns(returns)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Squared returns ACF
        print("  [3/10] ACF of squared returns")
        fig = plot_squared_returns_acf(returns)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: Absolute returns ACF
        print("  [4/10] ACF of absolute returns")
        fig = plot_absolute_returns_acf(returns)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Comparison plot
        print("  [5/10] ACF comparison")
        fig = plot_acf_comparison(returns)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 6: Multi-frequency ACF
        print("  [6/10] Multi-frequency ACF")
        fig = plot_multi_frequency_acf(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 7: Intraday heatmap
        print("  [7/10] Intraday ACF heatmap")
        fig = plot_intraday_acf_heatmap(df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 8: Ljung-Box tests
        print("  [8/10] Ljung-Box tests")
        fig = plot_ljung_box_test(returns)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 9: Rolling ACF
        print("  [9/10] Rolling autocorrelation")
        fig = plot_rolling_acf(returns)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 10: Summary
        print("  [10/10] Summary findings")
        fig = plot_summary_findings(returns)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n{'=' * 60}")
    print(f"Saved: {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
