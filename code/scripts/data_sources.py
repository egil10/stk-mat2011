
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import matplotlib.dates as mdates

# Optimization: Speed up date parsing
pd.options.mode.chained_assignment = None 

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROCESS_DIR = PROJECT_ROOT / "code" / "data" / "processed"
EDA_DIR = PROJECT_ROOT / "code" / "plots" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prep():
    print("Loading datasets...")
    # HistData (EST: UTC-5)
    hd = pd.read_parquet(PROCESS_DIR / "eurusd_bid_202601.parquet")
    hd['datetime'] = pd.to_datetime(hd['datetime']).dt.tz_localize(None) + pd.Timedelta(hours=5)
    
    # TrueFX (UTC)
    tfx = pd.read_parquet(PROCESS_DIR / "eurusd_truefx_bid_202601.parquet")
    tfx['datetime'] = pd.to_datetime(tfx['datetime']).dt.tz_localize(None)
    
    # Dukascopy (UTC)
    dk = pd.read_parquet(PROCESS_DIR / "eurusd_dukascopy_bid_202601.parquet")
    dk['datetime'] = pd.to_datetime(dk['datetime']).dt.tz_localize(None)
    
    return hd, tfx, dk

def page_overview(pdf, hd, tfx, dk):
    print("Generating Page 1: Overview...")
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    
    # Table Stats
    ax_table = axes[0]
    ax_table.axis('off')
    stats = []
    for name, df in [("HistData", hd), ("TrueFX", tfx), ("Dukascopy", dk)]:
        stats.append([
            name, f"{len(df):,}",
            df['datetime'].min().strftime('%Y-%m-%d'),
            df['price'].mean().__round__(5),
            f"{df['price'].std()*10000:.2f}"
        ])
    
    table = ax_table.table(
        cellText=stats, 
        colLabels=["Source", "Ticks", "Start", "Mean", "Vol (pips)"], 
        loc='center', cellLoc='center'
    )
    table.set_fontsize(12)
    table.scale(1, 4)
    ax_table.set_title("EUR/USD Bid Provider Comparison (Jan 2026)", fontsize=16, pad=20)

    # Activity Heatmap (New plot instead of density line)
    ax_heat = axes[1]
    dk['hour'] = dk['datetime'].dt.hour
    dk['dow'] = dk['datetime'].dt.dayofweek
    pivot = dk.groupby(['dow', 'hour']).size().unstack(fill_value=0)
    
    im = ax_heat.imshow(pivot, cmap='YlGnBu', aspect='auto')
    ax_heat.set_title("Market Activity Heatmap (Dukascopy Ticks)", fontsize=14)
    ax_heat.set_xlabel("Hour of Day (UTC)")
    ax_heat.set_ylabel("Day of Week (0=Mon)")
    plt.colorbar(im, ax=ax_heat, label='Ticks')
    
    plt.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close()

def page_interarrivals(pdf, hd, tfx, dk):
    print("Generating Page 2: Inter-arrival Dynamics...")
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    
    # Binned Inter-arrival histogram
    ax_hist = axes[0]
    for d, label, color in [(dk, 'Dukascopy', '#1f77b4'), (tfx, 'TrueFX', '#d62728')]:
        durations = d['datetime'].diff().dt.total_seconds().dropna()
        durations = durations[durations < 10]
        ax_hist.hist(durations, bins=100, alpha=0.5, label=label, density=True, color=color)
    
    ax_hist.set_title("Inter-Tick Duration Distribution (< 10s)", fontsize=14)
    ax_hist.set_xlabel("Seconds")
    ax_hist.legend()

    # Sampled CDF (Fixed: Reduced points)
    ax_cdf = axes[1]
    for d, label, color in [(dk, 'Dukascopy', 'blue'), (hd, 'HistData', 'orange'), (tfx, 'TrueFX', 'red')]:
        durations = d['datetime'].diff().dt.total_seconds().dropna()
        durations = durations[durations < 60]
        # Use only 1000 points for the CDF curve
        sorted_d = np.sort(durations)
        if len(sorted_d) > 1000:
            indices = np.linspace(0, len(sorted_d)-1, 1000).astype(int)
            plot_d = sorted_d[indices]
            plot_y = np.linspace(0, 1, 1000)
        else:
            plot_d = sorted_d
            plot_y = np.arange(len(sorted_d)) / len(sorted_d)
        
        ax_cdf.plot(plot_d, plot_y, label=label, color=color, alpha=0.8)
    
    ax_cdf.set_title("Sampled Cumulative Distribution (Update Speed)", fontsize=14)
    ax_cdf.set_xlim(0, 5)
    ax_cdf.set_ylabel("Probability")
    ax_cdf.legend()
    ax_cdf.grid(alpha=0.3)

    plt.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close()

def page_zooms(pdf, hd, tfx, dk):
    print("Generating Page 3: Latency Zooms...")
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    
    # 5 minute window
    start = pd.Timestamp("2026-01-15 15:00:00")
    end = start + pd.Timedelta(minutes=5)
    
    for ax, duration, title in [(axes[0], 300, "5 Minute Alignment"), (axes[1], 10, "10 Second Synchronicity")]:
        curr_end = start + pd.Timedelta(seconds=duration)
        mask_dk = (dk['datetime'] >= start) & (dk['datetime'] <= curr_end)
        mask_hd = (hd['datetime'] >= start) & (hd['datetime'] <= curr_end)
        mask_tfx = (tfx['datetime'] >= start) & (tfx['datetime'] <= curr_end)
        
        ax.step(dk.loc[mask_dk, 'datetime'], dk.loc[mask_dk, 'price'], label='Dukascopy', where='post', alpha=0.8)
        ax.step(hd.loc[mask_hd, 'datetime'], hd.loc[mask_hd, 'price'], label='HistData', where='post', linestyle='--', alpha=0.6)
        ax.step(tfx.loc[mask_tfx, 'datetime'], tfx.loc[mask_tfx, 'price'], label='TrueFX', where='post', color='red', alpha=0.8)
        
        ax.set_title(title, fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.legend(prop={'size': 8})
        ax.grid(alpha=0.2)

    plt.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close()

def page_discrepancies(pdf, hd, tfx, dk):
    print("Generating Page 4: Price Deviation...")
    # Resample to 1min for month-wide deviation analysis
    dk_m = dk.set_index('datetime')['price'].resample('1min').last().ffill()
    tfx_m = tfx.set_index('datetime')['price'].resample('1min').last().ffill()
    
    diff = (tfx_m - dk_m) * 10000 # Deviation in pips
    
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.plot(diff.index, diff.values, color='red', alpha=0.5, linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title("TrueFX vs Dukascopy: Price Deviation (1-min frequency)", fontsize=14)
    ax.set_ylabel("Difference (Pips)")
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 5)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def page_volatility_correl(pdf, hd, tfx, dk):
    print("Generating Page 5: Volatility & Correlation...")
    # 1-hour realized vol
    def get_vol(df):
        return df.set_index('datetime')['price'].resample('1h').last().ffill().pct_change().rolling(24).std()
    
    vol_dk = get_vol(dk)
    vol_tfx = get_vol(tfx)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.plot(vol_dk.index, vol_dk.values, label='Dukascopy Vol (24h Rolling)')
    ax.plot(vol_tfx.index, vol_tfx.values, label='TrueFX Vol', alpha=0.7, color='red')
    
    ax.set_title("Rolling Volatility Comparison", fontsize=14)
    ax.set_ylabel("Volatility (Std Dev of returns)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def main():
    hd, tfx, dk = load_and_prep()
    pdf_path = EDA_DIR / "data_sources.pdf"
    
    with PdfPages(pdf_path) as pdf:
        page_overview(pdf, hd, tfx, dk)
        page_interarrivals(pdf, hd, tfx, dk)
        page_zooms(pdf, hd, tfx, dk)
        page_discrepancies(pdf, hd, tfx, dk)
        page_volatility_correl(pdf, hd, tfx, dk)
        
    print(f"\nSUCCESS: Multi-page analytics PDF saved to {pdf_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
