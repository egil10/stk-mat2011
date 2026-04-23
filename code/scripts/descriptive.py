
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

class DESCRIPTIVE:
    """
    High-level exploratory data analysis for pair tick data.
    Focuses on liquidity windows, return distributions, and volatility clusters.
    """
    def __init__(self, df, name_a="Asset A", name_b="Asset B"):
        self.df = df.copy()
        self.name_a = name_a
        self.name_b = name_b
        
        # Ensure timestamp is index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
            
        self.df['hour'] = self.df.index.hour
        self.df['day'] = self.df.index.dayofweek

    def run_liquidity_report(self):
        """Analyze spreads and volume across trading hours to find optimal windows."""
        hourly_spread = self.df.groupby('hour')[['HalfSpread_A_bps', 'HalfSpread_B_bps']].median()
        
        print(f"\n=== LIQUIDITY PROFILE (Median Spreads) ===")
        # Find the 4-hour window with the lowest average spread
        rolling_4h = hourly_spread.mean(axis=1).rolling(window=4).mean()
        best_hour = rolling_4h.idxmin()
        
        print(f"Tightest 4h Window: {int(best_hour-3)}:00 to {int(best_hour)}:00 UTC")
        print(f"Overall Median Spread: A={self.df['HalfSpread_A_bps'].median():.2f} | B={self.df['HalfSpread_B_bps'].median():.2f} bps")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        hourly_spread.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        ax.set_title("Median Bid-Ask Spread by Hour (UTC)")
        ax.set_ylabel("Basis Points")
        plt.grid(alpha=0.3)
        plt.show()

    def run_stats_report(self):
        """Analyze the distribution of log returns for fat tails (Kurtosis)."""
        rets_a = self.df['Return_A'].dropna()
        rets_b = self.df['Return_B'].dropna()
        
        stats = {
            self.name_a: [rets_a.mean()*10000, rets_a.std()*10000, skew(rets_a), kurtosis(rets_a)],
            self.name_b: [rets_b.mean()*10000, rets_b.std()*10000, skew(rets_b), kurtosis(rets_b)]
        }
        
        stats_df = pd.DataFrame(stats, index=['Mean (bps)', 'Vol (bps)', 'Skew', 'Kurtosis']).T
        
        print(f"\n=== RETURN STATISTICS (Log Scale) ===")
        print(stats_df.to_string())
        
        # If Kurtosis > 10, warn about the Markov HMM
        if stats_df['Kurtosis'].max() > 10:
            print(f"\n[!] WARNING: High Kurtosis detected (>10). Gaussian Markov Models (HMM) \n    will likely struggle with false-positive 'Danger' regimes.")

    def plot_volatility_clusters(self):
        """Visualize realized volatility clusters over time."""
        # 100-bar rolling vol
        vol_a = self.df['Return_A'].rolling(100).std() * 10000
        vol_b = self.df['Return_B'].rolling(100).std() * 10000
        
        plt.figure(figsize=(12, 4))
        plt.plot(vol_a, label=f"{self.name_a} Vol", alpha=0.7)
        plt.plot(vol_b, label=f"{self.name_b} Vol", alpha=0.7)
        plt.title("Rolling Realized Volatility (100-bar window)")
        plt.ylabel("Bps")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def generate_full_eda(self):
        self.run_liquidity_report()
        self.run_stats_report()
        self.plot_volatility_clusters()