
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from scipy.stats import norm
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class SPREAD:
    """
    Ingests high-frequency tick data and outputs synchronized volume or tick
    bars for pairs trading. Now carries bid/ask explicitly so downstream can
    model half-spread slippage. Includes a diagnostic plotting suite.
    """
    def __init__(self, agg_type='volume', threshold=1000,
                 active_days=None, active_hours=(10, 14)):
        if agg_type not in ('volume', 'tick'):
            raise ValueError("agg_type must be 'volume' or 'tick'")
        self.agg_type = agg_type
        self.threshold = threshold
        self.active_days = active_days if active_days is not None else [0, 1, 2, 3, 4]
        self.active_hours = active_hours  # (start_hour_inclusive, end_hour_exclusive)
        self.data = None

    def _load_parquet(self, file_paths):
        if isinstance(file_paths, list):
            return pd.concat([pd.read_parquet(fp) for fp in file_paths])
        return pd.read_parquet(file_paths)

    def _apply_session_filter(self, df):
        """Use self.active_days and self.active_hours. Hour filter is [start, end)."""
        start_h, end_h = self.active_hours
        mask = (
            df['datetime'].dt.dayofweek.isin(self.active_days)
            & (df['datetime'].dt.hour >= start_h)
            & (df['datetime'].dt.hour < end_h)
        )
        return df[mask]

    def _aggregate_bars(self, ask_file, bid_file):
        df_ask = (self._load_parquet(ask_file)
                  .sort_values('datetime')
                  .rename(columns={'price': 'ask_price', 'volume': 'ask_volume'}))
        df_bid = (self._load_parquet(bid_file)
                  .sort_values('datetime')
                  .rename(columns={'price': 'bid_price', 'volume': 'bid_volume'}))

        df_ask = self._apply_session_filter(df_ask)
        df_bid = self._apply_session_filter(df_bid)

        df_ticks = pd.merge_asof(
            df_ask[['datetime', 'ask_price', 'ask_volume']],
            df_bid[['datetime', 'bid_price', 'bid_volume']],
            on='datetime',
            direction='backward'
        ).dropna()

        df_ticks['mid_price'] = (df_ticks['bid_price'] + df_ticks['ask_price']) / 2.0
        df_ticks['total_volume'] = df_ticks['bid_volume'] + df_ticks['ask_volume']

        if self.agg_type == 'volume':
            df_ticks['cum_volume'] = df_ticks['total_volume'].cumsum()
            df_ticks['bar_id'] = (df_ticks['cum_volume'] // self.threshold).astype(int)
        else:  # 'tick'
            df_ticks['bar_id'] = (np.arange(len(df_ticks)) // self.threshold).astype(int)

        # Keep bid/ask on the bar as the last observation within the bar
        bars = df_ticks.groupby('bar_id').agg(
            timestamp=('datetime', 'last'),
            close=('mid_price', 'last'),
            bid=('bid_price', 'last'),
            ask=('ask_price', 'last'),
        ).set_index('timestamp').sort_index()  # defensive sort
        return bars

    def build(self, file_paths):
        if len(file_paths) != 4:
            raise ValueError("Provide exactly 4 file paths: [ask_a, bid_a, ask_b, bid_b]")

        bars_a = self._aggregate_bars(file_paths[0], file_paths[1])
        bars_b = self._aggregate_bars(file_paths[2], file_paths[3])

        # Rename per-asset and prepare merge
        bars_a = bars_a.rename(columns={'close': 'Asset_A', 'bid': 'Bid_A', 'ask': 'Ask_A'})
        bars_b = bars_b.rename(columns={'close': 'Asset_B', 'bid': 'Bid_B', 'ask': 'Ask_B'})

        df_pairs = pd.merge_asof(
            bars_a, bars_b,
            left_index=True, right_index=True,
            direction='backward'
        ).dropna().sort_index()

        df_pairs['Log_A'] = np.log(df_pairs['Asset_A'])
        df_pairs['Log_B'] = np.log(df_pairs['Asset_B'])
        df_pairs['Return_A'] = df_pairs['Log_A'].diff()
        df_pairs['Return_B'] = df_pairs['Log_B'].diff()

        # Half-spread in bps at each bar, per asset (for slippage modeling)
        df_pairs['HalfSpread_A_bps'] = 0.5 * (df_pairs['Ask_A'] - df_pairs['Bid_A']) / df_pairs['Asset_A'] * 10000
        df_pairs['HalfSpread_B_bps'] = 0.5 * (df_pairs['Ask_B'] - df_pairs['Bid_B']) / df_pairs['Asset_B'] * 10000

        df_pairs = df_pairs.dropna()
        self.data = df_pairs
        print(f"built {len(self.data)} rows")
        return self.data

    def plot_diagnostics(self):
        """
        Plots 5 panels of (12, 3) each. 
        Shows raw time series, spread, returns, and distribution histograms (Log-Scaled for Fat Tails).
        """
        if self.data is None or self.data.empty:
            raise ValueError("Data has not been built yet. Please run .build() first.")

        df = self.data.copy()

        # Calculate a static OLS spread for diagnostic plotting
        Y = df['Log_A']
        X = sm.add_constant(df['Log_B'])
        model = sm.OLS(Y, X).fit()
        beta = model.params['Log_B']
        
        spread = df['Log_A'] - beta * df['Log_B']
        spread_returns = spread.diff().dropna()

        # 5 rows total, aiming for roughly 12x3 each -> 12x15 total figure
        fig = plt.figure(figsize=(12, 15))
        gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3)

        # --- Panel 1: Time Series of Assets (Shared X-axis but Twin Y-axis) ---
        ax1 = fig.add_subplot(gs[0, :])
        l1, = ax1.plot(df.index, df['Asset_A'], color='tab:blue', alpha=0.8, label='Asset A')
        ax1.set_ylabel('Asset A Price', color='tab:blue')
        
        ax1_twin = ax1.twinx()
        l2, = ax1_twin.plot(df.index, df['Asset_B'], color='tab:orange', alpha=0.8, label='Asset B')
        ax1_twin.set_ylabel('Asset B Price', color='tab:orange')
        
        ax1.set_title('Asset Prices (Time Series)', fontweight='bold')
        ax1.legend(handles=[l1, l2], loc='upper left')

        # --- Panel 2: Spread Time Series ---
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(spread.index, spread, color='tab:purple', alpha=0.9)
        ax2.axhline(spread.mean(), color='black', linestyle='--', alpha=0.5)
        ax2.set_title(f'Log Spread (Log A - {beta:.3f} * Log B)', fontweight='bold')
        ax2.set_ylabel('Spread Level')

        # --- Panel 3: Returns Time Series ---
        ax3 = fig.add_subplot(gs[2, :])
        ax3.plot(df.index, df['Return_A'], color='tab:blue', alpha=0.6, label='Return A')
        ax3.plot(df.index, df['Return_B'], color='tab:orange', alpha=0.6, label='Return B')
        ax3.set_title('Asset Returns', fontweight='bold')
        ax3.set_ylabel('Log Returns')
        ax3.legend(loc='upper left')

        # --- Panel 4: Spread Returns ---
        ax4 = fig.add_subplot(gs[3, :])
        ax4.plot(spread_returns.index, spread_returns, color='tab:green', alpha=0.8)
        ax4.set_title('Spread Returns', fontweight='bold')
        ax4.set_ylabel('Returns')

        # --- Panel 5: Histograms (Log Scale to expose Fat Tails) ---
        # Histogram A
        ax5a = fig.add_subplot(gs[4, 0])
        ax5a.hist(df['Return_A'].dropna(), bins=75, color='tab:blue', alpha=0.7, log=True)
        ax5a.set_title('Return A Dist (Log Scale)')
        ax5a.set_ylabel('Frequency (log)')
        
        # Histogram B
        ax5b = fig.add_subplot(gs[4, 1])
        ax5b.hist(df['Return_B'].dropna(), bins=75, color='tab:orange', alpha=0.7, log=True)
        ax5b.set_title('Return B Dist (Log Scale)')
        
        # Histogram Spread (Level)
        ax5c = fig.add_subplot(gs[4, 2])
        ax5c.hist(spread.dropna(), bins=75, color='tab:purple', alpha=0.7, log=True)
        ax5c.axvline(spread.mean(), color='black', linestyle='dashed', linewidth=1.5)
        ax5c.set_title('Spread Level Dist (Log Scale)')

        plt.show()