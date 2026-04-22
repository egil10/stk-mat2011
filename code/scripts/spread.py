
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from scipy.stats import norm
from arch import arch_model
import matplotlib.pyplot as plt

class SPREAD:
    """
    Ingests high-frequency tick data and outputs synchronized volume or tick
    bars for pairs trading. Now carries bid/ask explicitly so downstream can
    model half-spread slippage.
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