
import pandas as pd
import numpy as np

class SPREAD:
    """
    A minimal class to ingest high-frequency tick data and
    output synchronized volume bars for pairs trading
    """
    def __init__(self, vol_threshold=1000):
        self.vol_threshold = vol_threshold
        self.data = None

    def _aggregate_volume(self, ask_file, bid_file):
        """
        Internal method which converts raw bid/ask parquets
        into volume bars for a single asset
        """

        # load and rename
        df_ask = pd.read_parquet(ask_file).sort_values('datetime').rename(columns={'price': 'ask_price', 'volume': 'ask_volume'})
        df_bid = pd.read_parquet(bid_file).sort_values('datetime').rename(columns={'price': 'bid_price', 'volume': 'bid_volume'})
        
        # clipping active trading hours
        df_ask = df_ask[(df_ask['datetime'].dt.dayofweek < 5) & (df_ask['datetime'].dt.hour.between(10, 14))]
        df_bid = df_bid[(df_bid['datetime'].dt.dayofweek < 5) & (df_bid['datetime'].dt.hour.between(10, 14))]
        
        # asynchronous merge to avoid look-ahead bias
        df_ticks = pd.merge_asof(
            df_ask[['datetime', 'ask_price', 'ask_volume']],
            df_bid[['datetime', 'bid_price', 'bid_volume']],
            on='datetime',
            direction='backward'
        ).dropna()

        # calculate metrics
        df_ticks['mid_price'] = (df_ticks['bid_price'] + df_ticks['ask_price']) / 2
        df_ticks['total_volume'] = df_ticks['bid_volume'] + df_ticks['ask_volume']

        # volume clock agg
        df_ticks['cum_vol'] = df_ticks['total_volume'].cumsum()
        df_ticks['bar_id'] = (df_ticks['cum_vol'] // self.vol_threshold).astype(int)

        # compress into bars
        bars = df_ticks.groupby('bar_id').agg(
            timestamp=('datetime', 'last'),
            close=('mid_price', 'last')
        ).set_index('timestamp')

        return bars

    def build(self, file_paths):
        """
        Main method takes 4 file paths and build a final
        synchronized pairs dataset.
        Order = [ask_a, bid_a, ask_b, bid_b]
        """
        if len(file_paths) != 4:
            raise ValueError("Provide exactly 4 file paths: [ask_a, bid_a, ask_b, bid_b]")

        bars_a = self._aggregate_volume(file_paths[0], file_paths[1])
        bars_b = self._aggregate_volume(file_paths[2], file_paths[3])

        df_pairs = pd.merge_asof(
            bars_a.rename(columns={'close': 'Asset_A'}),
            bars_b.rename(columns={'close': 'Asset_B'}),
            left_index=True,
            right_index=True,
            direction='backward'
        ).dropna()

        df_pairs['Log_A'] = np.log(df_pairs['Asset_A'])
        df_pairs['Log_B'] = np.log(df_pairs['Asset_B'])

        self.data = df_pairs
        print(f"built {len(self.data)} rows")

        return self.data




