
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

class PAIRS:
    """
    A Hidden Markov Model framework for detecting
    structural regime shifts in high-frequency pairs
    trading spreads 
    """

    def __init__(self, file_a, file_b, data_dir="../data/processed/", resample_freq='1min'):
        self.data_dir = Path(data_dir)
        self.file_a = file_a
        self.file_b = file_b
        self.resample_freq = resample_freq

        # auto-extract names
        self.name_a = self._extract_symbol_name(self.file_a)
        self.name_b = self._extract_symbol_name(self.file_b)

        # state containers
        self.pair_df = None
        self.spread_mean = None
        self.spread_std = None
        self.modul = None
        self.result = None
        self.c0 = None
        self.c1 = None

    def extract_symbol_names(self, filename):
        """Extracts 'AUD/USD' from 'audusd_dukascopy_ask_202601.parquet'"""
        base = filename.split('_')[0].upper()
        if len(base) == 6:
            return f"{base[:3]}/{base[3:]}"
        return base

    def prepare_data(self):
        print(f"--- Preparing data: {self.name_a} versus {self.name_b} ---")

        df_a = pd.read_parquet(self.data_dir / self.file_a)
        df_b = pd.read_parquet(self.data_dir / self.file_b)

        # resampling
        print(f"Synchronizing time-grids ({self.resample_freq} bars)...")
        
        df_a = df_a.set_index('datetime').sort_index()
        df_b = df_b.set_index('datetime').sort_index()

        a_resampled = df_a['price'].resample(self.resample_freq).last()
        b_resampled = df_a['price'].resample(self.resample_freq).last()

        self.pair_df = pd.DataFrame({'Asset A': a_resampled, 'Asset_B': b_resampled}).dropna()

        # remove stale quotes
        a_changed = self.pair_df['Asset_A'].diff().ne(0)
        b_changed = self.pair_df['Asset_B'].diff().ne(0)
        
        either_moved = a_changed | b_changed 
        self.pair_df = self.pair_df[either_moved | (self.pair_df.index == self.pair_df.index[0])]

        if len(self.pair_df) < 100:
            raise ValueError(f"Only {len(self.pair_df)} aligned bars. Check data overlap.")

        # z-score standardization
        raw_log_spread = np.log(self.pair_df['Asset_A']) - np.log(self.pair_df['Asset_B'])
        self.spread_mean = raw_log_spread.mean()
        self.spread_std = raw_log_spread.std()

        if self.spread_std < 1e-10:
            raise ValueError("Spread std is ~0. Assets may be identical.")
        
        self.pair_df['Spread_Standardized'] = (raw_log_spread - self.spread_mean) / self.spread_std
        print(f"Data prepared. {len(self.pair_df):,} active-market bars aligned.")
        
        return self
    
    def fit(self):
        if self.pair_df is None: 
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print(f"Fitting Markov Autogregression...")
        spread_values = self.pair_df['Spread_Standardized'].dropna().values

        self.model = sm.tsa.MarkovAutoregression(
            spread_values,
            k_regimes = 2,
            order = 1,
            trend = 'c',
            switching_trend = True,
            switching_variance = False
        )

        self.model.initialize_known([0.5, 0.5])

        # surgical initialization
        initial_guess = self.model.start_params.copy()
        
    

        


        
        
