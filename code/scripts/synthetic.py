
import numpy as np
import pandas as pd
import os
from numba import njit

@njit
def _generate_ou_spread(n, theta, mu, sigma_0, sigma_1, p_00, p_11, jump_prob, jump_std):
    """
    Lightning-fast Numba compiled generator for the True Spread.
    Uses an Ornstein-Uhlenbeck process + Markov Switching Variance + Poisson Jumps.
    """
    spread = np.zeros(n)
    states = np.zeros(n, dtype=np.int32)
    
    # Initialize state
    curr_state = 0
    spread[0] = mu
    
    for t in range(1, n):
        # 1. Markov State Transition
        rand_trans = np.random.random()
        if curr_state == 0:
            if rand_trans > p_00: curr_state = 1
        else:
            if rand_trans > p_11: curr_state = 0
        states[t] = curr_state
        
        # 2. State-dependent volatility
        current_sigma = sigma_0 if curr_state == 0 else sigma_1
        
        # 3. Poisson Jumps (Fat Tails)
        jump = 0.0
        if np.random.random() < jump_prob:
            jump = np.random.normal(0, jump_std)
            
        # 4. OU Mean Reversion Step
        noise = np.random.normal(0, 1)
        # dS = theta * (mu - S) + sigma * dW + Jump
        spread[t] = spread[t-1] + theta * (mu - spread[t-1]) + (current_sigma * noise) + jump
        
    return spread, states

class SYNTHETIC:
    def __init__(self, n_ticks=250000, start_date="2026-01-01", 
                 symbol_a="SYN_A", symbol_b="SYN_B", random_seed=42): # <--- Add seed here
        self.n_ticks = n_ticks
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.random_seed = random_seed # <--- Store it
        
        # Microstructure params
        self.median_spread_bps = 0.8
        self.avg_volume = 1.5
        
    def generate_market(self, theta, mu, sigma_0, sigma_1, p_00, p_11, jump_prob, jump_std):
        print(f"Generating {self.n_ticks} synthetic ticks...")
        
        # --- LOCK THE RANDOM STATE ---
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # 1. Generate Timestamps (Poisson arrival times for ticks)
        # Avg 1 tick every 3 seconds
        time_deltas = np.random.exponential(scale=3.0, size=self.n_ticks)
        time_deltas[0] = 0
        timestamps = self.start_date + pd.to_timedelta(np.cumsum(time_deltas), unit='s')
        
        # 2. Generate Asset B (Geometric Brownian Motion)
        # Drift = 0, Daily Vol = ~100 bps
        dt = 1.0 / (252 * 24 * 60 * 20) # Approx time step
        vol_B = 0.10 * np.sqrt(dt)
        log_B = np.cumsum(np.random.normal(0, vol_B, self.n_ticks))
        mid_B = 1.0 * np.exp(log_B) # Start at 1.0
        
        # 3. Generate True Spread (OU + HMM + Jumps)
        spread, states = _generate_ou_spread(
            n=self.n_ticks,
            theta=theta,
            mu=mu,
            sigma_0=sigma_0,
            sigma_1=sigma_1,
            p_00=p_00,
            p_11=p_11,
            jump_prob=jump_prob,
            jump_std=jump_std
        )
        
        # 4. Generate Asset A
        # P_A = P_B * exp(Spread)
        mid_A = mid_B * np.exp(spread)
        
        self.data = pd.DataFrame({
            'datetime': timestamps,
            'mid_A': mid_A,
            'mid_B': mid_B,
            'true_state': states,
            'true_spread': spread
        })
        
        return self.data

    def _format_parquet(self, datetimes, symbol, price_type, prices):
        """Formats arrays into the exact DataFrame schema required."""
        df = pd.DataFrame({
            'datetime': datetimes,
            'symbol': symbol,
            'price_type': price_type,
            'price': prices,
            'volume': np.abs(np.random.normal(self.avg_volume, 0.5, len(prices))).round(2)
        })
        # Ensure volume is at least 0.01
        df['volume'] = df['volume'].clip(lower=0.01)
        return df

    def save_to_parquets(self, output_dir="../data/synthetic"):
        if not hasattr(self, 'data'):
            raise ValueError("Run generate_market() first.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply Bid/Ask microstructure (half-spread in absolute terms)
        half_spread_A = self.data['mid_A'] * (self.median_spread_bps / 2 / 10000)
        half_spread_B = self.data['mid_B'] * (self.median_spread_bps / 2 / 10000)
        
        # Generate 4 dataframes
        df_ask_a = self._format_parquet(self.data['datetime'], self.symbol_a, 'ASK', self.data['mid_A'] + half_spread_A)
        df_bid_a = self._format_parquet(self.data['datetime'], self.symbol_a, 'BID', self.data['mid_A'] - half_spread_A)
        df_ask_b = self._format_parquet(self.data['datetime'], self.symbol_b, 'ASK', self.data['mid_B'] + half_spread_B)
        df_bid_b = self._format_parquet(self.data['datetime'], self.symbol_b, 'BID', self.data['mid_B'] - half_spread_B)
        
        # Save them
        files = [
            f"{output_dir}/{self.symbol_a.lower()}_ask_202601.parquet",
            f"{output_dir}/{self.symbol_a.lower()}_bid_202601.parquet",
            f"{output_dir}/{self.symbol_b.lower()}_ask_202601.parquet",
            f"{output_dir}/{self.symbol_b.lower()}_bid_202601.parquet"
        ]
        
        df_ask_a.to_parquet(files[0], index=False)
        df_bid_a.to_parquet(files[1], index=False)
        df_ask_b.to_parquet(files[2], index=False)
        df_bid_b.to_parquet(files[3], index=False)
        
        print(f"Saved 4 Parquet files to {output_dir}")
        return files