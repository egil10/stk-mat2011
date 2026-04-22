
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

class SCREENER:
    """
    Gatekeeper: Engle-Granger cointegration + Ornstein-Uhlenbeck half-life.
    """
 
    def __init__(self, asset_a_series, asset_b_series):
        df = pd.concat([asset_a_series, asset_b_series], axis=1).dropna()
        self.asset_a = df.iloc[:, 0]
        self.asset_b = df.iloc[:, 1]
        self.log_a = np.log(self.asset_a)
        self.log_b = np.log(self.asset_b)
 
    def run_engle_granger(self):
        _, p_value, _ = coint(self.log_a, self.log_b)
        return p_value
 
    def calculate_half_life(self):
        X = sm.add_constant(self.log_b)
        X.columns = ['const', 'log_b']  # name explicitly
        beta = sm.OLS(self.log_a, X).fit().params['log_b']
 
        spread = self.log_a - beta * self.log_b
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
 
        X_hl = sm.add_constant(spread_lag)
        X_hl.columns = ['const', 'spread_lag']
        lambda_val = sm.OLS(spread_diff, X_hl).fit().params['spread_lag']
 
        if lambda_val >= 0:
            return np.inf
        half_life = -np.log(2) / lambda_val
        # Implausibly huge values -> treat as "essentially not reverting"
        if half_life > 1e6:
            return np.inf
        return half_life
 
    def generate_report(self):
        p_val = self.run_engle_granger()
        half_life = self.calculate_half_life()
 
        print(f"{'='*40}")
        print(f"{'COINTEGRATION SCREENER REPORT':^40}")
        print(f"{'='*40}")
        if p_val < 0.01:
            coint_status = "pass with 99% confidence"
        elif p_val < 0.05:
            coint_status = "pass with 95% confidence"
        else:
            coint_status = "fail (not cointegrated)"
        print(f"P-Value:    {p_val:.5f} -> {coint_status}")
        print(f"Half-Life:  {half_life:.2f} Bars")
        print(f"{'-'*40}")
        if p_val >= 0.05:
            print("Verdict: DO NOT TRADE. Spread is a random walk.")
        elif half_life > 1000:
            print("Verdict: CAUTION. Cointegrated, but reversion is too slow for intraday.")
        elif half_life < 5:
            print("Verdict: CAUTION. Reversion is so fast you might get killed by fees/slippage.")
        else:
            print("Verdict: APPROVED FOR ENGINE PIPELINE.")
        print(f"{'='*40}\n")
 
        return p_val, half_life