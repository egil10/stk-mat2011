
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

class SCREENER:
    """
    Diagnostic screener for pairs trading. Provides both a full-sample view
    (Engle-Granger p-value, half-life, hedge ratio) and a rolling view
    showing how those diagnostics evolve over sub-samples of the data.
    """
    def __init__(self, asset_a_series, asset_b_series):
        df = pd.concat([asset_a_series, asset_b_series], axis=1).dropna()
        self.asset_a = df.iloc[:, 0]
        self.asset_b = df.iloc[:, 1]
        self.log_a = np.log(self.asset_a)
        self.log_b = np.log(self.asset_b)
        self.rolling_df = None  

    # static helpers (per window)
    @staticmethod
    def _engle_granger_p(log_a, log_b):
        try:
            _, p, _ = coint(log_a, log_b)
            return p
        except Exception:
            return np.nan

    @staticmethod
    def _hedge_ratio(log_a, log_b):
        X = sm.add_constant(log_b)
        X.columns = ['const', 'log_b']
        return float(sm.OLS(log_a, X).fit().params['log_b'])

    @staticmethod
    def _half_life(spread):
        spread = pd.Series(spread).dropna()
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        X_hl = sm.add_constant(spread_lag)
        X_hl.columns = ['const', 'spread_lag']
        lam = sm.OLS(spread_diff, X_hl).fit().params['spread_lag']
        if lam >= 0:
            return np.inf
        hl = -np.log(2) / lam
        return hl if hl < 1e6 else np.inf

    # full sample view
    def run_engle_granger(self):
        return self._engle_granger_p(self.log_a, self.log_b)

    # calculating half life
    def calculate_half_life(self):
        beta = self._hedge_ratio(self.log_a, self.log_b)
        spread = self.log_a - beta * self.log_b
        return self._half_life(spread)

    # rolling view
    def run_rolling(self, window=2000, step=200):
        """
        Rolling diagnostics. Returns a DataFrame indexed by the window-end
        timestamp with columns: p_value, half_life, beta.
        """
        n = len(self.log_a)
        rows = []
        for start in range(0, n - window + 1, step):
            end = start + window
            la = self.log_a.iloc[start:end]
            lb = self.log_b.iloc[start:end]
            p = self._engle_granger_p(la, lb)
            try:
                beta = self._hedge_ratio(la, lb)
                hl = self._half_life(la - beta * lb)
            except Exception:
                beta, hl = np.nan, np.nan
            rows.append({
                'window_end': la.index[-1],
                'p_value': p,
                'half_life': hl,
                'beta': beta,
            })
        self.rolling_df = pd.DataFrame(rows).set_index('window_end')
        return self.rolling_df

    # reporting
    def generate_report(self, rolling_window=2000, rolling_step=200, plot=True):
        """
        Full diagnostic report: both full-sample numbers and rolling summary.
        """
        p_val = self.run_engle_granger()
        half_life = self.calculate_half_life()
        beta_full = self._hedge_ratio(self.log_a, self.log_b)

        self.run_rolling(window=rolling_window, step=rolling_step)
        rdf = self.rolling_df

        frac_p05 = (rdf['p_value'] < 0.05).mean()
        frac_p10 = (rdf['p_value'] < 0.10).mean()
        hl_valid = rdf['half_life'].replace([np.inf, -np.inf], np.nan).dropna()
        beta_std = rdf['beta'].std()
        beta_range = (rdf['beta'].min(), rdf['beta'].max())

        # printouts
        print(f"\n=== COINTEGRATION SCREENER ===")
        print(f"Full: p={p_val:.4f} | half-life={half_life:.1f} | β={beta_full:.4f}")
        print(f"\nRolling ({rolling_window}-bar, step {rolling_step}, n={len(rdf)}):")
        print(f"  p<0.05: {frac_p05:.1%} | p<0.10: {frac_p10:.1%}")
        if len(hl_valid):
            print(f"  half-life: {hl_valid.median():.1f} (IQR {hl_valid.quantile(0.25):.0f}–{hl_valid.quantile(0.75):.0f})")
        print(f"  β: {rdf['beta'].mean():.4f} ±{beta_std:.4f} [{beta_range[0]:.4f}, {beta_range[1]:.4f}]")

        if plot:
            self._plot_rolling()

        return p_val, half_life

    # plotting function
    def _plot_rolling(self):
        rdf = self.rolling_df
        beta_mean = rdf['beta'].mean()
        
        panels = [
            ('p_value',   'tab:blue',   'Rolling Engle-Granger p-value', 'p-value',         'linear', [(0.05, '--', 'red', '5%'), (0.10, ':', 'orange', '10%')]),
            ('half_life', 'tab:purple', 'Rolling Half-Life',             'half-life (bars)', 'log',   []),
            ('beta',      'tab:green',  'Rolling Hedge Ratio',           'beta',             'linear', [(beta_mean, '--', 'black', f'mean={beta_mean:.3f}')]),
        ]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        for ax, (col, color, title, ylabel, yscale, hlines) in zip(axes, panels):
            series = rdf[col].replace([np.inf, -np.inf], np.nan)
            ax.plot(rdf.index, series, color=color, linewidth=1.2)
            for y, ls, c, lbl in hlines:
                ax.axhline(y, color=c, linestyle=ls, alpha=0.6, label=lbl)
            ax.set(ylabel=ylabel, title=title, yscale=yscale)
            ax.grid(True, alpha=0.3)
            if hlines:
                ax.legend(loc='upper right')
        plt.tight_layout(); plt.show()