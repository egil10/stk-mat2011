
import numpy as np
import pandas as pd
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
        self.rolling_df = None  # populated by run_rolling()

    # ------------------------------------------------------------------
    # Static helpers (per window)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Full-sample view
    # ------------------------------------------------------------------
    def run_engle_granger(self):
        return self._engle_granger_p(self.log_a, self.log_b)

    def calculate_half_life(self):
        beta = self._hedge_ratio(self.log_a, self.log_b)
        spread = self.log_a - beta * self.log_b
        return self._half_life(spread)

    # ------------------------------------------------------------------
    # Rolling view
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
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

        print(f"{'='*56}")
        print(f"{'COINTEGRATION SCREENER REPORT':^56}")
        print(f"{'='*56}")
        print("Full-sample:")
        print(f"  P-value:    {p_val:.4f}")
        print(f"  Half-life:  {half_life:.1f} bars")
        print(f"  Beta:       {beta_full:.4f}")
        print(f"\nRolling ({rolling_window}-bar windows, step {rolling_step}, "
              f"{len(rdf)} windows):")
        print(f"  Fraction p < 0.05:   {frac_p05:.1%}")
        print(f"  Fraction p < 0.10:   {frac_p10:.1%}")
        if len(hl_valid):
            print(f"  Half-life median:    {hl_valid.median():.1f} bars "
                  f"(IQR {hl_valid.quantile(0.25):.0f}–{hl_valid.quantile(0.75):.0f})")
        print(f"  Beta mean:           {rdf['beta'].mean():.4f}")
        print(f"  Beta std:            {beta_std:.4f}")
        print(f"  Beta range:          [{beta_range[0]:.4f}, {beta_range[1]:.4f}]")
        print(f"{'-'*56}")

        # Interpretation — descriptive, not prescriptive
        if frac_p05 > 0.5:
            print("Interpretation: pair appears cointegrated most of the time.")
        elif frac_p05 > 0.2:
            print("Interpretation: pair shows intermittent cointegration — "
                  "potential regime-dependent mean reversion.")
        else:
            print("Interpretation: rarely cointegrated. If trading anyway, the "
                  "case must rest on something other than static coint.")

        if beta_std / abs(rdf['beta'].mean()) > 0.1:
            print("  + Hedge ratio drifts materially — rolling beta recommended.")
        else:
            print("  + Hedge ratio is relatively stable.")
        print(f"{'='*56}\n")

        if plot:
            self._plot_rolling()

        return p_val, half_life

    def _plot_rolling(self):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

        ax1.plot(self.rolling_df.index, self.rolling_df['p_value'],
                 color='tab:blue', linewidth=1.2)
        ax1.axhline(0.05, color='red', linestyle='--', alpha=0.6, label='5%')
        ax1.axhline(0.10, color='orange', linestyle=':', alpha=0.6, label='10%')
        ax1.set_ylabel('p-value'); ax1.set_title('Rolling Engle-Granger p-value')
        ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3)

        hl = self.rolling_df['half_life'].replace([np.inf, -np.inf], np.nan)
        ax2.plot(self.rolling_df.index, hl, color='tab:purple', linewidth=1.2)
        ax2.set_ylabel('half-life (bars)')
        ax2.set_title('Rolling Half-Life')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        ax3.plot(self.rolling_df.index, self.rolling_df['beta'],
                 color='tab:green', linewidth=1.2)
        ax3.axhline(self.rolling_df['beta'].mean(), color='black', linestyle='--',
                    alpha=0.5, label=f"mean={self.rolling_df['beta'].mean():.3f}")
        ax3.set_ylabel('beta'); ax3.set_title('Rolling Hedge Ratio')
        ax3.legend(loc='upper right'); ax3.grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()