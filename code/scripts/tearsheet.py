
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TEARSHEET:
    def __init__(self, df_results, df_params=None):
        self.df = df_results
        self.params = df_params # Now we store the engine parameters!

    def _calc_stats(self, returns_series, ann_factor):
        """Helper to calculate institutional metrics"""
        total_ret = returns_series.sum() * 10000
        vol = returns_series.std() * np.sqrt(ann_factor) * 10000
        sharpe = (returns_series.mean() * ann_factor) / (returns_series.std() * np.sqrt(ann_factor)) if returns_series.std() != 0 else 0
        
        # Downside Deviation (Sortino)
        downside_returns = returns_series[returns_series < 0]
        downside_dev = downside_returns.std() * np.sqrt(ann_factor) * 10000
        sortino = (total_ret) / downside_dev if downside_dev != 0 else 0

        # Max Drawdown
        cum_ret = returns_series.cumsum()
        max_dd = (cum_ret - cum_ret.cummax()).min() * 10000
        calmar = abs(total_ret / max_dd) if max_dd != 0 else 0

        # Trade Efficiency
        win_rate = (returns_series > 0).sum() / (returns_series != 0).sum() if (returns_series != 0).sum() > 0 else 0
        gross_profits = returns_series[returns_series > 0].sum()
        gross_losses = abs(returns_series[returns_series < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else 0

        return total_ret, vol, sharpe, sortino, max_dd, calmar, win_rate, profit_factor

    def generate_report(self):
        idx_series = pd.Series(self.df.index)
        unique_days_count = len(idx_series.dt.date.unique())
        ann_factor = len(self.df) / (unique_days_count / 252)

        stats_adapt = self._calc_stats(self.df['Return_Adaptive'], ann_factor)
        stats_base = self._calc_stats(self.df['Return_Baseline'], ann_factor)

        metrics = ["Total Return (Bps)", "Annual Vol (Bps)", "Sharpe Ratio", "Sortino Ratio", 
                   "Max Drawdown (Bps)", "Calmar Ratio", "Win Rate", "Profit Factor"]

        print(f"\n{'='*55}")
        print(f"{'INSTITUTIONAL PERFORMANCE REPORT':^55}")
        print(f"{'='*55}")
        print(f"{'Metric':<25} | {'Adaptive':<12} | {'Static Base':<12}")
        print(f"{'-'*55}")
        for i, metric in enumerate(metrics):
            val_a = f"{stats_adapt[i]:.2f}" if i < 6 else f"{stats_adapt[i]:.2%}" if i == 6 else f"{stats_adapt[i]:.2f}"
            val_b = f"{stats_base[i]:.2f}" if i < 6 else f"{stats_base[i]:.2%}" if i == 6 else f"{stats_base[i]:.2f}"
            print(f"{metric:<25} | {val_a:<12} | {val_b:<12}")
        print(f"{'='*55}\n")

    def plot_comparative_equity(self):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Equity Curve Comparison
        axes[0].plot(self.df.index, self.df['CumReturn_Adaptive'] * 10000, color='blue', linewidth=1.5, label='Adaptive Strategy (HMM/GARCH)')
        axes[0].plot(self.df.index, self.df['CumReturn_Baseline'] * 10000, color='gray', linewidth=1.0, alpha=0.7, label='Static Baseline (Fixed Z)')
        axes[0].set_title("Strategy Race: Adaptive vs Baseline")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Drawdown Comparison
        dd_adapt = (self.df['CumReturn_Adaptive'] - self.df['CumReturn_Adaptive'].cummax()) * 10000
        dd_base = (self.df['CumReturn_Baseline'] - self.df['CumReturn_Baseline'].cummax()) * 10000
        axes[1].fill_between(self.df.index, dd_adapt, 0, color='red', alpha=0.3, label='Adaptive Drawdown')
        axes[1].plot(self.df.index, dd_base, color='gray', alpha=0.5, label='Baseline Drawdown')
        axes[1].set_title("Underwater Plot Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_engine_parameters(self):
        if self.params is None:
            print("No Engine Parameters provided for plotting.")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # 1. Cointegration
        ax1.plot(self.params.index, self.params['Beta'], color='tab:blue', label='Beta (Hedge Ratio)')
        ax1_twin = ax1.twinx()  
        ax1_twin.plot(self.params.index, self.params['Alpha'], color='tab:orange', linestyle='--', label='Alpha')
        ax1.set_title('Cointegration Drift')
        ax1.legend(loc='upper left'); ax1_twin.legend(loc='upper right')

        # 2. Variances
        ax2.plot(self.params.index, self.params['Danger_Variance'], color='red', label='Danger Variance')
        ax2.plot(self.params.index, self.params['Safe_Variance'], color='green', label='Safe Variance')
        ax2.set_yscale('log')
        ax2.set_title('Markov-Switching Structural Breaks (Log Scale)')
        ax2.legend()

        # 3. Vol & AR
        ax3.plot(self.params.index, self.params['GARCH_Vol'], color='purple', label='GARCH Volatility')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.params.index, self.params['AR_Phi'], color='brown', alpha=0.6, label='AR(1) Phi')
        ax3_twin.axhline(0.995, color='red', linestyle=':', label='Reversion Limit')
        ax3.set_title('Market Dynamics: Volatility & Mean Reversion')
        ax3.legend(loc='upper left'); ax3_twin.legend(loc='upper right')

        plt.tight_layout()
        plt.show()