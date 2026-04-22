
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtester import BACKTESTER

class TEARSHEET:
    def __init__(self, df_results, df_params=None):
        self.df = df_results
        self.params = df_params
 
    def _calc_stats(self, returns_series, ann_factor):
        total_ret = returns_series.sum() * 10000
        vol = returns_series.std() * np.sqrt(ann_factor) * 10000
        sharpe = ((returns_series.mean() * ann_factor)
                  / (returns_series.std() * np.sqrt(ann_factor))
                  if returns_series.std() != 0 else 0)
 
        downside_returns = returns_series[returns_series < 0]
        downside_dev = downside_returns.std() * np.sqrt(ann_factor) * 10000
        sortino = (total_ret / downside_dev) if downside_dev != 0 else 0
 
        cum_ret = returns_series.cumsum()
        max_dd = (cum_ret - cum_ret.cummax()).min() * 10000
        calmar = abs(total_ret / max_dd) if max_dd != 0 else 0
 
        active = (returns_series != 0).sum()
        win_rate = (returns_series > 0).sum() / active if active > 0 else 0
        gross_profits = returns_series[returns_series > 0].sum()
        gross_losses = abs(returns_series[returns_series < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else 0
        return total_ret, vol, sharpe, sortino, max_dd, calmar, win_rate, profit_factor
 
    def generate_report(self):
        idx_series = pd.Series(self.df.index)
        unique_days_count = max(len(idx_series.dt.date.unique()), 1)
        ann_factor = len(self.df) / (unique_days_count / 252)
        stats_adapt = self._calc_stats(self.df['Return_Adaptive'], ann_factor)
        stats_base = self._calc_stats(self.df['Return_Baseline'], ann_factor)
 
        metrics = ["Total Return (Bps)", "Annual Vol (Bps)", "Sharpe Ratio",
                   "Sortino Ratio", "Max Drawdown (Bps)", "Calmar Ratio",
                   "Win Rate", "Profit Factor"]
        print(f"\n{'='*55}")
        print(f"{'INSTITUTIONAL PERFORMANCE REPORT':^55}")
        print(f"{'='*55}")
        print(f"{'Metric':<25} | {'Adaptive':<12} | {'Static Base':<12}")
        print(f"{'-'*55}")
        for i, metric in enumerate(metrics):
            if i == 6:  # win rate
                va, vb = f"{stats_adapt[i]:.2%}", f"{stats_base[i]:.2%}"
            else:
                va, vb = f"{stats_adapt[i]:.2f}", f"{stats_base[i]:.2f}"
            print(f"{metric:<25} | {va:<12} | {vb:<12}")
        print(f"{'='*55}\n")
 
    def plot_comparative_equity(self):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(self.df.index, self.df['CumReturn_Adaptive'] * 10000,
                     color='blue', linewidth=1.5, label='Adaptive (HMM/GARCH)')
        axes[0].plot(self.df.index, self.df['CumReturn_Baseline'] * 10000,
                     color='gray', linewidth=1.0, alpha=0.7, label='Static Baseline')
        axes[0].set_title("Strategy Race: Adaptive vs Baseline")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
 
        dd_a = (self.df['CumReturn_Adaptive'] - self.df['CumReturn_Adaptive'].cummax()) * 10000
        dd_b = (self.df['CumReturn_Baseline'] - self.df['CumReturn_Baseline'].cummax()) * 10000
        axes[1].fill_between(self.df.index, dd_a, 0, color='red', alpha=0.3, label='Adaptive DD')
        axes[1].plot(self.df.index, dd_b, color='gray', alpha=0.5, label='Baseline DD')
        axes[1].set_title("Underwater Plot Comparison")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()
 
    def plot_engine_parameters(self):
        if self.params is None:
            print("No Engine Parameters provided for plotting.")
            return
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
 
        ax1.plot(self.params.index, self.params['Beta'], color='tab:blue', label='Beta')
        ax1_tw = ax1.twinx()
        ax1_tw.plot(self.params.index, self.params['Alpha'], color='tab:orange',
                    linestyle='--', label='Alpha')
        ax1.set_title('Cointegration Drift'); ax1.legend(loc='upper left'); ax1_tw.legend(loc='upper right')
 
        ax2.plot(self.params.index, self.params['Danger_Variance'], color='red', label='Danger Var')
        ax2.plot(self.params.index, self.params['Safe_Variance'], color='green', label='Safe Var')
        ax2.set_yscale('log')
        ax2.set_title('Markov-Switching Variances (log)'); ax2.legend()
 
        ax3.plot(self.params.index, self.params['GARCH_Vol'], color='purple', label='GARCH Vol')
        ax3_tw = ax3.twinx()
        ax3_tw.plot(self.params.index, self.params['AR_Phi'], color='brown', alpha=0.6, label='AR(1) Phi')
        ax3_tw.axhline(0.995, color='red', linestyle=':', label='Reversion Limit')
        ax3.set_title('Market Dynamics: Vol & Reversion')
        ax3.legend(loc='upper left'); ax3_tw.legend(loc='upper right')
        plt.tight_layout(); plt.show()
 
    # -------- NEW: Robustness heatmap --------
    def plot_robustness_heatmap(self, engine_data,
                                base_z_grid=None,
                                danger_grid=None,
                                fixed_params=None,
                                metric='sharpe'):
        """
        Sweeps (base_z, danger_threshold), re-runs the BACKTESTER for each
        combo, and heatmaps the chosen metric for the Adaptive strategy.
 
        engine_data: the DataFrame that was fed to BACKTESTER (with Z_Score,
                     Danger_Regime_Prob, GARCH_Vol, AR_Phi, Spread_Return, ...)
        metric: 'sharpe' | 'total_ret' | 'max_dd'
        """
        if base_z_grid is None:
            base_z_grid = np.round(np.arange(0.75, 2.51, 0.25), 2)
        if danger_grid is None:
            danger_grid = np.round(np.arange(0.3, 0.96, 0.1), 2)
        fixed = {'exit_z': 0.0, 'ar_limit': 0.995, 'fee_bps': 0.5,
                 'slippage_mode': 'half_spread'}
        if fixed_params:
            fixed.update(fixed_params)
 
        idx_series = pd.Series(engine_data.index)
        ann_factor = len(engine_data) / (max(len(idx_series.dt.date.unique()), 1) / 252)
 
        heat = np.full((len(danger_grid), len(base_z_grid)), np.nan)
        for i, dth in enumerate(danger_grid):
            for j, bz in enumerate(base_z_grid):
                bt = BACKTESTER(engine_data)
                out = bt.run(base_z=bz, danger_threshold=dth, **fixed)
                r = out['Return_Adaptive']
                if metric == 'sharpe':
                    v = ((r.mean() * ann_factor)
                         / (r.std() * np.sqrt(ann_factor))
                         if r.std() != 0 else 0)
                elif metric == 'total_ret':
                    v = r.sum() * 10000
                elif metric == 'max_dd':
                    c = r.cumsum()
                    v = (c - c.cummax()).min() * 10000
                else:
                    raise ValueError(f"Unknown metric {metric}")
                heat[i, j] = v
 
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heat, aspect='auto', origin='lower', cmap='RdYlGn')
        ax.set_xticks(range(len(base_z_grid))); ax.set_xticklabels(base_z_grid)
        ax.set_yticks(range(len(danger_grid))); ax.set_yticklabels(danger_grid)
        ax.set_xlabel('base_z'); ax.set_ylabel('danger_threshold')
        ax.set_title(f'Robustness Heatmap — Adaptive {metric}')
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                if np.isfinite(heat[i, j]):
                    ax.text(j, i, f"{heat[i, j]:.2f}", ha='center', va='center',
                            fontsize=8, color='black')
        plt.colorbar(im, ax=ax)
        plt.tight_layout(); plt.show()
        return pd.DataFrame(heat, index=danger_grid, columns=base_z_grid)