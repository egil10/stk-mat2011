
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

class TEARSHEET:
    def __init__(self, df_results):
        self.df = df_results
        self.strats = ['Baseline', 'AR', 'MS_AR'] 

    def _calc_metrics(self, strat_name, ann_factor):
        ret_col = f'Return_{strat_name}'
        pos_col = f'Target_{strat_name}'
        
        if ret_col not in self.df.columns:
            return None
            
        returns = self.df[ret_col].fillna(0)
        positions = self.df[pos_col].fillna(0)
        
        # --- 1. Financial Metrics ---
        tot_ret = returns.sum() * 10000
        ann_ret = returns.mean() * ann_factor * 10000
        ann_vol = returns.std() * np.sqrt(ann_factor) * 10000
        sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
        
        downside = returns[returns < 0]
        down_vol = downside.std() * np.sqrt(ann_factor) * 10000
        sortino = (ann_ret / down_vol) if down_vol != 0 else 0
        
        cum_ret = returns.cumsum() * 10000
        roll_max = cum_ret.cummax()
        max_dd = (cum_ret - roll_max).min()
        calmar = abs(ann_ret / max_dd) if max_dd != 0 else 0
        
        gross_prof = returns[returns > 0].sum() * 10000
        gross_loss = abs(returns[returns < 0].sum() * 10000)
        prof_factor = (gross_prof / gross_loss) if gross_loss != 0 else np.nan

        # --- 2. Statistical & Trade Metrics ---
        trades = (positions.diff().abs() > 0).sum() / 2 
        active_bars = (returns != 0).sum()
        exposure = active_bars / len(returns) if len(returns) > 0 else 0
        win_rate = (returns > 0).sum() / active_bars if active_bars > 0 else 0
        
        avg_win = returns[returns > 0].mean() * 10000 if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() * 10000 if len(returns[returns < 0]) > 0 else 0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        
        skew = stats.skew(returns) if len(returns) > 2 else np.nan
        kurtosis = stats.kurtosis(returns) if len(returns) > 2 else np.nan
        
        var_95 = np.percentile(returns, 5) * 10000
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 10000

        return {
            "--- FINANCIAL METRICS ---": "",
            "Total Return (bps)": tot_ret,
            "Annual Return (bps)": ann_ret,
            "Annual Volatility (bps)": ann_vol,
            "Max Drawdown (bps)": max_dd,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Gross Profit (bps)": gross_prof,
            "Gross Loss (bps)": gross_loss,
            "Profit Factor": prof_factor,
            "--- STATISTICAL METRICS ---": "",
            "Win Rate": win_rate,
            "Number of Trades (Est.)": trades,
            "Market Exposure Time": exposure,
            "Average Win (bps)": avg_win,
            "Average Loss (bps)": avg_loss,
            "Payoff Ratio (Win/Loss)": payoff_ratio,
            "Return Skewness": skew,
            "Return Kurtosis": kurtosis,
            "Value at Risk (95%) bps": var_95,
            "Expected Shortfall (CVaR) bps": cvar_95
        }

    def generate_report(self):
        idx_series = pd.Series(self.df.index)
        unique_days_count = max(len(idx_series.dt.date.unique()), 1)
        ann_factor = len(self.df) / (unique_days_count / 252)

        results = {}
        for strat in self.strats:
            stats_dict = self._calc_metrics(strat, ann_factor)
            if stats_dict:
                results[strat] = stats_dict
        
        report_df = pd.DataFrame(results)
        
        print(f"\n{'='*75}")
        print(f"{'QUANTITATIVE STRATEGY TEARSHEET':^75}")
        print(f"{'='*75}")
        
        for index, row in report_df.iterrows():
            if "---" in index:
                print(f"\n{index}")
                continue
                
            row_str = f"{index:<30} |"
            for val in row:
                if isinstance(val, str):
                    row_str += f" {val:<12} |"
                elif pd.isna(val):
                    row_str += f" {'NaN':<12} |"
                elif "Win Rate" in index or "Exposure" in index:
                    row_str += f" {val:<12.2%} |"
                else:
                    row_str += f" {val:<12.2f} |"
            print(row_str)
        print(f"{'='*75}\n")

    def plot_performance(self):
        valid_strats = [s for s in self.strats if f'CumReturn_{s}' in self.df.columns]
        colors = {'Baseline': 'gray', 'AR': 'tab:blue', 'MS_AR': 'tab:purple'}
        
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        # --- Panel 1: Equity Curve ---
        ax1 = fig.add_subplot(gs[0])
        for strat in valid_strats:
            ax1.plot(self.df.index, self.df[f'CumReturn_{strat}'] * 10000, 
                     color=colors.get(strat, 'black'), linewidth=1.5, label=f'{strat} Equity')
        ax1.set_title("Cumulative Strategy Returns (bps)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Cumulative PnL (bps)")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # --- Panel 2: Underwater Plot (Drawdowns) ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        for strat in valid_strats:
            cum_ret = self.df[f'CumReturn_{strat}'] * 10000
            drawdown = cum_ret - cum_ret.cummax()
            ax2.fill_between(self.df.index, drawdown, 0, color=colors.get(strat, 'black'), 
                             alpha=0.3, label=f'{strat} DD')
        ax2.set_title("Underwater Plot (Drawdown Profile)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Drawdown (bps)")
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        # --- Panel 3: Return Distributions ---
        ax3 = fig.add_subplot(gs[2])
        for strat in valid_strats:
            returns = self.df[f'Return_{strat}'] * 10000
            active_returns = returns[returns != 0]
            if len(active_returns) > 0:
                ax3.hist(active_returns, bins=50, alpha=0.5, color=colors.get(strat, 'black'), 
                         label=f'{strat} (Active Bars)', density=True)
        ax3.set_title("Distribution of Active Bar Returns", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Density")
        ax3.set_xlabel("Return (bps)")
        ax3.axvline(0, color='black', linestyle='--', linewidth=1)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_positions_and_regimes(self):
        """Plots strategy positioning alongside the Markov danger probabilities."""
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        # Panel 1: Position Holding State
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.df.index, self.df['Target_Baseline'], color='gray', alpha=0.5, label='Baseline', lw=1)
        ax1.plot(self.df.index, self.df['Target_AR'] + 0.05, color='tab:blue', alpha=0.7, label='AR (+ offset)', lw=1)
        ax1.plot(self.df.index, self.df['Target_MS_AR'] - 0.05, color='tab:purple', alpha=0.9, label='MS-AR (- offset)', lw=1)
        ax1.set_yticks([-1, 0, 1])
        ax1.set_yticklabels(['Short (-1)', 'Flat (0)', 'Long (1)'])
        ax1.set_title("Strategy Position State Machine (When are we holding?)", fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Regime Classification
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(self.df.index, self.df['Danger_Regime_Prob'], color='red', alpha=0.8, lw=1.2)
        ax2.axhline(0.7, color='black', linestyle='--', label='Danger Kill-Switch Threshold (0.7)')
        ax2.fill_between(self.df.index, self.df['Danger_Regime_Prob'], 0.7, 
                         where=(self.df['Danger_Regime_Prob'] > 0.7), color='red', alpha=0.3, label='Trading Halted')
        ax2.set_title("Markov HMM: P(State = Danger | Data)", fontweight='bold')
        ax2.set_ylabel("Probability")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_markov_dynamics(self):
        """Plots how the Markov regime means, variances, and transition probabilities evolve over time."""
        if not hasattr(self, 'params') or self.params is None or 'Safe_Mean' not in self.params.columns:
            print("Markov parameters not found in df_params. Ensure ENGINE is tracking them.")
            return
            
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, hspace=0.3)
        
        # Panel 1: Regime Variances (Sigma^2)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.params.index, self.params['Danger_Variance'], color='red', label='Danger Regime Variance')
        ax1.plot(self.params.index, self.params['Safe_Variance'], color='green', label='Safe Regime Variance')
        ax1.set_yscale('log')
        ax1.set_title("Rolling Markov Variances ($\sigma^2$)", fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Regime Means (Mu)
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(self.params.index, self.params['Danger_Mean'], color='darkred', label='Danger Mean ($\mu$)')
        ax2.plot(self.params.index, self.params['Safe_Mean'], color='darkgreen', label='Safe Mean ($\mu$)')
        ax2.axhline(0, color='black', linestyle='--', lw=1)
        ax2.set_title("Rolling Markov Expected Returns ($\mu$ in bps)", fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Transition Probabilities Matrix diagonals
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(self.params.index, self.params['P_Danger_Danger'], color='salmon', label='P(Danger | Danger)')
        ax3.plot(self.params.index, self.params['P_Safe_Safe'], color='lightgreen', label='P(Safe | Safe)')
        ax3.set_title("Regime Persistence (Transition Matrix Diagonals)", fontweight='bold')
        ax3.set_ylabel("Probability")
        ax3.set_ylim(0, 1.05)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()