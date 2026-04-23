
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

class TEARSHEET:
    """
    Simplified but highly analytical tearsheet for comparing pairs trading strategies.
    Calculates 20 robust financial and statistical metrics and plots performance.
    """
    def __init__(self, df_results):
        self.df = df_results
        # Target strategies matching the new backtester output
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
        # A round trip trade is roughly 2 position changes (enter then exit)
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
        """Prints a heavily detailed 20-metric report comparing the strategies."""
        idx_series = pd.Series(self.df.index)
        unique_days_count = max(len(idx_series.dt.date.unique()), 1)
        ann_factor = len(self.df) / (unique_days_count / 252)

        results = {}
        for strat in self.strats:
            stats_dict = self._calc_metrics(strat, ann_factor)
            if stats_dict:
                results[strat] = stats_dict
        
        # Convert to DataFrame for a beautiful printed table
        report_df = pd.DataFrame(results)
        
        print(f"\n{'='*75}")
        print(f"{'QUANTITATIVE STRATEGY TEARSHEET':^75}")
        print(f"{'='*75}")
        
        for index, row in report_df.iterrows():
            if "---" in index:
                print(f"\n{index}")
                continue
                
            # Format percentages vs floats
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
        """Generates a 3-panel plot: Equity Curve, Underwater, and Distributions."""
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
            active_returns = returns[returns != 0] # Only plot active bars
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