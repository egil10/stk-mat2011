
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


class TEARSHEET:
    """
    Quant strategy tearsheet with expanded metrics and visuals.

    Expected columns in df_results:
      Return_{Baseline, AR, MS_AR}, Target_{...}, Danger_Regime_Prob (optional)
    """

    COLORS = {
        'Baseline': '#808080',
        'AR':       '#1f77b4',
        'MS_AR':    '#8856a7',
    }

    def __init__(self, df_results, df_params=None):
        self.df = df_results.copy()
        self.params = df_params
        self.strats = ['Baseline', 'AR', 'MS_AR']

        # Rebuild cumulative returns from raw returns — fixes WFO concat sawtooth
        for s in self.strats:
            if f'Return_{s}' in self.df.columns:
                self.df[f'CumReturn_{s}'] = self.df[f'Return_{s}'].fillna(0).cumsum()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calc_metrics(self, strat, ann_factor):
        ret_col = f'Return_{strat}'
        pos_col = f'Target_{strat}'
        if ret_col not in self.df.columns:
            return None

        r   = self.df[ret_col].fillna(0)
        pos = self.df[pos_col].fillna(0) if pos_col in self.df.columns else pd.Series(0, index=r.index)
        BPS = 10000

        # Core return stats
        tot_ret = r.sum() * BPS
        ann_ret = r.mean() * ann_factor * BPS
        ann_vol = r.std()  * np.sqrt(ann_factor) * BPS
        sharpe  = ann_ret / ann_vol if ann_vol else 0

        downside = r[r < 0]
        down_vol = downside.std() * np.sqrt(ann_factor) * BPS
        sortino  = ann_ret / down_vol if down_vol else 0

        cum = r.cumsum() * BPS
        dd  = cum - cum.cummax()
        max_dd = dd.min()
        calmar = abs(ann_ret / max_dd) if max_dd else 0

        # Drawdown duration (longest stretch underwater in bars)
        underwater = (dd < 0).astype(int)
        if underwater.any():
            groups = (underwater != underwater.shift()).cumsum()
            dd_durations = underwater.groupby(groups).sum()
            max_dd_duration = int(dd_durations.max())
        else:
            max_dd_duration = 0

        # Trade stats
        trades      = (pos.diff().abs() > 0).sum() / 2
        active_bars = (r != 0).sum()
        exposure    = active_bars / len(r) if len(r) else 0
        win_rate    = (r > 0).sum() / active_bars if active_bars else 0

        wins, losses = r[r > 0], r[r < 0]
        avg_win  = wins.mean()   * BPS if len(wins)   else 0
        avg_loss = losses.mean() * BPS if len(losses) else 0
        payoff   = abs(avg_win / avg_loss) if avg_loss else np.nan

        gross_prof = wins.sum()      * BPS
        gross_loss = abs(losses.sum() * BPS)
        profit_factor = gross_prof / gross_loss if gross_loss else np.nan

        # Risk & distribution
        skew     = stats.skew(r)     if len(r) > 2 else np.nan
        kurt     = stats.kurtosis(r) if len(r) > 2 else np.nan
        var_95   = np.percentile(r, 5) * BPS
        cvar_95  = r[r <= np.percentile(r, 5)].mean() * BPS

        # Tail ratio: |95th pct| / |5th pct|
        tail_ratio = (abs(np.percentile(r, 95)) /
                      abs(np.percentile(r, 5))) if np.percentile(r, 5) != 0 else np.nan

        # Percent of rolling-monthly periods positive
        monthly_ret = r.resample('ME').sum() if isinstance(r.index, pd.DatetimeIndex) else pd.Series(dtype=float)
        pct_pos_months = (monthly_ret > 0).mean() if len(monthly_ret) else np.nan

        # Best / worst month
        best_month  = monthly_ret.max() * BPS if len(monthly_ret) else np.nan
        worst_month = monthly_ret.min() * BPS if len(monthly_ret) else np.nan

        # Ulcer Index — RMS of drawdown over time
        ulcer = np.sqrt((dd ** 2).mean()) if len(dd) else np.nan

        return {
            "--- RETURN ---": "",
            "Total Return (bps)":      tot_ret,
            "Annual Return (bps)":     ann_ret,
            "Best Month (bps)":        best_month,
            "Worst Month (bps)":       worst_month,
            "Positive Months %":       pct_pos_months,

            "--- RISK ---": "",
            "Annual Volatility (bps)": ann_vol,
            "Max Drawdown (bps)":      max_dd,
            "Max DD Duration (bars)":  max_dd_duration,
            "Ulcer Index":             ulcer,
            "Value at Risk 95% (bps)": var_95,
            "CVaR 95% (bps)":          cvar_95,

            "--- RISK-ADJUSTED ---": "",
            "Sharpe Ratio":            sharpe,
            "Sortino Ratio":           sortino,
            "Calmar Ratio":            calmar,
            "Profit Factor":           profit_factor,
            "Payoff Ratio":            payoff,
            "Tail Ratio":              tail_ratio,

            "--- TRADING ---": "",
            "Number of Trades":        trades,
            "Win Rate":                win_rate,
            "Market Exposure":         exposure,
            "Avg Win (bps)":           avg_win,
            "Avg Loss (bps)":          avg_loss,

            "--- DISTRIBUTION ---": "",
            "Skewness":                skew,
            "Kurtosis":                kurt,
        }

    def generate_report(self):
        if isinstance(self.df.index, pd.DatetimeIndex):
            n_days = max((self.df.index.max() - self.df.index.min()).days, 1)
            ann_factor = len(self.df) / (n_days / 365.25)
        else:
            ann_factor = len(self.df) / (max(len(self.df.index.date.unique()), 1) / 252)

        results = {s: self._calc_metrics(s, ann_factor) for s in self.strats}
        results = {k: v for k, v in results.items() if v is not None}

        print("\n" + "=" * 85)
        print(f"{'QUANTITATIVE STRATEGY TEARSHEET':^85}")
        print("=" * 85)

        header = f"{'Metric':<30} |" + "".join(f" {s:<14} |" for s in results)
        print(header)
        print("-" * len(header))

        metric_names = list(next(iter(results.values())).keys())
        for m in metric_names:
            if "---" in m:
                print(f"\n{m}")
                continue
            row = f"{m:<30} |"
            for s in results:
                v = results[s][m]
                if isinstance(v, str):
                    row += f" {v:<14} |"
                elif pd.isna(v):
                    row += f" {'NaN':<14} |"
                elif 'Rate' in m or 'Exposure' in m or 'Months %' in m:
                    row += f" {v:<14.2%} |"
                elif 'Duration' in m or 'Trades' in m:
                    row += f" {v:<14.0f} |"
                else:
                    row += f" {v:<14.2f} |"
            print(row)
        print("=" * 85 + "\n")

    # ------------------------------------------------------------------
    # Main performance plot (fixed cumulative + richer panels)
    # ------------------------------------------------------------------

    def plot_performance(self):
        valid = [s for s in self.strats if f'Return_{s}' in self.df.columns]
        BPS = 10000

        fig = plt.figure(figsize=(14, 16))
        gs = gridspec.GridSpec(5, 2, height_ratios=[2, 1, 1.2, 1.2, 1.2],
                               hspace=0.45, wspace=0.25)

        # --- Panel 1: Equity Curve (full width) --------------------------
        ax1 = fig.add_subplot(gs[0, :])
        for s in valid:
            cum = self.df[f'Return_{s}'].fillna(0).cumsum() * BPS
            ax1.plot(self.df.index, cum, color=self.COLORS[s], lw=1.5, label=s)
        ax1.set_title("Cumulative Strategy Returns (bps)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Cumulative PnL (bps)")
        ax1.axhline(0, color='black', lw=0.8, alpha=0.5)
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Drawdown (full width) ------------------------------
        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        for s in valid:
            cum = self.df[f'Return_{s}'].fillna(0).cumsum() * BPS
            dd = cum - cum.cummax()
            ax2.fill_between(self.df.index, dd, 0,
                             color=self.COLORS[s], alpha=0.35, label=f'{s}')
        ax2.set_title("Drawdown (bps)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Drawdown")
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)

        # --- Panel 3: Monthly Returns Bar Chart (full width) -------------
        ax3 = fig.add_subplot(gs[2, :])
        if isinstance(self.df.index, pd.DatetimeIndex):
            width = 0.27
            monthly = {s: self.df[f'Return_{s}'].fillna(0).resample('ME').sum() * BPS
                       for s in valid}
            months = next(iter(monthly.values())).index
            x = np.arange(len(months))
            for i, s in enumerate(valid):
                ax3.bar(x + (i - 1) * width, monthly[s].values, width,
                        color=self.COLORS[s], label=s, alpha=0.85)
            ax3.set_xticks(x)
            ax3.set_xticklabels([d.strftime('%Y-%m') for d in months],
                                rotation=45, ha='right', fontsize=8)
            ax3.axhline(0, color='black', lw=0.8)
            ax3.set_title("Monthly Returns (bps)", fontsize=12, fontweight='bold')
            ax3.set_ylabel("Return (bps)")
            ax3.legend(loc='upper right')
            ax3.grid(True, axis='y', alpha=0.3)

        # --- Panel 4a: Rolling Volatility --------------------------------
        ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
        win = max(200, len(self.df) // 100)  # ~1% of bars
        for s in valid:
            rv = self.df[f'Return_{s}'].fillna(0).rolling(win).std() * np.sqrt(win) * BPS
            ax4.plot(self.df.index, rv, color=self.COLORS[s], lw=1.3, label=s)
        ax4.set_title(f"Rolling Volatility ({win}-bar window)", fontsize=11, fontweight='bold')
        ax4.set_ylabel("Vol (bps)")
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # --- Panel 4b: Rolling Sharpe ------------------------------------
        ax5 = fig.add_subplot(gs[3, 1], sharex=ax1)
        for s in valid:
            r = self.df[f'Return_{s}'].fillna(0)
            rs = r.rolling(win).mean() / r.rolling(win).std() * np.sqrt(win)
            ax5.plot(self.df.index, rs, color=self.COLORS[s], lw=1.3, label=s)
        ax5.axhline(0, color='black', lw=0.8, alpha=0.5)
        ax5.set_title(f"Rolling Sharpe ({win}-bar window)", fontsize=11, fontweight='bold')
        ax5.set_ylabel("Sharpe")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.3)

        # --- Panel 5a: Return Distribution (linear, clipped) -------------
        ax6 = fig.add_subplot(gs[4, 0])
        for s in valid:
            active = (self.df[f'Return_{s}'].fillna(0) * BPS)
            active = active[active != 0]
            if len(active):
                clip = np.percentile(np.abs(active), 99)
                ax6.hist(active.clip(-clip, clip), bins=60, alpha=0.5,
                         color=self.COLORS[s], label=s, density=True)
        ax6.axvline(0, color='black', ls='--', lw=1)
        ax6.set_title("Active Bar Returns (99% clipped)", fontsize=11, fontweight='bold')
        ax6.set_xlabel("Return (bps)")
        ax6.set_ylabel("Density")
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3)

        # --- Panel 5b: Tail Distribution (log) ---------------------------
        ax7 = fig.add_subplot(gs[4, 1])
        for s in valid:
            active = (self.df[f'Return_{s}'].fillna(0) * BPS)
            active = active[active != 0]
            if len(active):
                ax7.hist(active, bins=80, alpha=0.5, color=self.COLORS[s],
                         label=s, density=True, log=True)
        ax7.axvline(0, color='black', ls='--', lw=1)
        ax7.set_title("Return Distribution (log y, tails visible)",
                      fontsize=11, fontweight='bold')
        ax7.set_xlabel("Return (bps)")
        ax7.set_ylabel("Density (log)")
        ax7.legend(loc='upper right', fontsize=8)
        ax7.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Positions & regimes — readable version
    # ------------------------------------------------------------------

    def plot_positions_and_regimes(self):
        """Heatmap-style position display + danger regime overlay."""
        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 2], hspace=0.25)

        strat_rows = [('Baseline', 0), ('AR', 1), ('MS_AR', 2)]
        cmap = LinearSegmentedColormap.from_list(
            'pos_cmap',
            [(0, '#d62728'), (0.5, '#f0f0f0'), (1, '#2ca02c')]  # red / grey / green
        )

        # Positions as heatmap strips — much easier to read than line plots
        for name, row in strat_rows:
            ax = fig.add_subplot(gs[row])
            pos_col = f'Target_{name}'
            if pos_col not in self.df.columns:
                continue
            pos = self.df[pos_col].fillna(0).values.reshape(1, -1)
            ax.imshow(pos, aspect='auto', cmap=cmap, vmin=-1, vmax=1,
                      extent=[mdates.date2num(self.df.index[0]),
                              mdates.date2num(self.df.index[-1]), 0, 1])
            ax.set_yticks([])
            ax.set_ylabel(name, rotation=0, ha='right', va='center',
                          fontsize=11, fontweight='bold')
            ax.xaxis_date()
            if row < 2:
                ax.set_xticklabels([])

        # Exposure through time (stacked area)
        ax_exp = fig.add_subplot(gs[3])
        for s, _ in strat_rows:
            if f'Target_{s}' in self.df.columns:
                exposed = (self.df[f'Target_{s}'].fillna(0) != 0).astype(int)
                roll_exp = exposed.rolling(500, min_periods=1).mean() * 100
                ax_exp.plot(self.df.index, roll_exp, color=self.COLORS[s],
                            lw=1.2, label=s)
        ax_exp.set_ylabel("Exposure %\n(rolling 500-bar)", fontsize=9)
        ax_exp.set_ylim(0, 105)
        ax_exp.legend(loc='upper right', fontsize=8)
        ax_exp.grid(True, alpha=0.3)
        ax_exp.set_xticklabels([])

        # Danger regime probability — smoothed for readability
        ax_reg = fig.add_subplot(gs[4])
        if 'Danger_Regime_Prob' in self.df.columns:
            p = self.df['Danger_Regime_Prob']
            smoothed = p.rolling(100, min_periods=1).mean()
            ax_reg.fill_between(self.df.index, 0, smoothed,
                                color='#e41a1c', alpha=0.4,
                                label='P(Danger), 100-bar smoothed')
            ax_reg.plot(self.df.index, smoothed, color='#b30000', lw=0.8)
            ax_reg.axhline(0.7, color='black', ls='--', lw=1,
                           label='Kill-switch threshold')
        ax_reg.set_title("Smoothed P(State = Danger)", fontsize=11, fontweight='bold')
        ax_reg.set_ylabel("Probability")
        ax_reg.set_ylim(0, 1.05)
        ax_reg.legend(loc='upper right', fontsize=8)
        ax_reg.grid(True, alpha=0.3)

        # Legend for position heatmap
        fig.text(0.92, 0.82,
                 'Position\n\n'
                 '🟩 Long\n'
                 '⬜ Flat\n'
                 '🟥 Short',
                 fontsize=9, va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

        plt.suptitle("Strategy Positions Over Time (heatmap) + Regime Overlay",
                     fontweight='bold', fontsize=13, y=0.995)
        plt.show()

    # ------------------------------------------------------------------
    # Markov dynamics (unchanged from yours — small polish)
    # ------------------------------------------------------------------

    def plot_markov_dynamics(self):
        if self.params is None or 'Safe_Mean' not in self.params.columns:
            print("Markov parameters not found in df_params.")
            return

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, hspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.params.index, self.params['Danger_Variance'],
                 color='#d62728', label='Danger $\\sigma^2$', lw=1.3)
        ax1.plot(self.params.index, self.params['Safe_Variance'],
                 color='#2ca02c', label='Safe $\\sigma^2$', lw=1.3)
        ax1.set_yscale('log')
        ax1.set_title("Rolling Markov Variances (log scale)", fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(self.params.index, self.params['Danger_Mean'] * 10000,
                 color='#8b0000', label='Danger $\\mu$ (bps)', lw=1.3)
        ax2.plot(self.params.index, self.params['Safe_Mean'] * 10000,
                 color='#006400', label='Safe $\\mu$ (bps)', lw=1.3)
        ax2.axhline(0, color='black', ls='--', lw=1)
        ax2.set_title("Rolling Markov Expected Returns", fontweight='bold')
        ax2.set_ylabel("$\\mu$ (bps)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(self.params.index, self.params['P_Danger_Danger'],
                 color='#fc9272', label='P(Danger | Danger)', lw=1.3)
        ax3.plot(self.params.index, self.params['P_Safe_Safe'],
                 color='#a1d99b', label='P(Safe | Safe)', lw=1.3)
        ax3.set_title("Regime Persistence", fontweight='bold')
        ax3.set_ylabel("Probability")
        ax3.set_ylim(0, 1.05)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()