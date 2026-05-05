
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import skew, kurtosis

try:
    from plotting_utils import pdf_filename, save_figure_pdf, ECON, COL_A, COL_B, apply_econ_style
except ImportError:
    from .plotting_utils import pdf_filename, save_figure_pdf, ECON, COL_A, COL_B, apply_econ_style


class DESCRIPTIVE:
    """
    Microstructure EDA for pair tick data.
    All prints emitted first, then one multi-panel figure.
    """

    def __init__(self, df, name_a="Asset A", name_b="Asset B",
                 save_pdf=False, pdf_dir=None, pdf_prefix=None):
        self.df = df.copy()
        self.name_a = name_a
        self.name_b = name_b
        self.save_pdf = save_pdf
        self.pdf_dir = pdf_dir
        self.pdf_prefix = pdf_prefix or f"{name_a}_{name_b}"

        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

        self.df['hour']    = self.df.index.hour
        self.df['dow']     = self.df.index.dayofweek
        self.df['date']    = self.df.index.date

        # Sniff for optional columns so we degrade gracefully
        self._has_volume = any(c.lower().startswith('volume') for c in self.df.columns)
        self._vol_cols = {
            'A': next((c for c in self.df.columns if c.lower() in ('volume_a', 'vol_a', 'volume_a_total')), None),
            'B': next((c for c in self.df.columns if c.lower() in ('volume_b', 'vol_b', 'volume_b_total')), None),
        }

    # =================================================================
    # PRINT BLOCK
    # =================================================================

    def _print_header(self, title):
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print('=' * 70)

    def _print_overview(self):
        self._print_header("DATASET OVERVIEW")
        start, end = self.df.index.min(), self.df.index.max()
        span_days = (end - start).days
        trading_days = len(self.df['date'].unique())
        bars_per_day = len(self.df) / max(trading_days, 1)

        print(f"Pair:              {self.name_a}  vs  {self.name_b}")
        print(f"Bars:              {len(self.df):,}")
        print(f"Period:            {start}  →  {end}")
        print(f"Calendar span:     {span_days} days")
        print(f"Trading days:      {trading_days}")
        print(f"Bars per day:      {bars_per_day:.1f} (avg)")
        print(f"Hours covered:     {sorted(self.df['hour'].unique())}")

    def _print_liquidity(self):
        self._print_header("LIQUIDITY PROFILE")
        hs_a = self.df['HalfSpread_A_bps']
        hs_b = self.df['HalfSpread_B_bps']
        total_spread = hs_a + hs_b

        hourly = self.df.groupby('hour')[['HalfSpread_A_bps', 'HalfSpread_B_bps']].median()
        rolling_4h = hourly.mean(axis=1).rolling(4).mean()
        best_end = rolling_4h.idxmin()
        worst_end = rolling_4h.idxmax()

        print(f"Median spread:           A={hs_a.median():.2f}  B={hs_b.median():.2f}  bps")
        print(f"P95 spread (tail cost):  A={hs_a.quantile(0.95):.2f}  B={hs_b.quantile(0.95):.2f}  bps")
        print(f"Tightest 4h window:      {int(best_end-3):02d}:00 → {int(best_end):02d}:00 UTC")
        print(f"Widest   4h window:      {int(worst_end-3):02d}:00 → {int(worst_end):02d}:00 UTC")
        print(f"Round-trip cost (median total): {total_spread.median():.2f} bps")

    def _print_return_stats(self):
        self._print_header("RETURN STATISTICS")
        ra = self.df['Return_A'].dropna()
        rb = self.df['Return_B'].dropna()
        BPS = 10000

        def stats_row(r):
            return [
                r.mean() * BPS,
                r.std() * BPS,
                skew(r),
                kurtosis(r),
                np.percentile(r, 5)  * BPS,
                np.percentile(r, 95) * BPS,
                r.min() * BPS,
                r.max() * BPS,
            ]

        cols = [
            'Mean (bps)', 
            'Vol (bps)', 
            'Skew', 
            'Kurt', 
            'P5 (bps)', 
            'P95 (bps)', 
            'Min (bps)', 
            'Max (bps)'
        ]

        stats_df = pd.DataFrame(
            {self.name_a: stats_row(ra), self.name_b: stats_row(rb)},
            index=cols,
        ).T
        print(stats_df.to_string(float_format=lambda x: f"{x:>9.3f}"))

        if stats_df['Kurt'].max() > 10:
            print(f"\n[!] High Kurtosis (>10). Expect HMM false-positive 'Danger' regimes.")

        # Volatility clustering test — Ljung-Box on squared returns
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_a = acorr_ljungbox(ra**2, lags=[20], return_df=True)
            lb_b = acorr_ljungbox(rb**2, lags=[20], return_df=True)
            print(f"\nLjung-Box on r² (lag 20):  {self.name_a} p={lb_a['lb_pvalue'].iloc[0]:.4f}  "
                  f"{self.name_b} p={lb_b['lb_pvalue'].iloc[0]:.4f}")
            print("  (p < 0.05 = volatility clustering present)")
        except ImportError:
            pass

    def _print_pair_relation(self):
        self._print_header("PAIR RELATIONSHIP")
        ra, rb = self.df['Return_A'].dropna(), self.df['Return_B'].dropna()
        common = ra.index.intersection(rb.index)
        ra, rb = ra.loc[common], rb.loc[common]

        corr_pearson  = ra.corr(rb, method='pearson')
        corr_spearman = ra.corr(rb, method='spearman')
        beta = np.cov(ra, rb)[0, 1] / np.var(ra) if np.var(ra) else np.nan

        print(f"Pearson correlation:   {corr_pearson:.4f}")
        print(f"Spearman correlation:  {corr_spearman:.4f}   (rank-based, robust to outliers)")
        print(f"OLS β (B on A):        {beta:.4f}")

        rolling_corr = ra.rolling(500).corr(rb)
        print(f"Rolling corr (500-bar): min={rolling_corr.min():.3f}  "
              f"max={rolling_corr.max():.3f}  median={rolling_corr.median():.3f}")

    def _print_intraday(self):
        self._print_header("INTRADAY PROFILE")
        vol = self.df.groupby('hour')['Return_A'].std() * 10000
        print("Hourly volatility of Return_A (bps):")
        print(vol.to_frame('Vol').T.round(2).to_string())

        peak_h = vol.idxmax()
        quiet_h = vol.idxmin()
        print(f"\nMost volatile hour:  {peak_h:02d}:00  ({vol[peak_h]:.2f} bps)")
        print(f"Quietest hour:       {quiet_h:02d}:00  ({vol[quiet_h]:.2f} bps)")
        print(f"Peak/Quiet ratio:    {vol[peak_h]/vol[quiet_h]:.2f}×")

    def _print_weekday(self):
        self._print_header("DAY-OF-WEEK PROFILE")
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        grp = self.df.groupby('dow')
        summary = pd.DataFrame({
            'Bars':        grp.size(),
            'Vol A (bps)': grp['Return_A'].std() * 10000,
            'Vol B (bps)': grp['Return_B'].std() * 10000,
            'Spread A':    grp['HalfSpread_A_bps'].median(),
            'Spread B':    grp['HalfSpread_B_bps'].median(),
        })
        summary.index = [dow_names[i] for i in summary.index]
        print(summary.round(2).to_string())

    # =================================================================
    # PLOT GRID
    # =================================================================

    def _plot_everything(self, save_pdf=None, pdf_dir=None, filename=None):
        apply_econ_style()
        fig = plt.figure(figsize=(16, 22))
        gs = gridspec.GridSpec(7, 2, hspace=0.6, wspace=0.3,
                               height_ratios=[1, 1, 1, 1, 1, 1, 1])

        col_a, col_b = COL_A, COL_B
        BPS = 10000

        # ---- Row 0: Hourly spread + Hourly volatility ----
        ax = fig.add_subplot(gs[0, 0])
        hourly_sp = self.df.groupby('hour')[['HalfSpread_A_bps', 'HalfSpread_B_bps']].median()
        hourly_sp.plot(kind='bar', ax=ax, color=[col_a, col_b], alpha=0.85, width=0.85)
        ax.set_title("Median Spread by Hour (UTC)", fontweight='bold')
        ax.set_ylabel("bps")
        ax.set_xlabel("Hour")
        ax.grid(alpha=0.3)
        ax.legend([self.name_a, self.name_b], fontsize=8)

        ax = fig.add_subplot(gs[0, 1])
        hourly_vol = self.df.groupby('hour')[['Return_A', 'Return_B']].std() * BPS
        hourly_vol.plot(kind='bar', ax=ax, color=[col_a, col_b], alpha=0.85, width=0.85)
        ax.set_title("Return Volatility by Hour (UTC)", fontweight='bold')
        ax.set_ylabel("bps")
        ax.set_xlabel("Hour")
        ax.grid(alpha=0.3)
        ax.legend([self.name_a, self.name_b], fontsize=8)

        # ---- Row 1: Hourly mean return + Hourly bar count ----
        ax = fig.add_subplot(gs[1, 0])
        hourly_mean = self.df.groupby('hour')[['Return_A', 'Return_B']].mean() * BPS
        hourly_mean.plot(kind='bar', ax=ax, color=[col_a, col_b], alpha=0.85, width=0.85)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title("Mean Return by Hour (UTC)", fontweight='bold')
        ax.set_ylabel("bps")
        ax.set_xlabel("Hour")
        ax.grid(alpha=0.3)
        ax.legend([self.name_a, self.name_b], fontsize=8)

        ax = fig.add_subplot(gs[1, 1])
        self.df.groupby('hour').size().plot(kind='bar', ax=ax, color='#444', alpha=0.8)
        ax.set_title("Bar Count by Hour (activity proxy)", fontweight='bold')
        ax.set_ylabel("Bars")
        ax.set_xlabel("Hour")
        ax.grid(alpha=0.3)

        # ---- Row 2: Day-of-week vol + weekday bar count ----
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax = fig.add_subplot(gs[2, 0])
        dow_vol = self.df.groupby('dow')[['Return_A', 'Return_B']].std() * BPS
        dow_vol.index = [dow_names[i] for i in dow_vol.index]
        dow_vol.plot(kind='bar', ax=ax, color=[col_a, col_b], alpha=0.85, width=0.85)
        ax.set_title("Volatility by Weekday", fontweight='bold')
        ax.set_ylabel("bps")
        ax.set_xlabel("")
        ax.grid(alpha=0.3)
        ax.legend([self.name_a, self.name_b], fontsize=8)

        ax = fig.add_subplot(gs[2, 1])
        dow_cnt = self.df.groupby('dow').size()
        dow_cnt.index = [dow_names[i] for i in dow_cnt.index]
        dow_cnt.plot(kind='bar', ax=ax, color='#444', alpha=0.8)
        ax.set_title("Bar Count by Weekday", fontweight='bold')
        ax.set_ylabel("Bars")
        ax.set_xlabel("")
        ax.grid(alpha=0.3)

        # ---- Row 3: Rolling vol (time series) + Rolling correlation ----
        ax = fig.add_subplot(gs[3, 0])
        roll_vol_a = self.df['Return_A'].rolling(100).std() * BPS
        roll_vol_b = self.df['Return_B'].rolling(100).std() * BPS
        ax.plot(self.df.index, roll_vol_a, color=col_a, lw=0.8, alpha=0.75, label=self.name_a)
        ax.plot(self.df.index, roll_vol_b, color=col_b, lw=0.8, alpha=0.75, label=self.name_b)
        ax.set_title("Rolling 100-bar Realized Volatility", fontweight='bold')
        ax.set_ylabel("bps")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[3, 1])
        roll_corr = self.df['Return_A'].rolling(500).corr(self.df['Return_B'])
        ax.plot(self.df.index, roll_corr, color=ECON['green'], lw=0.9)
        ax.axhline(roll_corr.median(), color=ECON['navy'], ls='--', lw=1, label=f'Median {roll_corr.median():.2f}')
        ax.set_title(f"Rolling 500-bar Correlation ({self.name_a}, {self.name_b})", fontweight='bold')
        ax.set_ylabel("ρ")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ---- Row 4: Return histogram + QQ plot ----
        ax = fig.add_subplot(gs[4, 0])
        ra_bps = self.df['Return_A'].dropna() * BPS
        rb_bps = self.df['Return_B'].dropna() * BPS
        clip = np.percentile(np.abs(pd.concat([ra_bps, rb_bps])), 99)
        ax.hist(ra_bps.clip(-clip, clip), bins=80, alpha=0.5, color=col_a, label=self.name_a, density=True)
        ax.hist(rb_bps.clip(-clip, clip), bins=80, alpha=0.5, color=col_b, label=self.name_b, density=True)
        ax.axvline(0, color='black', ls='--', lw=1)
        ax.set_title("Return Distributions (99% clipped)", fontweight='bold')
        ax.set_xlabel("bps")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[4, 1])
        stats.probplot(ra_bps, dist='norm', plot=ax)
        ax.get_lines()[0].set_markerfacecolor(col_a)
        ax.get_lines()[0].set_markeredgecolor(col_a)
        ax.get_lines()[0].set_markersize(2)
        ax.get_lines()[1].set_color('black')
        ax.set_title(f"Q-Q Plot vs Normal ({self.name_a})", fontweight='bold')
        ax.grid(alpha=0.3)

        # ---- Row 5: Scatter A vs B + ACF of squared returns ----
        ax = fig.add_subplot(gs[5, 0])
        idx = np.random.default_rng(0).choice(len(ra_bps), size=min(5000, len(ra_bps)), replace=False)
        ra_s = ra_bps.iloc[idx] if hasattr(ra_bps, 'iloc') else ra_bps[idx]
        rb_s = rb_bps.iloc[idx] if hasattr(rb_bps, 'iloc') else rb_bps[idx]
        ax.scatter(ra_s, rb_s, s=3, alpha=0.25, color=ECON['grey'])
        if len(ra_s) > 2:
            m, b = np.polyfit(ra_s, rb_s, 1)
            xline = np.linspace(ra_s.min(), ra_s.max(), 50)
            ax.plot(xline, m*xline + b, color=ECON['red'], lw=1.5, label=f'β={m:.2f}')
            ax.legend(fontsize=8)
        ax.set_title(f"Return Scatter ({self.name_a} vs {self.name_b})", fontweight='bold')
        ax.set_xlabel(f"{self.name_a} (bps)")
        ax.set_ylabel(f"{self.name_b} (bps)")
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[5, 1])
        lags = np.arange(1, 51)
        ra = self.df['Return_A'].dropna()
        acf_sq = [ra.pow(2).autocorr(l) for l in lags]
        acf_rn = [ra.autocorr(l) for l in lags]
        ax.bar(lags - 0.2, acf_rn, width=0.4, color=col_a, alpha=0.7, label='Returns')
        ax.bar(lags + 0.2, acf_sq, width=0.4, color=ECON['red'], alpha=0.7, label='Squared Returns')
        ax.axhline(0, color=ECON['navy'], lw=0.8)
        ci = 1.96 / np.sqrt(len(ra))
        ax.axhline(ci, color='black', ls='--', lw=0.8, alpha=0.5)
        ax.axhline(-ci, color='black', ls='--', lw=0.8, alpha=0.5)
        ax.set_title(f"ACF of {self.name_a}: Returns vs r² (50 lags)", fontweight='bold')
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ---- Row 6: Rolling vol distribution + Spread vs Vol ----
        ax = fig.add_subplot(gs[6, 0])
        rv = roll_vol_a.dropna()
        ax.hist(rv, bins=60, color=col_a, alpha=0.7, density=True)
        ax.axvline(rv.median(), color=ECON['navy'], ls='--', lw=1.2, label=f'Median {rv.median():.1f}')
        ax.axvline(rv.quantile(0.95), color=ECON['red'], ls='--', lw=1.2, label=f'P95 {rv.quantile(0.95):.1f}')
        ax.set_title(f"Rolling Vol Distribution — {self.name_a}", fontweight='bold')
        ax.set_xlabel("Vol (bps)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[6, 1])
        hourly_sp_mean = self.df.groupby('hour')[['HalfSpread_A_bps', 'HalfSpread_B_bps']].median().mean(axis=1)
        hourly_vol_mean = (self.df.groupby('hour')[['Return_A', 'Return_B']].std() * BPS).mean(axis=1)
        ax.scatter(hourly_sp_mean, hourly_vol_mean, s=60, c=hourly_sp_mean.index, cmap='viridis', edgecolor='black')
        for h in hourly_sp_mean.index:
            ax.annotate(f'{h}', (hourly_sp_mean[h], hourly_vol_mean[h]),
                        fontsize=7, ha='center', va='center')
        ax.set_title("Spread vs Volatility (by Hour)", fontweight='bold')
        ax.set_xlabel("Median spread (bps)")
        ax.set_ylabel("Volatility (bps)")
        ax.grid(alpha=0.3)

        plt.suptitle(f"EDA: {self.name_a} vs {self.name_b}",
                     fontsize=15, fontweight='bold', y=0.995)
        save_figure_pdf(
            fig,
            filename or pdf_filename(self.pdf_prefix, "eda"),
            pdf_dir=pdf_dir or self.pdf_dir,
            enabled=self.save_pdf if save_pdf is None else save_pdf,
        )
        plt.show()

    # =================================================================
    # PUBLIC
    # =================================================================

    def generate_full_eda(self, save_pdf=None, pdf_dir=None, filename=None):
        # All prints first
        self._print_overview()
        self._print_liquidity()
        self._print_return_stats()
        self._print_pair_relation()
        self._print_intraday()
        self._print_weekday()

        # Then one big grid of plots
        self._plot_everything(save_pdf=save_pdf, pdf_dir=pdf_dir, filename=filename)