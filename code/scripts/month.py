"""
End-to-end pairs-trading runner for the `code/monthly/` notebooks.

The MONTH class wraps the existing pipeline (SPREAD -> ENGINE -> BACKTESTER ->
TEARSHEET) and exposes a handful of high-level methods so the notebooks stay
short and declarative.

    from month import MONTH

    m = MONTH('AUDUSD', 'NZDUSD')          # default config (see DEFAULT_CFG)
    m.run_months(['202408', '202409', '202508'])
    m.summary()                            # pretty tables + verdict
    m.tearsheets()                         # per-month report + plot
    m.sweep('202408')                      # 3 x 3 x 3 sensitivity grid

Override any default by passing it as a keyword to the constructor:

    m = MONTH('GBPUSD', 'EURUSD', train_days=5, k_regimes=3)
"""
import itertools

import numpy as np
import pandas as pd

from spread import SPREAD
from engine import ENGINE
from backtester import BACKTESTER
from tearsheet import TEARSHEET


class MONTH:
    DEFAULT_CFG = {
        # ---- bar aggregation ----
        'agg_type':       'tick',
        'bar_threshold':  500,
        'start_hour':     0,
        'end_hour':       24,

        # ---- engine (per-day rolling fit) ----
        'train_days':     3,
        'coint_window':   150,
        'z_window':       50,
        'k_regimes':      2,
        'winsorize_std':  4.0,
        'scaling':        10000,

        # ---- backtester trading rules ----
        'z_quiet':           1.3,
        'z_volatile':        2.5,
        'exit_z':            0.0,
        'danger_threshold':  0.30,
        'fee_bps':           0.5,
        'slippage_mode':     'half_spread',
        'flatten_eod':       True,

        # ---- metrics ----
        'ann_factor': 252 * 24 * 60,    # bars/year for tick-clock Sharpe
    }

    def __init__(self, name_a, name_b, data_dir='../data/processed', **overrides):
        self.name_a    = name_a
        self.name_b    = name_b
        self.pair_name = f'{name_a}_{name_b}'
        self.data_dir  = data_dir
        self.cfg       = {**self.DEFAULT_CFG, **overrides}

        self.runs       = {}    # month -> (results, df_params)
        self.summary_df = None
        self.sweep_df   = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _files(self, months):
        a, b = self.name_a.lower(), self.name_b.lower()
        d = self.data_dir
        return [
            [f'{d}/{a}_dukascopy_ask_{m}.parquet' for m in months],
            [f'{d}/{a}_dukascopy_bid_{m}.parquet' for m in months],
            [f'{d}/{b}_dukascopy_ask_{m}.parquet' for m in months],
            [f'{d}/{b}_dukascopy_bid_{m}.parquet' for m in months],
        ]

    def _sharpe(self, r):
        r = r.fillna(0)
        s = r.std()
        if not (s > 0 and np.isfinite(s)):
            return 0.0
        return float(r.mean() / s * np.sqrt(self.cfg['ann_factor']))

    def _metrics(self, results):
        out = {}
        for strat in ('Baseline', 'AR', 'MS_AR'):
            r = results[f'Return_{strat}'].fillna(0)
            out[f'{strat}_Sharpe'] = self._sharpe(r)
            out[f'{strat}_PnLbps'] = float(r.sum() * 1e4)
            out[f'{strat}_Trades'] = int((results[f'Target_{strat}'].diff().abs() > 0).sum() / 2)
        return out

    def _engine_run(self, month):
        c = self.cfg
        builder = SPREAD(
            agg_type     = c['agg_type'],
            threshold    = c['bar_threshold'],
            active_hours = (c['start_hour'], c['end_hour']),
        )
        df = builder.build(self._files([month]))
        live, df_params = ENGINE.walk_forward(
            df            = df,
            train_days    = c['train_days'],
            coint_window  = c['coint_window'],
            z_window      = c['z_window'],
            k_regimes     = c['k_regimes'],
            winsorize_std = c['winsorize_std'],
            scaling       = c['scaling'],
            print_freq    = 10**6,        # silence per-day prints
        )
        return df, live, df_params

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def run(self, month, verbose=True):
        """Build -> engine -> backtest for one month. Returns (results, df_params)."""
        df, live, df_params = self._engine_run(month)
        if verbose:
            n_days = df.index.normalize().unique().shape[0]
            print(f'  bars={len(df):,}  days={n_days}')

        c = self.cfg
        results = BACKTESTER(live).run(
            z_quiet          = c['z_quiet'],
            z_volatile       = c['z_volatile'],
            exit_z           = c['exit_z'],
            danger_threshold = c['danger_threshold'],
            fee_bps          = c['fee_bps'],
            slippage_mode    = c['slippage_mode'],
            flatten_eod      = c['flatten_eod'],
        )
        return results, df_params

    def run_months(self, months):
        """Sequentially run every month. Stores results in self.runs and self.summary_df."""
        rows = []
        for m in months:
            print(f'\n--- {m} ---')
            try:
                results, df_params = self.run(m)
                self.runs[m] = (results, df_params)
                mm = self._metrics(results)
                mm['Month'] = m
                rows.append(mm)
                print(f"  Sharpe  Baseline={mm['Baseline_Sharpe']:+.2f}  "
                      f"AR={mm['AR_Sharpe']:+.2f}  MS_AR={mm['MS_AR_Sharpe']:+.2f}")
            except Exception as e:
                print(f'  FAILED: {type(e).__name__}: {e}')

        self.summary_df = pd.DataFrame(rows).set_index('Month') if rows else None
        return self.summary_df

    def summary(self):
        """Pretty-print cross-month summary tables + verdict."""
        if self.summary_df is None or len(self.summary_df) == 0:
            print('No runs to summarise.')
            return

        s = self.summary_df
        print('=== SHARPE BY MONTH (tick-clock annualised) ===')
        print(s[['Baseline_Sharpe', 'AR_Sharpe', 'MS_AR_Sharpe']].round(2).to_string())

        print('\n=== PnL bps BY MONTH ===')
        print(s[['Baseline_PnLbps', 'AR_PnLbps', 'MS_AR_PnLbps']].round(1).to_string())

        print('\n=== TRADES BY MONTH ===')
        print(s[['Baseline_Trades', 'AR_Trades', 'MS_AR_Trades']].to_string())

        edge = pd.DataFrame({
            'MS_AR vs Baseline': s['MS_AR_Sharpe'] - s['Baseline_Sharpe'],
            'MS_AR vs AR':       s['MS_AR_Sharpe'] - s['AR_Sharpe'],
        })
        print('\n=== EDGE: MS_AR Sharpe minus baseline ===')
        print(edge.round(2).to_string())

        n      = len(edge)
        wins_b = int((edge['MS_AR vs Baseline'] > 0).sum())
        wins_a = int((edge['MS_AR vs AR']       > 0).sum())
        print(f'\nMS_AR beat Baseline in {wins_b}/{n} months  |  '
              f'MS_AR beat AR in {wins_a}/{n} months')

    def tearsheets(self, plot=True):
        """Full TEARSHEET report (and optional plot) for every month run."""
        for m, (results, df_params) in self.runs.items():
            print(f'\n{"="*30}  {m}  {"="*30}')
            ts = TEARSHEET(results, df_params=df_params,
                           pdf_prefix=f'{self.pair_name}_{m}')
            ts.generate_report()
            if plot:
                ts.plot_performance()

    def sweep(self, month, z_quiet=None, z_volatile=None, dthresh=None):
        """Sensitivity sweep on one chosen month. 3x3x3 by default."""
        z_quiet    = z_quiet    or [1.0, 1.3, 1.6]
        z_volatile = z_volatile or [2.0, 2.5, 3.5]
        dthresh    = dthresh    or [0.15, 0.30, 0.50]

        _, live, _ = self._engine_run(month)
        c = self.cfg

        rows = []
        for zq, zv, dt in itertools.product(z_quiet, z_volatile, dthresh):
            res = BACKTESTER(live).run(
                z_quiet=zq, z_volatile=zv, exit_z=c['exit_z'],
                danger_threshold=dt, fee_bps=c['fee_bps'],
                slippage_mode=c['slippage_mode'], flatten_eod=c['flatten_eod'],
            )
            rows.append({
                'z_quiet': zq, 'z_volatile': zv, 'danger_threshold': dt,
                'Baseline_Sharpe': self._sharpe(res['Return_Baseline']),
                'AR_Sharpe':       self._sharpe(res['Return_AR']),
                'MS_AR_Sharpe':    self._sharpe(res['Return_MS_AR']),
                'MS_AR_Trades':    int((res['Target_MS_AR'].diff().abs() > 0).sum() / 2),
            })

        sw = pd.DataFrame(rows)
        sw['MS_AR_vs_AR']       = sw['MS_AR_Sharpe'] - sw['AR_Sharpe']
        sw['MS_AR_vs_Baseline'] = sw['MS_AR_Sharpe'] - sw['Baseline_Sharpe']

        print(f'=== Sensitivity sweep, {month}, {self.pair_name} ===')
        print(sw.round(2).to_string(index=False))
        print(f'\nMS_AR > AR       across grid: '
              f'{int((sw["MS_AR_vs_AR"]       > 0).sum())}/{len(sw)}')
        print(f'MS_AR > Baseline across grid: '
              f'{int((sw["MS_AR_vs_Baseline"] > 0).sum())}/{len(sw)}')

        self.sweep_df = sw
        return sw
