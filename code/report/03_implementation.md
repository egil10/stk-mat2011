# 3. Implementation

This chapter describes the code that implements the methodology of
Chapter 2. The codebase is intentionally modular: each stage of the
pipeline (data → spread → engine → backtester → tearsheet) is owned by
exactly one class, and a higher-level `MONTH` orchestrator stitches them
into the multi-month experiment.

The aim is for any single stage to be replaceable. If next year's
experiment uses a different regime model, only `ENGINE.fit_markov_regimes`
changes — everything downstream still works.

---

## 3.1 Directory layout

```
code/
  scripts/                     # Reusable Python classes
    spread.py        # SPREAD       — tick → bar aggregation, log/return columns
    screener.py      # SCREENER     — cointegration diagnostics (full + rolling)
    engine.py        # ENGINE       — rolling cointegration + HMM walk-forward
    backtester.py    # BACKTESTER   — numba position generation + PnL
    tearsheet.py     # TEARSHEET    — metrics + plotting
    wfo.py           # WFO          — outer Optuna walk-forward (NOT used in
                     #                multi-month runs; kept for reference)
    descriptive.py   # DESCRIPTIVE  — exploratory data analysis report
    plotting.py      # Matplotlib styling + PDF helpers
    month.py         # MONTH        — high-level orchestrator
    colab.py         # 2-line Colab bootstrap

  monthly/                     # The multi-month validation notebooks
    audnzd-month.ipynb
    gbpeur-month.ipynb
    noksek-month.ipynb

  notebooks/                   # Older full-period notebooks (reference only)

  data/processed/              # Dukascopy parquet ticks (symlinked from
                               # Drive on Colab)
```

The split into `monthly/` and `notebooks/` is historical. The 2024–2025
two-year notebooks live in `notebooks/`; they used `wfo.py` with Optuna
and produced the inflated-Sharpe / many-thousand-trades results that
motivated the smaller multi-month design. The `monthly/` notebooks use
the `MONTH` class with fixed parameters.

---

## 3.2 The pipeline, end to end

A single notebook run executes:

```
parquet ticks
       │
       ▼
SPREAD.build(files)
       │  DataFrame:  Log_A, Log_B, Return_A, Return_B,
       │              HalfSpread_A_bps, HalfSpread_B_bps
       ▼
ENGINE.walk_forward(...)
       │  Per trading day in the month:
       │    • fit_cointegration  → β_t, α_t, Spread_Level, Z_Score, Spread_Return
       │    • fit_markov_regimes → MR_Prob, Danger_Regime_Prob (in-sample)
       │    • predict_oos        → MR_Prob, Danger_Regime_Prob (out-of-sample)
       │  Output: concatenated OOS DataFrame + per-day param_tracker
       ▼
BACKTESTER.run(...)
       │  Build positions for Baseline / AR / MS-AR (numba),
       │  compute gross/net returns, costs, trade counts.
       ▼
TEARSHEET.generate_report()
       │  Print metric tables; plot equity curves, drawdowns, regimes,
       │  cost impact, Markov dynamics.
       ▼
MONTH.summary() / MONTH.sweep()
          Cross-month aggregation; sensitivity sweep.
```

The `MONTH` class wraps the chain so that a notebook reads:

```python
from month import MONTH
m = MONTH('AUDUSD', 'NZDUSD')
m.run_months(['202408', '202409', '202508'])
m.summary()
m.tearsheets()
m.sweep('202408')
```

That is the entire notebook content.

---

## 3.3 `SPREAD` — `code/scripts/spread.py`

**Responsibility.** Convert two streams of Dukascopy tick parquet files
into synchronised mid-price bars on a common timestamp grid, and attach
the per-asset half-spread.

Key methods:

- `_load_parquet(file_paths)` — streams files one-by-one, applies the
  weekday/hour session filter (Mon–Fri, 00–24h by default), concatenates
  the filtered chunks. Memory-conservative: raw ticks are released after
  aggregation.
- `_aggregate_bars(ask_files, bid_files)` — merges ask and bid via
  `pandas.merge_asof(direction='backward')`, computes mid prices, then
  aggregates into bars of fixed tick count $\Delta = 500$. Stateful
  carryover across month boundaries preserves bar integrity.
- `build(file_paths)` — orchestrates the above for both assets, joins
  the two bar series with another `merge_asof`, and computes the log
  prices, log returns and half-spreads in basis points.

The output is a single `pandas.DataFrame` indexed by bar timestamp,
ready to be passed to the `ENGINE`.

---

## 3.4 `ENGINE` — `code/scripts/engine.py`

**Responsibility.** Estimate the rolling cointegration relationship and
fit the Markov-switching regime model on a rolling training window;
predict regime probabilities out-of-sample one day at a time.

This is the most mathematically dense file in the codebase. The methods
correspond 1-to-1 with the maths of §2.2 and §2.4:

| Method | Maths | Lines |
|---|---|---|
| `_rolling_ols(y, x, window)` | $\beta_t, \alpha_t$ via cumulative sums | 36–64 |
| `fit_cointegration(...)` | Spread, Z-score, Spread_Return | 66–84 |
| `fit_markov_regimes(...)` | EM fit of MS-AR(1), regime labelling, in-sample posteriors | 86–192 |
| `predict_oos(test_df, train_tail_df, ...)` | OOS analytical one-step posterior | 194–262 |
| `walk_forward(df, train_days, ...)` *(classmethod)* | The outer day-by-day loop | 264–311 |

Implementation notes worth flagging:

- **Cumulative-sum rolling OLS.** `_rolling_ols` uses precomputed cumulative
  sums of $x$, $y$, $x^2$, $xy$ so the full rolling regression is $O(N)$
  in the bar count. This is critical for the walk-forward where the
  same rolling OLS is rebuilt every trading day.
- **Linear pre-processing for EM stability.** The winsorisation +
  scaling steps before the EM fit are linear transforms, so the
  regime *structure* (AR coefficient, transition matrix, regime
  probabilities) is invariant. Only the magnitudes of `const` and
  `sigma` change, and they're unscaled in lines 130–135.
- **Per-regime parameter storage.** Lines 152–155 store full per-regime
  vectors `_regime_consts`, `_regime_ars`, `_regime_sigmas` so that
  `predict_oos` can compute the analytical posterior for any number of
  regimes, not just two.
- **Walk-forward fold skipping.** If `len(train_df) ≤ coint_window +
  z_window + 10` (i.e. not enough bars for both windows to fully
  populate) or `len(test_df) < 5`, the day is silently skipped. In
  practice this only happens on the first $T_{\text{train}}$ days of a
  month — every full subsequent day comfortably exceeds the threshold
  with 500-tick bars.
- **Param tracker.** Lines 297–305 record per-day β, α, regime variances,
  regime means, transition diagonals, and AR coefficients. These power
  the diagnostic plots in `TEARSHEET.plot_markov_dynamics`.

---

## 3.5 `BACKTESTER` — `code/scripts/backtester.py`

**Responsibility.** Generate positions for the three trading strategies
defined in §2.5, then compute gross/net returns with the cost model of
§2.6.

The hot loop is `_generate_positions`, a numba-`@njit` function. It
walks bars in order, maintains a single state variable `curr` (the
current position), and applies four rules in priority order. The full
loop:

```python
@njit
def _generate_positions(z_scores, entry_z_arr, exit_z, signals_allowed):
    n = len(z_scores)
    pos = np.zeros(n)
    curr = 0.0
    for i in range(n):
        if np.isnan(z_scores[i]):
            pos[i] = curr
            continue

        # Panic Button (only AR has signals_allowed != all-True)
        if curr != 0.0 and not signals_allowed[i]:
            curr = 0.0

        if curr == 0.0:
            if signals_allowed[i]:
                if   z_scores[i] < -entry_z_arr[i]: curr =  1.0
                elif z_scores[i] >  entry_z_arr[i]: curr = -1.0
        elif curr ==  1.0 and z_scores[i] >= -exit_z: curr = 0.0
        elif curr == -1.0 and z_scores[i] <=  exit_z: curr = 0.0

        pos[i] = curr
    return pos
```

That's the entire trading logic for all three strategies. The strategy
differences are encoded in how the two input arrays are constructed:

| Strategy | `entry_z_arr[i]` | `signals_allowed[i]` |
|---|---|---|
| **Baseline** | `z_quiet` (constant) | `True` (constant) |
| **AR** | `z_quiet` (constant) | `mr_probs[i] >= 1 - danger_threshold` |
| **MS-AR** | `mr_probs[i] * z_quiet + danger_probs[i] * z_volatile` | `True` (constant) |

Code, lines 59–78:

```python
base_allowed = np.ones(len(self.data), dtype=np.bool_)
hard_allowed = np.where(np.isfinite(mr_probs),
                        mr_probs >= (1.0 - danger_threshold), False)

static_entry_z  = np.full(len(z_scores), z_quiet)
dynamic_entry_z = (mr_probs * z_quiet) + (danger_probs * z_volatile)

pos_base  = gen(z_scores, static_entry_z,  exit_z, base_allowed)
pos_ar    = gen(z_scores, static_entry_z,  exit_z, hard_allowed)
pos_ms_ar = gen(z_scores, dynamic_entry_z, exit_z, base_allowed)
```

A few subtleties:

- **3-bar median smoothing of regime probabilities** (lines 56–57). Both
  `mr_probs` and `danger_probs` are passed through a 3-bar rolling
  median (`bfill`-padded) before they enter the strategy. This filters
  single-bar regime flickers that would otherwise cause spurious
  entries/exits.
- **End-of-day flattening** (lines 69–73). When `flatten_eod=True`, the
  position generator is wrapped in `_positions_daily`, which calls
  `_generate_positions` per-day and forces the day's last bar to zero.
- **One-bar lag on the position.** Lines 83–85 shift each strategy's
  position by one bar before turning it into a target — the standard
  fix for look-ahead.
- **Buy & Hold has no costs.** Lines 103–108 set `Target_BuyHold = 1.0`
  and don't apply any fee/slippage subtraction. This is intentional —
  Buy & Hold is the *zero-cost* benchmark.

Output columns added to `self.data` (per strategy `S ∈ {Baseline, AR,
MS_AR, BuyHold}`):

- `Target_S` — one-bar-lagged position
- `Return_S_Gross` — gross return
- `Return_S` — net return (after fee and slippage; equal to gross for BH)
- `CumReturn_S`, `CumReturn_S_Gross` — cumulative sums

---

## 3.6 `TEARSHEET` — `code/scripts/tearsheet.py`

**Responsibility.** Compute 20+ performance metrics per strategy and
generate the diagnostic plots.

Pure post-processing — no model fitting. It reads the columns produced
by `BACKTESTER` and prints / plots.

Key public methods:

- `generate_report(strats, gross=False, fast=False)` — prints a metrics
  table for every strategy in `strats`.
- `_calc_metrics(returns, strat_name)` — computes the metric dict.
  Implements all formulas in §2.7.
- `plot_performance(strats, gross)` — equity curves, drawdown overlay,
  monthly returns bar chart, rolling 252-bar volatility and Sharpe,
  return distribution (linear and log y).
- `plot_positions_and_regimes(...)` — per-strategy position heatmap +
  rolling exposure + smoothed Danger probability.
- `plot_markov_dynamics(df_params)` — rolling per-fold $\sigma^2$,
  $\mu$, $p_{ii}$ for the two regimes. These plots are the diagnostic
  for "what is the HMM actually saying day by day".
- `plot_cost_impact(strats)` — gross vs net equity curves side by side,
  highlighting cost drag.

The annualisation factor $F = 252 \cdot 24 \cdot 60$ is hard-coded in
`_calc_metrics`. See §2.7.1 for the caveat about absolute Sharpe
magnitudes.

---

## 3.7 `MONTH` — `code/scripts/month.py`

**Responsibility.** The high-level orchestrator that ties the whole
pipeline together for multi-month experiments. Used by every notebook
in `code/monthly/`.

### 3.7.1 Construction

```python
m = MONTH('AUDUSD', 'NZDUSD')              # all defaults
m = MONTH('GBPUSD', 'EURUSD', train_days=5, z_volatile=4.0)  # overrides
```

Defaults are stored in `MONTH.DEFAULT_CFG`:

```python
DEFAULT_CFG = {
    'bar_aggregation_method': 'tick',
    'tick_threshold':         500,
    'active_days':            (0, 1, 2, 3, 4),   # Mon–Fri
    'active_hours':           (0, 23),
    'train_days':             3,
    'coint_window':           150,
    'z_window':               50,
    'k_regimes':              2,
    'winsorize_std':          4.0,
    'scaling':                1e4,
    'print_freq':             10,
    'z_quiet':                1.3,
    'z_volatile':             2.5,
    'exit_z':                 0.0,
    'danger_threshold':       0.30,
    'fee_bps':                0.5,
    'slippage_mode':          'half_spread',
    'flatten_eod':            True,
}
```

Any default can be overridden at construction; `MONTH` carries the merged
config and feeds it into every downstream call.

### 3.7.2 Public methods

- `run(month_str) -> (results, df_params)` — runs the full pipeline for
  a single month string (e.g. `'202408'`).
- `run_months(months) -> summary_df` — runs `run` for every month in
  the list, stashes everything in `self.results` and `self.df_params`,
  builds a tidy summary DataFrame with columns `Sharpe_Baseline /
  Sharpe_AR / Sharpe_MS_AR / PnL_… / Trades_…` per month.
- `summary()` — prints a per-strategy cross-month table for Sharpe,
  PnL bps and trade count, an MS-AR-minus-Baseline edge column, and a
  per-strategy win count.
- `tearsheets()` — calls the full `TEARSHEET.generate_report()` +
  associated plots for every month in `self.results`.
- `sweep(month_str, z_quiet_grid, z_volatile_grid, danger_threshold_grid)`
  — runs the backtester on the already-fitted engine output for the
  chosen month over a 27-cell parameter grid, prints win-count tables
  for MS-AR vs Baseline and MS-AR vs AR.

### 3.7.3 What it doesn't do

- No fitting of `wfo.py`. The Optuna-tuned WFO loop is intentionally
  not used in the multi-month design — see §3.8 for why.
- No persistence. Results live in memory; the notebook is the
  reproducibility unit.

---

## 3.8 What we deliberately *don't* use: `wfo.py`

`code/scripts/wfo.py` implements a walk-forward Optuna optimisation
loop that tunes the trading-rule parameters (`z_quiet`, `z_volatile`,
`exit_z`, `danger_threshold`) by maximising in-sample Sharpe on a
validation window before applying to the test window.

We do not use it in the multi-month experiments. Two reasons:

1. **In-sample overfitting.** With Optuna free to set $z_v$ as high as it
   likes, it routinely picks corner-case parameter combinations that
   maximise validation Sharpe via *churn* — thousands of trades per
   month with marginal edges. The earlier two-year runs in
   `code/notebooks/` are saturated with this artefact (MS-AR producing
   9 235 trades over the two years and a Sharpe of −3.x).
2. **Goal mismatch.** The research question is whether the regime filter
   has *structural* value. Fixed sensible parameters give a clean test
   of that question; optimised parameters confound it.

`wfo.py` remains in the codebase as a reference implementation of WFO +
Optuna for future projects that want full optimisation.

---

## 3.9 The notebooks

Each notebook in `code/monthly/` has the same 5–6 cell skeleton.

**Cell 1 — Colab bootstrap (2 lines):**

```python
!curl -sL https://raw.githubusercontent.com/egil10/stk-mat2011/main/code/scripts/colab.py -o /content/colab.py
import sys; sys.path.insert(0, '/content'); from colab import setup; setup('code/monthly')
```

What it does (via `code/scripts/colab.py`):

1. `drive.mount('/content/drive')` — mounts Google Drive.
2. `git clone` (or `git pull`) the repo into `/content/stk-mat2011`.
3. `os.symlink(DRIVE_DATA, REPO_DATA)` so `code/data/processed`
   transparently points at the ~880 MB Drive cache of parquet ticks.
4. Add `code/scripts` to `sys.path`.
5. `os.chdir('code/monthly')` so the notebook's relative paths work.

**Cell 2 — imports (3 lines):**

```python
%pip install --quiet arch optuna
import warnings; warnings.filterwarnings('ignore')
from month import MONTH
```

**Cell 3 — run:**

```python
m = MONTH('AUDUSD', 'NZDUSD')      # or 'GBPUSD','EURUSD'  or 'EURNOK','EURSEK'
m.run_months(['202408', '202409', '202508'])
m.summary()
```

**Cell 4 — tearsheets:**

```python
m.tearsheets()
```

**Cell 5 — sensitivity sweep on one month:**

```python
m.sweep('202408')
```

That's the entire notebook. All the logic lives in `MONTH`, which in
turn delegates to the four core classes.

---

## 3.10 Reproducibility notes

- **Random seed.** `fit_markov_regimes` sets `np.random.seed(42)` before
  the EM fit, so given identical training data the EM converges to the
  same local optimum every run.
- **Numba caching.** The first call to `_generate_positions` JIT-compiles
  it; subsequent calls hit a process-local cache. Sweep performance
  benefits massively from this.
- **Data source.** Dukascopy historical tick archive. The download +
  parquetisation script is `code/scripts/p_duka.py`; the manifest of
  what's available is in `README.md`.
- **Pinned dependencies.** `requirements.txt` lists exact package
  versions (`arch`, `optuna`, `pandas`, `statsmodels`, `numba`,
  `matplotlib`) so a fresh clone reproduces the same numbers.
- **Hardware.** All multi-month runs complete in under five minutes on a
  free Colab CPU instance. No GPU is used anywhere.

---

## 3.11 Where the maths lives, by line number

A pointer for anyone reading both the code and the report side by side:

| Maths in §2 | Code location | Lines |
|---|---|---|
| Tick → bar aggregation | `code/scripts/spread.py` · `_aggregate_bars` | full method |
| Rolling OLS β, α | `code/scripts/engine.py` · `_rolling_ols` | 36–64 |
| Spread, Z-score, Spread_Return | `engine.py` · `fit_cointegration` | 66–84 |
| MS-AR(1) EM fit and labelling | `engine.py` · `fit_markov_regimes` | 86–192 |
| OOS one-step regime posterior | `engine.py` · `predict_oos` | 194–262 |
| Walk-forward outer loop | `engine.py` · `walk_forward` | 264–311 |
| Position generation (all 3 strats) | `code/scripts/backtester.py` · `_generate_positions` | 8–31 |
| EOD flattening wrapper | `backtester.py` · `_positions_daily` | 33–43 |
| Strategy dispatch (entry arrays + gates) | `backtester.py` · `BACKTESTER.run` | 50–78 |
| PnL, costs, cumulative return | `backtester.py` · `BACKTESTER.run` | 82–108 |
| Metric formulas | `code/scripts/tearsheet.py` · `_calc_metrics` | full method |

That table is the bridge between the maths in Chapter 2 and the code in
this chapter.
