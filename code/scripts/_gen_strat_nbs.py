"""One-shot generator for the three code/strats/ pair notebooks.

Run once after editing the template; the notebooks it produces are checked
in alongside the other artefacts. Delete this file after use if you like.
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'strats')

PAIRS = [
    ('AUDUSD', 'NZDUSD', 'audnzd.ipynb'),
    ('EURNOK', 'EURSEK', 'noksek.ipynb'),
    ('GBPUSD', 'EURUSD', 'gbpeur.ipynb'),
]


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


for pair_a, pair_b, fname in PAIRS:
    intro = (
        f"# {pair_a}/{pair_b} — independent-day strategies\n"
        "\n"
        "Four strategies on the cointegration spread, fit and back-tested "
        "**independently per trading day** on three months: **2024-08, 2024-09, 2025-08**.\n"
        "\n"
        "### Pipeline (paper §2–§4)\n"
        "1. **Pre-averaging** (paper eq 2.2): bid/ask ticks of leg A and leg B "
        "are synchronised at the tick level (union + ffill), then every L=200 "
        "consecutive synchronised ticks are collapsed into one bar by averaging "
        "the mids. Same L, same boundaries on both legs → pre-averaged legs "
        "perfectly co-time.\n"
        "2. **Rolling cointegration** (paper §3.1): hedge ratio $\\beta_t$ by "
        "rolling OLS with window $W_\\beta=50$ on the pre-averaged log prices; "
        "spread $S_t = \\log P^A_t - \\beta_t \\log P^B_t - \\alpha_t$.\n"
        "3. **Rolling $z$-score** (paper §3.2): $Z_t$ over $W_z=25$ bars.\n"
        "4. **MS-AR(1) on the spread** (paper §4.3): EM via Baum–Welch, $K=2$, "
        "multi-seed init.\n"
        "5. **Regime labelling by innovation variance** (paper §4.7): "
        "MR = regime with smallest $\\sigma^{(k)}$ (quiet, safe), "
        "DR = regime with largest $\\sigma^{(k)}$ (volatile, danger).\n"
        "6. **Four strategies**: Buy & Hold, Baseline (z-score), AR (binary "
        "gate $\\mathbf{1}\\{\\gamma_t^{MR} \\geq 1-\\delta\\}$), MS-AR (dynamic "
        "threshold $\\gamma_t^{MR} z_q + \\gamma_t^{DR} z_v$).\n"
        "\n"
        "The OOS forward filter is intentionally deferred — we use the "
        "in-sample smoothed posteriors $\\gamma_t^{MR}$ directly inside the "
        "strategies, so this is a *best-case* picture of how the regime "
        "information would be used if it were known.\n"
    )

    bootstrap = (
        "# Colab bootstrap (silently skipped on local environments).\n"
        "import importlib.util, sys, os\n"
        "if importlib.util.find_spec('google.colab') is not None:\n"
        "    import subprocess\n"
        "    subprocess.run(\n"
        "        ['curl', '-sL',\n"
        "         'https://raw.githubusercontent.com/egil10/stk-mat2011/main/code/scripts/colab.py',\n"
        "         '-o', '/content/colab.py'],\n"
        "        check=True,\n"
        "    )\n"
        "    sys.path.insert(0, '/content')\n"
        "    from colab import setup\n"
        "    setup('code/strats')\n"
        "else:\n"
        "    sys.path.insert(0, os.path.abspath('../scripts'))\n"
    )

    pip = "%pip install --quiet arch statsmodels numba optuna\n"

    config = (
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "from spread import SPREAD\n"
        "from engine import ENGINE\n"
        "from backtester import BACKTESTER\n"
        "from tearsheet import TEARSHEET\n"
        "\n"
        f"PAIR_A, PAIR_B = '{pair_a}', '{pair_b}'\n"
        "MONTHS         = ['202408', '202409', '202508']\n"
        "DATA_DIR       = '../data/processed'\n"
        "\n"
        "# Tick-synced pre-averaged bars (paper eq 2.2).\n"
        "# L=200 -> each bar is the mean of 200 SYNCHRONISED ticks on both legs.\n"
        "BAR_CFG = dict(\n"
        "    agg_type     = 'tick',\n"
        "    threshold    = 200,       # block size L\n"
        "    price_agg    = 'mean',    # 'mean' triggers tick-sync + pre-averaging\n"
        "    active_hours = (0, 24),\n"
        ")\n"
        "\n"
        "# Per-day fit: rolling beta, rolling z-score, MS-AR(1) on the day's spread.\n"
        "FIT_CFG = dict(\n"
        "    coint_window  = 50,    # W_beta\n"
        "    z_window      = 25,    # W_z\n"
        "    k_regimes     = 2,\n"
        "    winsorize_std = 4.0,\n"
        "    scaling       = 10000,\n"
        "    n_init        = 3,\n"
        ")\n"
        "\n"
        "# Strategies (paper sections on regime-aware trading).\n"
        "STRAT_CFG = dict(\n"
        "    z_quiet          = 1.3,\n"
        "    z_volatile       = 2.5,\n"
        "    exit_z           = 0.0,\n"
        "    danger_threshold = 0.30,\n"
        "    fee_bps          = 0.5,\n"
        "    slippage_mode    = 'half_spread',\n"
        "    flatten_eod      = True,\n"
        "    prob_smoothing   = 0,\n"
        ")\n"
        "\n"
        "ANN_FACTOR = 252 * 24 * 60   # tick-clock annualisation\n"
    )

    per_month_md = (
        "## Per-month run\n"
        "\n"
        "For each month we\n"
        "\n"
        "1. Aggregate raw bid/ask ticks into L-tick pre-averaged bars (`SPREAD`).\n"
        "2. Fit each trading day independently — rolling cointegration "
        "($\\beta_t, \\alpha_t$), rolling $z$-score, MS-AR(1) on the day's "
        "spread (`ENGINE.each_day`). Regimes labelled by $\\sigma$.\n"
        "3. Run Buy & Hold, Baseline, AR(1) gate, MS-AR(1) dynamic-threshold "
        "through the same back-test machinery (`BACKTESTER`).\n"
        "4. Score with `TEARSHEET`.\n"
    )

    per_month_code = (
        "def files(pair, month):\n"
        "    p = pair.lower()\n"
        "    return (\n"
        "        [f'{DATA_DIR}/{p}_dukascopy_ask_{month}.parquet'],\n"
        "        [f'{DATA_DIR}/{p}_dukascopy_bid_{month}.parquet'],\n"
        "    )\n"
        "\n"
        "\n"
        "def fmt_pnl(bt):\n"
        "    return {s: bt[f'Return_{s}'].fillna(0).sum() * 1e4\n"
        "            for s in ['BuyHold', 'Baseline', 'AR', 'MS_AR']}\n"
        "\n"
        "\n"
        "runs = {}\n"
        "for m in MONTHS:\n"
        "    print(f'\\n=== {m}  {PAIR_A}/{PAIR_B} ===')\n"
        "\n"
        "    ask_a, bid_a = files(PAIR_A, m)\n"
        "    ask_b, bid_b = files(PAIR_B, m)\n"
        "\n"
        "    df = SPREAD(**BAR_CFG).build([ask_a, bid_a, ask_b, bid_b], verbose=True)\n"
        "    n_bars = len(df)\n"
        "    n_days = df.index.normalize().unique().shape[0]\n"
        "    print(f'  {n_bars:,} pre-averaged bars over {n_days} trading days')\n"
        "\n"
        "    fitted, params = ENGINE.each_day(df, **FIT_CFG, verbose=True)\n"
        "    bt = BACKTESTER(fitted).run(**STRAT_CFG)\n"
        "    runs[m] = (bt, params, fitted)\n"
        "\n"
        "    pnl = fmt_pnl(bt)\n"
        "    bh, base, ar, ms = pnl['BuyHold'], pnl['Baseline'], pnl['AR'], pnl['MS_AR']\n"
        "    print(\n"
        "        f'  net pnl (bps):  BH={bh:+8.1f}  Base={base:+8.1f}  '\n"
        "        f'AR={ar:+8.1f}  MS-AR={ms:+8.1f}'\n"
        "    )\n"
    )

    tearsheets_md = "## Per-month tearsheets\n"
    tearsheets_code = (
        "for m, (bt, params, _) in runs.items():\n"
        "    print(f'\\n{\"=\"*28}  {m}  {\"=\"*28}')\n"
        "    ts = TEARSHEET(bt, df_params=params)\n"
        "    ts.generate_report()\n"
        "    ts.plot_performance()\n"
        "    ts.plot_positions_and_regimes()\n"
        "    ts.plot_markov_dynamics()\n"
    )

    summary_md = (
        "## Cross-month summary\n"
        "\n"
        "PnL in bps, Sharpe annualised on the tick-clock (`252 × 24 × 60` "
        "bars/year, matches `MONTH` default), and one round-trip per `Target` "
        "flip cycle.\n"
    )

    summary_code = (
        "def annualised_sharpe(returns):\n"
        "    r = returns.fillna(0)\n"
        "    sd = r.std()\n"
        "    return float(r.mean() / sd * np.sqrt(ANN_FACTOR)) if sd > 0 else 0.0\n"
        "\n"
        "\n"
        "rows = []\n"
        "for m, (bt, _, _) in runs.items():\n"
        "    row = {'Month': m}\n"
        "    for s in ['BuyHold', 'Baseline', 'AR', 'MS_AR']:\n"
        "        r = bt[f'Return_{s}']\n"
        "        row[f'{s}_PnL_bps'] = float(r.fillna(0).sum() * 1e4)\n"
        "        row[f'{s}_Sharpe']  = annualised_sharpe(r)\n"
        "        row[f'{s}_Trades']  = int((bt[f'Target_{s}'].diff().abs() > 0).sum() / 2)\n"
        "    rows.append(row)\n"
        "\n"
        "summary = pd.DataFrame(rows).set_index('Month')\n"
        "\n"
        "print('\\n=== PnL bps by month ===')\n"
        "print(summary[[c for c in summary.columns if c.endswith('PnL_bps')]].round(1).to_string())\n"
        "\n"
        "print('\\n=== Sharpe (tick-clock annualised) ===')\n"
        "print(summary[[c for c in summary.columns if c.endswith('Sharpe')]].round(2).to_string())\n"
        "\n"
        "print('\\n=== Round-trip trade counts ===')\n"
        "print(summary[[c for c in summary.columns if c.endswith('Trades')]].to_string())\n"
        "\n"
        "summary\n"
    )

    perday_md = (
        "### Per-day regime parameters\n"
        "\n"
        "Each row in `runs[m][1]` is the per-day fit summary: bar count, "
        "$\\beta$, regime $\\sigma$, regime $\\rho$, regime means and "
        "transition probabilities. Useful for sanity-checking the HMM and "
        "for the paper's tables.\n"
    )

    perday_code = (
        "for m, (_, params, _) in runs.items():\n"
        "    print(f'\\n=== {m}  per-day fits ===')\n"
        "    print(params.round(4).to_string())\n"
    )

    cells = [
        md(intro),
        code(bootstrap),
        code(pip),
        code(config),
        md(per_month_md),
        code(per_month_code),
        md(tearsheets_md),
        code(tearsheets_code),
        md(summary_md),
        code(summary_code),
        md(perday_md),
        code(perday_code),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path = os.path.join(OUT_DIR, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'wrote {path}')

print('done')
