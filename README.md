# Hidden Markov Models for High-Frequency FX Pairs Trading

Pairs trading is one of the oldest ideas in quantitative finance — find two assets that move together, bet on the spread reverting. At minute resolution across years of tick data, that simple idea collides with regime shifts, transaction costs, and the brutal honesty of out-of-sample testing.

This project tackles that collision head on. We build a regime-switching engine on top of cointegrated FX pairs, optimise it with walk-forward search, and judge it the only way that matters: on data it has never seen.

---

## The Idea

Spreads between cointegrated assets are mean-reverting — until they are not. A Hidden Markov Model identifies *when* mean-reversion is the right bet and *when* the spread has entered a danger regime and should be left alone. The HMM is the switch; the spread model is the trade.

- **Cointegration** finds pairs whose log-prices share a stochastic trend
- **Rolling OLS** estimates a time-varying hedge ratio
- **AR(1)-HMM** classifies each bar into a mean-revert or danger state
- **GARCH** sizes the z-score by current conditional volatility
- **Walk-forward optimisation** picks entry/exit thresholds without peeking

---

## The Three Systems

| Pair System | Notebook | What It Tests |
|---|---|---|
| **AUDUSD / NZDUSD** | `Y-AUDNZD.ipynb`, `m-audnzd.ipynb` | Antipodean commodity-linked cointegration |
| **GBPUSD / EURUSD** | `Y-GBPEUR.ipynb`, `m-gbpeur.ipynb` | European majors against the dollar |
| **EURNOK / EURSEK** | `Y-NOKSEK.ipynb`, `m-noksek.ipynb` | Nordic cross-rate co-movement |

`Y-*` notebooks run the full multi-year walk-forward. `m-*` notebooks isolate a single month for diagnostic depth.

---

## The Pipeline

| Module | Role |
|---|---|
| `screener.py` | Engle–Granger cointegration, half-life, rolling diagnostics |
| `engine.py` | Hedge ratio, AR(1)-HMM regime states, GARCH volatility |
| `backtester.py` | Numba-accelerated z-score entry/exit with regime gating |
| `wfo.py` | Optuna walk-forward search over entry/exit thresholds |
| `tearsheet.py` | Out-of-sample performance reporting |
| `plotting.py` · `descriptive.py` | Visual diagnostics and EDA |
| `synthetic.py` · `month.py` | Synthetic stress tests and monthly slicing |

---

## The Models

Each model is a building block. Understanding each step makes the next one obvious.

<br>

**AR(1)** — Autoregressive spread dynamics

$$y_t = c + \phi\, y_{t-1} + \varepsilon_t$$

<br>

**GARCH(1, 1)** — Conditional volatility

$$\sigma_t^2 = \omega + \alpha\, \varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2$$

<br>

**AR(1)-HMM** — Regime-switching mean reversion

$$y_t \mid S_t = k \;\sim\; \mathcal{N}\!\left(c_k + \phi_k\, y_{t-1},\; \sigma_k^2\right),\qquad \Pr(S_t = j \mid S_{t-1} = i) = P_{ij}$$

The HMM learns one $(c_k, \phi_k, \sigma_k)$ per regime and a transition matrix $P$. Trades are taken only in the regime where $|\phi_k| < 1$ and variance is contained — the rest of the time the system waits.

<br>

**Standardised spread** — what the bot actually trades

$$z_t = \frac{y_t - \mu_t}{\sigma_t}$$

Enter when $|z_t|$ crosses the entry threshold *and* the HMM allows it. Exit on mean-cross or regime flip.

---

## Data

Dukascopy tick data, resampled to one-minute bars, stored as Parquet.

| Coverage | Symbols | Span |
|---|---|---|
| **Majors** | EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD | 2019 – Feb 2026 |
| **Nordics** | EURNOK, EURSEK | Oct 2023 – Feb 2026 |
| **Cross-rates** | EURCHF, USDZAR | Nov 2025 – Feb 2026 |
| **Crypto** | BTCUSD, ETHUSD | Spot checks, 2024 – 2025 |

Sources: [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) via [dukascopy-python](https://pypi.org/project/dukascopy-python/), with [HistData](https://www.histdata.com/) and [TrueFX](https://www.truefx.com/truefx-historical-downloads/) for cross-validation. Small CSV samples live in `code/data/samples/` for quick experimentation; full Parquet archives in `code/data/processed/`.

---

## Quick Start

```bash
pip install -r requirements.txt
jupyter lab code/notebooks/Y-AUDNZD.ipynb
```

Notebooks read directly from `code/data/processed/`. Run any `Y-*.ipynb` end-to-end to reproduce the walk-forward results; rendered HTML and PDF copies live in `code/exports/`.

---

## Resources

- [Course Page (UiO STK-MAT2011)](https://www.uio.no/studier/emner/matnat/math/STK-MAT2011/)
- [Dukascopy Historical Market Data](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
- [HistData — Free Forex Historical Data](https://www.histdata.com/)
- [TrueFX — Historical Downloads](https://www.truefx.com/truefx-historical-downloads/)

---

<sub>University of Oslo · Department of Mathematics · Spring 2026</sub>
