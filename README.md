# Markov-Switching Strategies for High-Frequency FX Pairs Trading

Three cointegrated FX pairs. Four trading rules. One question: can a hidden Markov model tell us *when* the spread is mean-reverting and *when* it has slipped into a regime where the textbook trade quietly bleeds through transaction costs?

This repository is the codebase behind a University of Oslo `STK-MAT2011` project. The core experiments live in the `m-*` notebooks — three pairs, three months each, twelve trading days at a time — and the modules in `code/scripts/` are the engine they call.

<br>

## The Four Strategies

All four strategies see the same cointegration spread $S_t$ and the same rolling $z$-score $Z_t$. They differ only in **whether and how** the smoothed regime posterior $\gamma_t^{\mathrm{MR}}$ from a Markov-switching AR(1) enters the rule.

| # | Strategy | Entry rule | Regime gate $G_t$ | Reads HMM? |
|---|---|---|---|---|
| **(1)** | **Buy & Hold** | always long, $\pi_t^{\mathrm{BH}}=+1$ | — | No |
| **(2)** | **Baseline** | $\|Z_t\| > z_q$ | $G_t = 1$ | No |
| **(3)** | **AR** | $\|Z_t\| > z_q$ | $G_t = \mathbf{1}\{\gamma_t^{\mathrm{MR}} \geq 1-\delta\}$ | Yes — binary |
| **(4)** | **MS-AR** | $\|Z_t\| > z_q^{\mathrm{eff}}(t)$ | $G_t = 1$ | Yes — continuous |

**(1) Buy & Hold.** A passive reference. Long the spread at every bar, zero costs — the drift baseline every active rule has to beat.

**(2) Baseline.** Classic Bollinger-style mean reversion on the spread $z$-score. Position rule:

$$\pi_t = \begin{cases}+1 & \pi_{t-1}=0,\; Z_t < -z_q \\ -1 & \pi_{t-1}=0,\; Z_t > +z_q \\ 0 & \pi_{t-1}\neq 0,\; |Z_t| \leq z_x \\ \pi_{t-1} & \text{otherwise.}\end{cases}$$

Enter on a wide deviation, exit on the mean-cross. Knows nothing about regimes.

**(3) AR.** Baseline plus a **binary danger gate**. If the smoothed posterior of being in the mean-reverting regime drops below $1-\delta$, any open position is liquidated and no new entries are taken until the gate re-opens. A hard switch: trade or wait.

**(4) MS-AR.** Keeps the gate open at all times but **widens the entry threshold continuously** with the danger-regime weight:

$$z_q^{\mathrm{eff}}(t) \;=\; \gamma_t^{\mathrm{MR}}\, z_q \;+\; \gamma_t^{\mathrm{DR}}\, z_v, \qquad z_v \geq z_q.$$

In a clean mean-reverting regime $\gamma_t^{\mathrm{MR}}\to 1$ and the rule trades like the Baseline; as the danger regime takes over, the bar to enter rises smoothly toward $z_v$. A soft, continuous version of (3).

<br>

## The Pipeline

The `m-*` notebooks run the seven steps below **independently per trading day**. Day-by-day fitting keeps the regime parameters local: nothing learned on Monday leaks into Tuesday.

**1. Mid-price.** For each leg $x \in \{A, B\}$ with best bid $B_t^x$ and best ask $A_t^x$,

$$M_t^{x} \;=\; \tfrac{1}{2}\bigl(B_t^{x} + A_t^{x}\bigr).$$

**2. Half-spread in basis points.** Used as the per-leg slippage charge in the back-test:

$$\mathrm{HS}_t^{\,x} \;=\; \tfrac{1}{2}\,\frac{A_t^{x} - B_t^{x}}{M_t^{x}}\times 10^{4}.$$

**3. Tick-synced pre-averaging.** Bid/ask streams of both legs are union-aligned, forward-filled at tick resolution, and every block of $L=500$ synchronised ticks collapses into one bar by averaging the mids. Same $L$, same boundaries, both legs — so the two pre-averaged series stay perfectly co-timed. This is the noise-robust mid construction of Jacod et al.

**4. Rolling cointegration.** On the pre-averaged log prices, the hedge ratio $\beta_t$ and intercept $\alpha_t$ come from rolling OLS with window $W_\beta = 25$:

$$S_t \;=\; \log P_t^{\mathrm{mid},A} \;-\; \beta_t \log P_t^{\mathrm{mid},B} \;-\; \alpha_t, \qquad \Delta S_t \;=\; r_t^{A} - \beta_{t-1}\, r_t^{B}.$$

The lagged $\beta_{t-1}$ in $\Delta S_t$ is deliberate — it keeps the PnL strictly causal.

**5. Rolling $z$-score.** Over a window $W_z = 15$,

$$Z_t \;=\; \frac{S_t - \mu_t^{S}}{\sigma_t^{S}}.$$

**6. Markov-switching AR(1) on the spread.** With $K = 2$ latent regimes,

$$S_t \mid S_{t-1},\, K_t = k \;\sim\; \mathcal{N}\!\bigl(c^{(k)} + \rho^{(k)} S_{t-1},\; (\sigma^{(k)})^{2}\bigr), \qquad \Pr(K_t = j \mid K_{t-1} = i) = P_{ij}.$$

Fit by Expectation–Maximisation (Baum–Welch) with multi-seed initialisation. Before estimation the spread is winsorised at $\mathrm{median}(S) \pm 4\,\mathrm{std}(S)$ and rescaled by $10^{4}$.

**7. Regime labelling by innovation variance.** The mean-reverting regime is the one with the smaller residual std; the danger regime is its complement:

$$k_{\mathrm{MR}} = \arg\min_{k}\sigma^{(k)}, \qquad k_{\mathrm{DR}} = \arg\max_{k}\sigma^{(k)}.$$

Smoothed posteriors $\gamma_t^{\mathrm{MR}} = \mathbb{P}(K_t = k_{\mathrm{MR}} \mid S_1,\ldots,S_T)$ and $\gamma_t^{\mathrm{DR}} = 1 - \gamma_t^{\mathrm{MR}}$ feed into strategies (3) and (4).

> **A note on causality.** The `m-*` notebooks feed the in-sample *smoothed* posteriors $\gamma_t^{\mathrm{MR}}$ to the strategies. This is a *best-case* picture of the regime signal — it assumes perfect hindsight on regime identity within the day. The strictly forward-filtered, walk-forward extension lives in `code/notebooks/Y-*.ipynb`.

<br>

## Transaction Costs

Every position flip pays the fixed commission `fee_bps = 0.05` plus, when `slippage_mode = 'half_spread'`, the round-trip cross-the-spread cost

$$\mathrm{HS}^{A}_t + \mathrm{HS}^{B}_t \quad\text{(bps, both legs).}$$

`fee_bps = 0.05` is roughly $5 per $1M USD notional — the institutional ECN / prime-brokerage rate for spot FX. At this bar resolution, the half-spread term is by far the binding cost.

<br>

## The Three Pairs

| Pair | Notebook (monthly) | Notebook (multi-year) | Why this pair |
|---|---|---|---|
| **AUDUSD / NZDUSD** | `m-audnzd.ipynb` | `Y-AUDNZD.ipynb` | Antipodean commodity currencies — the textbook cointegrated FX pair |
| **GBPUSD / EURUSD** | `m-gbpeur.ipynb` | `Y-GBPEUR.ipynb` | European majors against the dollar — same shock, different beta |
| **EURNOK / EURSEK** | `m-noksek.ipynb` | `Y-NOKSEK.ipynb` | Nordic cross-rates — small open economies tied to oil and rates |

Each `m-*` notebook runs the four strategies on three sub-samples (**Dec 2024**, **Aug 2024**, **Apr 2025**), prints a four-column tearsheet (`BuyHold | Baseline | AR | MS_AR`), and dumps every panel as a stand-alone PDF for the thesis.

<br>

## Data

Tick-level bid/ask from [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) via [dukascopy-python](https://pypi.org/project/dukascopy-python/), stored as monthly Parquet files in `code/data/processed/`. Cross-checks against [HistData](https://www.histdata.com/) and [TrueFX](https://www.truefx.com/truefx-historical-downloads/). A small CSV sample lives in `code/data/samples/dukascopy_sample.csv`.

| Coverage | Symbols | Span |
|---|---|---|
| **Majors** | EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD | 2019 – Feb 2026 |
| **Nordics** | EURNOK, EURSEK | Oct 2023 – Feb 2026 |
| **Crosses** | EURCHF, EURNZD, EURPLN, USDZAR | 2024 – Feb 2026 |
| **Crypto** | BTCUSD, ETHUSD | spot checks, 2024 – 2025 |

<br>

## Repository Layout

```
code/
├── notebooks/
│   ├── m-audnzd.ipynb     <- per-day 4-strategy fit, AUDUSD/NZDUSD
│   ├── m-gbpeur.ipynb     <- per-day 4-strategy fit, GBPUSD/EURUSD
│   ├── m-noksek.ipynb     <- per-day 4-strategy fit, EURNOK/EURSEK
│   ├── Y-*.ipynb          <- multi-year walk-forward extensions
│   ├── duka.ipynb         <- Dukascopy tick download / Parquet build
│   └── visuals.ipynb      <- thesis figures
├── scripts/
│   ├── spread.py          <- mid + half-spread + tick-synced pre-averaging
│   ├── screener.py        <- Engle–Granger, half-life, rolling diagnostics
│   ├── engine.py          <- rolling OLS, z-score, MS-AR(1) per-day fit
│   ├── backtester.py      <- numba-accelerated 4-strategy back-test
│   ├── tearsheet.py       <- tearsheet + per-panel PDF export
│   ├── wfo.py             <- Optuna walk-forward search
│   ├── plotting.py        <- shared matplotlib helpers
│   ├── descriptive.py     <- EDA
│   ├── month.py           <- monthly slicing utilities
│   ├── synthetic.py       <- synthetic stress tests
│   └── colab.py           <- Colab bootstrap
├── data/{processed,samples,synthetic}/
└── exports/{html,markdown,webpdf}/   <- rendered notebooks
```

<br>

## Quick Start

```bash
pip install -r requirements.txt
jupyter lab code/notebooks/m-audnzd.ipynb
```

The `m-*` notebooks read directly from `code/data/processed/` and produce tearsheets plus per-panel PDFs into `code/plots/<PAIR>/`. Rendered HTML / markdown / PDF copies of every notebook live under `code/exports/`.

<br>

## References

- **Cointegration & pairs trading.** Engle, R. F., & Granger, C. W. J. (1987). *Co-integration and error correction: representation, estimation, and testing.* Econometrica, 55(2), 251–276.
- **Pairs trading, empirically.** Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs trading: performance of a relative-value arbitrage rule.* Review of Financial Studies, 19(3), 797–827.
- **Pairs trading, book-length.** Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis.* Wiley.
- **Markov-switching autoregressions.** Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle.* Econometrica, 57(2), 357–384.
- **EM for hidden Markov models.** Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). *A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains.* Annals of Mathematical Statistics, 41(1), 164–171.
- **Pre-averaging for noisy high-frequency data.** Jacod, J., Li, Y., & Mykland, P. A. (2017). *Statistical properties of microstructure noise.* Econometrica, 85(4), 1133–1174.
- **High-frequency econometrics.** Aït-Sahalia, Y., & Jacod, J. (2014). *High-Frequency Financial Econometrics.* Princeton University Press.
- **Course.** [UiO STK-MAT2011 — Project Work in Stochastic Modelling](https://www.uio.no/studier/emner/matnat/math/STK-MAT2011/).
- **Tick data.** [Dukascopy historical data](https://www.dukascopy.com/swiss/english/marketwatch/historical/) · [HistData](https://www.histdata.com/) · [TrueFX](https://www.truefx.com/truefx-historical-downloads/).

<br>

<sub>University of Oslo · Department of Mathematics · STK-MAT2011 · Spring 2026</sub>
