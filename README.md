# Hidden Markov Models for High-Frequency FX Pairs Trading

Three cointegrated FX pairs. Four trading rules. One question: can a hidden Markov model tell us *when* the spread is mean-reverting and *when* it has slipped into a regime where the textbook trade quietly bleeds through transaction costs?

This repository is the codebase behind a University of Oslo `STK-MAT2011` thesis. The core experiments live in the `m-*` notebooks — three pairs $\times$ three months, each fitted **independently per trading day** — and the modules in `code/scripts/` are the engine they call. The strictly-causal walk-forward extension lives in the `Y-*` notebooks.

<br>

## The Four Strategies

All four strategies see the same cointegration spread $S_t$ and the same rolling $z$-score $Z_t$. They differ only in **whether and how** the regime posterior $\gamma_t^{\mathrm{MR}}$ from a Markov-switching AR(1) enters the rule.

| # | Strategy | Entry rule | Regime gate $G_t$ | Reads HMM? |
|---|---|---|---|---|
| **(1)** | **Buy & Hold** | always long, $\pi_t^{\mathrm{BH}}=+1$ | — | No |
| **(2)** | **Baseline** | $\|Z_t\| > z_q$ | $G_t = 1$ | No |
| **(3)** | **AR** | $\|Z_t\| > z_q$ | $G_t = \mathbf{1}\{\gamma_t^{\mathrm{MR}} \geq 1-\delta\}$ | Yes — binary |
| **(4)** | **MS-AR** | $\|Z_t\| > z_q^{\mathrm{eff}}(t)$ | $G_t = 1$ | Yes — continuous |

**(1) Buy & Hold.** A passive drift reference. Long the spread at every bar, zero costs — the benchmark every active rule has to beat.

**(2) Baseline.** Classic Bollinger-style mean reversion on the cointegration $z$-score. With entry threshold $z_q$ and exit threshold $z_x$, the position $\pi_t \in \{-1, 0, +1\}$ is updated by

$$\pi_t \;=\; \begin{cases}+1 & \pi_{t-1}=0,\; Z_t < -z_q \\ -1 & \pi_{t-1}=0,\; Z_t > +z_q \\ \;\;\,0 & \pi_{t-1}\neq 0,\; |Z_t| \leq z_x \\ \pi_{t-1} & \text{otherwise.}\end{cases}$$

Knows nothing about regimes. Open on a wide deviation, close on the mean-cross.

**(3) AR.** Baseline plus a **binary danger gate**. With danger tolerance $\delta \in (0, 1)$,

$$G_t \;=\; \mathbf{1}\{\gamma_t^{\mathrm{MR}} \geq 1-\delta\}, \qquad \pi_t \;=\; \begin{cases} 0 & G_t = 0,\\ \pi_t^{\mathrm{baseline}} & G_t = 1. \end{cases}$$

The gate is binary: trade exactly like the baseline, or sit in cash. Existing positions are liquidated as soon as the gate closes.

**(4) MS-AR.** Keeps the gate open at all times but **widens the entry threshold continuously** with the drifting-regime weight $\gamma_t^{\mathrm{DR}} = 1 - \gamma_t^{\mathrm{MR}}$:

$$z_q^{\mathrm{eff}}(t) \;=\; \gamma_t^{\mathrm{MR}}\, z_q \;+\; \gamma_t^{\mathrm{DR}}\, z_v, \qquad z_v \geq z_q.$$

In a clean mean-reverting regime $\gamma_t^{\mathrm{MR}} \to 1$ and the rule trades like the Baseline; as the drifting regime takes over, the bar to enter rises smoothly toward $z_v$. A soft, continuous version of (3).

<br>

## The Pipeline

The `m-*` notebooks run the seven steps below **independently per trading day**. Day-by-day fitting keeps the regime parameters local: nothing learned on Monday leaks into Tuesday, and any look-ahead in the smoothed posteriors is bounded by a single session.

**1. Mid-price.** With $i \in \{A, B\}$ indexing the two legs of a pair and $P_t^{\mathrm{bid},i}$, $P_t^{\mathrm{ask},i}$ the prevailing top-of-book quotes,

$$P_t^{\mathrm{mid},i} \;=\; \tfrac{1}{2}\left(P_t^{\mathrm{bid},i} + P_t^{\mathrm{ask},i}\right).$$

**2. Half-spread in basis points.** Used as the per-leg slippage charge in the back-test:

$$\mathrm{HS}_t^{\,i} \;=\; \tfrac{1}{2}\,\frac{P_t^{\mathrm{ask},i} - P_t^{\mathrm{bid},i}}{P_t^{\mathrm{mid},i}}\times 10^{4}.$$

**3. Tick-synced pre-averaging.** Following Jacod, Li & Mykland (2017), each block of $L = 500$ synchronised ticks is collapsed into a single observation by averaging the mids:

$$\widetilde P_k^{\,i} \;=\; \tfrac{1}{L}\!\sum_{n=(k-1)L+1}^{kL} P_{t_n}^{\mathrm{mid},i}, \qquad k = 1, 2, \ldots$$

Same $L$, same block boundaries on both legs, so the pre-averaged series stay perfectly co-timed. Pre-averaging is preferred to fixed calendar-time sampling because it yields an index that is more nearly homogeneous in microstructure activity and substantially reduces the noise variance of the resulting return series. All downstream quantities are computed on the pre-averaged series; for notational simplicity we continue to write $P_t^{\mathrm{mid},i}$ for the price fed to the models, with $t$ now indexing pre-averaged blocks.

**4. Rolling cointegration.** Two non-stationary log-price series are cointegrated if some linear combination is stationary (Engle & Granger 1987). High-frequency FX is not globally cointegrated — $\beta$ drifts with liquidity, session structure, and pair-specific shocks — so we estimate $\beta_t$ by rolling OLS with window $W_\beta = 25$. Writing $x_s = \log P_s^{\mathrm{mid},B}$, $y_s = \log P_s^{\mathrm{mid},A}$, $\mathcal{W}_t = \{t-W_\beta+1,\ldots,t\}$:

$$\beta_t \;=\; \frac{W_\beta \sum_{s \in \mathcal{W}_t} x_s y_s - \left(\sum_{s \in \mathcal{W}_t} x_s\right)\left(\sum_{s \in \mathcal{W}_t} y_s\right)}{W_\beta \sum_{s \in \mathcal{W}_t} x_s^{2} - \left(\sum_{s \in \mathcal{W}_t} x_s\right)^{2}}, \qquad \alpha_t \;=\; \tfrac{1}{W_\beta}\left(\sum_{s \in \mathcal{W}_t} y_s - \beta_t \sum_{s \in \mathcal{W}_t} x_s\right).$$

The cointegration spread and its strictly-causal one-step return are

$$S_t \;=\; \log P_t^{\mathrm{mid},A} \;-\; \beta_t \log P_t^{\mathrm{mid},B} \;-\; \alpha_t, \qquad \Delta S_t \;=\; r_t^{A} - \beta_{t-1}\, r_t^{B}, \qquad r_t^{i} = \log P_t^{\mathrm{mid},i} - \log P_{t-1}^{\mathrm{mid},i}.$$

The lag in $\beta_{t-1}$ ensures the hedge ratio is observable at the moment the return is realised.

**5. Rolling $z$-score.** Over a window $W_z = 15$,

$$Z_t \;=\; \frac{S_t - \mu_t^{S}}{\sigma_t^{S}}, \quad \mu_t^{S} = \tfrac{1}{W_z}\sum_{s=t-W_z+1}^{t} S_s, \quad (\sigma_t^{S})^{2} = \tfrac{1}{W_z}\sum_{s=t-W_z+1}^{t}\left(S_s - \mu_t^{S}\right)^{2}.$$

**6. Markov-switching AR(1) on the spread.** An AR(1) on $S_t$ with all three parameters switching according to an unobserved Markov chain $K_t \in \{1, 2\}$ (Hamilton 1989; Krolzig 1997):

$$S_t \mid S_{t-1},\, K_t = k \;\sim\; \mathcal{N}\left(c^{(k)} + \rho^{(k)} S_{t-1},\; (\sigma^{(k)})^{2}\right), \qquad \mathbf{P} = \begin{pmatrix} p_{11} & p_{12} \\ p_{21} & p_{22} \end{pmatrix}, \qquad p_{ij} = \Pr(K_t = j \mid K_{t-1} = i).$$

For $|\rho^{(k)}| < 1$ the regime mean is $\mu^{(k)} = c^{(k)} / (1 - \rho^{(k)})$, giving the mean-deviation form

$$S_t \;=\; \mu^{(k)} + \rho^{(k)}\left(S_{t-1} - \mu^{(k)}\right) + \varepsilon_t^{(k)}, \qquad \varepsilon_t^{(k)} \sim \mathcal{N}\left(0,(\sigma^{(k)})^{2}\right).$$

Values of $\rho^{(k)}$ close to zero correspond to fast mean reversion; values close to unity, to near-random-walk behaviour. The parameter set $\Theta = \{\nu, \mathbf{P}, c^{(k)}, \rho^{(k)}, \sigma^{(k)}\}$ is estimated by Expectation–Maximisation (Baum–Welch), with the E-step computing the smoothed posteriors

$$\gamma_t(k) \;=\; \Pr(K_t = k \mid S_1, \ldots, S_T;\, \Theta)$$

via the forward–backward recursion, and the M-step updating $\Theta$ to maximise the expected complete-data log-likelihood. The EM is restarted from several random seeds to mitigate convergence to local maxima. Before estimation $S_t$ is winsorised at $\mathrm{median}(S) \pm 4\,\mathrm{std}(S)$ and rescaled by $10^{4}$ for numerical stability.

**7. Regime labelling.** The mean-reverting regime is the one with the smaller residual std; the drifting regime is its complement:

$$k_{\mathrm{MR}} = \arg\min_{k}\sigma^{(k)}, \qquad k_{\mathrm{DR}} = \arg\max_{k}\sigma^{(k)}, \qquad \gamma_t^{\mathrm{MR}} \;=\; \Pr(K_t = k_{\mathrm{MR}} \mid S_1, \ldots, S_T), \qquad \gamma_t^{\mathrm{DR}} = 1 - \gamma_t^{\mathrm{MR}}.$$

These feed strategies (3) and (4).

<br>

## Model Selection

The number of states $K$ and AR order $p$ are chosen by the Bayesian Information Criterion (Schwarz 1978). With $m$ free parameters, $T$ observations, and maximised likelihood $\hat L$,

$$\mathrm{BIC} \;=\; m\ln T - 2\ln\hat L.$$

BIC's $\ln T$ penalty is aggressive on the large samples produced by high-frequency tick data and is the standard choice for hidden Markov order selection, which is prone to over-fit under more lenient criteria. We sweep over a grid of $(K, p)$ and retain the configuration that minimises BIC. The headline results are reported for the BIC-selected $K = 2$, $p = 1$ specification.

<br>

## Transaction Costs and Causality

Every position flip in leg $i$ pays the half-spread $\mathrm{HS}_t^{\,i}$ plus a fixed commission `fee_bps = 0.05` (≈ $5 per $1M USD notional, the institutional ECN / prime-brokerage rate). The round-trip cost of one pair-position is therefore approximately

$$\mathrm{HS}^{A}_t + \widehat{\beta}\,\mathrm{HS}^{B}_t \quad\text{(bps).}$$

At this bar resolution, the half-spread term is by far the binding cost.

> **A note on causality.** The `m-*` notebooks feed the in-sample *smoothed* posteriors $\gamma_t^{\mathrm{MR}}$ to the strategies. This is a *best-case* picture of the regime signal — it assumes perfect hindsight on regime identity *within the day*. The strictly forward-filtered, walk-forward extension — re-estimating $\Theta$ and the hyperparameters $(W_\beta, W_z, z_q, z_x, z_v, \delta)$ on a training window and applying them to a disjoint OOS window — lives in `code/notebooks/Y-*.ipynb`.

<br>

## The Three Pairs

| Pair | Monthly notebook | Walk-forward | Why this pair |
|---|---|---|---|
| **AUDUSD / NZDUSD** | `m-audnzd.ipynb` | `Y-AUDNZD.ipynb` | Antipodean commodity currencies — the textbook cointegrated FX pair |
| **GBPUSD / EURUSD** | `m-gbpeur.ipynb` | `Y-GBPEUR.ipynb` | European majors against the dollar — same shock, different beta |
| **EURNOK / EURSEK** | `m-noksek.ipynb` | `Y-NOKSEK.ipynb` | Nordic cross-rates — small open economies tied to oil and rates |

Each `m-*` notebook runs the four strategies on three independent sub-samples — **August 2024**, **December 2024**, and **April 2025** — chosen for their distinct macro backdrops (summer dollar weakness, year-end positioning, the spring 2025 tariff episode). Each prints a four-column tearsheet (`BuyHold | Baseline | AR | MS_AR`) and dumps every panel as a stand-alone PDF for the thesis.

<br>

## Data

Tick-level bid/ask FX from [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) via [dukascopy-python](https://pypi.org/project/dukascopy-python/), stored as monthly Parquet files in `code/data/processed/`. Cross-checks against [HistData](https://www.histdata.com/) and [TrueFX](https://www.truefx.com/truefx-historical-downloads/). A small CSV sample lives in `code/data/samples/dukascopy_sample.csv`.

| Coverage | Symbols | Span |
|---|---|---|
| **Majors** | EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD | 2019 – Feb 2026 |
| **Nordics** | EURNOK, EURSEK | Oct 2023 – Feb 2026 |
| **Crosses** | EURCHF, EURNZD, EURPLN, USDZAR | 2024 – Feb 2026 |
| **Crypto** | BTCUSD, ETHUSD | spot checks, 2024 – 2025 |

Bid and ask streams are sorted by timestamp and aligned by a backward as-of merge before mid-price construction and pre-averaging.

<br>

## Performance Metrics

The tearsheet reports both statistical and capital-outcome metrics. With $R_t$ the strategy return at bar $t$, $\mu_R = \mathbb{E}[R]$, $\sigma_R = \sqrt{\mathrm{Var}(R)}$, $V_t$ the portfolio value with running peak $V_t^{\star} = \max_{s \leq t} V_s$, and drawdown $D_t = (V_t^{\star} - V_t)/V_t^{\star}$:

$$\sigma_{\mathrm{ann}} = \sigma_R\sqrt{T}, \qquad \mathrm{VaR}_\alpha = -Q_{1-\alpha}(R), \qquad \mathrm{CVaR}_\alpha = -\mathbb{E}[R \mid R \leq -\mathrm{VaR}_\alpha],$$

$$\mathrm{MDD} = \max_t D_t, \qquad \mathrm{UI} = \sqrt{\tfrac{1}{N}\textstyle\sum_t D_t^{2}}, \qquad \mathrm{SR} = \frac{\mu_R - R_f}{\sigma_R}, \qquad \mathrm{Sortino} = \frac{\mu_R - R_f}{\sigma_d}, \qquad \mathrm{Calmar} = \frac{\mathrm{AR}}{\mathrm{MDD}}.$$

The Sharpe annualisation factor is computed per run as $252 \times (n_{\mathrm{bars}}/n_{\mathrm{days}})$, so the magnitude matches the actual sampling frequency on each pair/month combination rather than assuming a fixed bar duration.

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
│   ├── screener.py        <- Engle-Granger, ADF, half-life, rolling diagnostics
│   ├── engine.py          <- rolling OLS, z-score, MS-AR(1) per-day fit (EM)
│   ├── backtester.py      <- numba-accelerated 4-strategy back-test
│   ├── tearsheet.py       <- tearsheet + per-panel PDF export
│   ├── wfo.py             <- Optuna walk-forward hyperparameter search
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

**Pairs trading and cointegration**

- Engle, R. F., & Granger, C. W. J. (1987). *Co-integration and error correction: representation, estimation, and testing.* Econometrica, 55(2), 251–276.
- Dickey, D. A., & Fuller, W. A. (1979). *Distribution of the estimators for autoregressive time series with a unit root.* Journal of the American Statistical Association, 74(366), 427–431.
- Said, S. E., & Dickey, D. A. (1984). *Testing for unit roots in autoregressive–moving average models of unknown order.* Biometrika, 71(3), 599–607.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs trading: performance of a relative-value arbitrage rule.* Review of Financial Studies, 19(3), 797–827.
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis.* Wiley.
- Krauss, C. (2017). *Statistical arbitrage pairs trading strategies: review and outlook.* Journal of Economic Surveys, 31(2), 513–545.

**Markov-switching models and HMM estimation**

- Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle.* Econometrica, 57(2), 357–384.
- Hamilton, J. D. (1994). *Time Series Analysis.* Princeton University Press.
- Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions: Modelling, Statistical Inference, and Application to Business Cycle Analysis.* Springer.
- Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition.* Proceedings of the IEEE, 77(2), 257–286.
- Cappé, O., Moulines, E., & Rydén, T. (2005). *Inference in Hidden Markov Models.* Springer.
- Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). *A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains.* Annals of Mathematical Statistics, 41(1), 164–171.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum likelihood from incomplete data via the EM algorithm.* JRSS B, 39(1), 1–38.
- Schwarz, G. (1978). *Estimating the dimension of a model.* Annals of Statistics, 6(2), 461–464.

**Microstructure and high-frequency econometrics**

- Roll, R. (1984). *A simple implicit measure of the effective bid–ask spread in an efficient market.* Journal of Finance, 39(4), 1127–1139.
- Hasbrouck, J. (2007). *Empirical Market Microstructure.* Oxford University Press.
- Hautsch, N. (2012). *Econometrics of Financial High-Frequency Data.* Springer.
- Jacod, J., Li, Y., & Mykland, P. A. (2017). *Statistical properties of microstructure noise.* Econometrica, 85(4), 1133–1174.
- Aït-Sahalia, Y., & Jacod, J. (2014). *High-Frequency Financial Econometrics.* Princeton University Press.
- Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues.* Quantitative Finance, 1(2), 223–236.

**Risk metrics and back-testing**

- Sharpe, W. F. (1994). *The Sharpe ratio.* Journal of Portfolio Management, 21(1), 49–58.
- Sortino, F. A., & van der Meer, R. (1991). *Downside risk.* Journal of Portfolio Management, 17(4), 27–31.
- Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). *Coherent measures of risk.* Mathematical Finance, 9(3), 203–228.
- Martin, P. G., & McCann, B. (1989). *The Investor's Guide to Fidelity Funds.* (Ulcer index.)
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Aldridge, I. (2013). *High-Frequency Trading.* 2nd ed., Wiley.

**Course and data**

- [UiO STK-MAT2011 — Project Work in Stochastic Modelling](https://www.uio.no/studier/emner/matnat/math/STK-MAT2011/).
- [Dukascopy historical data](https://www.dukascopy.com/swiss/english/marketwatch/historical/) · [HistData](https://www.histdata.com/) · [TrueFX](https://www.truefx.com/truefx-historical-downloads/).

<br>

<sub>University of Oslo · Department of Mathematics · STK-MAT2011 · Spring 2026</sub>
