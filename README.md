# Machine Learning for High-Frequency Time Series


## What This Is About

Tick-by-tick financial data is one of the most information-dense environments you can study. At millisecond resolution, markets reveal microstructure dynamics that are completely invisible in daily or hourly data — bid-ask bounce, order flow clustering, volatility regime shifts, and fleeting arbitrage signals. This project dives into that world using modern machine learning.

The goal is to go beyond textbook time series analysis and wrestle with data that is messy, non-stationary, and enormous by design. If you can build models that work here, you can build models that work anywhere.


## Why High-Frequency Data?

At the tick level, markets expose their mechanics:

- **Microstructure effects** — bid-ask bounce and order flow create autocorrelation patterns that break random walk assumptions
- **Regime switching** — markets cycle between calm and turbulent states that require adaptive, state-dependent models
- **Information asymmetry** — short-term price signals are embedded in order flow and tick patterns before they aggregate away
- **Scale** — millions of observations per symbol per month demand efficient, vectorized computation

### The Real Challenges

Working at this resolution is not just academically interesting — it is genuinely hard:

- **Noise** — tick prices are contaminated by microstructure; raw data is not your signal
- **Non-stationarity** — regimes shift intraday, requiring models that can adapt or detect change
- **Dimensionality** — traditional time series tools struggle with millions of correlated observations
- **Overfitting** — rich data invites spurious patterns; disciplined regularization is essential


## Methods Explored

**Regime-Switching Models**
Hidden Markov Models and Markov-switching autoregression to identify latent market states. Combined with GARCH for state-dependent volatility dynamics.

**Pre-averaging and Feature Engineering**
Aggregating raw ticks into meaningful windows, and extracting microstructure features: bid-ask spreads, order flow imbalance, inter-tick durations.

**Rolling Window Analysis**
Fitting AR models on rolling windows to capture time-varying autocorrelation. Visualizing how model parameters evolve across the trading day.

**Pairs Trading**
Cointegration analysis to find mean-reverting relationships between currency pairs, with ML-enhanced spread modeling for strategy development.


## The Models

Every model in this project is a building block in the same hierarchy — from simple linear structure to full conditional heteroskedasticity. Understanding each step makes the next one obvious.

<br>

**AR(p)** — Autoregressive

The price at time $t$ is a linear combination of its own past $p$ values. The simplest possible structure for temporal dependence.

$$y_t = c + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t$$

<br>

**MA(q)** — Moving Average

Instead of lagged prices, the model depends on lagged shocks. Short-lived, mean-reverting dynamics.

$$y_t = \mu + \varepsilon_t + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j}$$

<br>

**ARMA(p, q)** — Autoregressive Moving Average

Combines both: persistent autocorrelation from AR, transient shock responses from MA.

$$y_t = c + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j}$$

<br>

**ARIMA(p, d, q)** — Integrated

Non-stationary series are differenced $d$ times before fitting ARMA. Handles trending prices without detrending by hand.

$$\Delta^d y_t = c + \sum_{i=1}^{p} \phi_i\, \Delta^d y_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j}$$

<br>

**ARIMAX(p, d, q)** — With Exogenous Variables

ARIMA extended with external regressors $x_{k,t}$ — order flow, spread, or other microstructure signals enter as covariates.

$$\Delta^d y_t = c + \sum_{i=1}^{p} \phi_i\, \Delta^d y_{t-i} + \sum_{k=1}^{K} \beta_k\, x_{k,t} + \varepsilon_t + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j}$$

<br>

**ARCH(q)** — Autoregressive Conditional Heteroskedasticity

Volatility is not constant — it clusters. ARCH models the conditional variance as a function of past squared residuals.

$$\sigma_t^2 = \omega + \sum_{j=1}^{q} \alpha_j\, \varepsilon_{t-j}^2$$

<br>

**GARCH(p, q)** — Generalized ARCH

Adds lagged variance terms to ARCH, capturing the long memory of volatility with far fewer parameters. The workhorse of financial volatility modeling.

$$\sigma_t^2 = \omega + \sum_{i=1}^{p} \beta_i\, \sigma_{t-i}^2 + \sum_{j=1}^{q} \alpha_j\, \varepsilon_{t-j}^2$$

In all models, $\varepsilon_t \sim \mathcal{N}(0,\, \sigma_t^2)$.

<br>

## Quick Start

```bash
pip install -r requirements.txt
```

Data lives in `code/data/processed/` as compressed Parquet files. Small CSV samples (1 000 rows each) are available in `code/data/samples/` for quick experimentation.


## Data

Three data sources, covering EUR/USD, EUR/CHF, USD/ZAR and a broad set of forex pairs. All stored as Parquet for fast, memory-efficient loading.

### Sources

| Source | Coverage | Format | Link |
|--------|----------|--------|------|
| **HistData** | 32 pairs · Jan–Feb 2026 | NinjaTrader tick CSV | [histdata.com](https://www.histdata.com/) |
| **TrueFX** | 3 pairs · Nov 2025 – Jan 2026 | Tick CSV (bid/ask) | [truefx.com](https://www.truefx.com/truefx-historical-downloads/) |
| **Dukascopy** | 3 pairs · Nov 2025 – Jan 2026 | API (bid/ask + volume) | [dukascopy.com](https://www.dukascopy.com/swiss/english/marketwatch/historical/) · [PyPI](https://pypi.org/project/dukascopy-python/) |

### Volume Summary

| Source | Files | Symbols | Total Ticks | Size on Disk |
|--------|------:|--------:|------------:|-------------:|
| HistData | 72 | 32 | ~48.5M | ~423 MB |
| TrueFX | 18 | 3 | ~14.1M | ~122 MB |
| Dukascopy | 18 | 3 | ~36.1M | ~331 MB |
| **Total** | **108** | **35** | **~98.7M** | **~876 MB** |

> HistData `---` means the Ask/Bid file was not downloaded for that symbol (Last price only). EUR/USD, BCO/USD and USD/ZAR include Ask/Bid from ASCII format files.

### Processing Scripts

| Script | Source | Purpose |
|--------|--------|---------|
| `code/scripts/p_hist.py` | HistData | Converts NinjaTrader + ASCII CSVs to Parquet |
| `code/scripts/p_true.py` | TrueFX | Converts TrueFX tick CSVs to Parquet |
| `code/scripts/p_duka.py` | Dukascopy | Downloads tick data via API and saves as Parquet |
| `code/scripts/samp.py` | — | Generates small CSV samples from Parquet files |


## Resources

- [Course Page (UiO)](https://www.uio.no/studier/emner/matnat/math/STK-MAT2011/)
- [HistData — Free Forex Historical Data](https://www.histdata.com/)
- [TrueFX — Historical Downloads](https://www.truefx.com/truefx-historical-downloads/)
- [Dukascopy — Historical Market Data](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
- [dukascopy-python (PyPI)](https://pypi.org/project/dukascopy-python/)


<sub>University of Oslo · Department of Mathematics · Spring 2026</sub>
