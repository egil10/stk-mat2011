# STK-MAT2011 — Machine Learning for High-Frequency Time Series

> **University of Oslo** · Bachelor's Level · 10 Credits · Spring 2026

---

## Motivation

High-frequency financial time series present unique challenges and opportunities for machine learning. Unlike traditional time series analysis, tick-by-tick data operates at millisecond resolution, capturing microstructure effects that vanish at lower frequencies. This project explores how modern ML techniques can extract meaningful patterns from this noisy, high-dimensional data.

### Why High-Frequency Data?

At the tick level, financial markets reveal microstructure dynamics invisible in daily or hourly aggregates:

- **Market microstructure effects**: Bid-ask bounce, order flow clustering, and spread dynamics create autocorrelation patterns that violate standard random walk assumptions
- **Regime switching**: Markets transition between volatility regimes (calm vs. turbulent) that require state-dependent models
- **Information asymmetry**: Order flow and tick-level patterns contain signals about short-term price movements
- **Latency arbitrage**: Microsecond-level patterns matter for algorithmic trading strategies

### Challenges

High-frequency data introduces several complications:

- **Noise**: Tick-level prices are contaminated by bid-ask bounce and market microstructure noise
- **Non-stationarity**: Market regimes change throughout the trading day, requiring adaptive models
- **Curse of dimensionality**: Millions of observations with complex dependencies challenge traditional time series methods
- **Overfitting risk**: Rich data can easily lead to spurious patterns without proper regularization

### Machine Learning Approaches

This project investigates several ML frameworks:

**Regime-Switching Models**
- Hidden Markov Models (HMM) and Markov-switching autoregression to identify latent market states
- Combining regime detection with volatility modeling (GARCH) for state-dependent dynamics

**Pre-averaging and Feature Engineering**
- Aggregating tick data into meaningful windows (time-based or tick-count based)
- Extracting microstructure features: spreads, order flow imbalance, inter-tick durations

**Rolling Window Analysis**
- Fitting AR models on rolling windows to capture time-varying autocorrelation
- Visualizing how model parameters evolve throughout trading days

**Pairs Trading**
- Cointegration analysis to identify mean-reverting relationships between currency pairs
- ML-enhanced spread modeling for pairs trading strategies

---

## Data Availability

| Symbol | Status | Available Since | Notes |
|--------|--------|-----------------|-------|
| **EUR/USD** | Available | Jan 2026 | ~1.5M ticks, millisecond resolution |
| EUR/CHF | Not yet acquired | — | — |
| EUR/GBP | Not yet acquired | — | — |
| EUR/JPY | Not yet acquired | — | — |
| EUR/AUD | Not yet acquired | — | — |
| USD/CAD | Not yet acquired | — | — |
| USD/CHF | Not yet acquired | — | — |
| USD/JPY | Not yet acquired | — | — |
| USD/MXN | Not yet acquired | — | — |
| GBP/CHF | Not yet acquired | — | — |
| GBP/JPY | Not yet acquired | — | — |
| GBP/USD | Not yet acquired | — | — |
| AUD/JPY | Not yet acquired | — | — |
| AUD/USD | Not yet acquired | — | — |
| CHF/JPY | Not yet acquired | — | — |
| NZD/JPY | Not yet acquired | — | — |
| NZD/USD | Not yet acquired | — | — |
| **XAU/USD** | Not yet acquired | — | Target: metals pair |
| EUR/CAD | Not yet acquired | — | — |
| AUD/CAD | Not yet acquired | — | — |
| CAD/JPY | Not yet acquired | — | — |
| EUR/NZD | Not yet acquired | — | — |
| GRX/EUR | Not yet acquired | — | — |
| NZD/CAD | Not yet acquired | — | — |
| SGD/JPY | Not yet acquired | — | — |
| USD/HKD | Not yet acquired | — | — |
| USD/NOK | Not yet acquired | — | — |
| **USD/TRY** | Not yet acquired | — | Target: EM pair |
| XAU/AUD | Not yet acquired | — | — |
| AUD/CHF | Not yet acquired | — | — |
| AUX/AUD | Not yet acquired | — | — |
| EUR/HUF | Not yet acquired | — | — |
| EUR/PLN | Not yet acquired | — | — |
| FRX/EUR | Not yet acquired | — | — |
| HKX/HKD | Not yet acquired | — | — |
| NZD/CHF | Not yet acquired | — | — |
| SPX/USD | Not yet acquired | — | — |
| USD/HUF | Not yet acquired | — | — |
| USD/PLN | Not yet acquired | — | — |
| **USD/ZAR** | Not yet acquired | — | Target: EM pair |
| XAU/CHF | Not yet acquired | — | — |
| ZAR/JPY | Not yet acquired | — | — |
| BCO/USD | Not yet acquired | — | — |
| ETX/EUR | Not yet acquired | — | — |
| EUR/CZK | Not yet acquired | — | — |
| EUR/SEK | Not yet acquired | — | — |
| GBP/AUD | Not yet acquired | — | — |
| GBP/NZD | Not yet acquired | — | — |
| JPX/JPY | Not yet acquired | — | — |
| UDX/USD | Not yet acquired | — | — |
| USD/CZK | Not yet acquired | — | — |
| USD/SEK | Not yet acquired | — | — |
| WTI/USD | Not yet acquired | — | — |
| XAU/EUR | Not yet acquired | — | — |
| AUD/NZD | Not yet acquired | — | — |
| CAD/CHF | Not yet acquired | — | — |
| EUR/DKK | Not yet acquired | — | — |
| EUR/NOK | Not yet acquired | — | — |
| EUR/TRY | Not yet acquired | — | — |
| GBP/CAD | Not yet acquired | — | — |
| NSX/USD | Not yet acquired | — | — |
| UKX/GBP | Not yet acquired | — | — |
| USD/DKK | Not yet acquired | — | — |
| USD/SGD | Not yet acquired | — | — |
| XAG/USD | Not yet acquired | — | — |
| XAU/GBP | Not yet acquired | — | — |

**Data source:** [HistData.com](https://www.histdata.com/)

---

## Current Dataset (EUR/USD)

| Metric | Value |
|--------|-------|
| Time range | Jan 1 – Jan 30, 2026 |
| Total ticks | ~1.5 million |
| Bid range | 1.15778 – 1.20805 |
| Ask range | 1.15794 – 1.20809 |
| Avg spread | 0.33 pips |

---

## Quick Start

```bash
pip install -r requirements.txt
python code/scripts/visualize_forex.py
python code/scripts/tick_microstructure_acf.py
```

---

## Resources

- [Course Page (UiO)](https://www.uio.no/studier/emner/matnat/math/STK-MAT2011/)
- [HistData – Free Forex Historical Data](https://www.histdata.com/)

---

<sub>University of Oslo · Department of Mathematics · Spring 2026</sub>
