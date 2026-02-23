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

All tick data is stored as compressed Parquet files in `code/data/processed/`. For testing purposes, small CSV snippets (1,000 rows each) are available in `code/data/samples/`.

### HistData (NinjaTrader + ASCII)

Cell values show **ticks / MB** (parquet). Dash (`-`) = not downloaded.

| # | Symbol | Ask (Jan) | Ask (Feb) | Bid (Jan) | Bid (Feb) | Last (Jan) | Last (Feb) |
|:-:|--------|:---------:|:---------:|:---------:|:---------:|:----------:|:----------:|
| 1 | **AUD/CAD** | - | - | - | - | 751K / 6.1 | 362K / 3.2 |
| 2 | **AUD/CHF** | - | - | - | - | 609K / 5.1 | 303K / 2.7 |
| 3 | **AUD/JPY** | - | - | - | - | 927K / 7.7 | 536K / 4.5 |
| 4 | **AUD/NZD** | - | - | - | - | 623K / 5.2 | 343K / 3.0 |
| 5 | **AUD/USD** | - | - | - | - | 638K / 5.4 | 400K / 3.4 |
| 6 | **AUX/AUD** | - | - | - | - | 127K / 1.3 | 102K / 1.1 |
| 7 | **BCO/USD** | 713K / 6.3 | 448K / 3.8 | 713K / 6.3 | 448K / 3.8 | 406K / 3.6 | 268K / 2.4 |
| 8 | **CAD/CHF** | - | - | - | - | 474K / 4.1 | 235K / 2.1 |
| 9 | **CAD/JPY** | - | - | - | - | 815K / 6.7 | 452K / 3.8 |
| 10 | **CHF/JPY** | - | - | - | - | 939K / 7.8 | 490K / 4.2 |
| 11 | **EUR/AUD** | - | - | - | - | 1.02M / 8.4 | 568K / 4.8 |
| 12 | **EUR/CAD** | - | - | - | - | 882K / 7.2 | 490K / 4.1 |
| 13 | **EUR/CHF** | - | - | - | - | 603K / 5.0 | 310K / 2.7 |
| 14 | **EUR/CZK** | - | - | - | - | 325K / 2.8 | 98K / 1.0 |
| 15 | **EUR/DKK** | - | - | - | - | 222K / 2.0 | 103K / 1.0 |
| 16 | **EUR/GBP** | - | - | - | - | 627K / 5.1 | 326K / 2.8 |
| 17 | **EUR/HUF** | - | - | - | - | 396K / 3.5 | 147K / 1.5 |
| 18 | **EUR/JPY** | - | - | - | - | 1.08M / 8.8 | 586K / 5.0 |
| 19 | **EUR/NOK** | - | - | - | - | 974K / 8.5 | 375K / 3.6 |
| 20 | **EUR/NZD** | - | - | - | - | 868K / 7.4 | 424K / 3.7 |
| 21 | **EUR/PLN** | - | - | - | - | 311K / 2.8 | 103K / 1.1 |
| 22 | **EUR/SEK** | - | - | - | - | 774K / 6.9 | 264K / 2.6 |
| 23 | **EUR/TRY** | - | - | - | - | 704K / 6.7 | 353K / 3.6 |
| 24 | **EUR/USD** | 1.51M / 12.8 | - | 1.51M / 12.7 | - | 665K / 5.7 | 344K / 3.0 |
| 25 | **GBP/JPY** | - | - | - | - | 1.11M / 9.1 | 566K / 4.8 |
| 26 | **GBP/USD** | - | - | - | - | 741K / 6.3 | 378K / 3.3 |
| 27 | **NZD/JPY** | - | - | - | - | 769K / 6.3 | 448K / 3.8 |
| 28 | **NZD/USD** | - | - | - | - | 571K / 4.8 | 331K / 2.9 |
| 29 | **USD/CAD** | - | - | - | - | 754K / 6.1 | 374K / 3.2 |
| 30 | **USD/CHF** | - | - | - | - | 605K / 5.2 | 313K / 2.8 |
| 31 | **USD/ZAR** | 3.22M / 29.6 | 1.86M / 17.2 | 3.22M / 29.6 | 1.86M / 17.2 | - | - |
| 32 | **XAU/USD** | - | - | - | - | 1.53M / 14.1 | 740K / 7.0 |
| | **SUM** | **5.44M / 48.6** | **2.31M / 21.0** | **5.44M / 48.6** | **2.31M / 21.0** | **21.83M / 185.4** | **11.13M / 98.6** |

> **HistData subtotal:** 72 parquet files -- 32 symbols -- 48.5M total ticks -- 423 MB on disk
>
> Period: Jan-Feb 2026. EUR/USD Ask/Bid comes from ASCII files (~1.5M ticks).
> BCO/USD and USD/ZAR Ask/Bid are also derived from ASCII format files.

### TrueFX

Cell values show **ticks / MB** (parquet). Data includes bid and ask (no volume).

| # | Symbol | Bid (Nov 25) | Bid (Dec 25) | Bid (Jan 26) | Ask (Nov 25) | Ask (Dec 25) | Ask (Jan 26) |
|:-:|--------|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| 1 | **EUR/CHF** | 645K / 5.5 | 687K / 5.9 | 782K / 6.6 | 645K / 5.6 | 687K / 5.8 | 782K / 6.7 |
| 2 | **EUR/USD** | 665K / 5.8 | 694K / 5.9 | 755K / 6.7 | 665K / 5.8 | 694K / 5.9 | 755K / 6.7 |
| 3 | **USD/ZAR** | 863K / 7.6 | 926K / 8.1 | 1.02M / 8.9 | 863K / 7.6 | 926K / 8.2 | 1.02M / 8.9 |
| | **SUM** | **2.17M / 18.9** | **2.31M / 19.9** | **2.56M / 22.2** | **2.17M / 19.0** | **2.31M / 19.9** | **2.56M / 22.3** |

> **TrueFX subtotal:** 18 parquet files -- 3 symbols -- 14.1M total ticks -- 122 MB on disk
>
> Period: Nov 2025 - Jan 2026. Millisecond precision. No volume data.

### Dukascopy

Cell values show **ticks / MB** (parquet). Data includes bid, ask, and volume.

| # | Symbol | Bid (Nov 25) | Bid (Dec 25) | Bid (Jan 26) | Ask (Nov 25) | Ask (Dec 25) | Ask (Jan 26) |
|:-:|--------|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| 1 | **EUR/CHF** | 1.01M / 9.0 | 984K / 8.7 | 1.21M / 11.0 | 1.01M / 8.9 | 984K / 8.6 | 1.21M / 10.9 |
| 2 | **EUR/USD** | 1.23M / 11.6 | 1.30M / 12.0 | 1.51M / 13.9 | 1.23M / 11.5 | 1.30M / 12.0 | 1.51M / 13.9 |
| 3 | **USD/ZAR** | 4.52M / 41.6 | 3.06M / 27.9 | 3.22M / 30.0 | 4.52M / 41.7 | 3.06M / 27.9 | 3.22M / 30.0 |
| | **SUM** | **6.76M / 62.2** | **5.35M / 48.6** | **5.94M / 54.9** | **6.76M / 62.1** | **5.35M / 48.5** | **5.94M / 54.8** |

> **Dukascopy subtotal:** 18 parquet files -- 3 symbols -- 36.1M total ticks -- 331 MB on disk
>
> Period: Nov 2025 - Jan 2026. Millisecond precision. Includes bid/ask volume.

### Data Sources

| Source | Type | Link |
|--------|------|------|
| **HistData** | NinjaTrader tick CSV | [histdata.com](https://www.histdata.com/) |
| **TrueFX** | Tick CSV (bid/ask) | [truefx.com/truefx-historical-downloads](https://www.truefx.com/truefx-historical-downloads/) |
| **Dukascopy** | Tick API (bid/ask + volume) | [dukascopy.com](https://www.dukascopy.com/swiss/english/marketwatch/historical/) -- Python API: [dukascopy-python (PyPI)](https://pypi.org/project/dukascopy-python/) |

### Processing Scripts

| Script | Source | Description |
|--------|--------|-------------|
| `code/scripts/parquet_histdata.py` | HistData | Converts NinjaTrader + ASCII CSVs to Parquet |
| `code/scripts/parquet_truefx.py` | TrueFX | Converts TrueFX tick CSVs to Parquet |
| `code/scripts/parquet_dukascopy.py` | Dukascopy | Downloads tick data via API and saves as Parquet |
| `code/scripts/create_samples.py` | N/A | Generates small CSV samples from Parquet files |

---

## Quick Start

```bash
pip install -r requirements.txt
```

---

## Resources

- [Course Page (UiO)](https://www.uio.no/studier/emner/matnat/math/STK-MAT2011/)
- [HistData -- Free Forex Historical Data](https://www.histdata.com/)
- [TrueFX -- Historical Downloads](https://www.truefx.com/truefx-historical-downloads/)
- [Dukascopy -- Historical Market Data](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
- [dukascopy-python (PyPI)](https://pypi.org/project/dukascopy-python/)

---

<sub>University of Oslo · Department of Mathematics · Spring 2026</sub>
