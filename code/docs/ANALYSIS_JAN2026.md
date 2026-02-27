# January 2026 Tick Data Analysis

## Overview

This document describes the exploratory data analysis (EDA), pre-averaging, and AR(1) 
modelling performed on high-frequency tick data for three currency pairs during 
January 2026.

## Data

| Pair    | Source     | Price type               | Ticks (Jan) |
|---------|------------|--------------------------|-------------|
| EURUSD  | Dukascopy  | mid = (bid + ask) / 2    | ~1.5M       |
| USDZAR  | Dukascopy  | mid = (bid + ask) / 2    | ~3.2M       |
| XAUUSD  | HistData   | last (no bid/ask avail.) | ~1.5M       |

**Selection rationale:**  
- **EURUSD** — most liquid G7 pair, baseline behaviour  
- **USDZAR** — emerging-market exotic, wider spreads, different microstructure  
- **XAUUSD** — precious metal / USD, different asset class dynamics  

### Data loading

`code/scripts/data_jan.py` provides:
- `load_mid(pair, month)` — loads and merges Dukascopy bid/ask into mid, or HistData last  
- `load_three_pairs()` — convenience loader for all three pairs  

## Scripts

| Script         | Purpose                                        |
|----------------|------------------------------------------------|
| `data_jan.py`  | Data loading & mid-price construction          |
| `plots_jan.py` | EDA plots: ticks, gaps, microstructure, preavg |
| `AR1.py`       | Rolling / non-overlapping AR(1) estimation     |
| `hmm.py`       | HMM regime detection (skeleton)                |

## EDA (plots_jan.py)

### 1. Monthly overview
Each pair gets a downsampled full-month price chart showing the overall trend.

**Output:** `plots/jan/{pair}_202601_month.pdf`

### 2. Tick activity & gap detection
- **Top panel:** hourly bar chart of tick counts — reveals trading-session rhythm
- **Bottom panel:** scatter of gaps > 60 seconds — highlights weekends, holidays, illiquid hours

**Output:** `plots/jan/{pair}_202601_ticks.pdf`

### 3. Microstructure
Four-panel plot:
1. Inter-tick duration histogram (< 10 s)  
2. Inter-tick duration CDF (< 5 s)  
3. Tick-size (Δmid) histogram  
4. Tick-size CDF  

**Key observations:**
- EURUSD: median inter-tick ~15 ms, very tight tick-size distribution  
- USDZAR: highest tick count but many small duplicate ticks  
- XAUUSD: slightly wider ticks, no bid/ask microstructure  

**Output:** `plots/jan/{pair}_202601_micro.pdf`

### 4. Pre-averaging

Two modes:
- **Time-based:** resample mid prices in fixed time windows (e.g. 100 ms)  
- **Tick-based:** average every N ticks (e.g. 10 ticks)  

Each mode generates a plot overlaying raw mid and pre-averaged price for a single 
trading day. Three sample days are used: 2026-01-06, 2026-01-15, 2026-01-22.

**Output:** `plots/jan/{pair}_202601_{date}_{mode}.pdf`

## AR(1) Modelling (AR1.py)

### Model

The AR(1) model: `y_t = μ + φ · y_{t-1} + ε_t` is fitted via OLS on each window.

The key parameters tracked are:
- **φ (phi):** autoregressive coefficient — measures mean-reversion (φ < 0) 
  or momentum (φ > 0)  
- **σ (sigma):** residual standard deviation — captures local volatility  

### Estimation modes

| Mode             | Description                                        |
|------------------|----------------------------------------------------|
| Rolling          | Overlapping windows with step = window_size / 2    |
| Non-overlapping  | Disjoint, contiguous windows                       |

### Experiments run

For each pair (EURUSD, USDZAR, XAUUSD):

1. **Single day, multiple window sizes** (w=200, 500, 1000) on 2026-01-15  
   → Shows sensitivity of φ and σ estimates to window length  

2. **Single day with pre-averaging:**  
   - Time-based 100 ms window + w=200  
   - Tick-based 10-tick window + w=200  
   → Pre-averaging removes microstructure noise, stabilises φ  

3. **Multi-day overlay** (4 days: Jan 6, 8, 15, 22) with w=500  
   - Raw prices  
   - With 100 ms pre-averaging  
   → Look for intraday patterns that repeat across days  

### Key findings

- **φ is consistently negative** for raw tick data across all pairs, 
  consistent with bid-ask bounce / mean reversion at tick frequency  
- **Pre-averaging with 100 ms windows** reduces the negative φ, 
  as it smooths out microstructure noise  
- **σ varies through the day** — higher during overlap of London/NY sessions  
- **USDZAR shows the strongest negative φ** — consistent with wider spreads 
  and more pronounced bid-ask bounce  
- **Multi-day overlays** show consistent intraday patterns in φ, especially 
  the session-boundary effects  

**Output:** `plots/models/ar1_{pair}_*.pdf` (21 files total)

## Plot inventory

### EDA plots (plots/jan/) — 27 files

Per pair (× 3 pairs):
- `{pair}_202601_month.pdf` — full-month overview  
- `{pair}_202601_ticks.pdf` — tick count + gaps  
- `{pair}_202601_micro.pdf` — microstructure distributions  
- `{pair}_202601_{date}_time_100ms.pdf` — preavg (time, 100 ms) × 3 dates  
- `{pair}_202601_{date}_ticks_10.pdf` — preavg (tick, 10) × 3 dates  

### AR(1) plots (plots/models/) — 21 files

Per pair (× 3 pairs):
- `ar1_{pair}_2026-01-15_w200_rolling.pdf`  
- `ar1_{pair}_2026-01-15_w500_rolling.pdf`  
- `ar1_{pair}_2026-01-15_w1000_rolling.pdf`  
- `ar1_{pair}_2026-01-15_w200_rolling_pre100ms.pdf`  
- `ar1_{pair}_2026-01-15_w200_rolling_pre10t.pdf`  
- `ar1_{pair}_multiday_w500_rolling.pdf`  
- `ar1_{pair}_multiday_w500_rolling_pre100ms.pdf`  

## Next steps

- [ ] Implement HMM regime detection (2-3 states: low/high vol, trending/mean-reverting)
- [ ] Explore pairs trading — cointegration tests between pairs  
- [ ] Extend to ARCH/GARCH volatility models  
- [ ] Add non-overlapping AR(1) comparison  
- [ ] Test across all available currency pairs  

## Running

```bash
# From project root (with venv activated):
python code/scripts/data_jan.py      # verify data loads
python code/scripts/plots_jan.py     # generate EDA plots (~3 min)
python code/scripts/AR1.py           # generate AR(1) plots (~10 min)
```
