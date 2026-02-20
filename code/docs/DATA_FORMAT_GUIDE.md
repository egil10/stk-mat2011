# Data Format Guide

## File Structure Overview

### EUR/USD (Special Format)

EUR/USD has **three different data formats** available:

1. **ASCII Format** (`DAT_ASCII_EURUSD_T_*.csv`)
   - Format: `datetime,bid,ask,volume`
   - Example: `20260101 170401135,1.173870,1.175320,0`
   - **Best for microstructure analysis**: Contains both bid and ask prices
   - Allows calculation of spreads, mid-prices, and bid-ask bounce effects

2. **NT Format - Separate BID/ASK** (`DAT_NT_EURUSD_T_BID_*.csv` and `DAT_NT_EURUSD_T_ASK_*.csv`)
   - Format: `datetime;bid;volume` or `datetime;ask;volume`
   - Example: `20260101 170401;1.173870;0`
   - Semicolon-delimited
   - Requires merging BID and ASK files to reconstruct full order book

3. **NT Format - LAST Price** (`DAT_NT_EURUSD_T_LAST_*.csv`)
   - Format: `datetime;last_price;volume`
   - Example: `20260101 170401;1.174140;0`
   - Only contains the last executed trade price
   - **Loses bid-ask spread information**

### Other Pairs (Standard Format)

All other pairs (AUD/USD, EUR/CHF, XAU/USD, etc.) only have:

- **NT Format - LAST Price** (`DAT_NT_*_T_LAST_*.csv`)
  - Format: `datetime;last_price;volume`
  - Example: `20260101 170401;0.666170;0`
  - Semicolon-delimited
  - **No bid-ask spread data available**

## Why EUR/USD Has Special Treatment?

EUR/USD is the most liquid forex pair, so HistData provides:
- More detailed data formats (ASCII with bid/ask combined)
- Separate bid/ask files for deeper analysis
- Higher tick frequency

Other pairs typically only have "last price" data, which is the most common format from HistData.

---

## Data Suitability for HMM/ML Work

### ✅ **Ideal Data (EUR/USD ASCII Format)**

**What you have:**
- `DAT_ASCII_EURUSD_T_202601.csv` - Contains bid, ask, volume

**Why it's perfect:**
- **Microstructure features**: Can calculate spreads, mid-prices, bid-ask bounce
- **Regime detection**: Spread dynamics reveal market stress/calm regimes
- **HMM features**: Can use spread, mid-price returns, volume as emission variables
- **Pre-averaging**: Can average bid, ask, or mid-price separately

**Use cases:**
- Tick-level microstructure analysis
- Regime-switching models with spread as a feature
- Bid-ask bounce detection
- Volatility clustering at tick level

### ⚠️ **Limited Data (Other Pairs - LAST Price Only)**

**What you have:**
- `DAT_NT_*_T_LAST_*.csv` files - Only last executed price

**Limitations:**
- **No bid-ask spread**: Cannot analyze spread dynamics or bid-ask bounce
- **Less microstructure info**: Missing order book depth information
- **Still usable for**: Return modeling, volatility clustering, basic regime detection

**What you CAN still do:**
- Calculate returns from last prices
- Fit AR/GARCH models on returns
- Detect volatility regimes (high/low volatility states)
- Pairs trading (if both pairs have LAST prices)
- Basic HMM on returns/volatility features

**What you CANNOT do:**
- Microstructure analysis (no spread data)
- Bid-ask bounce modeling
- Order flow analysis
- Detailed spread-based regime detection

---

## Recommendations for HMM/ML Work

### Option 1: Focus on EUR/USD (Recommended)
- Use `DAT_ASCII_EURUSD_T_202601.csv` for detailed microstructure analysis
- This gives you the richest feature set for HMM/regime-switching models
- Can use spread, mid-price returns, volume as HMM emission variables

### Option 2: Multi-Pair Analysis (Limited Scope)
- Use LAST price files from multiple pairs
- Focus on return-based features only (no microstructure)
- Compare volatility regimes across pairs
- Pairs trading with cointegration (both pairs use LAST prices)

### Option 3: Hybrid Approach
- Deep microstructure analysis on EUR/USD (ASCII format)
- Broader multi-pair analysis using LAST prices for regime comparison
- Use EUR/USD as the "gold standard" and compare other pairs' behavior

---

## File Format Details

### ASCII Format (EUR/USD)
```
Column 1: datetime (YYYYMMDD HHMMSSmmm - milliseconds)
Column 2: bid price
Column 3: ask price  
Column 4: volume
Delimiter: comma
```

### NT Format (All Pairs)
```
Column 1: datetime (YYYYMMDD HHMMSS - seconds only, no milliseconds)
Column 2: price (bid/ask/last depending on file type)
Column 3: volume
Delimiter: semicolon
```

**Note**: NT format has **lower time resolution** (seconds) compared to ASCII (milliseconds).

---

## Summary

- **EUR/USD ASCII**: Best for microstructure HMM/ML work (has bid/ask/spread)
- **EUR/USD BID/ASK**: Good alternative, requires merging files
- **All other pairs (LAST)**: Limited to return-based analysis, no microstructure
- **For regime-switching HMM**: EUR/USD ASCII is ideal; LAST prices work but lose microstructure features
