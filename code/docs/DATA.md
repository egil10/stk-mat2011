# Data Management

Detailed information on the currency pairs, file formats, and processing pipeline.

---

## 🕒 Dataset (January 2026)

| Pair    | Source     | Price Type               | Rationale |
|---------|------------|--------------------------|-----------|
| EURUSD  | Dukascopy  | mid = (bid + ask) / 2    | Baseline liquidity |
| USDZAR  | Dukascopy  | mid = (bid + ask) / 2    | Exotic / High Spread |
| XAUUSD  | HistData   | last (no bid/ask)        | Different asset class |

---

## 📝 Raw Formats (HistData)

HistData provides several formats depending on the pair.

### 1. ASCII Format (EUR/USD)
- **Structure:** `datetime,bid,ask,volume`
- **Resolution:** Milliseconds (`YYYYMMDD HHMMSSmmm`)
- **Use Case:** Microstructure analysis, spreads, and bid-ask bounce.

### 2. NinjaTrader (NT) Format (All Pairs)
- **Structure:** `datetime;price;volume`
- **Resolution:** Seconds only.
- **Side:** Separate files for `BID` and `ASK` (require merging) or single `LAST` price file (Standard for non-EURUSD pairs).

---

## ⚡ Processing Pipeline

### 1. Ingestion (`scripts/p_*.py`)
- Raw CSVs from `code/data/raw/` are converted to **Snappy-compressed Parquet** files in `code/data/processed/`.
- This reduces file size (often by >80%) and dramatically increases read speed for iterative analysis.

### 2. Pre-averaging (`viz.py`)
To handle microstructure noise, prices can be pre-averaged:
- **Time-based**: Resample mid prices in fixed windows (e.g., 100ms).
- **Tick-based**: Average every N ticks.

---

## ✅ Suitability for HMM / ML

- **EUR/USD (ASCII)**: Ideal for microstructure features (spread, mid-returns).
- **Exotics (LAST)**: Limited to return and volatility modeling. Missing spread information prevents bid-ask bounce analysis.
