# Research & Modeling Findings

Documentation of the exploratory analysis, regime detection, and statistical modeling performed on the tick data.

---

## 📊 Exploratory Data Analysis (EDA)

EDA focuses on understanding the "rhythm" of the high-frequency market.

1.  **Tick Activity**: Hourly distributions reveal trading session transitions (London/New York overlap).
2.  **Gap Detection**: Identification of illiquid periods and service gaps.
3.  **Microstructure**: analysis of inter-tick durations and tick-size (spread) distributions.

---

## 📉 Statistical Modeling

### 1. AR(1) Model: `y_t = μ + φ · y_{t-1} + ε_t`
- **φ (phi)**: Measures mean-reversion (φ < 0) or momentum (φ > 0). Negative phi is dominant in raw tick data due to bid-ask bounce.
- **σ (sigma)**: Captures local volatility clustering.

### 2. Markov-Switching AR (MS-AR)
Used for regime detection on pre-averaged log-returns. Captures distinct states, such as:
- **Low-Volatility**: Calm periods with stable coefficients.
- **High-Volatility**: Market stress or high-activity sessions with switching variance.

### 3. MS-GARCH
Implements 2-regime sGARCH with Normal innovations via R's `MSGARCH` package. It tracks conditional volatility switching through the trading day.

---

## 🔬 Modeling Workflow

| Task | Script | Output |
|------|--------|--------|
| Load Data | `data.py` | Parquet cache |
| EDA Plots | `viz.py` | `plots/eda/` |
| AR(1) | `ar1.py` | `plots/models/` |
| MS-AR | `msar.py` | `plots/models/` |
| MS-GARCH | `msg.py` | `plots/models/` |

---

## 📓 Key Findings
- **Bid-Ask Bounce**: Consistently negative φ at high frequencies across all pairs.
- **Smoothing Effects**: Pre-averaging (e.g., 100ms) stabilizes phi estimates by removing microstructure noise.
- **Session Patterns**: Intraday seasonalities are visible across multiple days, matching global market hours.
