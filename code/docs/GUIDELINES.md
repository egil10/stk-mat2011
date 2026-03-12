# Project Guidelines

This document outlines the coding standards, data handling procedures, and AI safety protocols for the project.

---

## 🚀 Quickstart: Python for Tick Data

Read EUR/USD ASK tick data and plot it — nothing more, nothing less.

### 1 — Read the data
```python
import pandas as pd

# Path relative to script/notebook
df = pd.read_csv(
    "../data/raw/DAT_NT_EURUSD_T_ASK_202601.csv",
    sep=";",
    header=None,
    names=["datetime", "ask", "volume"],
    parse_dates=["datetime"],
    date_format="%Y%m%d %H%M%S",
)
```

### 2 — Resample to 1-minute bars
```python
df = df.set_index("datetime")
ohlc = df["ask"].resample("1min").ohlc().dropna()
```

---

## 💻 Coding Standards

### Performance
- **Parallel processing** is mandatory for independent tasks (plot generation, model fitting).
- Use `concurrent.futures.ThreadPoolExecutor` for I/O-bound work and `ProcessPoolExecutor` for CPU-bound work.
- **Vectorised NumPy / Pandas** is preferred over Python loops.

### Plotting
- Use **Plotly** for interactive plots; **Matplotlib** for static PDF/final outputs.
- Export as PDF to the appropriate `plots/` subfolder.
- Down-sample data before plotting if it exceeds ~5,000 points to keep file sizes manageable.

### Naming Conventions
- **Scripts:** lowercase, short, descriptive (e.g., `data.py`, `viz.py`).
- **Plots:** `{type}_{pair}_{date}_{variant}.pdf` (e.g., `ar1_eurusd_2026-01-15_pre100ms.pdf`).
- **Data:** `{pair}_{source}_{side}_{YYYYMM}.parquet` (e.g., `eurusd_duka_bid_202601.parquet`).

---

## 📂 Project Structure

```
stk-mat2011/
├── code/
│   ├── scripts/     # Analysis scripts (data.py, viz.py, ar1.py, msar.py, msg.py)
│   ├── data/
│   │   ├── raw/         # Raw CSV archives (gitignored)
│   │   ├── processed/   # Optimized Parquet data (gitignored)
│   │   └── samples/     # Small CSV samples (tracked)
│   ├── plots/
│   │   ├── eda/         # Exploration and pre-averaging plots
│   │   ├── models/      # AR(1), HMM, MS-AR/GARCH outputs
│   │   └── final/       # Polished figures for reports
│   ├── docs/            # Unified documentation
│   └── notebooks/       # Exploratory Jupyter notebooks
```

---

## 🤖 AI Safety & Privacy (Zero Absolute Path Policy)

To protect local environment details and ensure cross-system compatibility:

1.  **NO Absolute Paths**: Never hardcode paths like `C:\Users\...`.
2.  **Relative Resolution**: Always resolve paths relative to the script or project root.
    ```python
    from pathlib import Path
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parents[1]
    ```
3.  **Public Awareness**: Assume this repository is public. Avoid leaking usernames, local IPs, or system-specific configurations in scripts, logs, or documentation.
4.  **Non-Interactive Execution**: Ensure all scripts run headlessly without requiring user input or fixed local file structures.
