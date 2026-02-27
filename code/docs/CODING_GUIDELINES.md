# Coding Guidelines

## Performance

- **Always use parallel processing** for independent tasks (plot generation,  
  model fitting across different pairs/days/parameters).  Use  
  `concurrent.futures.ThreadPoolExecutor` for I/O-bound work (writing PDFs)  
  and `concurrent.futures.ProcessPoolExecutor` for CPU-bound work (heavy  
  numerical fitting).  Typical pattern:

  ```python
  from concurrent.futures import ThreadPoolExecutor, as_completed

  tasks = [(fn, args, kwargs), ...]
  with ThreadPoolExecutor(max_workers=6) as pool:
      futures = [pool.submit(fn, *a, **kw) for fn, a, kw in tasks]
      for f in as_completed(futures):
          f.result()
  ```

- Prefer **vectorised NumPy / Pandas** over Python loops.

## Project structure

```
code/
├── scripts/     # runnable scripts (data_jan.py, plots_jan.py, AR1.py, hmm.py,
│                #   ms_ar.py, ms_garch.py)
├── data/
│   ├── raw/         # original downloaded archives (gitignored)
│   ├── processed/   # parquet tick data (gitignored)
│   └── samples/     # small CSV samples (tracked)
├── plots/
│   ├── jan/         # EDA & pre-averaging plots
│   ├── models/      # AR(1), HMM, MS-AR, MS-GARCH PDFs
│   ├── plotly/      # interactive HTML plots (Plotly)
│   └── final/       # polished figures for reports
├── docs/            # markdown documentation
└── notebooks/       # exploratory Jupyter notebooks
```

## Naming

- **Script names:** lowercase, short, descriptive (`data_jan.py`, `AR1.py`).
- **Plot files:** `{type}_{pair}_{date}_{variant}.pdf`  
  e.g. `ar1_eurusd_2026-01-15_w500_rolling.pdf`
- **Parquet files:** `{pair}_{source}_{side}_{YYYYMM}.parquet`

## Plots

- Use **Plotly** (plotly.graph_objects / plotly.express) for interactive plots.
- Always **export as PDF** to the appropriate `plots/` subfolder.
- Use `plotly_white` template as default.
- Down-sample before plotting when data exceeds ~5 000 points.
