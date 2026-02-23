# Python Quickstart

Read EUR/USD ASK tick data and plot it — nothing more, nothing less.

---

## 1 — Read the data

```python
import pandas as pd

df = pd.read_csv(
    "../data/raw/DAT_NT_EURUSD_T_ASK_202601.csv",
    sep=";",
    header=None,
    names=["datetime", "ask", "volume"],
    parse_dates=["datetime"],
    date_format="%Y%m%d %H%M%S",
)
```

Quick sanity check:

```python
df.head()
```

```
             datetime      ask  volume
0 2026-01-01 17:04:01  1.17532       0
1 2026-01-01 17:04:37  1.17531       0
2 2026-01-01 17:05:00  1.17517       0
3 2026-01-01 17:05:01  1.17518       0
4 2026-01-01 17:05:30  1.17518       0
```

---

## 2 — Plot the full series

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(df["datetime"], df["ask"], linewidth=0.3, color="steelblue")
plt.title("EUR/USD Ask — Jan 2026 (tick-level)")
plt.xlabel("Date")
plt.ylabel("Ask Price")
plt.tight_layout()
plt.show()
```

---

## 3 — Resample to 1-minute bars

Tick data is noisy. Resample if you want something cleaner:

```python
df = df.set_index("datetime")
ohlc = df["ask"].resample("1min").ohlc().dropna()

plt.figure(figsize=(12, 4))
plt.plot(ohlc["close"], linewidth=0.4, color="steelblue")
plt.title("EUR/USD Ask — 1-min Close")
plt.xlabel("Date")
plt.ylabel("Ask Price")
plt.tight_layout()
plt.show()
```

---

## 4 — Compute returns

```python
ohlc["return"] = ohlc["close"].pct_change()

plt.figure(figsize=(12, 3))
plt.plot(ohlc["return"], linewidth=0.3, color="tomato", alpha=0.7)
plt.title("EUR/USD Ask — 1-min Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.tight_layout()
plt.show()
```

---

## 5 — Distribution of returns

```python
ohlc["return"].dropna().hist(bins=200, figsize=(8, 3), color="steelblue", edgecolor="none")
plt.title("Return Distribution")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```

---

## Notes

- **Path assumes you run from `code/notebooks/` or `code/scripts/`** — adjust `../data/raw/...` as needed.
- The raw file has **no header** — that's why we pass `header=None` and `names=[...]`.
- Delimiter is **semicolon** (`;`), not comma.
- The BID file (`DAT_NT_EURUSD_T_BID_202601.csv`) has the exact same format — just swap the filename.
