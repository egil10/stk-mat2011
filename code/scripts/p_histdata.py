"""
raw_to_parquet.py
=================
Converts tick CSV files from code/data/raw/ into compressed Parquet files
in code/data/processed/.

Supported formats:
  1. NinjaTrader:  DAT_NT_{SYMBOL}_T_{TYPE}_{YYYYMM}.csv
     - Semicolon-delimited, no header: datetime;price;volume
     - Output: {symbol}_{type}_{yyyymm}.parquet

  2. ASCII:        DAT_ASCII_{SYMBOL}_T_{YYYYMM}.csv
     - Comma-delimited, no header: datetime,bid,ask,volume
     - Output: {symbol}_ask_{yyyymm}.parquet  +  {symbol}_bid_{yyyymm}.parquet

Usage:
    python code/scripts/parquet.py              # process all
    python code/scripts/parquet.py --symbol EURUSD   # process one symbol
    python code/scripts/parquet.py --dry-run     # preview without writing
"""

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "code" / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "code" / "data" / "processed"

# Regex to parse NinjaTrader filenames
#   DAT_NT_EURUSD_T_LAST_202601.csv
#   groups: symbol, type (ASK/BID/LAST), period (YYYYMM)
NT_PATTERN = re.compile(
    r"^DAT_NT_([A-Z]+)_T_(ASK|BID|LAST)_(\d{6})\.csv$",
    re.IGNORECASE,
)

# Regex to parse ASCII filenames
#   DAT_ASCII_BCOUSD_T_202601.csv
#   groups: symbol, period (YYYYMM)
ASCII_PATTERN = re.compile(
    r"^DAT_ASCII_([A-Z]+)_T_(\d{6})\.csv$",
    re.IGNORECASE,
)


def discover_files(symbol_filter: str | None = None) -> list[dict]:
    """Find all tick CSV files and extract metadata from filenames."""
    files = []

    # NinjaTrader files
    for f in sorted(RAW_DIR.glob("DAT_NT_*.csv")):
        m = NT_PATTERN.match(f.name)
        if not m:
            continue
        symbol, price_type, period = m.group(1), m.group(2), m.group(3)
        if symbol_filter and symbol.upper() != symbol_filter.upper():
            continue
        files.append({
            "path": f,
            "symbol": symbol.upper(),
            "type": price_type.upper(),
            "period": period,
            "format": "NT",
        })

    # ASCII files
    for f in sorted(RAW_DIR.glob("DAT_ASCII_*.csv")):
        m = ASCII_PATTERN.match(f.name)
        if not m:
            continue
        symbol, period = m.group(1), m.group(2)
        if symbol_filter and symbol.upper() != symbol_filter.upper():
            continue
        files.append({
            "path": f,
            "symbol": symbol.upper(),
            "type": "ASCII",
            "period": period,
            "format": "ASCII",
        })

    return files


def convert_nt_file(meta: dict, compression: str = "snappy") -> list[dict]:
    """Read one NinjaTrader CSV and write a Parquet file. Returns list with one stats dict."""
    src = meta["path"]
    symbol = meta["symbol"].lower()
    price_type = meta["type"].lower()
    period = meta["period"]

    out_name = f"{symbol}_{price_type}_{period}.parquet"
    out_path = OUT_DIR / out_name

    df = pd.read_csv(
        src,
        sep=";",
        header=None,
        names=["datetime", "price", "volume"],
        dtype={"price": "float64", "volume": "int32"},
    )

    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S")
    df["symbol"] = meta["symbol"]
    df["price_type"] = meta["type"]
    df = df[["datetime", "symbol", "price_type", "price", "volume"]]

    df.to_parquet(out_path, engine="pyarrow", compression=compression, index=False)

    csv_size = src.stat().st_size
    pq_size = out_path.stat().st_size
    ratio = pq_size / csv_size * 100

    return [{
        "file": out_name,
        "rows": len(df),
        "csv_mb": csv_size / 1e6,
        "pq_mb": pq_size / 1e6,
        "ratio": ratio,
    }]


def convert_ascii_file(meta: dict, compression: str = "snappy") -> list[dict]:
    """Read one ASCII CSV (bid+ask) and write two Parquet files. Returns list of stats dicts."""
    src = meta["path"]
    symbol = meta["symbol"].lower()
    period = meta["period"]

    # ASCII format: datetime,bid,ask,volume (comma-delimited, no header)
    df = pd.read_csv(
        src,
        sep=",",
        header=None,
        names=["datetime", "bid", "ask", "volume"],
        dtype={"bid": "float64", "ask": "float64", "volume": "int32"},
    )

    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S%f")
    csv_size = src.stat().st_size

    results = []
    for price_type, col in [("ask", "ask"), ("bid", "bid")]:
        out_name = f"{symbol}_{price_type}_{period}.parquet"
        out_path = OUT_DIR / out_name

        out_df = pd.DataFrame({
            "datetime": df["datetime"],
            "symbol": meta["symbol"],
            "price_type": price_type.upper(),
            "price": df[col],
            "volume": df["volume"],
        })

        out_df.to_parquet(out_path, engine="pyarrow", compression=compression, index=False)

        pq_size = out_path.stat().st_size
        ratio = pq_size / csv_size * 100

        results.append({
            "file": out_name,
            "rows": len(out_df),
            "csv_mb": csv_size / 1e6,
            "pq_mb": pq_size / 1e6,
            "ratio": ratio,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert tick CSVs to Parquet (NinjaTrader + ASCII formats)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Process only this symbol (e.g. EURUSD)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "zstd", "none"],
        help="Parquet compression (default: snappy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without converting",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = discover_files(args.symbol)
    if not files:
        print("No tick CSV files found in", RAW_DIR)
        sys.exit(1)

    print(f"Found {len(files)} file(s) in {RAW_DIR}\n")

    if args.dry_run:
        for meta in files:
            sym = meta["symbol"]
            t = meta["type"]
            p = meta["period"]
            size_mb = meta["path"].stat().st_size / 1e6
            if meta["format"] == "ASCII":
                print(f"  {meta['path'].name:<45} -> {sym.lower()}_ask_{p}.parquet + {sym.lower()}_bid_{p}.parquet  ({size_mb:.1f} MB)")
            else:
                print(f"  {meta['path'].name:<45} -> {sym.lower()}_{t.lower()}_{p}.parquet  ({size_mb:.1f} MB)")
        print("\n(dry run -- no files written)")
        return

    total_csv = 0
    total_pq = 0
    t0 = time.time()

    file_count = 0
    for i, meta in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {meta['path'].name} ... ", end="", flush=True)

        if meta["format"] == "ASCII":
            results = convert_ascii_file(meta, compression=args.compression)
        else:
            results = convert_nt_file(meta, compression=args.compression)

        for stats in results:
            total_csv += stats["csv_mb"]
            total_pq += stats["pq_mb"]
            file_count += 1
            print(
                f"{stats['rows']:>10,} rows | "
                f"{stats['csv_mb']:>6.1f} MB -> {stats['pq_mb']:>6.1f} MB "
                f"({stats['ratio']:.0f}%) [{stats['file']}]"
            )
            if len(results) > 1 and stats != results[-1]:
                print(f"{'':>50} ", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output files: {file_count}")
    print(f"Total: {total_csv:.1f} MB CSV -> {total_pq:.1f} MB Parquet ({total_pq/total_csv*100:.0f}%)")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
