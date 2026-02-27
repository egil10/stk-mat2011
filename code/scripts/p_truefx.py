"""
parquet_truefx.py
=================
Converts tick CSV files from TrueFX (code/data/raw/) into compressed
Parquet files in code/data/processed/.

TrueFX format:
    EUR/USD,20251102 22:00:00.821,1.15291,1.15354
  = symbol, datetime, bid, ask  (comma-delimited, no header, no volume)

Filename pattern:
    EURCHF-2025-11.csv  ->  eurchf_truefx_bid_202511.parquet
                            eurchf_truefx_ask_202511.parquet

Usage:
    python code/scripts/parquet_truefx.py              # process all
    python code/scripts/parquet_truefx.py --symbol EURUSD   # one symbol
    python code/scripts/parquet_truefx.py --dry-run     # preview
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

# Regex: EURCHF-2025-11.csv -> groups: symbol, year, month
TRUEFX_PATTERN = re.compile(
    r"^([A-Z]{6})-(\d{4})-(\d{2})\.csv$",
    re.IGNORECASE,
)


def discover_files(symbol_filter: str | None = None) -> list[dict]:
    """Find all TrueFX-format CSV files and extract metadata."""
    files = []
    for f in sorted(RAW_DIR.glob("*-20??-??.csv")):
        m = TRUEFX_PATTERN.match(f.name)
        if not m:
            continue
        symbol = m.group(1).upper()
        year = m.group(2)
        month = m.group(3)
        period = f"{year}{month}"
        if symbol_filter and symbol != symbol_filter.upper():
            continue
        files.append({
            "path": f,
            "symbol": symbol,
            "period": period,
        })
    return files


def convert_truefx_file(meta: dict, compression: str = "snappy") -> list[dict]:
    """Read one TrueFX CSV and write bid + ask Parquet files."""
    src = meta["path"]
    symbol = meta["symbol"].lower()
    period = meta["period"]

    df = pd.read_csv(
        src,
        header=None,
        names=["pair", "datetime", "bid", "ask"],
        dtype={"bid": "float64", "ask": "float64"},
    )

    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H:%M:%S.%f")
    csv_size = src.stat().st_size

    results = []
    for price_type, col in [("bid", "bid"), ("ask", "ask")]:
        out_name = f"{symbol}_truefx_{price_type}_{period}.parquet"
        out_path = OUT_DIR / out_name

        out_df = pd.DataFrame({
            "datetime": df["datetime"],
            "symbol": meta["symbol"],
            "price_type": price_type.upper(),
            "price": df[col],
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
        description="Convert TrueFX tick CSVs to Parquet"
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Process only this symbol (e.g. EURUSD)",
    )
    parser.add_argument(
        "--compression", type=str, default="snappy",
        choices=["snappy", "gzip", "zstd", "none"],
        help="Parquet compression (default: snappy)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files without converting",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = discover_files(args.symbol)
    if not files:
        print("No TrueFX CSV files found in", RAW_DIR)
        sys.exit(1)

    print(f"Found {len(files)} TrueFX file(s) in {RAW_DIR}\n")

    if args.dry_run:
        for meta in files:
            s = meta["symbol"].lower()
            p = meta["period"]
            size_mb = meta["path"].stat().st_size / 1e6
            print(f"  {meta['path'].name:<30} -> {s}_truefx_bid_{p}.parquet + {s}_truefx_ask_{p}.parquet  ({size_mb:.1f} MB)")
        print("\n(dry run -- no files written)")
        return

    total_csv = 0
    total_pq = 0
    file_count = 0
    t0 = time.time()

    for i, meta in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {meta['path'].name} ... ", end="", flush=True)
        results = convert_truefx_file(meta, compression=args.compression)
        for stats in results:
            total_csv += stats["csv_mb"]
            total_pq += stats["pq_mb"]
            file_count += 1
            print(
                f"{stats['rows']:>10,} rows | "
                f"{stats['csv_mb']:>6.1f} MB -> {stats['pq_mb']:>6.1f} MB "
                f"({stats['ratio']:.0f}%) [{stats['file']}]"
            )
            if stats != results[-1]:
                print(f"{'':>50} ", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output files: {file_count}")
    if total_csv > 0:
        print(f"Total: {total_csv:.1f} MB CSV -> {total_pq:.1f} MB Parquet ({total_pq/total_csv*100:.0f}%)")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
