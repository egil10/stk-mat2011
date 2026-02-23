"""
parquet_dukascopy.py
====================
Downloads tick data from Dukascopy Bank SA via dukascopy-python and saves
as compressed Parquet files in code/data/processed/.

Output naming:
    {symbol}_dukascopy_bid_{yyyymm}.parquet
    {symbol}_dukascopy_ask_{yyyymm}.parquet

Default pairs and months are chosen to match the TrueFX data we already have.

Usage:
    python code/scripts/parquet_dukascopy.py                      # all defaults
    python code/scripts/parquet_dukascopy.py --symbol EURUSD      # one symbol
    python code/scripts/parquet_dukascopy.py --dry-run             # preview
    python code/scripts/parquet_dukascopy.py --months 202601      # one month
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import dukascopy_python as dp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "code" / "data" / "processed"

# ---------------------------------------------------------------------------
# Default configuration -- same pairs & months as TrueFX data
# ---------------------------------------------------------------------------
DEFAULT_PAIRS = [
    "EUR/CHF",
    "EUR/USD",
    "USD/ZAR",
]

DEFAULT_MONTHS = [
    "202511",  # Nov 2025
    "202512",  # Dec 2025
    "202601",  # Jan 2026
]


def month_range(yyyymm: str) -> tuple[datetime, datetime]:
    """Return (start, end) datetimes for a YYYYMM string."""
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])
    start = datetime(year, month, 1)
    # end = first day of next month
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start, end


def download_and_save(
    pair: str,
    yyyymm: str,
    compression: str = "snappy",
) -> list[dict]:
    """Download ticks for one pair/month from Dukascopy, write bid+ask parquet."""

    start, end = month_range(yyyymm)
    symbol = pair.replace("/", "").lower()

    print(f"  Downloading {pair} {yyyymm} ({start.date()} -> {end.date()}) ...", flush=True)

    df = dp.fetch(
        instrument=pair,
        interval=dp.INTERVAL_TICK,
        offer_side=dp.OFFER_SIDE_BID,  # ticks always return both bid & ask
        start=start,
        end=end,
    )

    if len(df) == 0:
        print(f"  WARNING: no data returned for {pair} {yyyymm}")
        return []

    print(f"  Received {len(df):,} ticks", flush=True)

    # df columns: bidPrice, askPrice, bidVolume, askVolume
    # index: timestamp (datetime, UTC)

    results = []
    for price_type, price_col, vol_col in [
        ("bid", "bidPrice", "bidVolume"),
        ("ask", "askPrice", "askVolume"),
    ]:
        out_name = f"{symbol}_dukascopy_{price_type}_{yyyymm}.parquet"
        out_path = OUT_DIR / out_name

        out_df = pd.DataFrame({
            "datetime": df.index,
            "symbol": pair.replace("/", ""),
            "price_type": price_type.upper(),
            "price": df[price_col].values,
            "volume": df[vol_col].values,
        })

        out_df.to_parquet(out_path, engine="pyarrow", compression=compression, index=False)

        pq_size = out_path.stat().st_size

        results.append({
            "file": out_name,
            "rows": len(out_df),
            "pq_mb": pq_size / 1e6,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download Dukascopy tick data and save as Parquet"
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Process only this pair (e.g. EURUSD or EUR/USD)",
    )
    parser.add_argument(
        "--months", nargs="*", default=None,
        help="YYYYMM months to download (default: 202511 202512 202601)",
    )
    parser.add_argument(
        "--compression", type=str, default="snappy",
        choices=["snappy", "gzip", "zstd", "none"],
        help="Parquet compression (default: snappy)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List planned downloads without fetching",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build job list
    pairs = DEFAULT_PAIRS
    if args.symbol:
        # Normalize: EURUSD -> EUR/USD
        s = args.symbol.upper().replace("/", "")
        pairs = [p for p in DEFAULT_PAIRS if p.replace("/", "") == s]
        if not pairs:
            # Try to use as-is with slash
            formatted = f"{s[:3]}/{s[3:]}"
            pairs = [formatted]

    months = args.months if args.months else DEFAULT_MONTHS

    jobs = [(p, m) for p in pairs for m in months]

    print(f"Planned: {len(jobs)} download(s)  ({len(pairs)} pair(s) x {len(months)} month(s))\n")

    if args.dry_run:
        for pair, m in jobs:
            sym = pair.replace("/", "").lower()
            print(f"  {pair} {m} -> {sym}_dukascopy_bid_{m}.parquet + {sym}_dukascopy_ask_{m}.parquet")
        print("\n(dry run -- no downloads)")
        return

    total_pq = 0
    file_count = 0
    t0 = time.time()

    for i, (pair, m) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {pair} {m}")
        try:
            results = download_and_save(pair, m, compression=args.compression)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        for stats in results:
            total_pq += stats["pq_mb"]
            file_count += 1
            print(
                f"  -> {stats['rows']:>10,} rows | "
                f"{stats['pq_mb']:>6.1f} MB [{stats['file']}]"
            )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output files: {file_count}")
    print(f"Total Parquet: {total_pq:.1f} MB")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
