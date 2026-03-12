from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROC_DIR = PROJECT_ROOT / "code" / "data" / "processed"


Pair = Literal["EURUSD", "USDZAR", "XAUUSD"]


@dataclass
class TickData:
    pair: Pair
    month: str
    df: pd.DataFrame


def _load_parquet(name: str) -> pd.DataFrame:
    path = PROC_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_mid(pair: Pair, month: str = "202601") -> TickData:
    """Load January 2026 tick data for one pair and return mid/last prices.

    - EURUSD, USDZAR: Dukascopy bid/ask -> mid = (bid+ask)/2
    - XAUUSD: HistData LAST -> mid = last
    """
    pair_u = pair.upper()
    if month != "202601":
        raise ValueError("This helper is currently scoped to January 2026 (month='202601').")

    if pair_u in {"EURUSD", "USDZAR"}:
        symbol = pair_u.lower()
        bid = _load_parquet(f"{symbol}_dukascopy_bid_{month}.parquet")
        ask = _load_parquet(f"{symbol}_dukascopy_ask_{month}.parquet")

        bid = bid[["datetime", "price"]].rename(columns={"price": "bid"})
        ask = ask[["datetime", "price"]].rename(columns={"price": "ask"})

        df = pd.merge_asof(
            bid.sort_values("datetime"),
            ask.sort_values("datetime"),
            on="datetime",
            direction="nearest",
        )
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
        df["pair"] = pair_u
        out = df[["datetime", "pair", "bid", "ask", "mid"]].sort_values("datetime")
        return TickData(pair=pair_u, month=month, df=out)

    if pair_u == "XAUUSD":
        df = _load_parquet(f"xauusd_last_{month}.parquet")
        df = df.rename(columns={"price": "last"})
        df["mid"] = df["last"]
        df["pair"] = pair_u
        out = df[["datetime", "pair", "last", "mid"]].sort_values("datetime")
        return TickData(pair=pair_u, month=month, df=out)

    raise ValueError(f"Unsupported pair: {pair}")


def load_three_pairs(month: str = "202601") -> dict[Pair, TickData]:
    """Convenience loader for EURUSD, USDZAR, XAUUSD for one month."""
    data: dict[Pair, TickData] = {}
    for p in ("EURUSD", "USDZAR", "XAUUSD"):
        data[p] = load_mid(p, month=month)
    return data


if __name__ == "__main__":
    all_data = load_three_pairs()
    for pair, td in all_data.items():
        df = td.df
        print(
            f"{pair} {td.month}: {len(df):,} ticks | "
            f"{df['datetime'].min()} -> {df['datetime'].max()}"
        )

