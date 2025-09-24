"""
ingest_prices.py
----------------
Batch-download OHLCV with yfinance and write a tidy combined parquet.

Output
------
data/processed/prices_{bar}.parquet with columns:
[timestamp, symbol, open, high, low, close, adj_close, volume]
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf

from src.core.utils import ensure_dir, to_parquet_safe

# Optional universe helpers (if present)
try:
    from src.core.universe import get_universe, list_universe_symbols
except Exception:
    get_universe = None  # type: ignore
    list_universe_symbols = None  # type: ignore


def _yf_interval(bar: str) -> str:
    if bar in ("1d", "1h"):
        return bar
    raise ValueError("bar must be '1d' or '1h'")


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase cols, ensure adj_close exists, tz-naive, sorted."""
    if df is None or df.empty:
        cols = ["open","high","low","close","adj_close","volume"]
        out = pd.DataFrame(columns=cols).astype({c: "float64" for c in cols})
        out.index = pd.to_datetime(out.index)
        return out

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([x for x in t if x]).lower() for t in df.columns]
    else:
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    rename = {
        "open":"open","high":"high","low":"low","close":"close",
        "adj_close":"adj_close","adjclose":"adj_close","adjusted_close":"adj_close",
        "volume":"volume"
    }
    df = df.rename(columns=rename)
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    df = df.sort_index()
    return df[["open","high","low","close","adj_close","volume"]].copy()


def _fetch(symbol: str, start: str, end: str, bar: str) -> pd.DataFrame:
    interval = _yf_interval(bar)
    # yfinance: 1h has ~730 day lookback limit → clamp start if needed
    if bar == "1h":
        end_ts = pd.to_datetime(end)
        min_start = end_ts - pd.Timedelta(days=730)
        start = max(pd.to_datetime(start), min_start).strftime("%Y-%m-%d")

    df = yf.download(
        symbol, start=start, end=end, interval=interval,
        auto_adjust=False, group_by="column", progress=False, threads=True
    )
    return _standardize(df)


def _resolve_symbols(explicit_csv: Optional[str], explicit_yaml: Optional[str], explicit_list: Optional[Iterable[str]]) -> List[str]:
    if explicit_list:
        out = [str(s).strip().upper().replace("/", "_") for s in explicit_list if str(s).strip()]
        if out:
            return sorted(set(out))
    if get_universe is not None:
        df = get_universe(symbols=None, csv_path=explicit_csv or "configs/universe.csv",
                          yaml_path=explicit_yaml or "configs/default.yaml", yaml_key="universe",
                          cache_to=None, fallback_symbols=["SPY","TLT"])
        if list_universe_symbols:
            return list_universe_symbols(df)
    return ["SPY", "TLT"]


def ingest_prices(
    symbols: List[str], start: str, end: str, bar: str = "1d",
    out_dir: str | Path = "data/processed", write_combined: bool = True
) -> Path:
    out_dir = ensure_dir(out_dir).parent if Path(out_dir).suffix else ensure_dir(out_dir)
    frames: List[pd.DataFrame] = []

    print(f"[PRICES] symbols={len(symbols)} {start}->{end} bar={bar}")
    for i, sym in enumerate(symbols, 1):
        df = _fetch(sym, start, end, bar)
        if df.empty:
            print(f"  - {i:>3}/{len(symbols)} {sym}: no data")
            continue
        df = df.copy()
        df.index.name = "timestamp"
        df["symbol"] = sym
        tidy = df.reset_index()[["timestamp","symbol","open","high","low","close","adj_close","volume"]]
        frames.append(tidy)
        print(f"  - {i:>3}/{len(symbols)} {sym}: {tidy['timestamp'].min()} → {tidy['timestamp'].max()}  rows={len(tidy)}")

    if not frames:
        raise SystemExit("[PRICES] No data retrieved. Check inputs.")

    combined = pd.concat(frames, ignore_index=True).sort_values(["symbol","timestamp"])
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    if write_combined:
        out_path = Path(out_dir) / f"prices_{bar}.parquet"
        to_parquet_safe(combined, out_path)
        print(f"[PRICES] wrote {out_path} rows={len(combined)} symbols={combined['symbol'].nunique()}")
        return out_path
    return Path(out_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OHLCV with yfinance and write a combined parquet.")
    p.add_argument("--symbols", type=str, default="", help="Comma-separated list (overrides CSV/YAML).")
    p.add_argument("--universe-csv", type=str, default="configs/universe.csv", help="CSV with a 'symbol' column.")
    p.add_argument("--universe-yaml", type=str, default="configs/default.yaml", help="YAML with key 'universe'.")
    p.add_argument("--start", type=str, default="2013-01-01")
    p.add_argument("--end", type=str, default="2025-01-01")
    p.add_argument("--bar", type=str, default="1d", choices=["1d","1h"])
    p.add_argument("--out-dir", type=str, default="data/processed")
    p.add_argument("--no-combined", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    explicit = [s for s in args.symbols.split(",")] if args.symbols else None
    syms = _resolve_symbols(args.universe_csv, args.universe_yaml, explicit)
    ingest_prices(syms, args.start, args.end, args.bar, args.out_dir, write_combined=(not args.no_combined))


if __name__ == "__main__":
    main()
