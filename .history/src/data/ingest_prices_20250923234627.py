"""
ingest_prices.py
----------------
Batch-ingest OHLCV data for a universe of symbols, with local caching.

What it does
------------
- Builds a symbol universe (explicit list, CSV, YAML, or config fallback).
- Uses loaders.get_price_data(...) to fetch data (and update per-symbol cache).
- Writes a combined, tidy parquet under data/processed/prices_{bar}.parquet:
    columns = [symbol, timestamp, open, high, low, close, adj_close, volume]

CLI
---
python -m src.data.ingest_prices \
  --start 2013-01-01 --end 2025-01-01 --bar 1d \
  --out-dir data/processed \
  --universe-csv configs/universe.csv \
  --universe-yaml configs/default.yaml

Or, pass explicit symbols (comma-separated):
python -m src.data.ingest_prices --symbols SPY,AAPL,MSFT --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Project imports
from src.data.loaders import get_price_data
from src.utils import ensure_dir, to_parquet_safe

# Optional universe sources
try:
    from src.core.universe import get_universe, list_universe_symbols
except Exception:
    get_universe = None  # type: ignore
    list_universe_symbols = None  # type: ignore

# Config fallback
try:
    from src.config import SYMBOLS as DEFAULT_SYMBOLS, START as DEFAULT_START, END as DEFAULT_END, BAR as DEFAULT_BAR
except Exception:
    DEFAULT_SYMBOLS = ["SPY", "TLT"]
    DEFAULT_START = "2013-01-01"
    DEFAULT_END = "2025-01-01"
    DEFAULT_BAR = "1d"


def _symbols_from_inputs(
    symbols_csv: Optional[str],
    symbols_yaml: Optional[str],
    explicit_symbols: Optional[Iterable[str]],
) -> List[str]:
    """
    Resolve a clean, uppercase list of symbols from (explicit | csv | yaml | config).
    Priority: explicit > CSV > YAML > DEFAULT_SYMBOLS
    """
    # 1) explicit
    if explicit_symbols:
        syms = [str(s).strip().upper().replace("/", "_") for s in explicit_symbols if str(s).strip()]
        if syms:
            return sorted(set(syms))

    # 2 / 3) CSV / YAML via universe module if present
    if get_universe is not None:
        df = get_universe(
            symbols=None,
            csv_path=symbols_csv or "configs/universe.csv",
            yaml_path=symbols_yaml or "configs/default.yaml",
            yaml_key="universe",
            cache_to=None,
            fallback_symbols=DEFAULT_SYMBOLS,
        )
        if list_universe_symbols is not None:
            return list_universe_symbols(df)

    # 4) fallback to config
    return sorted(set([s.upper().replace("/", "_") for s in DEFAULT_SYMBOLS]))


def ingest_symbols(
    symbols: List[str],
    start: str,
    end: str,
    bar: str = "1d",
    out_dir: str | Path = "data/processed",
    write_combined: bool = True,
) -> Path:
    """
    Ingest prices for `symbols` into the local cache and optionally write a combined parquet.

    Returns
    -------
    Path
        Path to the combined parquet file if write_combined=True,
        else the output directory path.
    """
    out_dir = ensure_dir(out_dir).parent if Path(out_dir).suffix else ensure_dir(out_dir)
    frames: List[pd.DataFrame] = []

    print(f"[INGEST] symbols={len(symbols)} range={start}->{end} bar={bar}")
    for i, sym in enumerate(symbols, 1):
        df = get_price_data(sym, start, end, bar)
        if df.empty:
            print(f"  - {i:>3}/{len(symbols)} {sym}: no data")
            continue
        df = df.copy()
        df.index.name = "timestamp"
        df["symbol"] = sym
        cols = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]
        # reorder and keep only expected columns
        kept = ["symbol"] + [c for c in cols if c in df.columns and c != "symbol"]
        tidy = df.reset_index()[["timestamp"] + kept]
        frames.append(tidy)
        print(f"  - {i:>3}/{len(symbols)} {sym}: rows={len(tidy)}  {tidy['timestamp'].min()} -> {tidy['timestamp'].max()}")

    if not frames:
        raise SystemExit("[INGEST] No data retrieved. Check symbols/date range/bar.")

    combined = pd.concat(frames, axis=0, ignore_index=True).sort_values(["symbol", "timestamp"])
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    if write_combined:
        out_path = Path(out_dir) / f"prices_{bar}.parquet"
        to_parquet_safe(combined, out_path)
        print(f"[INGEST] wrote combined: {out_path} (rows={len(combined)}, symbols={combined['symbol'].nunique()})")
        return out_path

    print(f"[INGEST] completed (combined not written).")
    return Path(out_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-ingest OHLCV data for a symbol universe.")
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols to ingest (overrides CSV/YAML).")
    p.add_argument("--universe-csv", type=str, default="configs/universe.csv", help="CSV file with a 'symbol' column.")
    p.add_argument("--universe-yaml", type=str, default="configs/default.yaml", help="YAML with key 'universe'.")
    p.add_argument("--start", type=str, default=DEFAULT_START, help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", type=str, default=DEFAULT_END, help="End date (YYYY-MM-DD).")
    p.add_argument("--bar", type=str, default=DEFAULT_BAR, choices=["1d", "1h"], help="Bar size.")
    p.add_argument("--out-dir", type=str, default="data/processed", help="Directory for combined parquet output.")
    p.add_argument("--no-combined", action="store_true", help="Skip writing combined parquet (refresh caches only).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    explicit = [s for s in args.symbols.split(",")] if args.symbols else None
    symbols = _symbols_from_inputs(args.universe_csv, args.universe_yaml, explicit)
    if not symbols:
        raise SystemExit("[INGEST] No symbols resolved.")
    ingest_symbols(
        symbols=symbols,
        start=args.start,
        end=args.end,
        bar=args.bar,
        out_dir=args.out_dir,
        write_combined=(not args.no_combined),
    )


if __name__ == "__main__":
    main()
