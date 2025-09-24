"""
ingest_fundamentals.py
----------------------
Batch-ingest fundamentals from one or more CSVs, standardize, (optionally) filter
to a universe, (optionally) align to trading sessions with forward-fill, and write
a tidy combined parquet to data/processed/fundamentals_{freq}.parquet.

Inputs (CSV)
------------
Must include at least:
  - symbol|ticker
  - date|report_date|ref_date|fiscal_date|filing_date|period_end

The file may be "wide" (metrics in columns) or already tidy. Extra columns are kept.

Examples
--------
# Quarterly fundamentals, filter to universe, align to daily sessions:
python -m src.data.ingest_fundamentals --csv data/raw/funda_q1.csv data/raw/funda_q2.csv \
  --freq Q --universe-csv configs/universe.csv --align 1d

# Annual fundamentals, explicit symbols (overrides CSV/YAML), no alignment:
python -m src.data.ingest_fundamentals --csv data/raw/funda_a.csv \
  --freq A --symbols SPY,AAPL,MSFT
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# Project utilities
from src.core.utils import ensure_dir, to_parquet_safe

# Optional: universe helpers
try:
    from src.core.universe import get_universe, list_universe_symbols
except Exception:  # pragma: no cover
    get_universe = None  # type: ignore
    list_universe_symbols = None  # type: ignore

# Optional: trading calendar for alignment
try:
    from src.core.calendar import get_trading_days
except Exception:  # pragma: no cover
    get_trading_days = None  # type: ignore


# ----------------------------
# Column normalization helpers
# ----------------------------

_DATE_CANDIDATES = ("date", "report_date", "ref_date", "fiscal_date", "filing_date", "period_end")
_SYMBOL_CANDIDATES = ("symbol", "ticker")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    # Direct match first
    for c in candidates:
        if c in df.columns:
            return c
    # Fuzzy: treat hyphens as underscores; try singular forms
    normalized = {c.replace("-", "_").rstrip("s"): c for c in df.columns}
    for c in candidates:
        key = c.replace("-", "_").rstrip("s")
        if key in normalized:
            return normalized[key]
    return None


def _normalize_symbol_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace("/", "_", regex=False).str.strip()


def _coerce_numeric(df: pd.DataFrame, exclude: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


# ----------------------------
# CSV loading & merging
# ----------------------------

def _load_one_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)
    df = _normalize_columns(df)

    sym_col = _find_column(df, _SYMBOL_CANDIDATES)
    date_col = _find_column(df, _DATE_CANDIDATES)
    if sym_col is None or date_col is None:
        raise ValueError(
            f"{p} must contain symbol and date columns. "
            f"Looked for symbol in {list(_SYMBOL_CANDIDATES)}, date in {list(_DATE_CANDIDATES)}. "
            f"Found columns: {list(df.columns)}"
        )

    df[sym_col] = _normalize_symbol_series(df[sym_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[sym_col, date_col])

    # Canonical names
    if sym_col != "symbol":
        df = df.rename(columns={sym_col: "symbol"})
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    # Coerce numerics (keep symbol/date as-is)
    df = _coerce_numeric(df, exclude=("symbol", "date"))

    # Deduplicate on (symbol, date) keeping last (assume later rows newer)
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")

    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _merge_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["symbol", "date"])

    base = frames[0]
    for part in frames[1:]:
        base = base.merge(part, on=["symbol", "date"], how="outer", suffixes=("", "_dup"))
        # Resolve *_dup columns by preferring original where available
        dupes = [c for c in base.columns if c.endswith("_dup")]
        for d in dupes:
            orig = d[:-4]
            if orig in base.columns:
                base[orig] = base[orig].where(base[orig].notna(), base[d])
            base = base.drop(columns=[d])

    return base.sort_values(["symbol", "date"]).reset_index(drop=True)


def _load_many_csv(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames = [_load_one_csv(p) for p in paths]
    return _merge_frames(frames)


# ----------------------------
# Universe filtering & alignment
# ----------------------------

def _resolve_universe(
    explicit_symbols: Optional[Iterable[str]],
    csv_path: Optional[str],
    yaml_path: Optional[str],
) -> List[str]:
    # 1) Explicit
    if explicit_symbols:
        syms = [str(s).strip().upper().replace("/", "_") for s in explicit_symbols if str(s).strip()]
        if syms:
            return sorted(set(syms))
    # 2) Via universe helpers if available
    if get_universe is not None:
        df = get_universe(
            symbols=None,
            csv_path=csv_path or "configs/universe.csv",
            yaml_path=yaml_path or "configs/default.yaml",
            yaml_key="universe",
            cache_to=None,
            fallback_symbols=["SPY", "TLT"],
        )
        if list_universe_symbols is not None:
            return list_universe_symbols(df)
    # 3) Fallback
    return ["SPY", "TLT"]


def _align_to_sessions(
    df: pd.DataFrame,
    *,
    bar: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Expand to trading sessions and forward-fill per symbol (requires core.calendar)."""
    if get_trading_days is None or df.empty:
        return df

    if start is None:
        start = str(df["date"].min())
    if end is None:
        end = str(df["date"].max())

    sessions = pd.DataFrame({"date": get_trading_days(start, end, bar=bar)})
    if sessions.empty:
        return df

    out = []
    metric_cols_cache: Optional[List[str]] = None

    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date")
        merged = sessions.merge(g, on="date", how="left")
        merged["symbol"] = sym
        if metric_cols_cache is None:
            metric_cols_cache = [c for c in merged.columns if c not in ("symbol", "date")]
        merged[metric_cols_cache] = merged[metric_cols_cache].ffill()
        out.append(merged)

    aligned = pd.concat(out, axis=0, ignore_index=True).sort_values(["symbol", "date"])
    return aligned


# ----------------------------
# Public API
# ----------------------------

def ingest_fundamentals(
    csv_files: Sequence[str | Path],
    *,
    freq: str = "Q",
    out_dir: str | Path = "data/processed",
    symbols: Optional[Iterable[str]] = None,
    universe_csv: Optional[str] = None,
    universe_yaml: Optional[str] = None,
    align: Optional[str] = None,   # '1d' or '1h' -> expand & ffill on trading sessions
    start: Optional[str] = None,   # alignment start bound (optional)
    end: Optional[str] = None,     # alignment end bound (optional)
) -> Path:
    """
    Ingest fundamentals and write a tidy combined parquet.

    Returns
    -------
    Path to written file.
    """
    if not csv_files:
        raise ValueError("No CSV files provided.")

    df = _load_many_csv(csv_files)
    if df.empty:
        raise SystemExit("[INGEST-FUND] No rows loaded from inputs.")

    # Universe filter
    resolved = _resolve_universe(symbols, universe_csv, universe_yaml) if (symbols or universe_csv or universe_yaml) else None
    if resolved:
        df = df[df["symbol"].isin(set(resolved))].copy()

    # Ensure canonical dtypes
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Optional alignment
    if align in ("1d", "1h"):
        df = _align_to_sessions(df, bar=align, start=start, end=end)

    # Write
    out_dir = ensure_dir(out_dir).parent if Path(out_dir).suffix else ensure_dir(out_dir)
    out_path = Path(out_dir) / f"fundamentals_{freq.upper()}.parquet"
    to_parquet_safe(df, out_path)
    print(
        f"[INGEST-FUND] wrote {out_path} rows={len(df)} symbols={df['symbol'].nunique()} "
        f"cols={len(df.columns)} range={df['date'].min().date()}->{df['date'].max().date()}"
    )
    return out_path


# ----------------------------
# CLI
# ----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest fundamentals CSVs → tidy combined parquet.")
    p.add_argument("--csv", nargs="+", required=True, help="One or more fundamentals CSV files.")
    p.add_argument("--freq", type=str, default="Q", choices=["Q", "A"], help="Output file frequency tag.")
    p.add_argument("--out-dir", type=str, default="data/processed", help="Destination directory.")
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols (overrides CSV/YAML).")
    p.add_argument("--universe-csv", type=str, default="configs/universe.csv", help="CSV with a 'symbol' column.")
    p.add_argument("--universe-yaml", type=str, default="configs/default.yaml", help="YAML with key 'universe'.")
    p.add_argument("--align", type=str, choices=["1d", "1h"], default=None, help="Align to trading sessions & ffill.")
    p.add_argument("--start", type=str, default=None, help="Alignment start (YYYY-MM-DD).")
    p.add_argument("--end", type=str, default=None, help="Alignment end (YYYY-MM-DD).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    explicit_syms = [s for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    ingest_fundamentals(
        csv_files=args.csv,
        freq=args.freq,
        out_dir=args.out_dir,
        symbols=explicit_syms,
        universe_csv=args.universe_csv,
        universe_yaml=args.universe_yaml,
        align=args.align,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
