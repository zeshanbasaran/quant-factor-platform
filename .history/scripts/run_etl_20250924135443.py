"""
run_etl.py
-----------
Batch ETL runner for prices & fundamentals.

Usage (from repo root)
----------------------
python -m scripts.run_etl \
  --symbols SPY,QQQ \
  --start 2013-01-01 --end 2025-01-01 \
  --bar 1d \
  --overwrite-cache

What it does
------------
1) Loads default config (configs/default.yaml) unless overridden by CLI.
2) Fetches prices via src.data.ingest_prices.ingest (or .run fallback).
3) Fetches fundamentals via src.data.ingest_fundamentals.ingest (or .run fallback).
4) Writes canonical parquet tables to data/processed/.
5) Optionally writes to SQL via src.data.store if configured.

Design notes
------------
- Defensive imports: missing modules are skipped with a warning.
- Idempotent writes: use --overwrite-cache to force refresh.
- Emits a concise summary table to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ----------------------------
# Soft imports
# ----------------------------

def _soft_import(path: str):
    try:
        return __import__(path, fromlist=["*"])
    except Exception as exc:
        print(f"[WARN] Could not import '{path}': {exc}")
        return None

_ing_px  = _soft_import("src.data.ingest_prices")
_ing_fn  = _soft_import("src.data.ingest_fundamentals")
_store   = _soft_import("src.data.store")

cfg_yaml = ROOT / "configs" / "default.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Simple YAML reader (with fallback if PyYAML missing)
# ----------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        print("[WARN] PyYAML not installed. Using minimal parser for key: value lines.")
        out: Dict[str, Any] = {}
        for ln in path.read_text().splitlines():
            if ":" in ln and not ln.strip().startswith("#"):
                k, v = ln.split(":", 1)
                out[k.strip()] = v.strip()
        return out
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {})


# ----------------------------
# I/O helpers
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _write_parquet(df: pd.DataFrame | pd.Series, path: Path, overwrite: bool = False) -> None:
    _ensure_dir(path)
    if path.exists() and not overwrite:
        print(f"[SKIP] {path.name} exists. Use --overwrite-cache to refresh.")
        return
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name or "value")
    try:
        df.to_parquet(path)
        print(f"[OK] Wrote {path.relative_to(ROOT)} ({len(df):,} rows)")
    except Exception as exc:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path)
        print(f"[OK/CSV] Parquet failed ({exc}); wrote {csv_path.relative_to(ROOT)}")


# ----------------------------
# Core ETL
# ----------------------------

def fetch_prices(symbols: List[str], start: str, end: str, bar: str, cache_dir: Optional[str], overwrite: bool) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if _ing_px is None:
        print("[WARN] src.data.ingest_prices not found; skipping prices.")
        return out

    if hasattr(_ing_px, "ingest"):
        for sym in symbols:
            try:
                df = _ing_px.ingest(sym, start, end, bar, cache_dir=cache_dir, overwrite=overwrite)  # type: ignore[arg-type]
                if isinstance(df, pd.DataFrame) and not df.empty:
                    out[sym] = df
                    _write_parquet(df, PROCESSED_DIR / f"prices_{sym}_{bar}.parquet", overwrite)
            except Exception as exc:
                print(f"[WARN] prices ingest failed for {sym}: {exc}")
    elif hasattr(_ing_px, "run"):
        try:
            res = _ing_px.run(symbols, start, end, bar)  # type: ignore[call-arg]
            if isinstance(res, dict):
                for sym, df in res.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out[sym] = df
                        _write_parquet(df, PROCESSED_DIR / f"prices_{sym}_{bar}.parquet", overwrite)
        except Exception as exc:
            print(f"[WARN] prices run() failed: {exc}")
    else:
        print("[WARN] ingest_prices has neither ingest() nor run().")

    return out


def fetch_fundamentals(symbols: List[str], start: str, end: str, overwrite: bool) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if _ing_fn is None:
        print("[INFO] Fundamentals module missing; skipping fundamentals.")
        return out

    if hasattr(_ing_fn, "ingest"):
        for sym in symbols:
            try:
                df = _ing_fn.ingest(sym, start, end)  # type: ignore[call-arg]
                if isinstance(df, pd.DataFrame) and not df.empty:
                    out[sym] = df
                    _write_parquet(df, PROCESSED_DIR / f"fundamentals_{sym}.parquet", overwrite)
            except Exception as exc:
                print(f"[WARN] fundamentals ingest failed for {sym}: {exc}")
    elif hasattr(_ing_fn, "run"):
        try:
            res = _ing_fn.run(symbols, start, end)  # type: ignore[call-arg]
            if isinstance(res, dict):
                for sym, df in res.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out[sym] = df
                        _write_parquet(df, PROCESSED_DIR / f"fundamentals_{sym}.parquet", overwrite)
        except Exception as exc:
            print(f"[WARN] fundamentals run() failed: {exc}")
    else:
        print("[INFO] ingest_fundamentals has neither ingest() nor run().")

    return out


# ----------------------------
# CLI & entrypoint
# ----------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run prices & fundamentals ETL")
    p.add_argument("--symbols", type=str, default=None, help="Comma-separated tickers (overrides config)")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--bar", type=str, default=None, choices=["1d", "1h"]) 
    p.add_argument("--overwrite-cache", action="store_true", help="Force refresh local cache & processed files")
    p.add_argument("--config", type=str, default=str(cfg_yaml), help="Path to YAML config")
    return p.parse_args(argv)


def load_defaults(ns: argparse.Namespace) -> Dict[str, Any]:
    # Default knobs if config missing or partial
    defaults = {
        "universe": {"symbols": ["SPY", "TLT"]},
        "dates": {"start": "2013-01-01", "end": "2025-01-01", "bar": "1d"},
        "storage": {"sql_url": None},
    }
    path = Path(ns.config) if ns and ns.config else cfg_yaml
    if path.exists():
        try:
            loaded = _read_yaml(path)
            defaults.update(loaded or {})
        except Exception as exc:
            print(f"[WARN] Failed reading {path}: {exc}")
    return defaults


def main(argv: Optional[Iterable[str]] = None) -> int:
    ns = parse_args(argv)
    cfg = load_defaults(ns)

    symbols = (ns.symbols.split(",") if ns.symbols else (cfg.get("universe", {}).get("symbols") or ["SPY"]))
    start   = ns.start or (cfg.get("dates", {}).get("start") or "2013-01-01")
    end     = ns.end   or (cfg.get("dates", {}).get("end") or "2025-01-01")
    bar     = ns.bar   or (cfg.get("dates", {}).get("bar") or "1d")

    symbols = [s.strip().upper() for s in symbols if s.strip()]
    print(f"[ETL] Symbols={symbols} Start={start} End={end} Bar={bar}")

    prices = fetch_prices(symbols, start, end, bar, cache_dir=str(ROOT / "data"), overwrite=ns.overwrite_cache)
    fun    = fetch_fundamentals(symbols, start, end, overwrite=ns.overwrite_cache)

    # Optional SQL sink if available and configured
    sql_url = (cfg.get("storage", {}) or {}).get("sql_url")
    if sql_url and _store is not None:
        try:
            if hasattr(_store, "to_sql"):
                for sym, df in prices.items():
                    _store.to_sql(df, table=f"prices_{sym}_{bar}", url=sql_url)  # type: ignore[call-arg]
                for sym, df in fun.items():
                    _store.to_sql(df, table=f"fundamentals_{sym}", url=sql_url)
                print("[OK] Wrote to SQL sinks via src.data.store.to_sql")
        except Exception as exc:
            print(f"[WARN] SQL write failed: {exc}")

    # Summary
    summary_rows = []
    for sym, df in prices.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            summary_rows.append({
                "type": "prices",
                "symbol": sym,
                "rows": len(df),
                "start": df.index.min(),
                "end": df.index.max(),
                "bar": bar,
            })
    for sym, df in fun.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            summary_rows.append({
                "type": "fundamentals",
                "symbol": sym,
                "rows": len(df),
                "start": getattr(df.index, "min", lambda: None)(),
                "end": getattr(df.index, "max", lambda: None)(),
                "bar": None,
            })

    if summary_rows:
        sm = pd.DataFrame(summary_rows)
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print("\n[SUMMARY]\n", sm.sort_values(["type", "symbol"]).to_string(index=False))
    else:
        print("[SUMMARY] Nothing fetched.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
