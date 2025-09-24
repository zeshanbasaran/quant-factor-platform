"""
refresh_universe.py
-------------------
Builds/refreshed the trading universe and writes it to data/processed/.

Usage
-----
python -m scripts.refresh_universe \
  --asof 2025-01-01 \
  --source sp500 \
  --min-price 3 --min-dollar-vol 2e6 \
  --include SPY,QQQ --exclude BRK.A

What it does
------------
1) Loads `configs/default.yaml` (unless overridden) for universe rules.
2) Calls `src.core.universe.build_universe(cfg)` if available; otherwise
   falls back to simple symbol list from config/CLI.
3) Optionally filters by price, ADV, market cap if columns are present.
4) Saves `data/processed/universe.parquet` (and CSV fallback).
5) Optionally writes the universe table to SQL if `storage.sql_url` is set
   and `src.data.store.to_sql` exists.

The script is defensive and will still produce a minimal universe if modules
are missing.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

cfg_yaml = ROOT / "configs" / "default.yaml"

# -------- soft imports --------

def _soft_import(path: str):
    try:
        return __import__(path, fromlist=["*"])
    except Exception as exc:
        print(f"[WARN] Could not import '{path}': {exc}")
        return None

_universe = _soft_import("src.core.universe")
_store    = _soft_import("src.data.store")

# -------- yaml (with fallback) --------

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


# -------- helpers --------

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    _ensure_parent(path)
    try:
        df.to_parquet(path, index=False)
        print(f"[OK] Wrote {path.relative_to(ROOT)} ({len(df):,} rows)")
    except Exception as exc:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK/CSV] Parquet failed ({exc}); wrote {csv_path.relative_to(ROOT)}")


# -------- core refresh --------
@dataclass
class UniArgs:
    asof: Optional[str]
    source: Optional[str]
    include: List[str]
    exclude: List[str]
    min_price: Optional[float]
    min_dollar_vol: Optional[float]
    min_mcap: Optional[float]


def _normalize_list(csv: Optional[str]) -> List[str]:
    if not csv:
        return []
    return [s.strip().upper() for s in csv.split(",") if s.strip()]


def build_universe(ns: UniArgs, defaults: Dict[str, Any]) -> pd.DataFrame:
    # Start with module builder if present
    if _universe is not None and hasattr(_universe, "build_universe"):
        try:
            cfg = (defaults.get("universe") or {}).copy()
            # CLI overrides
            if ns.source:
                cfg["source"] = ns.source
            if ns.include:
                cfg["include"] = list(set(_normalize_list(cfg.get("include", "")) + ns.include))  # type: ignore[arg-type]
            if ns.exclude:
                cfg["exclude"] = list(set(_normalize_list(cfg.get("exclude", "")) + ns.exclude))  # type: ignore[arg-type]
            if ns.asof:
                cfg["asof"] = ns.asof
            df = _universe.build_universe(cfg)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as exc:
            print(f"[WARN] build_universe(cfg) failed: {exc}")

    # Fallback: construct from symbols list only
    syms = defaults.get("universe", {}).get("symbols") or ["SPY"]
    syms = list(dict.fromkeys([s.strip().upper() for s in syms]))
    if ns.include:
        syms = list(dict.fromkeys(syms + ns.include))
    if ns.exclude:
        syms = [s for s in syms if s not in set(ns.exclude)]
    df = pd.DataFrame({"symbol": syms})
    if ns.asof:
        df["asof"] = ns.asof
    return df


def apply_filters(df: pd.DataFrame, ns: UniArgs) -> pd.DataFrame:
    out = df.copy()
    # Optional filters if columns exist
    if ns.min_price is not None and "last_price" in out.columns:
        out = out.loc[out["last_price"] >= ns.min_price]
    if ns.min_dollar_vol is not None and {"adv", "last_price"}.issubset(out.columns):
        out = out.loc[(out["adv"] * out["last_price"]) >= ns.min_dollar_vol]
    if ns.min_mcap is not None and "market_cap" in out.columns:
        out = out.loc[out["market_cap"] >= ns.min_mcap]
    return out.reset_index(drop=True)


# -------- CLI --------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh trading universe table")
    p.add_argument("--asof", type=str, default=None, help="Effective date for the universe snapshot (YYYY-MM-DD)")
    p.add_argument("--source", type=str, default=None, help="Universe source key (e.g., sp500, nasdaq100, custom)")
    p.add_argument("--include", type=str, default=None, help="Comma-separated tickers to force-include")
    p.add_argument("--exclude", type=str, default=None, help="Comma-separated tickers to exclude")
    p.add_argument("--min-price", type=float, default=None, help="Filter: minimum last price")
    p.add_argument("--min-dollar-vol", type=float, default=None, help="Filter: minimum ADV*$ price")
    p.add_argument("--min-mcap", type=float, default=None, help="Filter: minimum market cap")
    p.add_argument("--config", type=str, default=str(cfg_yaml), help="Path to YAML config")
    p.add_argument("--sql-table", type=str, default="universe", help="Optional SQL table name")
    return p.parse_args(argv)


def load_config(ns: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "universe": {"symbols": ["SPY", "TLT"], "source": "custom"},
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
    cfg = load_config(ns)

    args = UniArgs(
        asof=ns.asof,
        source=ns.source,
        include=_normalize_list(ns.include),
        exclude=_normalize_list(ns.exclude),
        min_price=ns.min_price,
        min_dollar_vol=ns.min_dollar_vol,
        min_mcap=ns.min_mcap,
    )

    uni = build_universe(args, cfg)
    uni = apply_filters(uni, args)

    # Write parquet
    out_path = PROCESSED_DIR / "universe.parquet"
    _write_parquet(uni, out_path)

    # Optional SQL sink
    sql_url = (cfg.get("storage", {}) or {}).get("sql_url")
    if sql_url and _store is not None and hasattr(_store, "to_sql"):
        try:
            _store.to_sql(uni, table=str(ns.sql_table), url=sql_url)  # type: ignore[call-arg]
            print(f"[OK] Wrote universe to SQL table '{ns.sql_table}'")
        except Exception as exc:
            print(f"[WARN] SQL write failed: {exc}")

    # Summary
    cols = [c for c in ["symbol", "name", "sector", "industry", "last_price", "adv", "market_cap"] if c in uni.columns]
    head = uni[cols] if cols else uni
    print("\n[SUMMARY] Universe sample:\n", head.head(10).to_string(index=False))
    print(f"\n[SUMMARY] Rows={len(uni):,} Columns={len(uni.columns)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
