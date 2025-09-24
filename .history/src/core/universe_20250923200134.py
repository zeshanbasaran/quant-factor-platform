"""
universe.py
-----------
Builds and refreshes the trading universe.

Sources (priority)
------------------
1) Explicit symbols passed in by the caller.
2) CSV at configs/universe.csv (columns: symbol, name, asset_class, currency, exchange, sector, industry).
3) YAML at configs/default.yaml (key: universe: [SPY, TLT, ...]) if available.
4) Fallback to src.config.SYMBOLS.

Outputs
-------
- Validated pandas DataFrame with at least a `symbol` column (UPPERCASE, deduped).
- Optional parquet cache at data/processed/universe.parquet

Functions
---------
- get_universe(...)
- load_csv_universe(...)
- load_yaml_universe(...)
- build_universe_table(...)
- save_universe_parquet(...)
- list_universe_symbols(...)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import yaml  # optional, only needed if you load from YAML
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

# Project defaults
try:
    from src.config import SYMBOLS as DEFAULT_SYMBOLS  # type: ignore
except Exception:  # pragma: no cover
    DEFAULT_SYMBOLS = ["SPY", "TLT"]


# ----------------------------
# Data model (optional)
# ----------------------------

@dataclass
class Security:
    symbol: str
    name: Optional[str] = None
    asset_class: Optional[str] = None     # e.g., "Equity", "ETF", "Future"
    currency: Optional[str] = None        # e.g., "USD"
    exchange: Optional[str] = None        # e.g., "NYSEARCA"
    sector: Optional[str] = None
    industry: Optional[str] = None

    def normalize(self) -> "Security":
        s = self.symbol.strip().upper().replace("/", "_")
        return Security(
            symbol=s,
            name=self.name.strip() if self.name else None,
            asset_class=(self.asset_class or None),
            currency=(self.currency or None),
            exchange=(self.exchange or None),
            sector=(self.sector or None),
            industry=(self.industry or None),
        )


# ----------------------------
# Helpers
# ----------------------------

_META_COLS = [
    "symbol", "name", "asset_class", "currency", "exchange", "sector", "industry"
]

def _to_df(objs: Iterable[Security]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(o.normalize()) for o in objs])
    # ensure core column
    if "symbol" not in df.columns:
        df["symbol"] = []
    # column order
    cols = [c for c in _META_COLS if c in df.columns] + [c for c in df.columns if c not in _META_COLS]
    df = df[cols]
    # validate & dedupe
    df["symbol"] = df["symbol"].astype(str).str.upper().str.replace("/", "_", regex=False).str.strip()
    df = df.loc[df["symbol"].ne("")].drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return df


def _symbols_to_df(symbols: Sequence[str]) -> pd.DataFrame:
    secs = [Security(symbol=s) for s in symbols]
    return _to_df(secs)


# ----------------------------
# Loaders
# ----------------------------

def load_csv_universe(csv_path: str | Path) -> pd.DataFrame:
    """
    Load universe from a CSV file.
    Expected columns (case-insensitive): symbol[, name, asset_class, currency, exchange, sector, industry]
    Extra columns are preserved.
    """
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame(columns=_META_COLS)
    df = pd.read_csv(p)
    # normalize columns to lowercase underscores
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    if "symbol" not in df.columns:
        raise ValueError(f"CSV '{csv_path}' must contain a 'symbol' column.")
    # reorder and clean
    return _to_df([Security(**{k: v for k, v in row.items() if k in _META_COLS})
                   for row in df.to_dict(orient="records")])


def load_yaml_universe(yaml_path: str | Path, key: str = "universe") -> List[str]:
    """
    Load symbols list from a YAML file under a specified key.
    Requires PyYAML if used.
    """
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is not installed. Add `pyyaml` to your dependencies to load YAML universes.")
    p = Path(yaml_path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    uni = data.get(key, [])
    if isinstance(uni, dict):
        # allow dict {symbols: [...]} or other nesting
        uni = uni.get("symbols", [])
    if not isinstance(uni, (list, tuple)):
        raise ValueError(f"YAML '{yaml_path}' key '{key}' must be a list of symbols.")
    return [str(s).strip().upper().replace("/", "_") for s in uni if str(s).strip()]


# ----------------------------
# Builder / Saver
# ----------------------------

def build_universe_table(
    symbols: Optional[Sequence[str]] = None,
    csv_path: str | Path = "configs/universe.csv",
    yaml_path: str | Path = "configs/default.yaml",
    yaml_key: str = "universe",
    fallback_symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a consolidated universe table:

    Priority:
      1) `symbols` if provided
      2) CSV at `csv_path`
      3) YAML key `yaml_key` at `yaml_path`
      4) `fallback_symbols` or DEFAULT_SYMBOLS

    Returns
    -------
    pd.DataFrame with at least a `symbol` column, deduped & uppercased.
    """
    # 1) explicit symbols
    if symbols and len(symbols) > 0:
        df = _symbols_to_df(symbols)
        if not df.empty:
            return df

    # 2) CSV
    df_csv = load_csv_universe(csv_path)
    if not df_csv.empty:
        return df_csv

    # 3) YAML (list under key)
    try:
        yaml_syms = load_yaml_universe(yaml_path, key=yaml_key)
    except Exception:
        yaml_syms = []
    if yaml_syms:
        return _symbols_to_df(yaml_syms)

    # 4) fallback
    fb = list(fallback_symbols) if fallback_symbols else list(DEFAULT_SYMBOLS)
    return _symbols_to_df(fb)


def save_universe_parquet(df: pd.DataFrame, out_path: str | Path = "data/processed/universe.parquet") -> Path:
    """
    Save universe table to parquet (falls back to CSV if parquet engine missing).
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False)
        return p
    except Exception as exc:  # pragma: no cover
        alt = Path(str(p) + ".csv")
        df.to_csv(alt, index=False)
        print(f"[WARN] Parquet write failed ({exc}). Wrote CSV fallback at: {alt}")
        return alt


# ----------------------------
# Public API
# ----------------------------

def get_universe(
    symbols: Optional[Sequence[str]] = None,
    *,
    csv_path: str | Path = "configs/universe.csv",
    yaml_path: str | Path = "configs/default.yaml",
    yaml_key: str = "universe",
    cache_to: Optional[str | Path] = "data/processed/universe.parquet",
    fallback_symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build the universe table and optionally cache it.

    Parameters
    ----------
    symbols : list[str] | None
        Explicit list of symbols to use (highest priority).
    csv_path : str | Path
        CSV file to load if `symbols` not provided.
    yaml_path : str | Path
        YAML file to consult if `symbols` and CSV not provided.
    yaml_key : str
        Key inside YAML that holds the list of symbols.
    cache_to : str | Path | None
        If provided, write the resulting table to parquet/CSV.
    fallback_symbols : list[str] | None
        Fallback list if all other sources are empty.

    Returns
    -------
    pd.DataFrame
        Universe table with at least a `symbol` column.
    """
    df = build_universe_table(
        symbols=symbols,
        csv_path=csv_path,
        yaml_path=yaml_path,
        yaml_key=yaml_key,
        fallback_symbols=fallback_symbols,
    )
    if cache_to:
        save_universe_parquet(df, cache_to)
    return df


def list_universe_symbols(df: Optional[pd.DataFrame] = None, **kwargs) -> List[str]:
    """
    Convenience helper to return a clean list of symbols.

    If `df` is None, it calls get_universe(**kwargs).
    """
    if df is None:
        df = get_universe(**kwargs)
    if "symbol" not in df.columns:
        raise ValueError("Universe DataFrame must contain a 'symbol' column.")
    return df["symbol"].astype(str).str.upper().str.replace("/", "_", regex=False).str.strip().tolist()


__all__ = [
    "Security",
    "get_universe",
    "build_universe_table",
    "load_csv_universe",
    "load_yaml_universe",
    "save_universe_parquet",
    "list_universe_symbols",
]
