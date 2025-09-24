"""
store.py
--------
Lightweight readers/writers for parquet/CSV and SQL targets.

Files
- write_parquet(df, path) / read_parquet(path)
- write_csv(df, path) / read_csv(path)

SQL (via SQLAlchemy)
- write_sql(df, table, *, db_url=None, engine=None, if_exists='append', chunksize=2_000, dtype=None, index=False)
- upsert_sql(df, table, unique_cols, *, db_url=None, engine=None, chunksize=1_000)
  - postgresql: INSERT ... ON CONFLICT (cols) DO UPDATE SET ...
  - mysql:      INSERT ... ON DUPLICATE KEY UPDATE ...
  - sqlite:     INSERT OR REPLACE ...

Convenience loaders
- load_prices(bar='1d', path=None)          -> data/processed/prices_{bar}.parquet
- load_fundamentals(freq='Q', path=None)    -> data/processed/fundamentals_{freq}.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

import pandas as pd

# Reuse core utils for safe parquet IO
from src.core.utils import ensure_dir, to_parquet_safe, read_parquet_safe

# typing-only imports to keep runtime deps optional while pleasing PyLance/mypy
if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

# Optional DB helpers at runtime
try:
    from sqlalchemy import create_engine, text  # type: ignore
except Exception:  # pragma: no cover
    create_engine = None  # type: ignore
    text = None  # type: ignore

try:
    from src.db.io import make_engine as _make_engine  # preferred if available
except Exception:  # pragma: no cover
    _make_engine = None  # type: ignore


# ----------------------------
# File writers/readers
# ----------------------------

def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Write DataFrame to parquet (falls back to CSV on failure).
    Returns the final Path (parquet or CSV fallback).
    """
    p = Path(path)
    ensure_dir(p)
    to_parquet_safe(df, p)
    return p if p.exists() else Path(str(p) + ".csv")


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read parquet (or CSV fallback)."""
    return read_parquet_safe(path)


def write_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Simple CSV writer with directory creation."""
    p = Path(path)
    ensure_dir(p)
    df.to_csv(p, index=index)
    return p


def read_csv(path: str | Path) -> pd.DataFrame:
    """Simple CSV reader."""
    return pd.read_csv(path)


# ----------------------------
# SQL writers
# ----------------------------

def _resolve_engine(db_url: Optional[str], engine: Optional["Engine"]) -> "Engine":
    """
    Prefer provided engine; else build from db_url; else error.
    """
    if engine is not None:
        return engine
    if db_url:
        if _make_engine is not None:
            return _make_engine(db_url)
        if create_engine is None:  # pragma: no cover
            raise RuntimeError("SQLAlchemy is not available.")
        return create_engine(db_url, future=True)  # type: ignore[call-arg]
    raise ValueError("Provide either an SQLAlchemy engine or a db_url.")


def write_sql(
    df: pd.DataFrame,
    table: str,
    *,
    db_url: Optional[str] = None,
    engine: Optional["Engine"] = None,
    if_exists: str = "append",
    chunksize: int = 2_000,
    dtype: Optional[Dict] = None,
    index: bool = False,
) -> int:
    """
    Write a DataFrame to a SQL table using pandas.to_sql.

    Returns
    -------
    int
        Number of rows written (len(df) if non-empty).
    """
    if df is None or df.empty:
        return 0
    eng = _resolve_engine(db_url, engine)
    df.to_sql(
        table,
        con=eng,
        if_exists=if_exists,
        index=index,
        chunksize=chunksize,
        method=None,
        dtype=dtype,
    )
    return int(len(df))


def _dialect_name(eng: "Engine") -> str:
    try:
        return eng.dialect.name  # 'postgresql' | 'mysql' | 'sqlite' | ...
    except Exception:  # pragma: no cover
        return "unknown"


def _chunks(iterable: Iterable, size: int) -> Iterable[List]:
    buf: List = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def upsert_sql(
    df: pd.DataFrame,
    table: str,
    unique_cols: Sequence[str],
    *,
    db_url: Optional[str] = None,
    engine: Optional["Engine"] = None,
    chunksize: int = 1_000,
) -> int:
    """
    UPSERT rows into a SQL table using dialect-specific syntaxes.

    Requirements
    ------------
    - The target table must exist with a UNIQUE or PRIMARY KEY on `unique_cols`.
    - DataFrame column names must match DB column names.

    Returns
    -------
    int
        Number of rows attempted (len(df)).
    """
    if df is None or df.empty:
        return 0

    eng = _resolve_engine(db_url, engine)
    dialect = _dialect_name(eng)

    cols = list(df.columns)
    if not set(unique_cols).issubset(cols):
        missing = [c for c in unique_cols if c not in cols]
        raise ValueError(f"unique_cols missing from DataFrame: {missing}")

    # Quote identifiers per dialect (backticks for MySQL, double-quotes otherwise)
    def q(c: str) -> str:
        return f"`{c}`" if dialect == "mysql" else f'"{c}"'

    col_list = ", ".join(q(c) for c in cols)
    placeholders = ", ".join([f":{c}" for c in cols])

    if dialect == "postgresql":
        conflict_cols = ", ".join(q(c) for c in unique_cols)
        update_assign = ", ".join(f'{q(c)}=EXCLUDED.{q(c)}' for c in cols if c not in unique_cols)
        sql_tpl = (
            f'INSERT INTO {q(table)} ({col_list}) VALUES ({placeholders}) '
            f'ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_assign};'
        )
    elif dialect == "mysql":
        update_assign = ", ".join(f"{q(c)}=VALUES({q(c)})" for c in cols if c not in unique_cols)
        sql_tpl = (
            f'INSERT INTO {q(table)} ({col_list}) VALUES ({placeholders}) '
            f'ON DUPLICATE KEY UPDATE {update_assign};'
        )
    elif dialect == "sqlite":
        # Works when a UNIQUE constraint exists on unique_cols
        sql_tpl = f'INSERT OR REPLACE INTO {q(table)} ({col_list}) VALUES ({placeholders});'
    else:
        # Fallback: append only
        return write_sql(df, table, engine=eng, if_exists="append", chunksize=chunksize)

    if text is None:  # pragma: no cover
        raise RuntimeError("SQLAlchemy text() is unavailable; cannot perform upsert.")

    total = 0
    with eng.begin() as conn:
        for batch in _chunks(df.to_dict(orient="records"), chunksize):
            conn.execute(text(sql_tpl), batch)  # type: ignore[arg-type]
            total += len(batch)
    return total


# ----------------------------
# Convenience loaders
# ----------------------------

def load_prices(bar: str = "1d", path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load combined prices parquet produced by ingest_prices.py.
    """
    p = Path(path) if path else Path(f"data/processed/prices_{bar}.parquet")
    df = read_parquet_safe(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def load_fundamentals(freq: str = "Q", path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load combined fundamentals parquet produced by ingest_fundamentals.py.
    """
    p = Path(path) if path else Path(f"data/processed/fundamentals_{freq.upper()}.parquet")
    df = read_parquet_safe(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


__all__ = [
    # files
    "write_parquet", "read_parquet", "write_csv", "read_csv",
    # SQL
    "write_sql", "upsert_sql",
    # loaders
    "load_prices", "load_fundamentals",
]
