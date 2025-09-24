"""
utils.py
--------
General-purpose utilities for the Quantitative Factor Analytics & Visualization Platform.

Highlights
----------
- Filesystem helpers: ensure_dir
- Reproducibility: set_seed
- Time/granularity: periods_per_year_from_bar, infer_periods_per_year
- DataFrame hygiene: validate_price_df, align_on_index
- Risk helpers: drawdown_series
- Formatting: fmt_pct, fmt_money, fmt_float
- Timing: time_block context manager
- Safe parquet I/O: to_parquet_safe, read_parquet_safe
- Misc: stable_hash
"""

from __future__ import annotations

import contextlib
import hashlib
import random
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ----------------------------
# Filesystem / reproducibility
# ----------------------------


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists; return Path to the original target.
    If 'path' points to a file, its parent directory is created.

    Parameters
    ----------
    path : str | Path
        Directory or file path.

    Returns
    -------
    Path
        The original path as a Path object.
    """
    p = Path(path)
    target = p if p.suffix == "" else p.parent
    target.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for Python's PRNG and NumPy for reproducibility.

    Parameters
    ----------
    seed : int, default 42
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------
# Time & cadence utilities
# ----------------------------


def periods_per_year_from_bar(bar: str) -> int:
    """
    Map a bar string to an annualized period count.

    - '1d' -> 252 (trading days)
    - '1h' -> ~252 * 6.5 (trading hours)
    Fallback to 252 if unknown.

    Parameters
    ----------
    bar : str
        Bar size label ("1d", "1h", ...).

    Returns
    -------
    int
        Estimated periods per year.
    """
    b = (bar or "").lower().strip()
    if b == "1d":
        return 252
    if b == "1h":
        return int(252 * 6.5)
    return 252


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """
    Infer periods/year from a DatetimeIndex spacing.

    Heuristic:
      - If median spacing >= ~12 hours: treat as daily -> 252
      - Else: compute 365 days (in seconds) / median spacing.

    Parameters
    ----------
    index : pd.DatetimeIndex

    Returns
    -------
    int
        Estimated periods per year.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252

    # Ensure monotonic to compute diffs properly
    idx = index.sort_values()
    deltas = np.diff(idx.view("i8"))  # nanoseconds as int64
    if len(deltas) == 0:
        return 252

    med_ns = float(np.median(deltas))
    sec = med_ns / 1e9
    if sec >= 12 * 3600:  # ~half-day or more → daily-like
        return 252

    periods = int(round((365 * 24 * 3600) / sec))
    return max(52, min(periods, 24 * 365))  # clamp to sane bounds


# ----------------------------
# DataFrame validation & alignment
# ----------------------------

OHLCV_COLS = ("open", "high", "low", "close", "adj_close", "volume")


def validate_price_df(
    df: pd.DataFrame, require: Iterable[str] = ("adj_close",)
) -> pd.DataFrame:
    """
    Validate that `df` looks like a price DataFrame used by the engine.

    - Ensures DatetimeIndex (tz-naive), sorted, unique.
    - Lowercases column names.
    - Checks required columns exist (default: 'adj_close').

    Returns a cleaned copy of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate price frame.
    require : Iterable[str], default ('adj_close',)
        Columns that must exist.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with normalized columns/index.
    """
    if df is None or df.empty:
        out = pd.DataFrame(columns=list(OHLCV_COLS))
        for c in out.columns:
            out[c] = out[c].astype("float64")
        out.index = pd.to_datetime(out.index)
        return out

    out = df.copy()

    # Normalize columns
    out.columns = [str(c).lower().strip().replace(" ", "_") for c in out.columns]

    # Index → DatetimeIndex (tz-naive), sorted, unique
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # Required columns exist?
    missing = [c for c in require if c not in out.columns]
    if missing:
        raise ValueError(
            f"Price DataFrame missing required columns: {missing}. "
            f"Have: {list(out.columns)}"
        )

    # Coerce numeric types where present
    for c in OHLCV_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def align_on_index(*objs: Union[pd.Series, pd.DataFrame]) -> Tuple:
    """
    Align multiple pandas objects on the intersection of their indices.
    Preserves the type of each object and returns them in the same order.

    Parameters
    ----------
    *objs : pd.Series | pd.DataFrame

    Returns
    -------
    tuple
        Objects aligned to the common index.
    """
    if not objs:
        return tuple()
    idx = None
    for o in objs:
        if not hasattr(o, "index"):
            raise TypeError("All objects must be pandas Series or DataFrames.")
        idx = o.index if idx is None else idx.intersection(o.index)
    aligned = []
    for o in objs:
        aligned.append(o.loc[idx])
    return tuple(aligned)


# ----------------------------
# Risk helpers
# ----------------------------


def drawdown_series(equity: pd.Series) -> Tuple[float, pd.Series]:
    """
    Compute drawdown series and its minimum value.

    Parameters
    ----------
    equity : pd.Series
        Equity curve (portfolio value over time).

    Returns
    -------
    (min_drawdown, dd_series) : (float, pd.Series)
        min_drawdown is a negative fraction (e.g., -0.23 for -23%).
    """
    e = pd.Series(equity).astype(float).dropna()
    if e.empty:
        return 0.0, pd.Series(dtype=float, index=getattr(equity, "index", None))
    roll_max = e.cummax()
    dd = e / roll_max - 1.0
    return float(dd.min()), dd.reindex(equity.index) if hasattr(equity, "index") else dd


# ----------------------------
# Formatting helpers
# ----------------------------


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    """Format a number as a percentage string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}%}"


def fmt_money(x: Optional[float], digits: int = 2) -> str:
    """Format a number as USD money string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"${x:,.{digits}f}"


def fmt_float(x: Optional[float], digits: int = 4) -> str:
    """Format a float with fixed precision."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


# ----------------------------
# Timing / profiling
# ----------------------------


@contextlib.contextmanager
def time_block(label: str = "block"):
    """
    Context manager to measure elapsed wall time of a code block.

    Example
    -------
        with time_block("simulate"):
            result = simulate(...)

    Parameters
    ----------
    label : str, default "block"
        A label to print with elapsed time.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIMER] {label}: {dt:.3f}s")


# ----------------------------
# Parquet I/O (safe)
# ----------------------------


def to_parquet_safe(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Write DataFrame to parquet using pyarrow/fastparquet if available.
    Gracefully falls back to CSV if parquet engine is missing.

    Parameters
    ----------
    df : pd.DataFrame
    path : str | Path
    """
    p = ensure_dir(path)
    try:
        df.to_parquet(p, index=True)
    except Exception as exc:  # pragma: no cover - fallback path
        alt = Path(str(p) + ".csv")
        df.to_csv(alt, index=True)
        print(f"[WARN] Parquet write failed ({exc}). Wrote CSV fallback at: {alt}")


def read_parquet_safe(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read DataFrame from parquet, falling back to CSV(.csv) if needed.
    Returns empty DataFrame if nothing is available.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"[WARN] Parquet read failed ({exc}). Trying CSV fallback...")
    csv_p = Path(str(p) + ".csv")
    if csv_p.exists():
        return pd.read_csv(csv_p, parse_dates=True, index_col=0)
    return pd.DataFrame()


# ----------------------------
# Misc
# ----------------------------


def stable_hash(text: str) -> str:
    """
    Deterministic short hash for IDs/filenames from arbitrary text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        12-character hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


__all__ = [
    # FS / repro
    "ensure_dir",
    "set_seed",
    # cadence
    "periods_per_year_from_bar",
    "infer_periods_per_year",
    # data
    "validate_price_df",
    "align_on_index",
    # risk
    "drawdown_series",
    # formatting
    "fmt_pct",
    "fmt_money",
    "fmt_float",
    # timing
    "time_block",
    # parquet
    "to_parquet_safe",
    "read_parquet_safe",
    # misc
    "stable_hash",
]
