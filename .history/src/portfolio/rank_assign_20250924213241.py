"""
rank_assign.py
--------------
Utilities to assign per-date (and optional per-group) portfolio ranks/buckets
from a continuous score (e.g., factor composite). Typical use is quintiles.

Core APIs
---------
- assign_ranks(...): per-date (optionally per-group) bins 1..n_bins (5=best by default)
- make_long_short_flags(...): convenience flags for long/short buckets
- filter_top_bottom(...): return only top and/or bottom bins

Notes
-----
- Uses quantile binning (qcut) by default, falling back to rank-based binning
  if there are too few unique values for exact quantiles on a given cross-section.
- Higher score = better rank by default (bin `n_bins` is the best).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RankConfig:
    n_bins: int = 5                         # number of buckets (e.g., quintiles)
    best_is_high: bool = True               # True: higher score => higher bin
    min_assets: int = 5                     # min names to attempt binning
    method: Literal["quantile", "rank"] = "quantile"  # primary method
    rank_col: str = "rank"                  # output column name (1..n_bins)
    score_col: str = "composite"            # input score column
    date_col: str = "date"
    ticker_col: str = "ticker"
    group_col: Optional[str] = None         # e.g., 'gics_sector'


def _normalize(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Have: {list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _bin_via_qcut(values: pd.Series, n_bins: int) -> pd.Series:
    """
    Try quantile bins via qcut. If too few unique values, fall back to rank-based bins.
    Returns labels 1..n_bins (int).
    """
    try:
        b = pd.qcut(values, q=n_bins, labels=list(range(1, n_bins + 1)))
        return b.astype("float64")
    except Exception:
        # Fall back: dense rank (1..K), then map to bins via percentiles
        r = values.rank(method="average", na_option="keep")  # ascending by value
        pct = r / r.max()  # in (0,1]
        # cut into n equal-width bins
        b = pd.cut(pct, bins=n_bins, labels=list(range(1, n_bins + 1)))
        return b.astype("float64")


def _assign_bins_block(
    block: pd.DataFrame,
    *,
    cfg: RankConfig
) -> pd.DataFrame:
    """
    Assign bins within a date (and optionally group). Expects `block` to be the
    relevant slice for one cross-section.
    """
    out = block.copy()

    # Not enough assets to bin
    valid_mask = out[cfg.score_col].notna()
    if valid_mask.sum() < cfg.min_assets:
        out[cfg.rank_col] = np.nan
        return out

    vals = out.loc[valid_mask, cfg.score_col].astype(float)

    # High-is-best -> bin on ascending or descending?
    # We always qcut on ascending values, but if best_is_high=False, invert.
    if not cfg.best_is_high:
        vals = -vals

    if cfg.method == "quantile":
        bins = _bin_via_qcut(vals, cfg.n_bins)
    elif cfg.method == "rank":
        # Pure rank-based bins (no attempt at exact quantiles)
        r = vals.rank(method="average", na_option="keep")
        pct = r / r.max()
        bins = pd.cut(pct, bins=cfg.n_bins, labels=list(range(1, cfg.n_bins + 1))).astype("float64")
    else:
        raise ValueError("RankConfig.method must be 'quantile' or 'rank'")

    # We want labels 1..n where n=best (highest score)
    # _bin_via_qcut produced 1..n with 1=lowest value. If best_is_high=True,
    # we map to make n=best. If best_is_high=False we already inverted values,
    # so n=best is still correct.
    # Convert to 1..n already; ensure dtype float for merge compatibility.
    out[cfg.rank_col] = np.nan
    out.loc[valid_mask, cfg.rank_col] = bins

    # Make sure it's 1..n_bins with n being best:
    # After qcut, 1=smallest, n=largest -> already correct for best_is_high=True.
    # If best_is_high=False we inverted values, so largest of (-values) is worst in original scale;
    # but because we inverted beforehand, bins still map with n=best in the inverted space,
    # which equates to 1=best in original space. To fix, flip bins when best_is_high=False.
    if not cfg.best_is_high:
        out.loc[valid_mask, cfg.rank_col] = (cfg.n_bins + 1) - out.loc[valid_mask, cfg.rank_col]

    return out


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------

def assign_ranks(
    df: pd.DataFrame,
    cfg: RankConfig = RankConfig(),
) -> pd.DataFrame:
    """
    Assign per-date ranks 1..n_bins (with n being BEST by default) from a score column.

    Parameters
    ----------
    df : DataFrame containing at least [date_col, ticker_col, score_col]
    cfg: RankConfig with n_bins, best_is_high, method, etc.

    Returns
    -------
    DataFrame with an added column `cfg.rank_col` of type float (1..n_bins).
    """
    req = (cfg.date_col, cfg.ticker_col, cfg.score_col)
    data = _normalize(df, req)

    # Group keys: per-date, optionally within group_col
    if cfg.group_col and cfg.group_col in data.columns:
        grouped = data.sort_values([cfg.date_col, cfg.group_col, cfg.ticker_col]) \
                      .groupby([cfg.date_col, cfg.group_col], group_keys=False)
    else:
        grouped = data.sort_values([cfg.date_col, cfg.ticker_col]) \
                      .groupby(cfg.date_col, group_keys=False)

    out = grouped.apply(lambda b: _assign_bins_block(b, cfg=cfg))
    return out.reset_index(drop=True)


def make_long_short_flags(
    ranked_df: pd.DataFrame,
    *,
    rank_col: str = "rank",
    n_bins: int = 5,
    long_bin: int = 5,
    short_bin: int = 1,
    flag_col_long: str = "is_long",
    flag_col_short: str = "is_short",
) -> pd.DataFrame:
    """
    Given a DataFrame with a `rank_col` (1..n_bins), add boolean flags
    for long and short buckets.

    Returns a new DataFrame with two columns `flag_col_long` / `flag_col_short`.
    """
    df = ranked_df.copy()
    if rank_col not in df.columns:
        raise ValueError(f"make_long_short_flags: missing '{rank_col}'")

    df[flag_col_long] = (df[rank_col] == float(long_bin))
    df[flag_col_short] = (df[rank_col] == float(short_bin))
    return df


def filter_top_bottom(
    ranked_df: pd.DataFrame,
    *,
    rank_col: str = "rank",
    keep_top: bool = True,
    keep_bottom: bool = True,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Filter to only top and/or bottom bins.
    """
    df = ranked_df.copy()
    if rank_col not in df.columns:
        raise ValueError(f"filter_top_bottom: missing '{rank_col}'")

    mask = pd.Series(False, index=df.index)
    if keep_top:
        mask |= df[rank_col] == float(n_bins)
    if keep_bottom:
        mask |= df[rank_col] == float(1)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------
# Example convenience for end-to-end usage
# ---------------------------------------------------------------------

def rank_from_composite(
    factor_df: pd.DataFrame,
    *,
    score_col: str = "composite",
    n_bins: int = 5,
    best_is_high: bool = True,
    group_col: Optional[str] = None,
    method: Literal["quantile", "rank"] = "quantile",
    min_assets: int = 5,
    rank_col: str = "rank",
) -> pd.DataFrame:
    """
    Thin wrapper to assign ranks directly from a DataFrame that already
    contains a per-date `score_col` (e.g., engine.combine_factors output).
    """
    cfg = RankConfig(
        n_bins=n_bins,
        best_is_high=best_is_high,
        min_assets=min_assets,
        method=method,
        rank_col=rank_col,
        score_col=score_col,
        group_col=group_col,
    )
    return assign_ranks(factor_df, cfg=cfg)

# ---------------------------------------------------------------------
# Back-compat helper for tests: per-date quintiles {1..5}
# ---------------------------------------------------------------------

def assign_quintiles(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy helper used by tests.
    Input: rows=dates, cols=symbols, values=factor scores.
    Output: same shape DataFrame with quintile ranks in {1..5}; NaN in -> NaN out.
    """
    if not isinstance(scores.index, pd.DatetimeIndex):
        try:
            scores.index = pd.to_datetime(scores.index)
        except Exception:
            pass

    def _rank_row(row: pd.Series) -> pd.Series:
        s = row.copy()
        mask = s.notna()
        if mask.sum() < 2:
            return pd.Series(np.nan, index=row.index)
        # qcut over ranks (stable with duplicates)
        r = s[mask].rank(method="first")
        try:
            q = pd.qcut(r, 5, labels=False, duplicates="drop") + 1
        except Exception:
            # fallback: equal-width on percentile of ranks
            pct = r / r.max()
            q = pd.cut(pct, bins=5, labels=False) + 1
        out = pd.Series(np.nan, index=row.index, dtype="float64")
        out.loc[mask] = q.astype("float64")
        return out

    ranked = scores.apply(_rank_row, axis=1)
    return ranked
