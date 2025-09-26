"""
correlations.py
---------------
Correlation utilities for factors and factor returns.

Use cases
---------
1) Cross-sectional correlations between factor scores on each date
   (e.g., value_z vs momentum_z vs quality_z across tickers).
2) Time-series correlations between factor *returns* (e.g., long-minus-short sleeves).
3) Rolling correlations (time-series) for heatmaps or monitoring.

Inputs
------
- Factor scores: tidy DataFrame with ['date','ticker', <factor cols>]
- Factor returns: tidy DataFrame with ['date', <factor return cols>] or
  long-form ['date','factor','ret'] (we’ll pivot it wide).

Outputs
-------
- Correlation matrices per date, averaged matrices, and heatmap-ready long frames.

Notes
-----
- For cross-sectional correlations, Spearman (rank) is often preferable for robustness.
- All functions avoid look-ahead by operating on already-formed inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers / normalization
# ---------------------------------------------------------------------

def _normalize(df: pd.DataFrame, required_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["date"])
    return out


def _ensure_unique(seq: Iterable[str]) -> List[str]:
    s = [str(c).lower().strip() for c in seq]
    if len(set(s)) != len(s):
        raise ValueError("Duplicate factor/return column names detected.")
    return s


def _corr(df: pd.DataFrame, method: Literal["pearson","spearman"]) -> pd.DataFrame:
    if method == "pearson":
        return df.corr(method="pearson")
    elif method == "spearman":
        # rank transform, then Pearson on ranks
        r = df.rank(axis=0)
        return r.corr(method="pearson")
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")


def _stack_heatmap(mat: pd.DataFrame, *, value_col: str = "corr") -> pd.DataFrame:
    """
    Convert a square correlation matrix to long form suitable for heatmaps.
    """
    long = mat.stack().reset_index()
    long.columns = ["row", "col", value_col]
    return long


# ---------------------------------------------------------------------
# 1) Cross-sectional correlations between factor scores
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CSConfig:
    factor_cols: Tuple[str, ...] = ("value_z", "momentum_z", "quality_z")
    method: Literal["pearson","spearman"] = "spearman"
    min_names: int = 5  # minimum tickers per date to compute correlation


def cs_corr_by_date(
    factor_df: pd.DataFrame,
    *,
    cfg: CSConfig = CSConfig(),
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Cross-sectional correlations between `factor_cols` for each date.

    Returns
    -------
    dict: {date -> correlation matrix DataFrame}
    """
    req = ("date", "ticker") + tuple(cfg.factor_cols)
    data = _normalize(factor_df, req)

    # Per date: pivot (tickers as rows) with factor columns
    corr_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    for d, block in data.groupby("date"):
        # Keep rows with at least one non-null across factors
        B = block[list(cfg.factor_cols)].astype(float)
        # Need enough non-null names per pair; corr() handles pairwise, we enforce min_names globally
        if (B.notna().sum(axis=1) > 0).sum() < cfg.min_names:
            continue
        mat = _corr(B, method=cfg.method)
        corr_by_date[pd.Timestamp(d)] = mat
    return corr_by_date


def cs_corr_mean(
    factor_df: pd.DataFrame,
    *,
    cfg: CSConfig = CSConfig(),
    how: Literal["simple","weighted"] = "simple",
) -> Optional[pd.DataFrame]:
    """
    Average cross-sectional correlation matrix over time.

    - 'simple': arithmetic mean of daily matrices (ignoring NaNs per cell)
    - 'weighted': weight by number of valid tickers per day (approx via non-null counts)

    Returns
    -------
    DataFrame correlation matrix, or None if insufficient data.
    """
    mats = cs_corr_by_date(factor_df, cfg=cfg)
    if not mats:
        return None

    # Stack matrices into 3D array per cell
    keys = list(mats.keys())
    cols = list(mats[keys[0]].columns)
    idx = list(mats[keys[0]].index)

    acc = pd.DataFrame(0.0, index=idx, columns=cols)
    weight = pd.DataFrame(0.0, index=idx, columns=cols)

    if how == "simple":
        for m in mats.values():
            acc = acc.add(m, fill_value=0.0)
            weight = weight.add(m.notna().astype(float), fill_value=0.0)
    else:
        # Weighted by cross-sectional breadth ≈ count of tickers with any factor value
        req = ("date", "ticker") + tuple(cfg.factor_cols)
        data = _normalize(factor_df, req)
        breadth_by_date = data.groupby("date")["ticker"].nunique()

        for d, m in mats.items():
            w = float(breadth_by_date.get(d, np.nan))
            if not np.isfinite(w) or w <= 0:
                w = 1.0
            acc = acc.add(m * w, fill_value=0.0)
            weight = weight.add(m.notna().astype(float) * w, fill_value=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_mat = acc / weight.replace(0.0, np.nan)
    return mean_mat


def cs_corr_heatmap_long(
    factor_df: pd.DataFrame,
    *,
    cfg: CSConfig = CSConfig(),
) -> Optional[pd.DataFrame]:
    """
    Convenience: average CS correlation matrix and return heatmap-ready long form.
    """
    mat = cs_corr_mean(factor_df, cfg=cfg, how="simple")
    if mat is None:
        return None
    return _stack_heatmap(mat, value_col="cs_corr")


# ---------------------------------------------------------------------
# 2) Time-series correlations between factor RETURNS
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TSConfig:
    ret_cols: Tuple[str, ...] = ("ret_value", "ret_momentum", "ret_quality")
    method: Literal["pearson","spearman"] = "pearson"
    min_periods: int = 30  # for rolling windows etc.


def ts_corr_matrix(
    returns_df: pd.DataFrame,
    *,
    cfg: TSConfig = TSConfig(),
    dropna: Literal["any","all","none"] = "any",
) -> pd.DataFrame:
    """
    Time-series correlation matrix between factor return columns.

    Returns
    -------
    DataFrame square correlation matrix.
    """
    req = ("date",) + tuple(cfg.ret_cols)
    data = _normalize(returns_df, req).set_index("date").sort_index()
    X = data[list(cfg.ret_cols)].astype(float)

    if dropna == "any":
        X = X.dropna(how="any")
    elif dropna == "all":
        X = X.dropna(how="all")
    # 'none' keeps NaNs; corr() handles pairwise

    return _corr(X, method=cfg.method)


def ts_corr_heatmap_long(
    returns_df: pd.DataFrame,
    *,
    cfg: TSConfig = TSConfig(),
) -> pd.DataFrame:
    """
    Convenience: TS correlation matrix in heatmap-ready long form.
    """
    mat = ts_corr_matrix(returns_df, cfg=cfg)
    return _stack_heatmap(mat, value_col="ts_corr")


# ---------------------------------------------------------------------
# 3) Rolling time-series correlations
# ---------------------------------------------------------------------

def rolling_ts_corr(
    returns_df: pd.DataFrame,
    *,
    cols: Tuple[str, str],
    window: int = 63,
    method: Literal["pearson","spearman"] = "pearson",
) -> pd.DataFrame:
    """
    Rolling time-series correlation between two return columns.

    Parameters
    ----------
    returns_df : DataFrame with ['date', cols[0], cols[1]]
    cols       : tuple of two column names to correlate (e.g., ('ret_value','ret_momentum'))
    window     : rolling window length (trading days)
    method     : 'pearson' or 'spearman'

    Returns
    -------
    DataFrame with ['date','rolling_corr'].
    """
    req = ("date", cols[0], cols[1])
    data = _normalize(returns_df, req).set_index("date").sort_index()
    A = data[[cols[0], cols[1]]].astype(float)

    if method == "pearson":
        roll = A[cols[0]].rolling(window).corr(A[cols[1]])
    else:
        # Spearman: rank each series within the rolling window before corr
        def _spearman_corr(x: pd.DataFrame) -> float:
            if x.shape[0] < window:
                return np.nan
            r = x.rank()
            return r.iloc[:, 0].corr(r.iloc[:, 1])

        roll = A[[cols[0], cols[1]]].rolling(window).apply(
            lambda arr: _spearman_corr(pd.DataFrame(arr.reshape(-1, 2), columns=[cols[0], cols[1]])),
            raw=True
        )

    out = roll.rename("rolling_corr").reset_index()
    # If spearman branch yields a 2-col output, take the second column's correlation
    if isinstance(out, pd.DataFrame) and "rolling_corr" not in out.columns:
        # It can return a 2-column frame; take the last non-date column
        cc = [c for c in out.columns if c != "date"]
        out = out[["date", cc[-1]]].rename(columns={cc[-1]: "rolling_corr"})
    return out


def rolling_ts_corr_matrix(
    returns_df: pd.DataFrame,
    *,
    ret_cols: Tuple[str, ...],
    window: int = 63,
    method: Literal["pearson","spearman"] = "pearson",
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Rolling correlation matrices across multiple return columns.
    Produces a dict keyed by window-end date.

    Returns
    -------
    {date -> correlation matrix DataFrame}
    """
    req = ("date",) + tuple(ret_cols)
    data = _normalize(returns_df, req).set_index("date").sort_index()
    X = data[list(ret_cols)].astype(float)

    result: Dict[pd.Timestamp, pd.DataFrame] = {}
    for end_ix in range(window - 1, len(X)):
        window_slice = X.iloc[end_ix - window + 1 : end_ix + 1]
        mat = _corr(window_slice, method=method)
        result[pd.Timestamp(X.index[end_ix])] = mat
    return result


# ---------------------------------------------------------------------
# 4) Convenience for long-form inputs
# ---------------------------------------------------------------------

def pivot_returns_long_to_wide(
    long_df: pd.DataFrame,
    *,
    factor_col: str = "factor",
    ret_col: str = "ret",
) -> pd.DataFrame:
    """
    Convert long-form factor returns ['date','factor','ret'] to wide ['date', ret_<factor>...].
    """
    data = _normalize(long_df, ["date", factor_col, ret_col])
    wide = data.pivot_table(index="date", columns=factor_col, values=ret_col, aggfunc="first").sort_index()
    wide.columns = [f"ret_{c}" for c in wide.columns]
    wide = wide.reset_index()
    return wide


def corr_matrix_to_long(
    mat: pd.DataFrame,
    *,
    value_col: str = "corr",
) -> pd.DataFrame:
    """
    Public alias for stacking a correlation matrix to long form.
    """
    return _stack_heatmap(mat, value_col=value_col)

# ---------------------------------------------------------------------
# Back-compat helpers expected by tests
# ---------------------------------------------------------------------

def corr_matrix(df: pd.DataFrame, method: str = "pearson", min_periods: int | None = None) -> pd.DataFrame:
    """
    Legacy helper: plain correlation matrix for a wide DataFrame.
    Supports 'pearson' and 'spearman' (rank-then-pearson).
    """
    X = df.copy()
    # pandas' min_periods is supported by .corr for pearson; for spearman we rank first
    if method.lower() == "spearman":
        X = X.rank(axis=0)
        return X.corr(method="pearson", min_periods=min_periods)
    return X.corr(method="pearson", min_periods=min_periods)


def corr_melt(df: pd.DataFrame, method: str = "pearson", drop_self: bool = True) -> pd.DataFrame:
    """
    Legacy helper: melt a correlation matrix into long form with unique pairs.
    Returns columns: ['var1','var2','corr'].
    """
    M = corr_matrix(df, method=method)
    vars_ = list(M.columns)
    rows = []
    # only take upper triangle (i < j) when drop_self=True; else include diagonal too
    start_j = 1 if drop_self else 0
    for i in range(len(vars_)):
        for j in range(i + start_j, len(vars_)):
            rows.append({"var1": vars_[i], "var2": vars_[j], "corr": float(M.iloc[i, j])})
    return pd.DataFrame(rows)
