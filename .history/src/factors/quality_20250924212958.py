"""
quality.py
----------
Quality factor computations: ROE, D/E, and a composite score.

Signals
-------
- ROE  = NetIncome_TTM / Shareholders' Equity
- D/E  = Total Debt / Shareholders' Equity  (lower is better)
- Composite (raw):  quality_raw = z(ROE) - z(D/E)
  Composite (scored): cross-sectional winsorized z-score per date.

Inputs
------
- prices:        ['date','ticker','adj_close']  (daily or higher)
- fundamentals:  ['date','ticker', ...] with any of:
    * 'net_income_ttm'         (preferred for ROE numerator)
    * 'shareholders_equity' or 'total_equity' or 'book_equity'
    * 'total_debt' or 'debt_total' (if absent, D/E not computed)
  Optional helpers (not required):
    * 'short_long_term_debt_total' or similar debt fields
    * 'liabilities_total'  (if used as fallback for debt, not ideal)
    * 'shares_out' (unused here but common in datasets)

Behavior
--------
- As-of merge fundamentals to daily price dates per ticker (no look-ahead).
- Safe divisions (avoid +/-inf, divide-by-zero).
- Optional industry/sector neutralization via `group_col`.

Outputs
-------
- compute_quality_signals(...) -> ['date','ticker','roe','de','quality_raw']
- score_quality(...)           -> adds ['quality_z','roe_z','de_z']
- compute_and_score_quality(...) -> end-to-end convenience

Notes
-----
- Negative or near-zero equity -> ROE and D/E set to NaN (to avoid extreme noise).
- You may pre-clean fundamentals upstream (e.g., map vendor column names).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class WinsorZ:
    lower_pct: float = 0.01   # clamp bottom 1%
    upper_pct: float = 0.99   # clamp top 1%
    min_assets: int = 5       # minimum cross-section size to compute z


def _normalize(df: pd.DataFrame, required_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    return out


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    res = numer.astype(float) / denom.replace({0: np.nan}).astype(float)
    return res.replace([np.inf, -np.inf], np.nan)


def _winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if s.size == 0:
        return s
    q_lo = s.quantile(lo)
    q_hi = s.quantile(hi)
    return s.clip(lower=q_lo, upper=q_hi)


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def _asof_merge_per_ticker(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tk: str = "ticker",
    dt: str = "date",
) -> pd.DataFrame:
    left = left.sort_values([tk, dt])
    right = right.sort_values([tk, dt])
    parts = []
    for tkr, lgrp in left.groupby(tk, sort=False):
        rgrp = right[right[tk] == tkr]
        if rgrp.empty:
            parts.append(lgrp.copy())
            continue
        merged = pd.merge_asof(
            lgrp.sort_values(dt),
            rgrp.sort_values(dt),
            by=tk,
            on=dt,
            direction="backward",
            allow_exact_matches=True,
        )
        parts.append(merged)
    return pd.concat(parts, axis=0).sort_values([tk, dt])


# ---------------------------------------------------------------------
# Core quality signals
# ---------------------------------------------------------------------

def _pick_equity_column(cols: list[str]) -> Optional[str]:
    candidates = ["shareholders_equity", "total_equity", "book_equity", "equity"]
    for c in candidates:
        if c in cols:
            return c
    return None


def _pick_debt_column(cols: list[str]) -> Optional[str]:
    candidates = ["total_debt", "debt_total", "short_long_term_debt_total"]
    for c in candidates:
        if c in cols:
            return c
    return None


def compute_quality_signals(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    min_equity_abs: float = 1e3,
    clamp_de_ratio: float | None = 20.0,
) -> pd.DataFrame:
    """
    Compute ROE and D/E aligned to price dates (as-of by ticker), plus a raw composite.

    Parameters
    ----------
    prices : DataFrame with ['date','ticker','adj_close']
    fundamentals : DataFrame with ['date','ticker', ...]
    min_equity_abs : treat |equity| < min_equity_abs as invalid -> NaN (avoid noise)
    clamp_de_ratio : if not None, cap D/E at this value to tame outliers

    Returns
    -------
    DataFrame with columns:
        ['date','ticker','roe','de','quality_raw']
    """
    px = _normalize(prices, ["date", "ticker", "adj_close"])
    f = fundamentals.copy()
    f.columns = [str(c).lower().strip() for c in f.columns]
    f = _normalize(f, ["date", "ticker"])  # ensures date/ticker, other cols optional

    # Identify source columns
    cols = list(f.columns)
    eq_col = _pick_equity_column(cols)
    ni_col = "net_income_ttm" if "net_income_ttm" in cols else None
    debt_col = _pick_debt_column(cols)

    # As-of merge fundamentals onto daily price grid
    df = _asof_merge_per_ticker(px[["date", "ticker"]], f)

    # Compute ROE
    roe = pd.Series(np.nan, index=df.index, dtype="float64")
    if ni_col and eq_col:
        equity = df[eq_col].astype(float)
        # Invalidate near-zero/negative equity (common source of blowups / sign inversions)
        equity_valid = equity.where(equity.abs() >= float(min_equity_abs), np.nan)
        net_inc = df[ni_col].astype(float)
        roe = _safe_div(net_inc, equity_valid)

    # Compute D/E
    de = pd.Series(np.nan, index=df.index, dtype="float64")
    if debt_col and eq_col:
        equity = df[eq_col].astype(float)
        equity_valid = equity.where(equity.abs() >= float(min_equity_abs), np.nan)
        debt = df[debt_col].astype(float)
        de = _safe_div(debt, equity_valid)
        if clamp_de_ratio is not None:
            de = de.clip(upper=float(clamp_de_ratio))

    out = df.loc[:, ["date", "ticker"]].copy()
    out["roe"] = roe.replace([np.inf, -np.inf], np.nan)
    out["de"] = de.replace([np.inf, -np.inf], np.nan)

    # Raw composite: high ROE good, low D/E good
    # We form per-date z-scores (temporary) to combine robustly; if not enough names on a date, leave NaN.
    tmp = out.copy()

    def _per_date_combine(block: pd.DataFrame) -> pd.DataFrame:
        # z of ROE
        rz = _zscore(_winsorize(block["roe"].astype(float), 0.01, 0.99)) if block["roe"].notna().sum() >= 5 else pd.Series(np.nan, index=block.index)
        # z of D/E (invert since lower is better)
        dz_raw = block["de"].astype(float)
        dz = _zscore(_winsorize(dz_raw, 0.01, 0.99)) if dz_raw.notna().sum() >= 5 else pd.Series(np.nan, index=block.index)
        q_raw = rz - dz
        out_block = block.copy()
        out_block["quality_raw"] = q_raw
        out_block["roe_z_tmp"] = rz
        out_block["de_z_tmp"] = dz
        return out_block

    tmp = tmp.groupby("date", group_keys=False).apply(_per_date_combine)
    out["quality_raw"] = tmp["quality_raw"]
    # Keep intermediate z's for optional diagnostics in `score_quality`
    out["_roe_z_pre"] = tmp["roe_z_tmp"]
    out["_de_z_pre"] = tmp["de_z_tmp"]

    return out


# ---------------------------------------------------------------------
# Cross-sectional scoring
# ---------------------------------------------------------------------

def score_quality(
    factor_df: pd.DataFrame,
    *,
    winsor: WinsorZ = WinsorZ(),
    group_col: Optional[str] = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    quality_col: str = "quality_raw",
    out_col: str = "quality_z",
    expose_component_z: bool = True,
) -> pd.DataFrame:
    """
    Cross-sectional winsorize + z-score of the Quality composite per date,
    optionally neutralized within `group_col`.

    If `_roe_z_pre` and `_de_z_pre` exist, also output stabilized component z-scores:
      - `roe_z` (winsorized again for consistency)
      - `de_z`  (winsorized again for consistency)
    """
    df = factor_df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    req = [date_col, ticker_col, quality_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"score_quality: missing columns {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, quality_col])

    def _score_block(block: pd.DataFrame) -> pd.DataFrame:
        if block.shape[0] < winsor.min_assets:
            block[out_col] = np.nan
            if expose_component_z:
                block["roe_z"] = np.nan
                block["de_z"] = np.nan
            return block

        x = block[quality_col].astype(float)
        x_w = _winsorize(x, winsor.lower_pct, winsor.upper_pct)
        block[out_col] = _zscore(x_w)

        if expose_component_z:
            if "_roe_z_pre" in block.columns:
                r_w = _winsorize(block["_roe_z_pre"].astype(float), winsor.lower_pct, winsor.upper_pct)
                block["roe_z"] = _zscore(r_w)
            else:
                block["roe_z"] = np.nan

            if "_de_z_pre" in block.columns:
                d_w = _winsorize(block["_de_z_pre"].astype(float), winsor.lower_pct, winsor.upper_pct)
                block["de_z"] = _zscore(d_w)
            else:
                block["de_z"] = np.nan

        return block

    if group_col and group_col in df.columns:
        out = (
            df.sort_values([date_col, group_col, ticker_col])
              .groupby([date_col, group_col], group_keys=False)
              .apply(_score_block)
        )
    else:
        out = (
            df.sort_values([date_col, ticker_col])
              .groupby(date_col, group_keys=False)
              .apply(_score_block)
        )

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------

def compute_and_score_quality(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    min_equity_abs: float = 1e3,
    clamp_de_ratio: float | None = 20.0,
    winsor: WinsorZ = WinsorZ(),
    classifications: Optional[pd.DataFrame] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    One-stop pipeline:
      1) compute ROE, D/E, and quality_raw aligned to price dates
      2) optionally as-of attach `classifications` (e.g., sector)
      3) cross-sectional score to `quality_z` (+ component z's)

    Returns
    -------
    DataFrame with:
      ['date','ticker','roe','de','quality_raw','quality_z']
      plus ['roe_z','de_z'] if expose_component_z=True,
      plus [group_col] if classifications provided.
    """
    q = compute_quality_signals(
        prices=prices,
        fundamentals=fundamentals,
        min_equity_abs=min_equity_abs,
        clamp_de_ratio=clamp_de_ratio,
    )

    if classifications is not None and group_col:
        cls = _normalize(classifications, ["date", "ticker", group_col])
        q = _asof_merge_per_ticker(q, cls)

    scored = score_quality(
        q,
        winsor=winsor,
        group_col=(group_col if group_col and (group_col in q.columns) else None),
        date_col="date",
        ticker_col="ticker",
        quality_col="quality_raw",
        out_col="quality_z",
        expose_component_z=True,
    )
    # Cleanup temporary columns if present
    drop_cols = [c for c in ["_roe_z_pre", "_de_z_pre"] if c in scored.columns]
    if drop_cols:
        scored = scored.drop(columns=drop_cols)

    return scored

# ---------------------------------------------------------------------
# Back-compat helper for tests: simple composite from ROE and D/E
# ---------------------------------------------------------------------

def quality_score(funda: pd.DataFrame) -> pd.Series:
    """
    Legacy helper used by tests:
      Input: DataFrame with columns:
        - 'roe' (higher is better)
        - 'de'  (lower is better)
      Output: Series of composite scores (z(roe) - z(de)).
    NaNs are handled component-wise; if both NaN for a row -> NaN.
    """
    df = funda.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    roe = df.get("roe", pd.Series(index=df.index, dtype=float)).astype(float)
    de  = df.get("de",  pd.Series(index=df.index, dtype=float)).astype(float)

    # z-score each component (ignore rows with all-NaN later)
    def _z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    z_roe = _z(roe)
    z_de  = _z(de)

    comp = pd.concat([z_roe, -z_de], axis=1)
    out = comp.mean(axis=1, skipna=True)  # if one side missing, use the other
    out.name = "quality_score"
    # if both missing -> NaN (mean over empty -> NaN already)
    return out
