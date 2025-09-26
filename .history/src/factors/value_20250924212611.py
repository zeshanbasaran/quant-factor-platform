"""
value.py
--------
Value factor computations.

Primary signal
--------------
- Earnings Yield (E/P) = 1 / (P/E)
  If P/E unavailable, we approximate:
    E/P ≈ TTM Net Income / Market Cap

Input expectations
------------------
- `prices`: daily price data with columns:
    ['date','ticker','adj_close']  (additional OHLCV ignored)
- `fundamentals`: lower-frequency (quarterly or annual) data with *one* of:
    (A) 'pe'  (price-to-earnings ratio)
        Optional: 'pe_ttm' if you store trailing P/E; if both exist, 'pe_ttm' wins.
    (B) 'net_income_ttm', 'shares_out', and (optionally) 'market_cap'
        If 'market_cap' is missing we compute: market_cap = adj_close * shares_out
  Columns must include: ['date','ticker', ...fields...]

Key operations
--------------
- As-of merge (forward-fill fundamentals to daily price dates per ticker)
- Compute earnings_yield
- Optional: winsorization & cross-sectional z-scoring per date

Outputs
-------
- `compute_earnings_yield(...)` -> DataFrame with columns:
    ['date','ticker','earnings_yield']
- `score_value(...)` -> adds ['value_z'] after winsorize+zscore by date
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Config / helpers
# ----------------------------

# --- Compatibility adapter for tests expecting `earnings_yield(pe)` ---

from typing import Union

def earnings_yield(pe: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """
    Invert P/E -> earnings yield for simple inputs.
    Accepts:
      - Series of P/E values
      - DataFrame with 'pe_ttm' (preferred) or 'pe'
    Rules:
      - Only positive PE produce finite EY; non-positive/missing -> NaN.
    """
    if isinstance(pe, pd.DataFrame):
        if "pe_ttm" in pe.columns:
            series = pe["pe_ttm"].astype(float)
        elif "pe" in pe.columns:
            series = pe["pe"].astype(float)
        else:
            raise ValueError("DataFrame must contain 'pe_ttm' or 'pe'.")
    else:
        series = pd.Series(pe, copy=False).astype(float)

    ey = pd.Series(np.nan, index=series.index, name="earnings_yield")
    mask = series > 0
    ey.loc[mask] = 1.0 / series.loc[mask]
    return ey


@dataclass(frozen=True)
class WinsorZ:
    """
    Winsorize + Z-score configuration.
    """
    lower_pct: float = 0.01   # clamp bottom 1%
    upper_pct: float = 0.99   # clamp top 1%
    min_assets: int = 5       # minimum cross-section size to compute z


def _to_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {list(out.columns)}")
    # Normalize date
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        out = out.dropna(subset=["date"])
    return out


def _asof_merge_fundamentals_to_prices(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    on: Tuple[str, str] = ("ticker", "date"),
) -> pd.DataFrame:
    """
    Join lower-freq fundamentals to daily prices using per-ticker as-of merge.

    Returns a daily frame with columns from both inputs.
    """
    tk, dt = on
    p = _to_frame(prices, [tk, dt, "adj_close"]).sort_values([tk, dt])
    f = _to_frame(fundamentals, [tk, dt]).sort_values([tk, dt])

    # Use merge_asof per ticker (group-apply for clarity & robustness)
    out_parts = []
    for tkr, p_grp in p.groupby(tk, sort=False):
        f_grp = f.loc[f[tk] == tkr]
        if f_grp.empty:
            # no fundamentals for this ticker; keep price rows with NaNs for flds
            out_parts.append(p_grp.copy())
            continue
        merged = pd.merge_asof(
            p_grp.sort_values(dt),
            f_grp.sort_values(dt),
            by=tk,
            on=dt,
            direction="backward",
            allow_exact_matches=True,
        )
        out_parts.append(merged)
    out = pd.concat(out_parts, axis=0).sort_values([tk, dt])
    return out


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = numer.astype(float) / denom.replace({0: np.nan}).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


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


# ----------------------------
# Factor: Earnings Yield (E/P)
# ----------------------------

def compute_earnings_yield(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    prefer: Literal["pe_ttm", "pe"] = "pe_ttm",
    allow_alt_from_income: bool = True,
) -> pd.DataFrame:
    """
    Compute daily earnings yield (E/P) by as-of joining fundamentals to prices.

    Hierarchy:
      1) If `{prefer}` exists (e.g., 'pe_ttm'), earnings_yield = 1 / {prefer}
      2) Else if 'pe' exists, earnings_yield = 1 / pe
      3) Else if allow_alt_from_income and {'net_income_ttm' and (market_cap or shares_out)} exist:
           earnings_yield ≈ net_income_ttm / market_cap
           where market_cap = adj_close * shares_out if not provided
      4) Otherwise -> NaN

    Parameters
    ----------
    prices : DataFrame with ['date','ticker','adj_close']
    fundamentals : DataFrame with ['date','ticker', ...]
    prefer : which P/E field to invert first if available
    allow_alt_from_income : allow fallback using net income & market cap

    Returns
    -------
    DataFrame with ['date','ticker','earnings_yield']
    """
    # Normalize & as-of merge
    merged = _asof_merge_fundamentals_to_prices(prices, fundamentals, on=("ticker", "date"))

    cols = {c.lower(): c for c in merged.columns}
    has_pe_pref = prefer in cols
    has_pe = ("pe" in cols) or has_pe_pref
    # Map canonical lowercase access
    def col(name: str) -> Optional[pd.Series]:
        return merged[name] if name in merged.columns else None

    ey = pd.Series(np.nan, index=merged.index, dtype="float64")

    # 1) Preferred P/E
    if has_pe_pref:
        pe_pref = col(prefer).astype(float)
        ey_pref = _safe_div(1.0, pe_pref)
        ey = ey.fillna(ey_pref)

    # 2) Plain 'pe'
    if "pe" in merged.columns:
        pe = col("pe").astype(float)
        ey_pe = _safe_div(1.0, pe)
        ey = ey.fillna(ey_pe)

    # 3) Fallback via income & market cap
    if allow_alt_from_income and ey.isna().any():
        ni = col("net_income_ttm")
        mcap = col("market_cap")
        sh_out = col("shares_out")
        px = col("adj_close")

        if ni is not None and (mcap is not None or (sh_out is not None and px is not None)):
            if mcap is None:
                mcap = (px.astype(float) * sh_out.astype(float)).rename("market_cap")
            ey_alt = _safe_div(ni.astype(float), mcap.astype(float))
            ey = ey.fillna(ey_alt)

    out = merged.loc[:, ["date", "ticker"]].copy()
    out["earnings_yield"] = ey.astype(float)
    return out


# ----------------------------
# Cross-sectional scoring
# ----------------------------

def score_value(
    factor_df: pd.DataFrame,
    *,
    winsor: WinsorZ = WinsorZ(),
    group_col: Optional[str] = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    signal_col: str = "earnings_yield",
    z_col: str = "value_z",
) -> pd.DataFrame:
    """
    Cross-sectional winsorize + z-score by date (optionally by group within date).

    Parameters
    ----------
    factor_df : DataFrame with columns [date_col, ticker_col, signal_col]
    winsor : WinsorZ settings for clamping tails before z-scoring
    group_col : Optional industry/sector column to score within each group per date
    date_col : name of the date column
    ticker_col : name of the ticker column
    signal_col : the raw factor column (higher E/P = cheaper = higher value score)
    z_col : output z-score column name

    Returns
    -------
    DataFrame including z-scores in `z_col`.
    """
    df = factor_df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    req = [date_col, ticker_col, signal_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"score_value: missing columns {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, signal_col])

    def _score_block(block: pd.DataFrame) -> pd.DataFrame:
        if block.shape[0] < winsor.min_assets:
            block[z_col] = np.nan
            return block
        x = block[signal_col].astype(float)
        x_w = _winsorize(x, winsor.lower_pct, winsor.upper_pct)
        block[z_col] = _zscore(x_w)
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


# ----------------------------
# Convenience pipeline
# ----------------------------

def compute_and_score_value(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    prefer: Literal["pe_ttm", "pe"] = "pe_ttm",
    allow_alt_from_income: bool = True,
    winsor: WinsorZ = WinsorZ(),
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    One-stop Value factor pipeline: compute earnings_yield and add cross-sectional z-scores.

    Returns DataFrame with:
      ['date','ticker','earnings_yield','value_z'] (+ group_col if present)
    """
    ey = compute_earnings_yield(
        prices=prices,
        fundamentals=fundamentals,
        prefer=prefer,
        allow_alt_from_income=allow_alt_from_income,
    )
    if group_col and group_col in fundamentals.columns:
        # bring group labels (e.g., sector/industry) via as-of merge (static grouping also OK)
        grp = fundamentals.loc[:, ["date", "ticker", group_col]].copy()
        ey = _asof_merge_fundamentals_to_prices(ey, grp, on=("ticker", "date"))

    scored = score_value(
        ey,
        winsor=winsor,
        group_col=group_col if (group_col and group_col in ey.columns) else None,
        date_col="date",
        ticker_col="ticker",
        signal_col="earnings_yield",
        z_col="value_z",
    )
    return scored
