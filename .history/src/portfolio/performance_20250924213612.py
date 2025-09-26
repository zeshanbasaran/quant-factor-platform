"""
performance.py
---------------
Portfolio & factor performance utilities.

Includes:
- Asset returns from prices (simple/log)
- Portfolio returns from weights (with trading delay)
- Turnover (per-date and average)
- Drawdown series & stats
- Performance summary (CAGR, vol, Sharpe, Sortino, Calmar, hit rate, max DD)
- Information Coefficient (IC) for factors vs. forward returns

Conventions
-----------
- Prices must include ['date','ticker','adj_close'].
- Weights must include ['date','ticker','weight'].
- By default, weights at date T are applied to returns realized from T->T+1
  via a 1-day shift to avoid look-ahead (configurable via `delay_days`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers / Normalization
# ---------------------------------------------------------------------

def _normalize(df: pd.DataFrame, required_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {list(out.columns)}")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])
    return out


def _ann_factor(freq: Literal["daily","weekly","monthly"], trading_days: int = 252) -> int:
    if freq == "daily":
        return trading_days
    if freq == "weekly":
        return 52
    if freq == "monthly":
        return 12
    raise ValueError("freq must be 'daily' | 'weekly' | 'monthly'")


# ---------------------------------------------------------------------
# Asset returns
# ---------------------------------------------------------------------

def asset_returns_from_prices(
    prices: pd.DataFrame,
    *,
    method: Literal["simple","log"] = "simple",
) -> pd.DataFrame:
    """
    Compute per-ticker returns from adjusted close.
    Returns DataFrame ['date','ticker','ret'].
    """
    px = _normalize(prices, ["date","ticker","adj_close"]).sort_values(["ticker","date"])
    def _one(g: pd.DataFrame) -> pd.DataFrame:
        p = g["adj_close"].astype(float)
        if method == "simple":
            r = p.pct_change()
        elif method == "log":
            r = np.log(p).diff()
        else:
            raise ValueError("method must be 'simple' or 'log'")
        out = g[["date","ticker"]].copy()
        out["ret"] = r
        return out
    return px.groupby("ticker", group_keys=False).apply(_one).reset_index(drop=True)


# ---------------------------------------------------------------------
# Portfolio returns & turnover
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioConfig:
    freq: Literal["daily","weekly","monthly"] = "daily"  # for annualization
    method: Literal["simple","log"] = "simple"           # asset return method
    delay_days: int = 1                                  # apply weights with this lag (>=0)
    # If your weights are already aligned to the return period, set delay_days=0.


def compute_portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    cfg: PortfolioConfig = PortfolioConfig(),
) -> pd.DataFrame:
    """
    Compute portfolio returns by applying (delayed) weights to asset returns.

    Returns
    -------
    DataFrame with ['date','port_ret'] and an intermediate ['date','ticker','ret','weight_eff'] if needed.
    """
    rets = asset_returns_from_prices(prices, method=cfg.method)
    w = _normalize(weights, ["date","ticker","weight"]).sort_values(["ticker","date"])

    # Delay weights within each ticker by `delay_days`
    if cfg.delay_days > 0:
        w["weight_eff"] = w.groupby("ticker")["weight"].shift(cfg.delay_days)
    else:
        w["weight_eff"] = w["weight"]

    # Merge on (date, ticker) and compute contribution = w_eff * ret
    merged = pd.merge(rets, w[["date","ticker","weight_eff"]], on=["date","ticker"], how="left")
    merged["contrib"] = merged["weight_eff"] * merged["ret"]

    # Sum across tickers per date
    port = merged.groupby("date", as_index=False)["contrib"].sum().rename(columns={"contrib":"port_ret"})
    return port.sort_values("date").reset_index(drop=True)


def compute_turnover(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Turnover per date: 0.5 * sum(|w_t - w_{t-1}|) over tickers (standard definition).
    Returns ['date','turnover'].
    """
    w = _normalize(weights, ["date","ticker","weight"]).sort_values(["ticker","date"])
    def _one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["dw"] = g["weight"].astype(float).diff()  # by ticker
        return g
    dw = w.groupby("ticker", group_keys=False).apply(_one)

    # sum abs change over tickers per date, then * 0.5
    turn = dw.groupby("date", as_index=False)["dw"].apply(lambda s: 0.5 * np.nansum(np.abs(s))).rename(columns={"dw":"turnover"})
    return turn.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------
# Drawdowns & summary stats
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SummaryConfig:
    freq: Literal["daily","weekly","monthly"] = "daily"
    trading_days: int = 252
    rf_rate_annual: float = 0.0  # risk-free (annualized) for Sharpe
    downside_threshold: float = 0.0  # target for Sortino (e.g., 0 for standard Sortino)


def _series_from_df(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.Series(df[col].values, index=pd.DatetimeIndex(df["date"].values, name="date"))
    s = s.sort_index()
    return s


def compute_drawdown_curve(
    returns_df: pd.DataFrame,
    *,
    col: str = "port_ret",
) -> pd.DataFrame:
    """
    From a return series, compute cumulative wealth and drawdown series.
    Returns ['date','wealth','peak','drawdown'].
    """
    s = _series_from_df(returns_df, col).astype(float).fillna(0.0)
    wealth = (1.0 + s).cumprod()
    peak = wealth.cummax()
    dd = (wealth / peak) - 1.0
    out = pd.DataFrame({"date": wealth.index, "wealth": wealth.values, "peak": peak.values, "drawdown": dd.values})
    return out


def compute_performance_summary(
    returns_df: pd.DataFrame,
    *,
    cfg: SummaryConfig = SummaryConfig(),
    turnover_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute common performance metrics from a return series.
    Returns a dict of metrics.
    """
    s = _series_from_df(returns_df, "port_ret").astype(float).dropna()
    if s.empty:
        return {}

    ann_n = _ann_factor(cfg.freq, cfg.trading_days)

    # CAGR
    wealth = (1.0 + s).cumprod()
    n_years = len(s) / float(ann_n)
    cagr = wealth.iloc[-1] ** (1.0 / max(n_years, 1e-12)) - 1.0

    # Volatility (annualized)
    vol_ann = s.std(ddof=0) * np.sqrt(ann_n)

    # Sharpe
    rf_per_period = (1.0 + cfg.rf_rate_annual) ** (1.0 / ann_n) - 1.0
    excess = s - rf_per_period
    sharpe = (excess.mean() / excess.std(ddof=0)) * np.sqrt(ann_n) if excess.std(ddof=0) > 0 else np.nan

    # Sortino
    downside = np.minimum(s - cfg.downside_threshold, 0.0)
    downside_dev = np.sqrt(np.mean(downside**2))  # per-period downside deviation
    sortino = ((s.mean() - cfg.downside_threshold) / downside_dev) * np.sqrt(ann_n) if downside_dev > 0 else np.nan

    # Max drawdown
    dd_df = compute_drawdown_curve(returns_df)
    max_dd = dd_df["drawdown"].min()
    max_dd_abs = abs(max_dd)

    # Calmar
    calmar = (cagr / max_dd_abs) if max_dd_abs > 0 else np.nan

    # Hit rate, avg win/loss
    hit_rate = (s > 0).mean()
    avg_win = s[s > 0].mean() if (s > 0).any() else np.nan
    avg_loss = s[s < 0].mean() if (s < 0).any() else np.nan

    # Turnover
    avg_turn = None
    if turnover_df is not None and "turnover" in turnover_df.columns:
        avg_turn = float(turnover_df["turnover"].mean())

    return {
        "cagr": float(cagr),
        "vol_annual": float(vol_ann),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "hit_rate": float(hit_rate),
        "avg_win": float(avg_win) if avg_win is not None else np.nan,
        "avg_loss": float(avg_loss) if avg_loss is not None else np.nan,
        "avg_turnover": float(avg_turn) if avg_turn is not None else np.nan,
        "periods": int(len(s)),
    }


# ---------------------------------------------------------------------
# Information Coefficient (IC)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ICConfig:
    horizon_days: int = 21
    method: Literal["pearson","spearman"] = "spearman"
    group_neutral: bool = False
    group_col: Optional[str] = None  # e.g., 'gics_sector'


def _forward_returns(
    prices: pd.DataFrame,
    *,
    horizon_days: int,
    method: Literal["simple","log"] = "simple",
) -> pd.DataFrame:
    """
    Compute forward returns over `horizon_days` for each ticker.
    Returns ['date','ticker','fwd_ret'] where 'date' is the formation date.
    """
    px = _normalize(prices, ["date","ticker","adj_close"]).sort_values(["ticker","date"])
    def _one(g: pd.DataFrame) -> pd.DataFrame:
        p = g["adj_close"].astype(float)
        if method == "simple":
            fwd = p.shift(-horizon_days) / p - 1.0
        else:
            fwd = np.log(p.shift(-horizon_days)) - np.log(p)
        out = g[["date","ticker"]].copy()
        out["fwd_ret"] = fwd
        return out
    return px.groupby("ticker", group_keys=False).apply(_one).reset_index(drop=True)


def information_coefficient(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    factor_col: str = "composite",
    cfg: ICConfig = ICConfig(),
    return_method: Literal["simple","log"] = "simple",
) -> dict:
    """
    Cross-sectional correlation between factor values at date t and forward returns
    over `horizon_days`.

    Parameters
    ----------
    factor_df : DataFrame with ['date','ticker', factor_col] (and optionally group_col)
    prices    : DataFrame with ['date','ticker','adj_close']
    factor_col: column to evaluate (e.g., 'value_z', 'momentum_z', 'quality_z', 'composite')

    Returns
    -------
    dict with:
      - 'ic_series': pd.Series indexed by date (mean correlation per date)
      - 'ic_mean', 'ic_std', 'ic_ir' (mean / std), 'n_obs'
    """
    fac = _normalize(factor_df, ["date","ticker",factor_col]).sort_values(["date","ticker"])
    if cfg.group_neutral and not cfg.group_col:
        raise ValueError("group_neutral=True requires group_col to be set in ICConfig")
    if cfg.group_neutral and cfg.group_col not in fac.columns:
        raise ValueError(f"group_col '{cfg.group_col}' not found in factor_df")

    fwd = _forward_returns(prices, horizon_days=cfg.horizon_days, method=return_method)

    df = pd.merge(fac, fwd, on=["date","ticker"], how="inner")
    df = df.dropna(subset=[factor_col,"fwd_ret"])

    def _cs_corr(block: pd.DataFrame) -> float:
        if cfg.group_neutral and cfg.group_col in block.columns:
            # Weighted average of within-group correlations by group size
            vals = []
            sizes = []
            for _, grp in block.groupby(cfg.group_col):
                if grp[factor_col].nunique() < 2 or grp["fwd_ret"].nunique() < 2:
                    continue
                if cfg.method == "pearson":
                    c = grp[factor_col].corr(grp["fwd_ret"])
                else:
                    c = grp[factor_col].rank().corr(grp["fwd_ret"].rank())
                if pd.notna(c):
                    vals.append(c)
                    sizes.append(len(grp))
            if len(vals) == 0:
                return np.nan
            return float(np.average(vals, weights=sizes))
        else:
            if block[factor_col].nunique() < 2 or block["fwd_ret"].nunique() < 2:
                return np.nan
            if cfg.method == "pearson":
                return float(block[factor_col].corr(block["fwd_ret"]))
            else:
                return float(block[factor_col].rank().corr(block["fwd_ret"].rank()))

    ic_by_date = df.groupby("date").apply(_cs_corr).astype(float)
    ic_mean = ic_by_date.mean(skipna=True)
    ic_std = ic_by_date.std(ddof=0, skipna=True)
    ic_ir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan

    return {
        "ic_series": ic_by_date,
        "ic_mean": float(ic_mean) if pd.notna(ic_mean) else np.nan,
        "ic_std": float(ic_std) if pd.notna(ic_std) else np.nan,
        "ic_ir": float(ic_ir) if pd.notna(ic_ir) else np.nan,
        "n_obs": int(ic_by_date.notna().sum()),
        "horizon_days": cfg.horizon_days,
        "method": cfg.method,
        "group_neutral": cfg.group_neutral,
        "group_col": cfg.group_col,
    }

# -----------------------------------------------------------------------------
# Back-compat helpers expected by tests
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np

def portfolio_returns(w: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    """
    Legacy helper: row-wise dot product of weights and asset returns.
    Inputs are wide DataFrames (rows=dates, cols=tickers).
    """
    W = w.copy().astype(float)
    R = asset_returns.copy().astype(float).reindex(index=W.index, columns=W.columns)
    pr = (W.fillna(0.0) * R.fillna(0.0)).sum(axis=1)
    pr.name = "portfolio_return"
    return pr

def turnover_series(w: pd.DataFrame) -> pd.Series:
    """
    Legacy helper: Turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|.
    Input is a wide weights DataFrame (rows=dates, cols=tickers).
    """
    W = w.copy().astype(float)
    to = 0.5 * W.diff().abs().sum(axis=1)
    to.name = "turnover"
    return to

def drawdown_curve(r: pd.Series) -> pd.Series:
    """
    Legacy helper: return the drawdown series only.
    (Your modern API returns a DataFrame via compute_drawdown_curve.)
    """
    s = pd.Series(r, copy=False).astype(float).fillna(0.0)
    wealth = (1.0 + s).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    dd.name = "drawdown"
    return dd

def max_drawdown(r: pd.Series) -> float:
    """Legacy helper: minimum of the drawdown series."""
    return float(drawdown_curve(r).min())

# Preserve the modern IC function, and provide a dispatcher that also
# supports the test's legacy signature: information_coefficient(scores, fwd_returns, method=...)
_information_coefficient_modern = information_coefficient  # keep reference to your new API

def information_coefficient(
    arg1,
    arg2,
    *args,
    method: str = "spearman",
    **kwargs,
):
    """
    Dispatch:
      - Modern usage (current project): information_coefficient(factor_df, prices, factor_col=..., cfg=..., return_method=...)
        -> returns dict (unchanged)
      - Legacy tests: information_coefficient(scores_wide, fwd_returns_wide, method="spearman"|"pearson")
        -> returns pd.Series of per-date IC
    """
    # Heuristic: if arg2 looks like a prices DataFrame with 'adj_close', use modern API.
    if isinstance(arg2, pd.DataFrame) and ("adj_close" in {c.lower() for c in arg2.columns}):
        return _information_coefficient_modern(arg1, arg2, *args, **kwargs)

    # Legacy path: arg1 = scores (wide, rows=dates, cols=tickers),
    #               arg2 = fwd_returns (same shape)
    scores = arg1.copy()
    fwd = arg2.copy()

    # Make sure indexes align and are datetime-like
    if not isinstance(scores.index, pd.DatetimeIndex):
        scores.index = pd.to_datetime(scores.index, errors="coerce")
    if not isinstance(fwd.index, pd.DatetimeIndex):
        fwd.index = pd.to_datetime(fwd.index, errors="coerce")

    dates = scores.index.intersection(fwd.index)
    ic_vals = []
    for dt in dates:
        a = scores.loc[dt]
        b = fwd.loc[dt]
        pair = pd.concat([a, b], axis=1).dropna()
        if pair.shape[0] < 2:
            ic_vals.append(np.nan)
            continue
        if method == "pearson":
            ic = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        else:
            ic = float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))
        ic_vals.append(ic)

    ic_series = pd.Series(ic_vals, index=dates, name="IC")
    return ic_series
