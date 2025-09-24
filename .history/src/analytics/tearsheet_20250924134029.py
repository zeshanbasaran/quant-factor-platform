"""
tearsheet.py
------------
Summary tables & helper utilities to assemble a compact, dashboard-ready
“tearsheet” for a quant factor portfolio.

What this module provides
-------------------------
- High-level one-call generator: `make_tearsheet(...)`
- Focused builders for specific blocks:
    * performance: CAGR/Vol/Sharpe/Sortino/MaxDD/Calmar/HitRate/Turnover
    * drawdown table: top N drawdowns with depth/duration/recovery
    * IC table: mean/std/IR per factor
    * correlation blocks: cross-sectional (factor z’s) and time-series (factor L/S returns)

Input conventions (tidy)
------------------------
- `prices`:        ['date','ticker','adj_close']
- `weights`:       ['date','ticker','weight']
- `factor_scores`: ['date','ticker', 'value_z','momentum_z','quality_z','composite']  (any subset ok)
- `factor_rets_wide` (optional): ['date','ret_value','ret_momentum','ret_quality', ...]
- `turnover_df` (optional):      ['date','turnover']

Notes
-----
- This file only builds *data* (DataFrames/dicts). Plotting is left to the app layer.
- Relies on metrics in `performance.py` and `correlations.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Local deps
from ..portfolio.performance import (
    PortfolioConfig,
    SummaryConfig,
    asset_returns_from_prices,
    compute_portfolio_returns,
    compute_turnover,
    compute_drawdown_curve,
    compute_performance_summary,
    ICConfig,
    information_coefficient,
)
from .correlations import (
    CSConfig,
    TSConfig,
    cs_corr_mean,
    ts_corr_matrix,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _norm(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {list(out.columns)}")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values(["date"])
    return out


def _fmt_pct(x: float) -> float:
    """Return x already as decimal (e.g., 0.123), round for neat tables."""
    if x is None or not np.isfinite(x):
        return np.nan
    return float(np.round(x, 4))  # 4dp (~basis-point precision)


def _fmt2(x: float) -> float:
    if x is None or not np.isfinite(x):
        return np.nan
    return float(np.round(x, 2))


# ---------------------------------------------------------------------
# Drawdown blocks
# ---------------------------------------------------------------------

def _drawdown_episodes(returns_df: pd.DataFrame, *, col: str = "port_ret") -> pd.DataFrame:
    """
    Identify drawdown episodes with start/end/valley dates, depth and duration.
    """
    dd = compute_drawdown_curve(returns_df, col=col)
    wealth = dd["wealth"].values
    peak = dd["peak"].values
    draw = dd["drawdown"].values
    dates = pd.to_datetime(dd["date"]).to_list()

    episodes: List[dict] = []
    in_dd = False
    start_ix = 0
    valley_ix = 0
    for i in range(len(draw)):
        if not in_dd and draw[i] < 0:
            in_dd = True
            start_ix = i
            valley_ix = i
        if in_dd:
            if draw[i] < draw[valley_ix]:
                valley_ix = i
            # recovery: wealth back to prior peak (drawdown returns to 0)
            if np.isclose(draw[i], 0.0, atol=1e-12):
                end_ix = i
                episodes.append({
                    "start": dates[start_ix],
                    "valley": dates[valley_ix],
                    "end": dates[end_ix],
                    "depth": float(draw[valley_ix]),
                    "length_days": int((dates[end_ix] - dates[start_ix]).days),
                    "recovery_days": int((dates[end_ix] - dates[valley_ix]).days),
                })
                in_dd = False

    # If unrecovered drawdown persists to last date, record open episode
    if in_dd:
        end_ix = len(draw) - 1
        episodes.append({
            "start": dates[start_ix],
            "valley": dates[valley_ix],
            "end": pd.NaT,
            "depth": float(draw[valley_ix]),
            "length_days": int((dates[end_ix] - dates[start_ix]).days),
            "recovery_days": np.nan,
        })

    ep = pd.DataFrame(episodes).sort_values("depth")  # most negative first
    return ep.reset_index(drop=True)


def build_drawdown_table(returns_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Top N drawdowns by depth.
    """
    r = _norm(returns_df, ["date", "port_ret"])
    ep = _drawdown_episodes(r, col="port_ret")
    if ep.empty:
        return pd.DataFrame(columns=["#","Start","Valley","End","Depth","Length (days)","Recovery (days)"])
    ep = ep.sort_values("depth").head(top_n)
    ep["rank"] = range(1, len(ep) + 1)
    out = ep.rename(columns={
        "rank": "#",
        "start": "Start",
        "valley": "Valley",
        "end": "End",
        "depth": "Depth",
        "length_days": "Length (days)",
        "recovery_days": "Recovery (days)",
    })[["#","Start","Valley","End","Depth","Length (days)","Recovery (days)"]]
    # format Depth as decimal (negative)
    out["Depth"] = out["Depth"].apply(_fmt_pct)
    return out


# ---------------------------------------------------------------------
# Performance blocks
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PerfInputs:
    prices: pd.DataFrame
    weights: pd.DataFrame
    turnover: Optional[pd.DataFrame] = None
    # If turnover is None, it will be computed from weights.


def build_performance_blocks(
    *,
    inputs: PerfInputs,
    port_cfg: PortfolioConfig = PortfolioConfig(),
    sum_cfg: SummaryConfig = SummaryConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (perf_summary_table, returns_df)
    """
    prices = _norm(inputs.prices, ["date", "ticker", "adj_close"])
    weights = _norm(inputs.weights, ["date", "ticker", "weight"])

    port_rets = compute_portfolio_returns(prices, weights, cfg=port_cfg)
    turn_df = inputs.turnover if inputs.turnover is not None else compute_turnover(weights)

    summary = compute_performance_summary(port_rets, cfg=sum_cfg, turnover_df=turn_df)
    table = pd.DataFrame.from_records([{
        "CAGR": _fmt_pct(summary.get("cagr")),
        "Vol (ann.)": _fmt_pct(summary.get("vol_annual")),
        "Sharpe": _fmt2(summary.get("sharpe")),
        "Sortino": _fmt2(summary.get("sortino")),
        "Max Drawdown": _fmt_pct(summary.get("max_drawdown")),
        "Calmar": _fmt2(summary.get("calmar")),
        "Hit Rate": _fmt_pct(summary.get("hit_rate")),
        "Avg Turnover": _fmt_pct(summary.get("avg_turnover")),
        "Periods": int(summary.get("periods", 0)),
    }])

    return table, port_rets


# ---------------------------------------------------------------------
# IC blocks
# ---------------------------------------------------------------------

def build_ic_table(
    factor_frames: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    *,
    factor_col_map: Dict[str, str] | None = None,
    ic_cfg: ICConfig = ICConfig(horizon_days=21, method="spearman"),
    return_method: Literal["simple","log"] = "simple",
) -> pd.DataFrame:
    """
    factor_frames: dict like {
        "Value": df_val[['date','ticker','value_z']],
        "Momentum": df_mom[['date','ticker','momentum_z']],
        ...
    }
    factor_col_map: optional override mapping {"Value": "value_z", ...}
    """
    px = _norm(prices, ["date", "ticker", "adj_close"])
    rows = []
    for name, df in factor_frames.items():
        df2 = _norm(df, ["date", "ticker"])
        col = (factor_col_map or {}).get(name, None)
        if not col:
            # heuristics: take the last non [date,ticker] column
            cols = [c for c in df2.columns if c not in ("date", "ticker")]
            if not cols:
                continue
            col = cols[-1]
        stats = information_coefficient(df2[["date","ticker", col]], px, factor_col=col, cfg=ic_cfg, return_method=return_method)
        rows.append({
            "Factor": name,
            "IC Mean": _fmt2(stats.get("ic_mean")),
            "IC Std": _fmt2(stats.get("ic_std")),
            "IC IR": _fmt2(stats.get("ic_ir")),
            "Horizon (days)": int(stats.get("horizon_days", ic_cfg.horizon_days)),
            "Method": stats.get("method", ic_cfg.method),
            "Obs (dates)": int(stats.get("n_obs", 0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Correlation blocks
# ---------------------------------------------------------------------

def build_correlation_tables(
    factor_scores: pd.DataFrame,
    factor_rets_wide: Optional[pd.DataFrame] = None,
    *,
    cs_cfg: CSConfig = CSConfig(),
    ts_cfg: TSConfig = TSConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of named tables:
      - 'cs_corr_mean': average cross-sectional corr matrix of factor scores
      - 'ts_corr': time-series corr matrix of factor returns (if provided)
    """
    out: Dict[str, pd.DataFrame] = {}

    # CS corr (factor z-scores)
    fs = _norm(factor_scores, ["date", "ticker"] + list(cs_cfg.factor_cols))
    cs_mat = cs_corr_mean(fs, cfg=cs_cfg, how="simple")
    if cs_mat is not None:
        out["cs_corr_mean"] = cs_mat

    # TS corr (factor L/S return series)
    if factor_rets_wide is not None:
        fr = _norm(factor_rets_wide, ["date"] + list(ts_cfg.ret_cols))
        out["ts_corr"] = ts_corr_matrix(fr, cfg=ts_cfg)

    return out


# ---------------------------------------------------------------------
# Convenience: One-call tearsheet builder
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TearsheetConfig:
    # Portfolio calc configs
    port_cfg: PortfolioConfig = PortfolioConfig()
    sum_cfg: SummaryConfig = SummaryConfig()
    # IC config (applied to each factor frame)
    ic_cfg: ICConfig = ICConfig(horizon_days=21, method="spearman")
    ic_return_method: Literal["simple","log"] = "simple"
    # Corr configs
    cs_cfg: CSConfig = CSConfig()
    ts_cfg: TSConfig = TSConfig()


def make_tearsheet(
    *,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    factor_scores: pd.DataFrame,
    factor_frames: Optional[Dict[str, pd.DataFrame]] = None,
    factor_rets_wide: Optional[pd.DataFrame] = None,
    turnover_df: Optional[pd.DataFrame] = None,
    cfg: TearsheetConfig = TearsheetConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Assemble a dict of DataFrames suitable for rendering in a dashboard:

    Keys:
      - 'performance'    : single-row summary table
      - 'drawdowns'      : top drawdown episodes
      - 'cs_corr_mean'   : avg cross-sectional corr matrix (z-scores)
      - 'ts_corr'        : time-series corr matrix (if factor returns provided)
      - 'ic_table'       : IC stats per factor (if factor frames provided)
      - 'returns'        : ['date','port_ret'] time series (for charts)

    Any missing inputs (e.g., factor returns or frames) simply omit those tables.
    """
    results: Dict[str, pd.DataFrame] = {}

    # Performance & drawdowns
    perf_table, port_rets = build_performance_blocks(
        inputs=PerfInputs(prices=prices, weights=weights, turnover=turnover_df),
        port_cfg=cfg.port_cfg,
        sum_cfg=cfg.sum_cfg,
    )
    results["performance"] = perf_table
    results["returns"] = port_rets

    results["drawdowns"] = build_drawdown_table(port_rets, top_n=5)

    # Correlations
    corr_tabs = build_correlation_tables(
        factor_scores=factor_scores,
        factor_rets_wide=factor_rets_wide,
        cs_cfg=cfg.cs_cfg,
        ts_cfg=cfg.ts_cfg,
    )
    results.update(corr_tabs)

    # IC table (optional)
    if factor_frames:
        results["ic_table"] = build_ic_table(
            factor_frames=factor_frames,
            prices=prices,
            ic_cfg=cfg.ic_cfg,
            return_method=cfg.ic_return_method,
        )

    return results
