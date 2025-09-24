"""
engine.py
---------
Factor orchestration: compute Value, Momentum, and Quality; apply cross-sectional
scoring (winsorized z-scores), combine to a composite, and optionally assign
quintile ranks per date (and per group if desired).

Expected inputs
---------------
- prices:        ['date','ticker','adj_close']
- fundamentals:  ['date','ticker', ...]  (see individual factor modules)
- classifications (optional): ['date','ticker', group_col] for neutralization

Key APIs
--------
- FactorEngineConfig: tunable parameters for each factor + scoring.
- compute_factors(...): returns DataFrame with z-scores per factor.
- combine_factors(...): builds a weighted composite and (optionally) ranks.
- run_pipeline(...): one-stop: compute -> combine -> (optional) ranks.

Notes
-----
- Uses per-ticker as-of merges inside factor modules to avoid look-ahead.
- If `group_col` is provided (and present in `classifications`), neutralization
  happens inside each factor’s scoring step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd

from .value import compute_and_score_value, WinsorZ as WinsorZ_Value
from .momentum import compute_and_score_momentum, WinsorZ as WinsorZ_Mom
from .quality import compute_and_score_quality, WinsorZ as WinsorZ_Qual


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ValueConfig:
    prefer: Literal["pe_ttm", "pe"] = "pe_ttm"
    allow_alt_from_income: bool = True
    winsor: WinsorZ_Value = WinsorZ_Value()
    # neutralization handled at engine level via group_col/classifications


@dataclass(frozen=True)
class MomentumConfig:
    lookback_months: int = 6
    skip_months: int = 1
    method: Literal["simple", "log"] = "simple"
    min_history_days: int = 120
    winsor: WinsorZ_Mom = WinsorZ_Mom()


@dataclass(frozen=True)
class QualityConfig:
    min_equity_abs: float = 1e3
    clamp_de_ratio: float | None = 20.0
    winsor: WinsorZ_Qual = WinsorZ_Qual()


@dataclass(frozen=True)
class CompositeConfig:
    # weights applied to the z-scores; must reference columns produced below
    weights: Dict[str, float] = field(default_factory=lambda: {
        "value_z": 1/3,
        "momentum_z": 1/3,
        "quality_z": 1/3,
    })
    # If True, re-standardize the composite per date (cross-sectional z-score)
    zscore_composite: bool = True
    # Optional: cap the composite before z-scoring to reduce influence
    winsor: Tuple[float, float] = (0.01, 0.99)


@dataclass(frozen=True)
class FactorEngineConfig:
    value: ValueConfig = ValueConfig()
    momentum: MomentumConfig = MomentumConfig()
    quality: QualityConfig = QualityConfig()
    composite: CompositeConfig = CompositeConfig()
    # Neutralization
    group_col: Optional[str] = None  # e.g., "gics_sector"
    # Output options
    include_raw_columns: bool = False  # keep raw signals from factor modules
    include_component_z: bool = False  # keep component z (roe_z, de_z) from quality


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _normalize(df: pd.DataFrame, required_cols: Tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Have: {list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    return out


def _winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if s.size == 0:
        return s
    q_lo, q_hi = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=q_lo, upper=q_hi)


def _per_date_zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


# -----------------------------------------------------------------------------
# Core engine
# -----------------------------------------------------------------------------

def compute_factors(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    classifications: Optional[pd.DataFrame] = None,
    cfg: FactorEngineConfig = FactorEngineConfig(),
) -> pd.DataFrame:
    """
    Compute Value, Momentum, and Quality z-scores (optionally group-neutralized).
    Returns a DataFrame indexed by ['date','ticker'] with columns:
        ['value_z','momentum_z','quality_z'] (+ optional raw/components)
        (+ group_col if provided)
    """
    px = _normalize(prices, ("date", "ticker", "adj_close"))
    f = _normalize(fundamentals, ("date", "ticker"))

    cls = None
    if classifications is not None and cfg.group_col:
        cls = _normalize(classifications, ("date", "ticker", cfg.group_col))

    # Value
    val = compute_and_score_value(
        prices=px,
        fundamentals=f,
        prefer=cfg.value.prefer,
        allow_alt_from_income=cfg.value.allow_alt_from_income,
        winsor=cfg.value.winsor,
        group_col=cfg.group_col if cls is not None else None,
    )

    # Momentum
    mom = compute_and_score_momentum(
        prices=px,
        lookback_months=cfg.momentum.lookback_months,
        skip_months=cfg.momentum.skip_months,
        method=cfg.momentum.method,
        min_history_days=cfg.momentum.min_history_days,
        winsor=cfg.momentum.winsor,
        classifications=cls if cls is not None else None,
        group_col=cfg.group_col if cls is not None else None,
    )

    # Quality
    qual = compute_and_score_quality(
        prices=px,
        fundamentals=f,
        min_equity_abs=cfg.quality.min_equity_abs,
        clamp_de_ratio=cfg.quality.clamp_de_ratio,
        winsor=cfg.quality.winsor,
        classifications=cls if cls is not None else None,
        group_col=cfg.group_col if cls is not None else None,
    )

    # Select columns to keep
    keep_cols = ["date", "ticker", "value_z", "momentum_z", "quality_z"]
    if cfg.group_col and (cfg.group_col in val.columns):
        keep_cols.append(cfg.group_col)

    # Raw columns (optional)
    if cfg.include_raw_columns:
        for c in ("earnings_yield", "mom_ret", "roe", "de", "quality_raw"):
            if c in val.columns:
                keep_cols.append(c)
            if c in mom.columns and c not in keep_cols:
                keep_cols.append(c)
            if c in qual.columns and c not in keep_cols:
                keep_cols.append(c)

    # Component z (optional, from quality)
    if cfg.include_component_z:
        for c in ("roe_z", "de_z"):
            if c in qual.columns:
                keep_cols.append(c)

    # Merge
    out = (
        val[keep_cols].merge(
            mom[["date", "ticker", "momentum_z"]],
            on=["date", "ticker"], how="outer"
        ).merge(
            qual[[c for c in ["date", "ticker", "quality_z", "roe_z", "de_z"] if c in qual.columns]],
            on=["date", "ticker"], how="outer"
        ).sort_values(["ticker", "date"]).reset_index(drop=True)
    )

    return out


def combine_factors(
    factor_df: pd.DataFrame,
    *,
    cfg: FactorEngineConfig = FactorEngineConfig(),
    rank_quintiles: bool = False,
    rank_col: str = "composite_rank",
) -> pd.DataFrame:
    """
    Given factor z-scores, produce a weighted composite and optional per-date ranks.

    Returns original columns plus:
        - 'composite_raw' : weighted sum of z-scores
        - 'composite'     : (optionally) per-date z-scored composite
        - 'composite_rank': 1..5 if rank_quintiles=True (5 = best)
    """
    df = factor_df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker"])

    # Weighted sum over available z-score columns
    weights = cfg.composite.weights
    missing_keys = [k for k in weights.keys() if k not in df.columns]
    if missing_keys:
        raise ValueError(f"combine_factors: missing columns for weights: {missing_keys}")

    def _dot(row):
        s = 0.0
        wsum = 0.0
        for k, w in weights.items():
            v = row[k]
            if pd.notna(v):
                s += float(w) * float(v)
                wsum += abs(float(w))
        return s if wsum > 0 else np.nan

    df["composite_raw"] = df.apply(_dot, axis=1)

    # Optional winsorize + per-date z-score on the composite
    lo, hi = cfg.composite.winsor
    def _by_date(block: pd.DataFrame) -> pd.DataFrame:
        x = block["composite_raw"].astype(float)
        x_w = _winsorize(x, lo, hi)
        block["composite"] = _per_date_zscore(x_w) if cfg.composite.zscore_composite else x_w
        if rank_quintiles:
            # Rank ascending -> lower scores worse; we want 5 = best (highest composite)
            valid = block["composite"].notna()
            block.loc[valid, rank_col] = pd.qcut(block.loc[valid, "composite"], 5, labels=[1,2,3,4,5])
            block[rank_col] = block[rank_col].astype("float64")
        return block

    df = df.groupby("date", group_keys=False).apply(_by_date)

    return df.reset_index(drop=True)


def run_pipeline(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    classifications: Optional[pd.DataFrame] = None,
    cfg: FactorEngineConfig = FactorEngineConfig(),
    rank_quintiles: bool = True,
) -> pd.DataFrame:
    """
    One-stop factor pipeline:
      1) compute_factors -> value_z, momentum_z, quality_z (optionally with group neutralization)
      2) combine_factors -> composite (+ optional quintile ranks)

    Returns a tidy DataFrame with:
      ['date','ticker','value_z','momentum_z','quality_z','composite','composite_rank'(opt)]
      plus any extra columns requested via cfg flags.
    """
    facs = compute_factors(
        prices=prices,
        fundamentals=fundamentals,
        classifications=classifications,
        cfg=cfg,
    )
    out = combine_factors(
        facs,
        cfg=cfg,
        rank_quintiles=rank_quintiles,
    )
    # Sort for readability
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
