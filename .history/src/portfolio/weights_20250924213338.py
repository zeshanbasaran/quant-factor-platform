"""
weights.py
----------
Portfolio weighting utilities.

Supported schemes
-----------------
1) Equal-weight
2) Cap-weight (uses 'market_cap' column)
3) Long-only, short-only, or dollar-neutral long/short
4) Optional group neutrality (e.g., sector) and per-name max-weight caps

Typical workflow
----------------
- Select your investable sleeve (e.g., quintile rank == 5 for long, == 1 for short)
- Call a weighting function to get ['date','ticker','weight']
- (Optional) Post-process with `cap_max_weight` or `enforce_group_neutral`

Notes
-----
- All functions operate cross-sectionally *by date*.
- Dollar-neutral long/short targets split gross exposure between long and short sleeves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class WeightConfig:
    # Exposure configuration
    mode: Literal["long_only", "short_only", "long_short"] = "long_short"
    gross_exposure: float = 1.0          # |long| + |short| (e.g., 1.0 = 100% gross)
    long_share_of_gross: float = 0.5     # only used when mode="long_short" (0.5 => 50/50)
    # Binning / selection
    rank_col: str = "rank"
    long_bin: int = 5
    short_bin: int = 1
    # Group neutralization
    group_col: Optional[str] = None      # e.g., 'gics_sector'
    neutralize_groups: bool = False
    # Per-name cap (applied *after* raw scheme, before normalization)
    max_weight_abs: Optional[float] = None  # absolute cap per name (e.g., 0.05)
    # Cap-weighted scheme column
    mcap_col: str = "market_cap"
    # Required identity columns
    date_col: str = "date"
    ticker_col: str = "ticker"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _normalize_df(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Have: {list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _apply_cap(series: pd.Series, cap: float) -> pd.Series:
    """
    Symmetric cap for both long (positive) and short (negative) weights by absolute value.
    """
    s = series.copy()
    too_big = s.abs() > cap
    if too_big.any():
        s.loc[too_big] = np.sign(s.loc[too_big]) * cap
    return s


def _sum_preserving_rescale(series: pd.Series, target_sum: float) -> pd.Series:
    """
    Linearly rescale to hit a target sum (handles sign).
    If current sum is ~0, return zeros to avoid blow-ups.
    """
    cur = series.sum()
    if np.isfinite(cur) and abs(cur) > 1e-12:
        return series * (target_sum / cur)
    return series * 0.0


def _groupwise_rescale(
    weights: pd.Series,
    groups: pd.Series,
    target_by_group: pd.Series,
) -> pd.Series:
    """
    Rescale weights within each group so that group sums match `target_by_group[group]`.
    """
    out = weights.copy()
    for g, idx in groups.groupby(groups).groups.items():
        tgt = float(target_by_group.get(g, 0.0))
        out.loc[idx] = _sum_preserving_rescale(out.loc[idx], tgt)
    return out


def _exposure_targets(
    cfg: WeightConfig,
) -> Tuple[float, float]:
    """
    Return (long_target, short_target) sums given config.
    """
    gross = float(cfg.gross_exposure)
    if cfg.mode == "long_only":
        return (gross, 0.0)
    if cfg.mode == "short_only":
        return (0.0, -gross)
    # long/short:
    long_tgt = gross * float(cfg.long_share_of_gross)
    short_tgt = -(gross - long_tgt)
    return (long_tgt, short_tgt)


# ---------------------------------------------------------------------
# Core weighting blocks
# ---------------------------------------------------------------------

def _equal_weight_block(
    block: pd.DataFrame,
    cfg: WeightConfig,
) -> pd.DataFrame:
    """
    Compute equal-weights for the selected sleeves inside a *single date* block.
    """
    out = block.copy()
    long_mask = (cfg.mode in ("long_only", "long_short")) & (out[cfg.rank_col] == float(cfg.long_bin))
    short_mask = (cfg.mode in ("short_only", "long_short")) & (out[cfg.rank_col] == float(cfg.short_bin))

    out["weight"] = 0.0
    long_target, short_target = _exposure_targets(cfg)

    # Long sleeve
    if long_mask.any():
        if cfg.neutralize_groups and (cfg.group_col and cfg.group_col in out.columns):
            # allocate long_target equally across groups, then within group equally across names
            groups = out.loc[long_mask, cfg.group_col]
            group_sizes = groups.value_counts()
            grp_target = (long_target / len(group_sizes)) if len(group_sizes) > 0 else 0.0
            for g, size in group_sizes.items():
                idx = out.index[long_mask & (out[cfg.group_col] == g)]
                if size > 0:
                    out.loc[idx, "weight"] = grp_target / size
        else:
            n = long_mask.sum()
            out.loc[long_mask, "weight"] = long_target / float(n)

    # Short sleeve
    if short_mask.any():
        if cfg.neutralize_groups and (cfg.group_col and cfg.group_col in out.columns):
            groups = out.loc[short_mask, cfg.group_col]
            group_sizes = groups.value_counts()
            grp_target = (abs(short_target) / len(group_sizes)) if len(group_sizes) > 0 else 0.0
            # short weights are negative
            for g, size in group_sizes.items():
                idx = out.index[short_mask & (out[cfg.group_col] == g)]
                if size > 0:
                    out.loc[idx, "weight"] = -(grp_target / size)
        else:
            n = short_mask.sum()
            out.loc[short_mask, "weight"] = short_target / float(n)

    # Per-name cap (optional), then renormalize sleeves to hit targets again
    if cfg.max_weight_abs is not None:
        out["weight"] = _apply_cap(out["weight"], float(cfg.max_weight_abs))
        # Re-normalize long and short sleeves separately
        if long_mask.any():
            out.loc[long_mask, "weight"] = _sum_preserving_rescale(out.loc[long_mask, "weight"], long_target)
        if short_mask.any():
            out.loc[short_mask, "weight"] = _sum_preserving_rescale(out.loc[short_mask, "weight"], short_target)

    return out


def _cap_weight_block(
    block: pd.DataFrame,
    cfg: WeightConfig,
) -> pd.DataFrame:
    """
    Compute cap-weighted sleeves for a *single date* block.
    Requires `cfg.mcap_col` to be present where sleeves are selected.
    """
    out = block.copy()
    out["weight"] = 0.0

    long_mask = (cfg.mode in ("long_only", "long_short")) & (out[cfg.rank_col] == float(cfg.long_bin))
    short_mask = (cfg.mode in ("short_only", "long_short")) & (out[cfg.rank_col] == float(cfg.short_bin))
    long_target, short_target = _exposure_targets(cfg)

    # Helper to raw-cap-weight within an index set
    def _raw_cap_weights(idx: pd.Index) -> pd.Series:
        mc = out.loc[idx, cfg.mcap_col].astype(float)
        mc = mc.where(mc > 0, np.nan)
        # If all missing or non-positive, fall back to equal-weight
        if mc.notna().sum() == 0:
            return pd.Series(1.0, index=idx)
        return mc / mc.sum(skipna=True)

    # Long: positive scaling to long_target
    if long_mask.any():
        idx = out.index[long_mask]
        w = _raw_cap_weights(idx)
        if cfg.neutralize_groups and (cfg.group_col and cfg.group_col in out.columns):
            # allocate long_target equally across groups, within group prop to cap
            groups = out.loc[idx, cfg.group_col]
            group_indices = {g: out.index[long_mask & (out[cfg.group_col] == g)] for g in groups.unique()}
            grp_target = (long_target / len(group_indices)) if len(group_indices) > 0 else 0.0
            for g, gidx in group_indices.items():
                wg = _raw_cap_weights(gidx)
                out.loc[gidx, "weight"] = wg * grp_target
        else:
            out.loc[idx, "weight"] = w * long_target

    # Short: negative scaling to short_target
    if short_mask.any():
        idx = out.index[short_mask]
        w = _raw_cap_weights(idx)
        if cfg.neutralize_groups and (cfg.group_col and cfg.group_col in out.columns):
            groups = out.loc[idx, cfg.group_col]
            group_indices = {g: out.index[short_mask & (out[cfg.group_col] == g)] for g in groups.unique()}
            grp_target = (abs(short_target) / len(group_indices)) if len(group_indices) > 0 else 0.0
            for g, gidx in group_indices.items():
                wg = _raw_cap_weights(gidx)
                out.loc[gidx, "weight"] = -(wg * grp_target)
        else:
            out.loc[idx, "weight"] = -(w * abs(short_target))

    # Per-name cap (optional), then sleeve-wise renormalize to targets
    if cfg.max_weight_abs is not None:
        out["weight"] = _apply_cap(out["weight"], float(cfg.max_weight_abs))
        if long_mask.any():
            out.loc[long_mask, "weight"] = _sum_preserving_rescale(out.loc[long_mask, "weight"], long_target)
        if short_mask.any():
            out.loc[short_mask, "weight"] = _sum_preserving_rescale(out.loc[short_mask, "weight"], short_target)

    return out


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------

def make_weights_equal(
    df: pd.DataFrame,
    cfg: WeightConfig = WeightConfig(),
) -> pd.DataFrame:
    """
    Equal-weight construction.

    Parameters
    ----------
    df : DataFrame with at least [date_col, ticker_col, rank_col] (+group_col if neutralizing)
    cfg: WeightConfig (mode, gross_exposure, long_bin/short_bin, etc.)

    Returns
    -------
    DataFrame with ['date','ticker','weight'] (and any passthrough columns).
    """
    req = (cfg.date_col, cfg.ticker_col, cfg.rank_col)
    data = _normalize_df(df, req)

    grouped = data.sort_values([cfg.date_col, cfg.ticker_col]).groupby(cfg.date_col, group_keys=False)
    out = grouped.apply(lambda b: _equal_weight_block(b, cfg))
    return out[[cfg.date_col, cfg.ticker_col, "weight"] + [c for c in data.columns if c not in {cfg.date_col, cfg.ticker_col, "weight"}]] \
             .reset_index(drop=True)


def make_weights_cap(
    df: pd.DataFrame,
    cfg: WeightConfig = WeightConfig(),
) -> pd.DataFrame:
    """
    Cap-weighted construction using `cfg.mcap_col`.

    Parameters
    ----------
    df : DataFrame with [date_col, ticker_col, rank_col, mcap_col] (+group_col if neutralizing)
    cfg: WeightConfig

    Returns
    -------
    DataFrame with ['date','ticker','weight'] (and passthrough columns).
    """
    req = (cfg.date_col, cfg.ticker_col, cfg.rank_col, cfg.mcap_col)
    data = _normalize_df(df, req)

    grouped = data.sort_values([cfg.date_col, cfg.ticker_col]).groupby(cfg.date_col, group_keys=False)
    out = grouped.apply(lambda b: _cap_weight_block(b, cfg))
    return out[[cfg.date_col, cfg.ticker_col, "weight"] + [c for c in data.columns if c not in {cfg.date_col, cfg.ticker_col, "weight"}]] \
             .reset_index(drop=True)


def cap_max_weight(
    weights_df: pd.DataFrame,
    *,
    max_weight_abs: float,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Post-process: cap absolute weight per name and re-normalize *by date* to preserve
    total long and short sums separately.
    """
    df = _normalize_df(weights_df, (date_col, ticker_col, "weight"))

    def _renorm(block: pd.DataFrame) -> pd.DataFrame:
        b = block.copy()
        b["weight"] = _apply_cap(b["weight"], float(max_weight_abs))
        long_mask = b["weight"] > 0
        short_mask = b["weight"] < 0
        if long_mask.any():
            b.loc[long_mask, "weight"] = _sum_preserving_rescale(b.loc[long_mask, "weight"], b.loc[long_mask, "weight"].sum())
        if short_mask.any():
            b.loc[short_mask, "weight"] = _sum_preserving_rescale(b.loc[short_mask, "weight"], b.loc[short_mask, "weight"].sum())
        return b

    return df.groupby(date_col, group_keys=False).apply(_renorm).reset_index(drop=True)


def enforce_group_neutral(
    weights_df: pd.DataFrame,
    *,
    group_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Post-process: force per-date group neutrality (sum of weights per group = 0 for L/S,
    or = original proportion for long-only).

    Implementation:
    - If both long and short present on the date, set each group's net to zero by
      allocating group totals proportionally to existing weights within group.
    - If only long (or only short) on the date, preserve the original total and
      distribute by existing weights within groups.
    """
    req = (date_col, ticker_col, "weight", group_col)
    df = _normalize_df(weights_df, req)

    def _neutralize(block: pd.DataFrame) -> pd.DataFrame:
        b = block.copy()
        if b[group_col].isna().all():
            return b

        long_total = b.loc[b["weight"] > 0, "weight"].sum()
        short_total = b.loc[b["weight"] < 0, "weight"].sum()

        # Target per-group sums:
        if long_total > 0 and short_total < 0:
            # Dollar-neutral target: 0 per group
            target_by_group = pd.Series(0.0, index=b[group_col].dropna().unique())
        else:
            # Long-only or short-only: keep relative group shares of gross
            gross = b["weight"].abs().sum()
            if gross == 0:
                return b
            grp_abs = b.groupby(group_col)["weight"].apply(lambda s: s.abs().sum())
            target_by_group = (grp_abs / grp_abs.sum()) * b["weight"].sum()

        # Rescale within groups to hit the targets
        b["weight"] = _groupwise_rescale(b["weight"], b[group_col], target_by_group)
        return b

    return df.groupby(date_col, group_keys=False).apply(_neutralize).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Back-compat helpers for tests
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np

def equal_weight_long_only(ranks: pd.DataFrame, pick_quintile: int = 5) -> pd.DataFrame:
    """
    Legacy helper:
      Input: DataFrame of ranks (rows=dates, cols=symbols) with values in {1..5}
      Output: DataFrame of weights summing to 1.0 each date, equal across pick_quintile.
    """
    assert isinstance(ranks, pd.DataFrame)
    W = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    for dt, row in ranks.iterrows():
        mask = row == float(pick_quintile)
        n = int(mask.sum())
        if n > 0:
            W.loc[dt, mask] = 1.0 / n
    return W

def long_short_weights(
    ranks: pd.DataFrame,
    long_quintile: int = 5,
    short_quintile: int = 1,
    gross: float = 1.0,
    net: float = 0.0,
) -> pd.DataFrame:
    """
    Legacy helper:
      Market-neutral by default (net=0). |weights| sum to `gross` each date.
      Equal-weight within long and short sleeves.
    """
    assert isinstance(ranks, pd.DataFrame)
    W = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)

    # Target long/short sums given gross & net
    long_target = (gross + net) / 2.0
    short_target = -(gross - net) / 2.0

    for dt, row in ranks.iterrows():
        L = (row == float(long_quintile))
        S = (row == float(short_quintile))
        nL, nS = int(L.sum()), int(S.sum())

        if nL > 0 and nS > 0:
            W.loc[dt, L] = long_target / nL
            W.loc[dt, S] = short_target / nS
        elif nL > 0 and nS == 0:
            # Only long side available → hit desired net with all long exposure
            target = long_target + (-short_target)  # equals gross if net=0
            W.loc[dt, L] = (net if net > 0 else target) / nL
        elif nS > 0 and nL == 0:
            target = (-short_target) + long_target
            W.loc[dt, S] = (net if net < 0 else -target) / nS
        # else: leave zeros if neither side available

    return W
