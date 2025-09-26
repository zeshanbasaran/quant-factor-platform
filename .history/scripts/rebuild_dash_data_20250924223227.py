# scripts/rebuild_dash_data.py

from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.factors.momentum import compute_and_score_momentum
from src.portfolio.rank_assign import rank_from_composite
from src.portfolio.weights import make_weights_equal, WeightConfig
from src.portfolio.performance import (
    asset_returns_from_prices, compute_portfolio_returns,
    compute_drawdown_curve, compute_performance_summary, SummaryConfig
)
from src.analytics.correlations import cs_corr_mean, CSConfig

PROCESSED = Path("data/processed")
DASH = PROCESSED / "dashboard"
DASH.mkdir(parents=True, exist_ok=True)

prices_fp = PROCESSED / "prices.parquet"
prices = pd.read_parquet(prices_fp)[["date","ticker","adj_close"]].copy()
prices["date"] = pd.to_datetime(prices["date"])

# --- portfolio returns / risk (what you already had) ---
rets = asset_returns_from_prices(prices, method="simple")
# (re-)build a simple LS momentum portfolio so we also have factors / ranks / weights
mom = compute_and_score_momentum(prices, lookback_months=6, skip_months=1)
factors = mom[["date","ticker","momentum_z"]].copy()
factors["composite"] = factors["momentum_z"]

ranks = rank_from_composite(factors, score_col="composite", n_bins=5, best_is_high=True)
wcfg = WeightConfig(mode="long_short", gross_exposure=1.0, long_bin=5, short_bin=1)
weights = make_weights_equal(ranks, cfg=wcfg)

port = compute_portfolio_returns(prices, weights)
dd = compute_drawdown_curve(port)
wealth = (1.0 + pd.Series(port["port_ret"].values, index=pd.to_datetime(port["date"]))).cumprod()

# summary + rolling risk + histogram
sum_cfg = SummaryConfig()
summary = compute_performance_summary(port, cfg=sum_cfg)
(pd.Series(wealth.values, index=pd.to_datetime(port["date"]))).to_frame("equity") \
    .to_parquet(DASH / "equity.parquet")
port.to_parquet(DASH / "returns.parquet", index=False)
dd.set_index(pd.to_datetime(dd["date"]))[["drawdown"]].to_parquet(DASH / "drawdown.parquet")

# rolling stats
ret_s = port.set_index(pd.to_datetime(port["date"]))["port_ret"].astype(float)
rvol = ret_s.rolling(63).std() * np.sqrt(252)
rvol.to_frame("rolling_vol").to_parquet(DASH / "rolling_vol.parquet")
roll_sharpe = (ret_s.rolling(63).mean() / ret_s.rolling(63).std()) * np.sqrt(252)
roll_sharpe.to_frame("rolling_sharpe").to_parquet(DASH / "rolling_sharpe.parquet")

# histogram (simple fixed bins)
bins = np.linspace(ret_s.min(), ret_s.max(), 41)
hist, edges = np.histogram(ret_s.dropna(), bins=bins)
pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "freq": hist}).to_parquet(DASH / "returns_hist.parquet", index=False)

# --- NEW: tables for the “Tables” tab ---
factors.to_parquet(DASH / "factors.parquet", index=False)
ranks.to_parquet(DASH / "ranks.parquet", index=False)
weights.to_parquet(DASH / "weights.parquet", index=False)

# --- NEW: correlations for the “Analytics” tab ---
# With only momentum this will be a 1x1 matrix; add more factor z columns later for a richer heatmap.
corr = cs_corr_mean(factors, cfg=CSConfig(factor_cols=("momentum_z",)))
if corr is not None and not corr.empty:
    corr.to_parquet(DASH / "correlations.parquet")

# optional: write the perf summary JSON the Overview tab reads
summary_out = {
    "cagr": summary.get("cagr", 0.0),
    "vol_ann": summary.get("vol_annual", 0.0),
    "sharpe": summary.get("sharpe", 0.0),
    "max_dd": summary.get("max_drawdown", 0.0),
}
(DASH / "perf_summary.json").write_text(json.dumps(summary_out))
print("[DASH] Wrote dashboard artifacts to", DASH)
