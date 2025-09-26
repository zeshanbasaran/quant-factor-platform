# scripts/rebuild_dash_data.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.factors.momentum import compute_and_score_momentum
from src.portfolio.rank_assign import rank_from_composite
from src.portfolio.weights import make_weights_equal, WeightConfig
from src.portfolio.performance import (
    asset_returns_from_prices,
    compute_portfolio_returns,
    compute_drawdown_curve,
    compute_performance_summary,
    SummaryConfig,
)
from src.analytics.correlations import cs_corr_mean, CSConfig

PROCESSED = Path("data/processed")
DASH = PROCESSED / "dashboard"
DASH.mkdir(parents=True, exist_ok=True)

PRICES_FP = PROCESSED / "prices.parquet"


def load_prices_tidy(path: Path) -> pd.DataFrame:
    """Return prices as tidy ['date','ticker','adj_close'] from multiple possible schemas."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Seed prices first.")

    df = pd.read_parquet(path)

    # If date lives in the index, bring it out
    idx_name = str(getattr(df.index, "name", "")).lower()
    if "date" not in [idx_name] and "date" not in [str(c).lower() for c in df.columns]:
        df = df.reset_index()

    # normalize columns
    df.columns = [str(c).strip().lower() for c in df.columns]

    # alias common names
    alias = {}
    if "date" not in df.columns and "index" in df.columns:
        alias["index"] = "date"
    if "ticker" not in df.columns:
        for a in ("symbol", "tic", "name"):
            if a in df.columns:
                alias[a] = "ticker"
                break
    if "adj_close" not in df.columns:
        if "adjclose" in df.columns:
            alias["adjclose"] = "adj_close"
        elif "adjusted_close" in df.columns:
            alias["adjusted_close"] = "adj_close"
        elif "close" in df.columns:
            alias["close"] = "adj_close"
    if alias:
        df = df.rename(columns=alias)

    # If still not tidy, try to melt wide format like adj_close_aapl, ...
    if "adj_close" not in df.columns or "ticker" not in df.columns:
        wide_cols = [c for c in df.columns if c.startswith("adj_close_")]
        if wide_cols:
            long = df.melt(
                id_vars=[c for c in df.columns if c == "date"],
                value_vars=wide_cols,
                var_name="tmp",
                value_name="adj_close",
            )
            long["ticker"] = long["tmp"].str.replace("adj_close_", "", regex=False).str.upper()
            df = long[["date", "ticker", "adj_close"]]
        else:
            raise ValueError(
                f"prices.parquet is not tidy and no adj_close_* columns were found. Columns: {list(df.columns)}"
            )

    out = df[["date", "ticker", "adj_close"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def main() -> None:
    print("[DASH] Loading prices…")
    prices = load_prices_tidy(PRICES_FP)
    print(f"[DASH] Prices: {len(prices):,} rows, {prices['ticker'].nunique()} tickers.")

    # Asset returns (not strictly needed later, but good to have)
    _ = asset_returns_from_prices(prices, method="simple")

    # Factors: use 6M momentum with 1M skip; use momentum_z as 'composite'
    print("[DASH] Computing momentum & scores…")
    mom = compute_and_score_momentum(prices, lookback_months=6, skip_months=1)
    factors = mom[["date", "ticker", "momentum_z"]].copy()
    factors["composite"] = factors["momentum_z"]

    # Ranks → equal-weight long-short (Q5 long / Q1 short)
    print("[DASH] Ranks → weights…")
    ranks = rank_from_composite(factors, score_col="composite", n_bins=5, best_is_high=True)
    wcfg = WeightConfig(mode="long_short", gross_exposure=1.0, long_bin=5, short_bin=1)
    weights = make_weights_equal(ranks, cfg=wcfg)

    # Portfolio returns & wealth/drawdown
    print("[DASH] Portfolio returns & risk…")
    port = compute_portfolio_returns(prices, weights)
    ret_s = port.set_index(pd.to_datetime(port["date"]))["port_ret"].astype(float).sort_index()
    wealth = (1.0 + ret_s.fillna(0.0)).cumprod()
    dd = compute_drawdown_curve(port)

    # Summary metrics
    summary = compute_performance_summary(port, cfg=SummaryConfig())

    # Rolling risk stats
    rvol = ret_s.rolling(63).std() * np.sqrt(252)
    rshp = (ret_s.rolling(63).mean() / ret_s.rolling(63).std()) * np.sqrt(252)

    # Histogram
    if ret_s.dropna().empty:
        hist_df = pd.DataFrame(columns=["bin_left", "bin_right", "freq"])
    else:
        bins = np.linspace(ret_s.min(), ret_s.max(), 41)
        hist, edges = np.histogram(ret_s.dropna(), bins=bins)
        hist_df = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "freq": hist})

    # Correlations (Analytics tab) – single factor => 1×1 matrix (fine)
    corr = cs_corr_mean(factors, cfg=CSConfig(factor_cols=("momentum_z",)))

    # --- Write artifacts the Dash app reads ---
    print(f"[DASH] Writing artifacts to {DASH} …")
    wealth.rename("equity").to_frame().to_parquet(DASH / "equity.parquet")
    port.to_parquet(DASH / "returns.parquet", index=False)
    dd.set_index(pd.to_datetime(dd["date"]))[["drawdown"]].to_parquet(DASH / "drawdown.parquet")
    rvol.rename("rolling_vol").to_frame().to_parquet(DASH / "rolling_vol.parquet")
    rshp.rename("rolling_sharpe").to_frame().to_parquet(DASH / "rolling_sharpe.parquet")
    hist_df.to_parquet(DASH / "returns_hist.parquet", index=False)

    factors.to_parquet(DASH / "factors.parquet", index=False)
    ranks.to_parquet(DASH / "ranks.parquet", index=False)
    weights.to_parquet(DASH / "weights.parquet", index=False)

    if corr is not None and not corr.empty:
        corr.to_parquet(DASH / "correlations.parquet")

    # Perf summary JSON (Overview KPIs)
    summary_out = {
        "cagr": summary.get("cagr", 0.0),
        "vol_ann": summary.get("vol_annual", 0.0),
        "sharpe": summary.get("sharpe", 0.0),
        "max_dd": summary.get("max_drawdown", 0.0),
    }
    (DASH / "perf_summary.json").write_text(json.dumps(summary_out))

    print("[DASH] Done.")


if __name__ == "__main__":
    main()
