# src/pilots/build_all.py
from pathlib import Path
import re
import pandas as pd

from src.analytics.correlations import CSConfig
from src.analytics.tearsheet import make_tearsheet, TearsheetConfig
from src.factors.momentum import compute_and_score_momentum
from src.portfolio.rank_assign import rank_from_composite
from src.portfolio.weights import WeightConfig, make_weights_equal
from src.portfolio.performance import (
    PortfolioConfig, SummaryConfig,
    compute_portfolio_returns, compute_turnover, compute_performance_summary
)
from src.analytics.tearsheet import make_tearsheet, TearsheetConfig

PRICES_PARQUET = Path("data/processed/prices.parquet")

def _norm_cols(cols):
    def norm(c):
        c = str(c).strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        return re.sub(r"_+", "_", c).strip("_")
    return [norm(c) for c in cols]

def load_prices() -> pd.DataFrame:
    if not PRICES_PARQUET.exists():
        raise FileNotFoundError(f"Missing {PRICES_PARQUET}. Seed prices first.")

    df = pd.read_parquet(PRICES_PARQUET)

    # If date is the index, move it to a column
    if "date" not in {str(getattr(df.index, "name", "")).lower()} and "date" not in [str(c).lower() for c in df.columns]:
        df = df.reset_index()

    # Normalize column names
    df.columns = _norm_cols(df.columns)

    # If we don't have a 'date' col yet, map 'index' -> 'date'
    if "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})

    # --- Wide-to-long fix: columns like adj_close_aapl, adj_close_msft, ... ---
    wide_adj_cols = [c for c in df.columns if c.startswith("adj_close_")]
    if wide_adj_cols:
        if "date" not in df.columns:
            raise ValueError("Could not find a 'date' column to unpivot wide prices.")
        # Build long frame by stacking each ticker column
        long_parts = []
        for c in wide_adj_cols:
            tkr = c.replace("adj_close_", "").upper()
            tmp = pd.DataFrame({"date": df["date"], "ticker": tkr, "adj_close": df[c]})
            long_parts.append(tmp)
        out = pd.concat(long_parts, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date", "adj_close"])
        return out[["date", "ticker", "adj_close"]].sort_values(["ticker", "date"]).reset_index(drop=True)

    # --- Long/normal path with common aliases ---
    alias_map = {}
    if "adj_close" not in df.columns:
        if "adjclose" in df.columns: alias_map["adjclose"] = "adj_close"
        elif "adjusted_close" in df.columns: alias_map["adjusted_close"] = "adj_close"
        elif "close" in df.columns: alias_map["close"] = "adj_close"
    if "ticker" not in df.columns:
        if "symbol" in df.columns: alias_map["symbol"] = "ticker"
        elif "tic" in df.columns: alias_map["tic"] = "ticker"
        elif "name" in df.columns: alias_map["name"] = "ticker"
    if alias_map:
        df = df.rename(columns=alias_map)

    required = {"date", "ticker", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prices.parquet is missing required columns {missing}. Found: {list(df.columns)}")

    df = df[["date", "ticker", "adj_close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)

def main():
    print("[BUILD] Loading prices…", flush=True)
    prices = load_prices()
    print(f"[BUILD] Prices rows: {len(prices):,}  cols: {list(prices.columns)}  "
          f"tickers: {prices['ticker'].nunique()}", flush=True)

    print("[BUILD] Momentum factor…", flush=True)
    mom = compute_and_score_momentum(prices)
    assert "momentum_z" in mom.columns, "momentum_z not computed"
    factors = mom[["date","ticker","momentum_z"]].copy()
    factors["composite"] = factors["momentum_z"]

    print("[BUILD] Ranks → Weights…", flush=True)
    ranked = rank_from_composite(factors, score_col="composite", n_bins=5, best_is_high=True)
    wcfg = WeightConfig(mode="long_short", gross_exposure=1.0, long_share_of_gross=0.5,
                        rank_col="rank", long_bin=5, short_bin=1)
    weights = make_weights_equal(ranked, cfg=wcfg)

    print("[BUILD] Portfolio returns…", flush=True)
    port_cfg = PortfolioConfig(freq="daily", method="simple", delay_days=1)
    port_rets = compute_portfolio_returns(prices, weights, cfg=port_cfg)
    turnover = compute_turnover(weights)
    summary = compute_performance_summary(port_rets, cfg=SummaryConfig(), turnover_df=turnover)

    from src.portfolio.performance import compute_drawdown_curve

    # ... after summary/ts are computed and before writing out:
    dd_curve = compute_drawdown_curve(port_rets)            # ['date','wealth','peak','drawdown']
    equity = dd_curve[['date','wealth']].copy()

    outd = Path("reports"); outd.mkdir(exist_ok=True)

    # existing writes ...
    pd.DataFrame([summary]).to_csv(outd / "performance_summary.csv", index=False)
    ts["returns"].to_csv(outd / "portfolio_returns.csv", index=False)
    ts["drawdowns"].to_csv(outd / "drawdowns.csv", index=False)

    # NEW: files most dashboards look for
    dd_curve.to_csv(outd / "drawdown_curve.csv", index=False)
    equity.to_csv(outd / "equity.csv", index=False)


    print("[BUILD] Tearsheet tables…", flush=True)
    ts = make_tearsheet(
        prices=prices,
        weights=weights,
        factor_scores=factors[["date","ticker","momentum_z"]],   # only what we have
        factor_frames={"Momentum": factors[["date","ticker","momentum_z"]]},
        cfg=TearsheetConfig(
            cs_cfg=CSConfig(factor_cols=("momentum_z",))         # <-- key line
        ),
    )

    outd = Path("reports"); outd.mkdir(exist_ok=True)
    pd.DataFrame([summary]).to_csv(outd / "performance_summary.csv", index=False)
    ts["returns"].to_csv(outd / "portfolio_returns.csv", index=False)
    ts["drawdowns"].to_csv(outd / "drawdowns.csv", index=False)
    if "cs_corr_mean" in ts: ts["cs_corr_mean"].to_csv(outd / "cs_corr_mean.csv")

    print("[DONE] Wrote reports/. Summary:", summary, flush=True)

if __name__ == "__main__":
    main()
