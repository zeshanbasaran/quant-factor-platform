# src/pilots/build_all.py
from pathlib import Path
import pandas as pd

from src.factors.momentum import compute_and_score_momentum
from src.portfolio.rank_assign import rank_from_composite
from src.portfolio.weights import WeightConfig, make_weights_equal
from src.portfolio.performance import (
    PortfolioConfig, SummaryConfig,
    compute_portfolio_returns, compute_turnover, compute_performance_summary
)
from src.analytics.tearsheet import make_tearsheet, TearsheetConfig

PRICES_PARQUET = Path("data/processed/prices.parquet")  # created by scripts/seed_prices.py

def load_prices() -> pd.DataFrame:
    if not PRICES_PARQUET.exists():
        raise FileNotFoundError(f"Missing {PRICES_PARQUET}. Seed prices first.")
    df = pd.read_parquet(PRICES_PARQUET)
    # keep only what’s needed
    keep = [c for c in df.columns if c.lower() in ("date","ticker","adj_close")]
    return df[keep].rename(columns=str.lower)

def main():
    print("[BUILD] Loading prices…")
    prices = load_prices()
    print(f"[BUILD] Prices rows: {len(prices):,}  tickers: {prices['ticker'].nunique()}")

    print("[BUILD] Momentum factor…")
    mom = compute_and_score_momentum(prices)
    factors = mom[["date","ticker","momentum_z"]].copy()
    factors["composite"] = factors["momentum_z"]

    print("[BUILD] Ranks → Weights…")
    ranked = rank_from_composite(factors, score_col="composite", n_bins=5, best_is_high=True)
    wcfg = WeightConfig(mode="long_short", gross_exposure=1.0, long_share_of_gross=0.5,
                        rank_col="rank", long_bin=5, short_bin=1)
    weights = make_weights_equal(ranked, cfg=wcfg)

    print("[BUILD] Portfolio returns…")
    port_cfg = PortfolioConfig(freq="daily", method="simple", delay_days=1)
    port_rets = compute_portfolio_returns(prices, weights, cfg=port_cfg)
    turnover = compute_turnover(weights)
    summary = compute_performance_summary(port_rets, cfg=SummaryConfig(), turnover_df=turnover)

    print("[BUILD] Tearsheet tables…")
    ts = make_tearsheet(
        prices=prices,
        weights=weights,
        factor_scores=factors[["date","ticker","momentum_z","composite"]],
        factor_frames={"Momentum": factors[["date","ticker","momentum_z"]]},
        cfg=TearsheetConfig(),
    )

    outd = Path("reports"); outd.mkdir(exist_ok=True)
    pd.DataFrame([summary]).to_csv(outd / "performance_summary.csv", index=False)
    ts["returns"].to_csv(outd / "portfolio_returns.csv", index=False)
    ts["drawdowns"].to_csv(outd / "drawdowns.csv", index=False)
    if "cs_corr_mean" in ts: ts["cs_corr_mean"].to_csv(outd / "cs_corr_mean.csv")
    print("[DONE] Wrote reports/. Summary:", summary)

if __name__ == "__main__":
    main()
