# scripts/rebuild_dash_data.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DASH = ROOT / "data" / "processed" / "dashboard"
DASH.mkdir(parents=True, exist_ok=True)

def _find_returns() -> pd.DataFrame:
    # try the most likely names first
    candidates = [
        REPORTS / "returns.parquet",
        REPORTS / "returns.csv",
        REPORTS / "port_returns.parquet",
        REPORTS / "portfolio_returns.parquet",
    ]
    for p in candidates:
        if p.exists():
            return (pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))
    # last resort: look for any *returns*.parquet in reports/
    for p in REPORTS.glob("*returns*.parquet"):
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    return pd.DataFrame()

def _to_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.Series(df[col].values, index=pd.to_datetime(df["date"]), name=col).sort_index()
    return s

def main():
    # 1) Core series copied straight through if present
    #    (these are what your Overview tab already uses)
    for name in ["equity", "drawdown", "returns", "perf_summary"]:
        src_pq = REPORTS / f"{name}.parquet"
        src_csv = REPORTS / f"{name}.csv"
        src_json = REPORTS / f"{name}.json"
        dst = DASH / f"{name}.parquet" if name != "perf_summary" else DASH / f"{name}.json"
        try:
            if name == "perf_summary":
                if src_json.exists():
                    dst.write_text(src_json.read_text())
            else:
                if src_pq.exists():
                    pd.read_parquet(src_pq).to_parquet(dst, index=False)
                elif src_csv.exists():
                    pd.read_csv(src_csv).to_parquet(dst, index=False)
        except Exception:
            pass

    # 2) Build Risk artifacts from returns
    rets_df = _find_returns()
    if not rets_df.empty and {"date","port_ret"}.issubset(rets_df.columns):
        r = _to_series(rets_df, "port_ret").astype(float)

        # rolling vol (ann) and rolling sharpe (ann) with 63d window
        win = 63
        ann = 252
        r_rolling_vol = r.rolling(win).std(ddof=0) * np.sqrt(ann)
        r_rolling_sharpe = (r.rolling(win).mean() / r.rolling(win).std(ddof=0)) * np.sqrt(ann)

        pd.DataFrame({"date": r_rolling_vol.index, "rolling_vol": r_rolling_vol.values}).to_parquet(DASH / "rolling_vol.parquet", index=False)
        pd.DataFrame({"date": r_rolling_sharpe.index, "rolling_sharpe": r_rolling_sharpe.values}).to_parquet(DASH / "rolling_sharpe.parquet", index=False)

        # return histogram for last ~3Y (optional)
        x = r.dropna()
        if len(x) > 0:
            bins = np.histogram(x.values, bins=50)
            hist = pd.DataFrame({
                "bin_left": bins[1][:-1],
                "bin_right": bins[1][1:],
                "freq": bins[0],
            })
            hist.to_parquet(DASH / "returns_hist.parquet", index=False)
    else:
        print("[WARN] No returns found with columns ['date','port_ret']; Risk tab will be empty.")

    # 3) Optional tables if present
    for name in ["factors", "ranks", "weights", "correlations", "regressions"]:
        for ext in [".parquet", ".csv"]:
            src = REPORTS / f"{name}{ext}"
            if src.exists():
                df = pd.read_parquet(src) if ext == ".parquet" else pd.read_csv(src)
                df.to_parquet(DASH / f"{name}.parquet", index=False)
                break

if __name__ == "__main__":
    main()
