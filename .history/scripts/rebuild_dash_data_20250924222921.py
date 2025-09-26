# scripts/rebuild_dash_data.py
from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DASH = ROOT / "data" / "processed" / "dashboard"
DASH.mkdir(parents=True, exist_ok=True)

def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    if path.suffix == ".parquet":
        try: return pd.read_parquet(path)
        except Exception: pass
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # move index to column if needed
    if "date" not in df.columns:
        idxname = str(getattr(df.index, "name", "")).lower()
        if idxname == "date" or np.issubdtype(getattr(df.index, "dtype", object), np.datetime64):
            df = df.reset_index().rename(columns={idxname or "index": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df

def _find_file(basenames: list[str]) -> pd.DataFrame:
    locations = [REPORTS, DASH]
    exts = [".parquet", ".csv"]
    for loc in locations:
        for base in basenames:
            for ext in exts:
                df = _norm(_read_any(loc / f"{base}{ext}"))
                if not df.empty:
                    return df
    # wildcard last resort
    for loc in locations:
        for p in list(loc.glob("*returns*.parquet")) + list(loc.glob("*returns*.csv")):
            df = _norm(_read_any(p))
            if not df.empty:
                return df
    return pd.DataFrame()

def _load_returns() -> pd.DataFrame:
    # Try common filenames
    df = _find_file(["returns", "portfolio_returns", "port_returns"])
    if df.empty:
        return df

    # Try to discover the return column
    candidates = ["port_ret", "ret", "return", "portfolio_return", "portfolio_ret", "r"]
    ret_col = next((c for c in candidates if c in df.columns), None)

    # If not found and there’s only one non-date numeric column, take it
    if ret_col is None:
        num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 1:
            ret_col = num_cols[0]

    if ret_col is None:
        return pd.DataFrame()

    return df[["date", ret_col]].rename(columns={ret_col: "port_ret"}).sort_values("date")

def _load_equity() -> pd.Series:
    df = _find_file(["equity"])
    if df.empty: return pd.Series(dtype=float)
    # pick the first numeric col if needed
    cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    if not cols: return pd.Series(dtype=float)
    s = pd.Series(df[cols[0]].values, index=pd.to_datetime(df["date"]), name="equity").sort_index()
    return s

def _copy_if_exists(name: str):
    # pass-through for overview/tab tables
    for ext in [".parquet", ".csv", ".json"]:
        src = REPORTS / f"{name}{ext}"
        if src.exists():
            if ext == ".json":
                (DASH / f"{name}.json").write_text(src.read_text())
            else:
                _read_any(src).to_parquet(DASH / f"{name}.parquet", index=False)
            return

def main():
    # Copy overview artifacts if present
    for base in ["equity", "drawdown", "returns", "perf_summary",
                 "factors", "ranks", "weights", "correlations", "regressions"]:
        _copy_if_exists(base)

    # Load/derive returns
    rets = _load_returns()
    if rets.empty:
        # fallback: derive from equity
        eq = _load_equity()
        if not eq.empty:
            r = eq.pct_change().rename("port_ret").dropna()
            rets = pd.DataFrame({"date": r.index, "port_ret": r.values})
            print("[INFO] Derived returns from equity curve.")
    if rets.empty or {"date","port_ret"} - set(rets.columns):
        print("[WARN] No usable returns; Risk tab will remain empty.")
        return

    # Rolling metrics
    r = pd.Series(rets["port_ret"].values, index=pd.to_datetime(rets["date"])).astype(float).sort_index()
    win, ann = 63, 252
    rvol = r.rolling(win).std(ddof=0) * np.sqrt(ann)
    rshp = (r.rolling(win).mean() / r.rolling(win).std(ddof=0)) * np.sqrt(ann)

    pd.DataFrame({"date": rvol.index, "rolling_vol": rvol.values}).to_parquet(DASH / "rolling_vol.parquet", index=False)
    pd.DataFrame({"date": rshp.index, "rolling_sharpe": rshp.values}).to_parquet(DASH / "rolling_sharpe.parquet", index=False)

    # Histogram
    x = r.dropna().values
    if x.size:
        freq, edges = np.histogram(x, bins=50)
        hist = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "freq": freq})
        hist.to_parquet(DASH / "returns_hist.parquet", index=False)

    print("[DASH] Risk artifacts written to", DASH)

if __name__ == "__main__":
    main()
