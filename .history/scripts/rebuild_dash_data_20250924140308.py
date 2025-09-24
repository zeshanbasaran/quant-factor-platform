"""
rebuild_dash_data.py
--------------------
Materializes tidy, dashboard-ready datasets from the factor backtesting
pipeline into `data/processed/dashboard/`.

This script is intentionally **defensive** and will adapt to whichever
modules are present. It can:
- Run the full pipeline (via `src.pilots.build_all.run`) **or**
- Load already-processed artifacts from `data/processed/` and derive the
  dashboard tables/series.

Outputs (all under data/processed/dashboard/):
- equity.parquet            (Series: portfolio equity)
- returns.parquet           (Series: daily/period returns)
- perf_summary.json         (dict: CAGR, vol, Sharpe, maxDD, start/end)
- drawdown.parquet          (DataFrame: equity, peak, drawdown)
- rolling_vol.parquet       (Series)
- rolling_sharpe.parquet    (Series)
- returns_hist.parquet      (DataFrame: bins, freq)
- factors.parquet           (DataFrame: long table if available)
- ranks.parquet             (DataFrame)
- weights.parquet           (DataFrame: wide, dt × symbol)
- correlations.parquet      (DataFrame) if present
- regressions.parquet       (DataFrame) if present

Usage examples
--------------
python -m scripts.rebuild_dash_data \
  --symbols SPY,QQQ --start 2015-01-01 --end 2025-01-01 --bar 1d

# Only reshape from existing processed artifacts, no pipeline run
python -m scripts.rebuild_dash_data --no-build
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
DASH_DIR  = PROCESSED / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)

# Soft imports

def _soft_import(path: str):
    try:
        return __import__(path, fromlist=["*"])
    except Exception as exc:
        print(f"[WARN] Could not import '{path}': {exc}")
        return None

_build_all = _soft_import("src.pilots.build_all")

# Config
cfg_yaml = ROOT / "configs" / "default.yaml"


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        print("[WARN] PyYAML not installed. Using minimal parser for key: value lines.")
        out: Dict[str, Any] = {}
        for ln in path.read_text().splitlines():
            if ":" in ln and not ln.strip().startswith("#"):
                k, v = ln.split(":", 1)
                out[k.strip()] = v.strip()
        return out
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {})


# -------- helpers --------

def _ppy(bar: str) -> int:
    b = (bar or "1d").lower().strip()
    if b == "1d":
        return 252
    if b == "1h":
        return int(252 * 6.5)
    return 252


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _dump_df(obj: pd.DataFrame | pd.Series, path: Path) -> None:
    _ensure_parent(path)
    if isinstance(obj, pd.Series):
        obj.to_frame(obj.name or "value").to_parquet(path)
    else:
        obj.to_parquet(path)
    print(f"[OK] wrote {path.relative_to(ROOT)} ({len(obj):,} rows)")


def _dump_json(obj: Dict[str, Any], path: Path) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"[OK] wrote {path.relative_to(ROOT)}")


def _calc_perf_summary(returns: pd.Series, bar: str) -> Dict[str, Any]:
    if returns.empty:
        return {}
    re = returns.dropna()
    ppy = _ppy(bar)
    # CAGR
    cum = (1 + re).prod()
    years = max(1e-9, len(re) / ppy)
    cagr = cum ** (1 / years) - 1
    # Vol & Sharpe (rf=0)
    vol = re.std() * math.sqrt(ppy)
    sharpe = (re.mean() * ppy) / (vol + 1e-12)
    # Max DD
    equity = (1 + re).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1).min()
    return {
        "start": str(re.index.min()),
        "end": str(re.index.max()),
        "bars": int(len(re)),
        "cagr": float(cagr),
        "vol_ann": float(vol),
        "sharpe": float(sharpe),
        "max_dd": float(dd),
    }


def _calc_drawdown(returns: pd.Series) -> pd.DataFrame:
    re = returns.fillna(0.0)
    eq = (1 + re).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    return pd.DataFrame({"equity": eq, "peak": peak, "drawdown": dd})


def _rolling(series: pd.Series, window: int, fn: str) -> pd.Series:
    if series.empty:
        return series
    if fn == "vol":
        return series.rolling(window).std()
    if fn == "sharpe":
        mu = series.rolling(window).mean()
        sd = series.rolling(window).std()
        return mu / (sd + 1e-12)
    return series


def _histogram(series: pd.Series, bins: int = 50) -> pd.DataFrame:
    s = series.dropna()
    if s.empty:
        return pd.DataFrame({"bin_left": [], "bin_right": [], "freq": []})
    hist, edges = np.histogram(s.values, bins=bins)
    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "freq": hist,
    })


# -------- loaders for processed artifacts --------

def _load_if_exists(name: str) -> Optional[pd.DataFrame]:
    p = PROCESSED / f"{name}.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return pd.read_csv(p.with_suffix(".csv")) if p.with_suffix(".csv").exists() else None
    return None


# -------- main routine --------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild dashboard data artifacts")
    p.add_argument("--symbols", type=str, default=None, help="Comma-separated tickers (overrides config)")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--bar", type=str, default=None, choices=["1d", "1h"]) 
    p.add_argument("--no-build", action="store_true", help="Skip running pipeline; only reshape existing data")
    p.add_argument("--config", type=str, default=str(cfg_yaml), help="Path to YAML config")
    return p.parse_args(argv)


def load_cfg(ns: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "universe": {"symbols": ["SPY", "TLT"]},
        "dates": {"start": "2013-01-01", "end": "2025-01-01", "bar": "1d"},
    }
    path = Path(ns.config) if ns and ns.config else cfg_yaml
    if path.exists():
        try:
            loaded = _read_yaml(path)
            defaults.update(loaded or {})
        except Exception as exc:
            print(f"[WARN] Failed reading {path}: {exc}")
    return defaults


def maybe_run_pipeline(ns: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if ns.no_build:
        print("[DASH] --no-build set; skipping pipeline run.")
        return {}
    if _build_all is None or not hasattr(_build_all, "RunConfig") or not hasattr(_build_all, "run"):
        print("[DASH] build_all not available; skipping pipeline run.")
        return {}
    symbols = (ns.symbols.split(",") if ns.symbols else (cfg.get("universe", {}).get("symbols") or ["SPY"]))
    start   = ns.start or (cfg.get("dates", {}).get("start") or "2013-01-01")
    end     = ns.end   or (cfg.get("dates", {}).get("end") or "2025-01-01")
    bar     = ns.bar   or (cfg.get("dates", {}).get("bar") or "1d")

    rcfg = _build_all.RunConfig(symbols=[s.strip().upper() for s in symbols], start=start, end=end, bar=bar)  # type: ignore[attr-defined]
    print(f"[DASH] Running pipeline for {rcfg.symbols} {rcfg.start}->{rcfg.end} {rcfg.bar}")
    return _build_all.run(rcfg)  # type: ignore[call-arg]


def reshape_to_dashboard(ns: argparse.Namespace, artifacts: Dict[str, Any]) -> None:
    bar = ns.bar or (artifacts.get("bar") if isinstance(artifacts, dict) else None) or "1d"

    # Prefer fresh artifacts; fallback to processed files
    perf = artifacts.get("perf") if isinstance(artifacts, dict) else None
    returns = perf.get("returns") if isinstance(perf, dict) else None
    equity  = perf.get("equity") if isinstance(perf, dict) else None

    if not isinstance(returns, pd.Series):
        rt = _load_if_exists("returns")
        returns = rt.iloc[:, 0] if isinstance(rt, pd.DataFrame) and not rt.empty else pd.Series(dtype=float)
    if not isinstance(equity, pd.Series):
        eq = _load_if_exists("equity")
        equity = eq.iloc[:, 0] if isinstance(eq, pd.DataFrame) and not eq.empty else pd.Series(dtype=float)

    # Derived metrics
    if isinstance(returns, pd.Series) and not returns.empty:
        _dump_df(returns.rename("return"), DASH_DIR / "returns.parquet")
        _dump_df(equity.rename("equity"), DASH_DIR / "equity.parquet") if isinstance(equity, pd.Series) and not equity.empty else None

        summary = _calc_perf_summary(returns, bar)
        _dump_json(summary, DASH_DIR / "perf_summary.json")

        dd = _calc_drawdown(returns)
        _dump_df(dd, DASH_DIR / "drawdown.parquet")

        # 21-day rolling for daily, 130 for ~6 months
        win_vol = 21 if bar == "1d" else 130
        win_shp = 63 if bar == "1d" else 390
        _dump_df(_rolling(returns, win_vol, "vol").rename("rolling_vol"), DASH_DIR / "rolling_vol.parquet")
        _dump_df(_rolling(returns, win_shp, "sharpe").rename("rolling_sharpe"), DASH_DIR / "rolling_sharpe.parquet")

        hist = _histogram(returns, bins=60)
        _dump_df(hist, DASH_DIR / "returns_hist.parquet")
    else:
        print("[WARN] No returns found; core dashboard charts will be empty.")

    # Optional tables
    factors = artifacts.get("factors") if isinstance(artifacts, dict) else None
    if not isinstance(factors, pd.DataFrame):
        factors = _load_if_exists("factors")
    if isinstance(factors, pd.DataFrame) and not factors.empty:
        _dump_df(factors, DASH_DIR / "factors.parquet")

    ranks = artifacts.get("ranks") if isinstance(artifacts, dict) else None
    if not isinstance(ranks, pd.DataFrame):
        ranks = _load_if_exists("ranks")
    if isinstance(ranks, pd.DataFrame) and not ranks.empty:
        _dump_df(ranks, DASH_DIR / "ranks.parquet")

    weights = artifacts.get("weights") if isinstance(artifacts, dict) else None
    if not isinstance(weights, pd.DataFrame):
        weights = _load_if_exists("weights")
    if isinstance(weights, pd.DataFrame) and not weights.empty:
        _dump_df(weights, DASH_DIR / "weights.parquet")

    # Analytics (optional)
    corr = artifacts.get("correlations") if isinstance(artifacts, dict) else None
    if isinstance(corr, pd.DataFrame) and not corr.empty:
        _dump_df(corr, DASH_DIR / "correlations.parquet")

    regs = artifacts.get("regressions") if isinstance(artifacts, dict) else None
    if isinstance(regs, pd.DataFrame) and not regs.empty:
        _dump_df(regs, DASH_DIR / "regressions.parquet")


# -------- entrypoint --------

def main(argv: Optional[Iterable[str]] = None) -> int:
    ns = parse_args(argv)
    cfg = load_cfg(ns)

    artifacts = maybe_run_pipeline(ns, cfg)
    reshape_to_dashboard(ns, artifacts)
    print("[DASH] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
