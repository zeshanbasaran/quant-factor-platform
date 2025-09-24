"""
build_all.py
-------------
One-click pipeline runner for **quant-factor-platform**.

Stages
------
1) Load config (configs/default.yaml) with sensible fallbacks
2) ETL: prices + fundamentals → data/processed/
3) Factors: value / momentum / quality → factor scores
4) Portfolio: rank assignment → weights → performance metrics
5) Analytics: correlations, regressions, tearsheet
6) Persist artifacts (parquet/SQL) + console summary

Notes
-----
- This script is *defensive*: it will gracefully skip stages if a module
  or function is missing, and print a clear warning instead of crashing.
- You can override config via CLI flags (symbols, dates, bar, etc.).
- Designed to be run from repo root:

    python -m src.pilots.build_all \
        --symbols SPY,QQQ \
        --start 2015-01-01 --end 2025-01-01 --bar 1d

"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# ----------------------------
# Repo-relative imports (defensive)
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Soft imports with fallbacks

def _soft_import(path: str):
    try:
        module = __import__(path, fromlist=["*"])
        return module
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not import '{path}': {exc}")
        return None

cfg_yaml = ROOT / "configs" / "default.yaml"

# Core & data
_universe = _soft_import("src.core.universe")
_calendar = _soft_import("src.core.calendar")
_utils    = _soft_import("src.core.utils") or _soft_import("src.core.utils")
_ing_px   = _soft_import("src.data.ingest_prices")
_ing_fun  = _soft_import("src.data.ingest_fundamentals")
_store    = _soft_import("src.data.store")

# Factors
_f_value  = _soft_import("src.factors.value")
_f_mom    = _soft_import("src.factors.momentum")
_f_qual   = _soft_import("src.factors.quality")
_f_engine = _soft_import("src.factors.engine")

# Portfolio
_rank     = _soft_import("src.portfolio.rank_assign")
_weights  = _soft_import("src.portfolio.weights")
_perf     = _soft_import("src.portfolio.performance")

# Analytics
_corr     = _soft_import("src.analytics.correlations")
_regs     = _soft_import("src.analytics.regressions")
_tear     = _soft_import("src.analytics.tearsheet")

# DB (optional)
_db_models = _soft_import("src.db.models")
_db_io     = _soft_import("src.db.io")


# ----------------------------
# Small helpers
# ----------------------------

def _ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        print("[WARN] PyYAML not installed. Using minimal parser for key: value lines.")
        # minimal fallback: key: value per line
        out: Dict[str, Any] = {}
        for ln in path.read_text().splitlines():
            if ":" in ln and not ln.strip().startswith("#"):
                k, v = ln.split(":", 1)
                out[k.strip()] = v.strip()
        return out
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _periods_per_year_from_bar(bar: str) -> int:
    b = (bar or "").lower().strip()
    if b == "1d":
        return 252
    if b == "1h":
        return int(252 * 6.5)
    return 252


@dataclass
class RunConfig:
    symbols: List[str]
    start: str
    end: str
    bar: str = "1d"
    universe_cfg: Dict[str, Any] | None = None
    rebal_freq: str | None = None
    risk: Dict[str, float] | None = None
    out_dir: Path = ROOT / "data" / "processed"
    reports_dir: Path = ROOT / "reports"
    cache_dir: Path = ROOT / "data"
    sql_url: Optional[str] = None


# ----------------------------
# Stage runners (each returns dict of artifacts)
# ----------------------------

def stage_universe(cfg: RunConfig) -> Dict[str, Any]:
    if _universe and hasattr(_universe, "build_universe"):
        uni = _universe.build_universe(cfg.universe_cfg or {})
    else:
        # Fallback: use symbols from cfg
        uni = pd.DataFrame({"symbol": cfg.symbols})
    return {"universe": uni}


def stage_prices(cfg: RunConfig) -> Dict[str, Any]:
    prices: Dict[str, pd.DataFrame] = {}
    if _ing_px and hasattr(_ing_px, "ingest"):
        for sym in cfg.symbols:
            prices[sym] = _ing_px.ingest(sym, cfg.start, cfg.end, cfg.bar, cache_dir=str(cfg.cache_dir))
    elif _ing_px and hasattr(_ing_px, "run"):
        # run() may accept list of symbols
        out = _ing_px.run(cfg.symbols, cfg.start, cfg.end, cfg.bar)
        if isinstance(out, dict):
            prices.update(out)
    else:
        print("[WARN] Missing ingest_prices.*. Skipping price ETL.")
    return {"prices": prices}


def stage_fundamentals(cfg: RunConfig) -> Dict[str, Any]:
    fun: Dict[str, pd.DataFrame] = {}
    if _ing_fun and hasattr(_ing_fun, "ingest"):
        for sym in cfg.symbols:
            fun[sym] = _ing_fun.ingest(sym, cfg.start, cfg.end)
    elif _ing_fun and hasattr(_ing_fun, "run"):
        out = _ing_fun.run(cfg.symbols, cfg.start, cfg.end)
        if isinstance(out, dict):
            fun.update(out)
    else:
        print("[INFO] No fundamentals ETL found. Continuing without fundamentals.")
    return {"fundamentals": fun}


def stage_factors(cfg: RunConfig, prices: Dict[str, pd.DataFrame], fundamentals: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    # Compose per-symbol factor tables and a combined long table
    frames: List[pd.DataFrame] = []
    for sym, df in prices.items():
        if df is None or df.empty:
            continue
        fparts: Dict[str, pd.Series] = {}
        # Value
        if _f_value and hasattr(_f_value, "compute"):
            try:
                fparts["value"] = _f_value.compute(df, fundamentals.get(sym))
            except Exception as exc:
                print(f"[WARN] value.compute failed for {sym}: {exc}")
        # Momentum
        if _f_mom and hasattr(_f_mom, "compute"):
            try:
                fparts["momentum"] = _f_mom.compute(df)
            except Exception as exc:
                print(f"[WARN] momentum.compute failed for {sym}: {exc}")
        # Quality
        if _f_qual and hasattr(_f_qual, "compute"):
            try:
                fparts["quality"] = _f_qual.compute(df, fundamentals.get(sym))
            except Exception as exc:
                print(f"[WARN] quality.compute failed for {sym}: {exc}")

        if not fparts:
            print(f"[WARN] No factor components computed for {sym}; skipping.")
            continue
        fdf = pd.DataFrame(fparts)
        fdf["symbol"] = sym
        frames.append(fdf)

    if not frames:
        print("[WARN] No factor data assembled.")
        return {"factors": pd.DataFrame()}

    factors = pd.concat(frames).sort_index()

    # Optional orchestration/score combine
    if _f_engine and hasattr(_f_engine, "score"):
        try:
            combined = _f_engine.score(factors)
            factors = combined
        except Exception as exc:
            print(f"[WARN] engine.score failed; keeping raw factors: {exc}")

    return {"factors": factors}


def stage_portfolio(cfg: RunConfig, factors: pd.DataFrame, prices: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    if factors is None or factors.empty:
        print("[WARN] Empty factors; skipping portfolio stage.")
        return {"ranks": pd.DataFrame(), "weights": pd.DataFrame(), "perf": {}}

    # Rank assignment
    if _rank and hasattr(_rank, "assign"):
        ranks = _rank.assign(factors, method="quintile", by_date=True)
    else:
        # simple rank: higher score → higher rank
        ranks = factors.copy()
        if "score" in ranks.columns:
            ranks["rank"] = ranks.groupby(level=0)["score"].rank(pct=True)
        else:
            cols = [c for c in ["value", "momentum", "quality", "score"] if c in ranks.columns]
            ranks["rank"] = ranks.groupby(level=0)[cols].mean(axis=1).rank(pct=True)

    # Weights
    if _weights and hasattr(_weights, "make"):
        w = _weights.make(ranks, scheme="equal_weight_long_short", top=0.2, bottom=0.2)
    else:
        # default: go long top 20%, short bottom 20%, equal-weight by symbol
        def _lw(x: pd.Series) -> float:
            return 1.0 / max(1, x.size)
        wlist = []
        for dt, grp in ranks.groupby(level=0):
            g = grp.reset_index()
            g = g.sort_values("rank")
            n = len(g)
            long_cut = int(max(1, round(0.8 * n)))
            top = g.iloc[long_cut:]
            bot = g.iloc[: max(1, n - long_cut)]
            wl = pd.Series(_lw(top["symbol"]) if not top.empty else 0.0, index=top["symbol"])  # type: ignore
            ws = -pd.Series(_lw(bot["symbol"]) if not bot.empty else 0.0, index=bot["symbol"])  # type: ignore
            wlist.append(pd.DataFrame({"dt": dt, "symbol": wl.index, "w": wl.values}))
            wlist.append(pd.DataFrame({"dt": dt, "symbol": ws.index, "w": ws.values}))
        w = pd.concat(wlist).pivot_table(index="dt", columns="symbol", values="w", fill_value=0.0)

    # Performance
    if _perf and hasattr(_perf, "backtest"):
        perf = _perf.backtest(w, prices, init_cash=100_000, rebal_freq="M")
    else:
        # Minimal backtest: daily rebalancing using adj_close returns
        # Align returns matrix
        px = {sym: df["adj_close"].pct_change().rename(sym) for sym, df in prices.items() if "adj_close" in df}
        rets = pd.concat(px.values(), axis=1).fillna(0.0)
        w_aligned = w.reindex(rets.index).fillna(method="ffill").fillna(0.0)
        port_ret = (w_aligned * rets).sum(axis=1)
        equity = (1 + port_ret).cumprod() * 100_000
        perf = {"returns": port_ret, "equity": equity}

    return {"ranks": ranks, "weights": w, "perf": perf}


def stage_analytics(cfg: RunConfig, perf: Dict[str, Any], factors: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Correlations
    if _corr and hasattr(_corr, "build"):
        try:
            out["correlations"] = _corr.build(factors)
        except Exception as exc:
            print(f"[WARN] correlations.build failed: {exc}")

    # Regressions
    if _regs and hasattr(_regs, "run"):
        try:
            out["regressions"] = _regs.run(perf.get("returns"), factors)
        except Exception as exc:
            print(f"[WARN] regressions.run failed: {exc}")

    # Tearsheets
    if _tear and hasattr(_tear, "save"):
        try:
            out["tearsheet"] = _tear.save(perf, factors, cfg.reports_dir)
        except Exception as exc:
            print(f"[WARN] tearsheet.save failed: {exc}")

    return out


# ----------------------------
# Persistence helpers
# ----------------------------

def persist_artifacts(cfg: RunConfig, art: Dict[str, Any]) -> None:
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _dump_df(name: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        p = _ensure_dir(out_dir / f"{name}.parquet")
        try:
            df.to_parquet(p, index=True)
        except Exception:
            df.to_csv(str(p) + ".csv")

    # Dump what we have
    if "factors" in art:
        _dump_df("factors", art["factors"])
    if "ranks" in art:
        _dump_df("ranks", art["ranks"])
    if "weights" in art:
        w = art["weights"]
        if isinstance(w, pd.DataFrame):
            _dump_df("weights", w)
    if "perf" in art and isinstance(art["perf"], dict):
        eq = art["perf"].get("equity")
        re = art["perf"].get("returns")
        if isinstance(eq, pd.Series):
            _dump_df("equity", eq.to_frame("equity"))
        if isinstance(re, pd.Series):
            _dump_df("returns", re.to_frame("return"))


# ----------------------------
# Main
# ----------------------------

def run(cfg: RunConfig) -> Dict[str, Any]:
    print("[BUILD] Universe…")
    uni = stage_universe(cfg)

    print("[BUILD] Prices ETL…")
    px = stage_prices(cfg)

    print("[BUILD] Fundamentals ETL…")
    fn = stage_fundamentals(cfg)

    print("[BUILD] Factors…")
    fac = stage_factors(cfg, prices=px.get("prices", {}), fundamentals=fn.get("fundamentals", {}))

    print("[BUILD] Portfolio (ranks → weights → performance)…")
    port = stage_portfolio(cfg, factors=fac.get("factors", pd.DataFrame()), prices=px.get("prices", {}))

    print("[BUILD] Analytics…")
    an = stage_analytics(cfg, perf=port.get("perf", {}), factors=fac.get("factors", pd.DataFrame()))

    # Persist core artifacts
    persist_artifacts(cfg, {**fac, **port})

    out = {**uni, **px, **fn, **fac, **port, **an}
    return out


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click pipeline runner")
    p.add_argument("--symbols", type=str, default=None, help="Comma-separated tickers (overrides config)")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--bar", type=str, default=None, choices=["1d", "1h"]) 
    p.add_argument("--sql", type=str, default=None, help="SQLAlchemy DB URL override")
    p.add_argument("--config", type=str, default=str(cfg_yaml), help="Path to YAML config")
    return p.parse_args(argv)


def load_config(ns: argparse.Namespace) -> RunConfig:
    # Defaults
    defaults = {
        "universe": {"symbols": ["SPY", "TLT"]},
        "dates": {"start": "2013-01-01", "end": "2025-01-01", "bar": "1d"},
        "portfolio": {"rebal_freq": "M"},
        "risk": {"max_dd": 0.2, "var_95": 0.025, "vol_ann": 0.25},
        "storage": {"sql_url": None},
    }
    cfg = defaults
    path = Path(ns.config) if ns and ns.config else cfg_yaml
    if path.exists():
        try:
            loaded = _read_yaml(path)
            # Merge shallowly, tolerate flexible schemas
            cfg.update(loaded or {})
        except Exception as exc:
            print(f"[WARN] Failed reading {path}: {exc}")

    symbols = (ns.symbols.split(",") if ns.symbols else (cfg.get("universe", {}).get("symbols") or ["SPY"]))
    start   = ns.start or (cfg.get("dates", {}).get("start") or "2013-01-01")
    end     = ns.end   or (cfg.get("dates", {}).get("end") or "2025-01-01")
    bar     = ns.bar   or (cfg.get("dates", {}).get("bar") or "1d")

    return RunConfig(
        symbols=[s.strip().upper() for s in symbols],
        start=start,
        end=end,
        bar=bar,
        universe_cfg=cfg.get("universe"),
        rebal_freq=(cfg.get("portfolio", {}) or {}).get("rebal_freq"),
        risk=cfg.get("risk"),
        sql_url=ns.sql or (cfg.get("storage", {}) or {}).get("sql_url"),
    )


if __name__ == "__main__":
    ns = parse_args()
    rcfg = load_config(ns)
    print("[CONFIG]", json.dumps({
        "symbols": rcfg.symbols,
        "start": rcfg.start,
        "end": rcfg.end,
        "bar": rcfg.bar,
    }, indent=2))
    artifacts = run(rcfg)

    # Console summary
    eq = artifacts.get("perf", {}).get("equity") if isinstance(artifacts.get("perf"), dict) else None
    if isinstance(eq, pd.Series) and not eq.empty:
        ret = eq.pct_change().fillna(0.0)
        ppy = _periods_per_year_from_bar(rcfg.bar)
        ann = (1 + ret).prod() ** (ppy / max(1, len(ret))) - 1
        dd = (eq / eq.cummax() - 1).min()
        print(f"[SUMMARY] {rcfg.start}..{rcfg.end} | AnnReturn={ann:.2%} | MaxDD={dd:.2%} | Bars={len(eq):,}")
    else:
        print("[SUMMARY] No equity series computed.")
