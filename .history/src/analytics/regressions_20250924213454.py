"""
regressions.py
---------------
Factor regression utilities:

1) Time-series OLS with Newey–West (HAC) standard errors
   - time_series_regression(...)
   - rolling_ts_regression(...)

2) Cross-sectional (Fama–MacBeth) regressions
   - fama_macbeth_cs(...)

Features
--------
- Optional intercept
- Newey–West HAC for TS (alpha/betas) and for FM (avg betas across time)
- Winsorization of regressors (per-date for CS)
- Optional industry/sector dummies for CS via `group_col`
- Robust handling of collinearity (pseudo-inverse) and small samples

Inputs (tidy)
-------------
Time-series:
  returns_df: ['date', y_col, X1, X2, ...]
Cross-sectional (FM):
  cs_df: ['date','ticker', 'fwd_ret', <factor cols>, (optional) group_col]

Notes
-----
- FM procedure:
    For each date t: r_i,t = a_t + b_t' * x_i,t + e_i,t (cross-section)
    Average coefficients: b_hat = mean_t(b_t)
    Newey–West across time on the sequence {b_t}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers / math
# ---------------------------------------------------------------------

def _normalize(df: pd.DataFrame, required_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    miss = [c for c in required_cols if c not in out.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}. Have: {list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def _winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if s.size == 0:
        return s
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)


def _add_intercept(X: np.ndarray, add: bool) -> np.ndarray:
    if not add:
        return X
    n = X.shape[0]
    return np.c_[np.ones(n), X]


def _ols_fit(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Return (beta, resid, s2, X) using pseudo-inverse for stability.
    """
    # Handle all-NaN or empty
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y2 = y[mask]
    X2 = X[mask, :]
    if X2.shape[0] == 0 or X2.shape[1] == 0:
        return np.full(X.shape[1], np.nan), np.full_like(y, np.nan, dtype=float), np.nan, X
    beta = np.linalg.pinv(X2) @ y2
    # residuals aligned to original index
    resid = np.full_like(y, np.nan, dtype=float)
    resid[mask] = y2 - (X2 @ beta)
    # homoskedastic variance estimate (unused directly when HAC is applied)
    s2 = np.nan
    return beta, resid, s2, X2


def _newey_west_cov(resid: np.ndarray, X: np.ndarray, lags: int) -> np.ndarray:
    """
    HAC (Newey–West) covariance for time-series OLS:
      V = (X'X)^(-1) * S * (X'X)^(-1)
      S = sum_{k=-L..L} w_k * Gamma_k, with Bartlett weights.
    resid, X must be aligned (non-NaN rows only).
    """
    # filter rows with finite residuals and features
    mask = np.isfinite(resid) & np.all(np.isfinite(X), axis=1)
    u = resid[mask]
    Z = X[mask, :]
    n, k = Z.shape
    if n == 0 or k == 0:
        return np.full((k, k), np.nan)

    XtX_inv = np.linalg.pinv(Z.T @ Z)
    # meat of the sandwich
    # Gamma_0
    S = (Z * u[:, None]).T @ (Z * u[:, None])
    # serial covariances
    L = max(int(lags), 0)
    for h in range(1, L + 1):
        w = 1.0 - h / (L + 1.0)  # Bartlett weight
        # u_t * u_{t-h}
        uh = u[h:]
        u0 = u[:-h]
        Z0 = Z[:-h, :]
        Zh = Z[h:, :]
        Gamma_h = (Z0.T @ (u0[:, None] * Zh))  # sum_t Z_{t-h} u_{t-h} * Z_t u_t
        # add both +h and -h
        S += w * (Gamma_h + Gamma_h.T)

    V = XtX_inv @ S @ XtX_inv
    return V


# ---------------------------------------------------------------------
# 1) Time-series regression
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TSConfig:
    add_intercept: bool = True
    hac_lags: int = 4                 # e.g., monthly ≈ 3, daily ≈ 5–10 depending on autocorr
    dropna: Literal["any","all","none"] = "any"  # how to treat NaNs across columns


def time_series_regression(
    returns_df: pd.DataFrame,
    *,
    y_col: str,
    X_cols: Iterable[str],
    cfg: TSConfig = TSConfig(),
) -> dict:
    """
    OLS y_t = a + b' x_t + e_t  (with Newey–West HAC SEs)

    Returns dict:
      - params: pd.Series (alpha first if add_intercept=True)
      - stderr, tstat: pd.Series
      - r2, n_obs
    """
    req = ("date", y_col, *X_cols)
    df = _normalize(returns_df, req).set_index("date")
    Y = df[y_col].astype(float)
    X = df[list(X_cols)].astype(float)

    if cfg.dropna == "any":
        Z = pd.concat([Y, X], axis=1).dropna(how="any")
        Y = Z.iloc[:, 0]
        X = Z.iloc[:, 1:]
    elif cfg.dropna == "all":
        Z = pd.concat([Y, X], axis=1).dropna(how="all")
        Y = Z.iloc[:, 0]
        X = Z.iloc[:, 1:]
    # 'none' keeps NaNs; OLS fitter handles row-wise mask

    y = Y.values
    Xn = _add_intercept(X.values, cfg.add_intercept)

    beta, resid, _, X_used = _ols_fit(y, Xn)
    # R2 (using rows actually used)
    mask = np.isfinite(resid)
    if mask.sum() > 1:
        y_used = y[mask]
        yhat = (Xn[mask, :] @ beta)
        ssr = np.nansum((y_used - yhat) ** 2)
        sst = np.nansum((y_used - np.nanmean(y_used)) ** 2)
        r2 = 1.0 - (ssr / sst) if sst > 0 else np.nan
    else:
        r2 = np.nan

    # HAC covariance + SE
    V = _newey_west_cov(resid, Xn, cfg.hac_lags)
    se = np.sqrt(np.diag(V))
    # Names
    names = (["alpha"] if cfg.add_intercept else []) + list(X.columns)
    params = pd.Series(beta[:len(names)], index=names)
    stderr = pd.Series(se[:len(names)], index=names)
    tstat = params / stderr

    return {
        "params": params,
        "stderr": stderr,
        "tstat": tstat,
        "r2": float(r2),
        "n_obs": int(mask.sum()),
        "hac_lags": cfg.hac_lags,
        "add_intercept": cfg.add_intercept,
    }


def rolling_ts_regression(
    returns_df: pd.DataFrame,
    *,
    y_col: str,
    X_cols: Iterable[str],
    window: int = 63,
    cfg: TSConfig = TSConfig(),
) -> pd.DataFrame:
    """
    Rolling TS regression; returns a DataFrame with betas (and alpha if configured)
    indexed by window-end date.
    """
    req = ("date", y_col, *X_cols)
    df = _normalize(returns_df, req).set_index("date").astype(float)
    cols = (["alpha"] if cfg.add_intercept else []) + list(X_cols)
    out = []

    dates = df.index.unique()
    for i in range(window - 1, len(dates)):
        sl = df.loc[dates[i - window + 1] : dates[i]]
        res = time_series_regression(sl.reset_index(), y_col=y_col, X_cols=X_cols, cfg=cfg)
        row = {"date": dates[i]}
        for k in cols:
            row[k] = res["params"].get(k, np.nan)
        out.append(row)

    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------
# 2) Fama–MacBeth cross-sectional regression
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FMConfig:
    factor_cols: Tuple[str, ...]
    fwd_ret_col: str = "fwd_ret"     # forward return over horizon
    add_intercept: bool = True
    per_date_winsor: Tuple[float, float] = (0.01, 0.99)   # on factors (not on returns)
    min_names_per_date: int = 10
    hac_lags_time: int = 4          # NW across time on beta_t sequences
    group_col: Optional[str] = None # industry/sector dummies (as-of labels in cs_df)


def _make_group_dummies(cs_block: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Create group dummies (omit one to avoid the dummy variable trap).
    """
    d = pd.get_dummies(cs_block[group_col].astype("category"), prefix=group_col)
    if d.shape[1] > 0:
        d = d.iloc[:, 1:]  # drop first category
    return d


def fama_macbeth_cs(
    cs_df: pd.DataFrame,
    *,
    cfg: FMConfig,
) -> dict:
    """
    Run Fama–MacBeth:
      For each date t: regress fwd_ret on factors (and optional group dummies).
      Then average betas across t and compute NW HAC SE over time.

    Returns dict:
      - params: pd.Series (avg betas; includes 'alpha' if add_intercept)
      - stderr, tstat: pd.Series (HAC across time)
      - by_date: DataFrame of daily coefficients (rows dates, cols params)
      - n_dates_used, mean_cs_obs
    """
    req = ("date", "ticker", cfg.fwd_ret_col, *cfg.factor_cols)
    df = _normalize(cs_df, req)

    # per-date winsorization for regressors
    lo, hi = cfg.per_date_winsor

    betas: List[pd.Series] = []
    dates_used: List[pd.Timestamp] = []

    # Track per-date coefficients in a consistent column order
    base_cols = list(cfg.factor_cols)
    if cfg.group_col and (cfg.group_col in df.columns):
        # we will add dummies per date; but column set may vary by date.
        # We'll collect union of names encountered and reindex each beta series later.
        pass

    # Container for all names (factors + dummies + intercept)
    all_names: set = set((["alpha"] if cfg.add_intercept else []) + list(cfg.factor_cols))

    # Per-date regressions
    for d, block in df.groupby("date"):
        B = block.copy()
        # Enough names?
        if B["ticker"].nunique() < cfg.min_names_per_date:
            continue

        # Winsorize factor columns (per-date)
        for c in cfg.factor_cols:
            B[c] = _winsorize(B[c].astype(float), lo, hi)

        # Build design matrix
        X_parts = [B[list(cfg.factor_cols)].astype(float)]

        # Optional group dummies
        if cfg.group_col and (cfg.group_col in B.columns):
            D = _make_group_dummies(B, cfg.group_col)
            if D.shape[1] > 0:
                X_parts.append(D.astype(float))
                all_names.update(list(D.columns))

        X_mat = np.column_stack([p.values for p in X_parts]) if len(X_parts) > 1 else X_parts[0].values
        y = B[cfg.fwd_ret_col].astype(float).values

        # Add intercept if requested
        add_int = cfg.add_intercept
        Xn = _add_intercept(X_mat, add_int)

        # Fit cross-section
        beta_t, resid_t, _, X_used = _ols_fit(y, Xn)

        # Name alignment
        names = (["alpha"] if add_int else []) + list(cfg.factor_cols)
        if len(X_parts) > 1:
            names += list(D.columns)

        # Store
        s = pd.Series(beta_t[:len(names)], index=names, dtype=float)
        betas.append(s)
        dates_used.append(pd.Timestamp(d))
        all_names.update(names)

    if len(betas) == 0:
        return {
            "params": pd.Series(dtype=float),
            "stderr": pd.Series(dtype=float),
            "tstat": pd.Series(dtype=float),
            "by_date": pd.DataFrame(),
            "n_dates_used": 0,
            "mean_cs_obs": 0.0,
        }

    # Align all daily beta series to union of names
    all_names = list(all_names)
    beta_df = pd.DataFrame(betas, index=dates_used).sort_index()
    beta_df = beta_df.reindex(columns=all_names)

    # Average coefficients across time
    params = beta_df.mean(axis=0, skipna=True)

    # HAC across time on each coefficient series
    stderr = {}
    tstat = {}
    for col in beta_df.columns:
        series = beta_df[col].values.astype(float)
        # Build TS regression of beta_t on constant (to get HAC SE of the mean)
        y = series
        X = np.ones((len(series), 1))
        _, resid, _, _ = _ols_fit(y, X)
        V = _newey_west_cov(resid, X, cfg.hac_lags_time)
        se = np.sqrt(V[0, 0]) if np.isfinite(V[0, 0]) else np.nan
        stderr[col] = se
        tstat[col] = params[col] / se if (se and np.isfinite(se) and se > 0) else np.nan

    # Summary
    # Approx mean cross-sectional obs used (rough; uses total tickers per used date)
    cs_obs = df[df["date"].isin(beta_df.index)].groupby("date")["ticker"].nunique()
    mean_cs = float(cs_obs.mean()) if not cs_obs.empty else 0.0

    return {
        "params": params,
        "stderr": pd.Series(stderr).reindex(params.index),
        "tstat": pd.Series(tstat).reindex(params.index),
        "by_date": beta_df,
        "n_dates_used": int(beta_df.shape[0]),
        "mean_cs_obs": mean_cs,
        "hac_lags_time": cfg.hac_lags_time,
        "add_intercept": cfg.add_intercept,
        "winsor": cfg.per_date_winsor,
        "group_col": cfg.group_col,
    }

# ---------------------------------------------------------------------
# Back-compat helpers expected by tests
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd

def ts_regression(y: pd.Series, X: pd.DataFrame, add_const: bool = True) -> dict:
    """
    Legacy wrapper around time_series_regression(y_t ~ X_t).
    Returns keys expected by tests: params, tstats, resid, rsq.
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Align and drop rows with any NaNs (test-friendly)
    df = pd.concat([y.rename("y"), X], axis=1).dropna(how="any")
    if df.empty:
        return {
            "params": pd.Series(dtype=float),
            "tstats": pd.Series(dtype=float),
            "resid": pd.Series(dtype=float),
            "rsq": np.nan,
        }

    returns_df = df.reset_index().rename(columns={"index": "date"})
    res = time_series_regression(
        returns_df,
        y_col="y",
        X_cols=list(X.columns),
        cfg=TSConfig(add_intercept=add_const, dropna="any"),
    )

    # Recompute residuals on the used rows for the exact output contract
    Xn = returns_df[list(X.columns)].to_numpy(dtype=float)
    if add_const:
        Xn = np.c_[np.ones(len(Xn)), Xn]
        beta = np.concatenate([[res["params"].get("alpha", np.nan)], res["params"].drop(labels=["alpha"], errors="ignore").to_numpy(dtype=float)])
    else:
        beta = res["params"].to_numpy(dtype=float)
    yhat = Xn @ beta
    resid = returns_df["y"].to_numpy(dtype=float) - yhat
    resid_series = pd.Series(resid, index=returns_df["date"], name="resid")

    # Rename 'alpha' -> 'const', and 'tstat' -> 'tstats'
    params = res["params"].rename(index={"alpha": "const"})
    tstats = res["tstat"].rename(index={"alpha": "const"})

    return {
        "params": params,
        "tstats": tstats,
        "resid": resid_series,
        "rsq": float(res["r2"]),
    }


def fama_macbeth(scores: pd.DataFrame, fwd_returns: pd.DataFrame) -> dict:
    """
    Legacy wrapper to run a 1-factor Fama–MacBeth on cross-sectional 'scores'
    vs forward returns. Returns keys the tests expect.
      - scores: DataFrame (rows=dates, cols=tickers)
      - fwd_returns: same shape/index/columns as scores
    """
    if not isinstance(scores.index, pd.DatetimeIndex):
        scores.index = pd.to_datetime(scores.index, errors="coerce")
    if not isinstance(fwd_returns.index, pd.DatetimeIndex):
        fwd_returns.index = pd.to_datetime(fwd_returns.index, errors="coerce")

    # Long format cross-section
    s_long = scores.stack(dropna=False).rename("score")
    r_long = fwd_returns.stack(dropna=False).rename("fwd_ret")
    cs = pd.concat([s_long, r_long], axis=1).dropna(how="any").reset_index()
    cs = cs.rename(columns={"level_0": "date", "level_1": "ticker"})
    if cs.empty:
        return {
            "beta_mean": np.nan,
            "alpha_mean": np.nan,
            "beta_t": np.nan,
            "alpha_t": np.nan,
            "by_date": pd.DataFrame(),
        }

    res = fama_macbeth_cs(
        cs_df=cs,
        cfg=FMConfig(factor_cols=("score",), fwd_ret_col="fwd_ret", add_intercept=True),
    )

    # Map to the legacy keys
    alpha_mean = float(res["params"].get("alpha", np.nan))
    beta_mean = float(res["params"].get("score", np.nan))
    alpha_t = float(res["tstat"].get("alpha", np.nan))
    beta_t = float(res["tstat"].get("score", np.nan))

    by_date = res["by_date"].copy()
    if "score" in by_date.columns:
        by_date = by_date.rename(columns={"score": "beta"})
    # Ensure at least ['alpha','beta'] exist for the test
    for col in ["alpha", "beta"]:
        if col not in by_date.columns:
            by_date[col] = np.nan
    by_date = by_date[["alpha", "beta"]]

    return {
        "beta_mean": beta_mean,
        "alpha_mean": alpha_mean,
        "beta_t": beta_t,
        "alpha_t": alpha_t,
        "by_date": by_date,
    }
