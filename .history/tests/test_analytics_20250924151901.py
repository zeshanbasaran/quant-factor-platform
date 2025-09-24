# tests/test_analytics.py
import numpy as np
import pandas as pd
import pytest

from src.analytics.correlations import corr_matrix, corr_melt
from src.analytics.regressions import fama_macbeth, ts_regression
from src.analytics.tearsheet import build_factor_tearsheet


# ----------------------------
# Helpers: reproducible toy data
# ----------------------------
rng = np.random.default_rng(0)


def _mk_corr_df(n=200):
    # A ~ N(0,1); B ≈ A + small noise; C independent
    A = rng.standard_normal(n)
    B = A + 0.05 * rng.standard_normal(n)
    C = rng.standard_normal(n)
    df = pd.DataFrame({"A": A, "B": B, "C": C})
    return df


def _mk_fmb_inputs(T=24, N=50, beta_true=0.5, noise=0.05):
    """
    Build panel-like cross-sectional data for Fama-MacBeth:
      r_it = a_t + beta_true * score_it + eps_it
    """
    dates = pd.date_range("2020-01-31", periods=T, freq="M")
    syms = [f"S{i:03d}" for i in range(N)]

    # scores: persistent cross-sectional variation (normalize per-date)
    base_scores = rng.standard_normal((T, N))
    scores = pd.DataFrame(base_scores, index=dates, columns=syms)
    scores = scores.apply(lambda s: (s - s.mean()) / s.std(ddof=0), axis=1)

    # per-date alpha (near zero)
    alphas = 0.01 * rng.standard_normal(T)

    # returns generated from the linear model + noise
    eps = noise * rng.standard_normal((T, N))
    rets = pd.DataFrame(
        alphas[:, None] + beta_true * scores.values + eps,
        index=dates,
        columns=syms,
    )

    return scores, rets, beta_true


def _mk_ts_reg_inputs(T=250, alpha=0.001, beta=1.2, noise=0.01):
    """
    Build time-series CAPM-like inputs:
      y_t = alpha + beta * x_t + eps_t
    """
    idx = pd.bdate_range("2020-01-01", periods=T)
    x = rng.standard_normal(T) * 0.01  # "market" excess return ~1% vol
    eps = noise * rng.standard_normal(T)

    y = alpha + beta * x + eps
    y = pd.Series(y, index=idx, name="port_excess")
    X = pd.DataFrame({"mkt_excess": x}, index=idx)
    return y, X, alpha, beta


# ----------------------------
# CORRELATIONS
# ----------------------------
def test_corr_matrix_and_melt_shapes_and_values():
    df = _mk_corr_df(n=300)
    M = corr_matrix(df, method="pearson")
    assert isinstance(M, pd.DataFrame)
    assert M.shape == (3, 3)
    # symmetry & diagonals
    assert np.allclose(M.values, M.values.T, atol=1e-12)
    assert np.allclose(np.diag(M.values), 1.0, atol=1e-12)

    # A and B should be highly correlated; C roughly independent
    assert M.loc["A", "B"] > 0.95
    assert abs(M.loc["A", "C"]) < 0.2
    assert abs(M.loc["B", "C"]) < 0.2

    melted = corr_melt(df, method="pearson", drop_self=True)
    assert isinstance(melted, pd.DataFrame)
    assert set(melted.columns) >= {"var1", "var2", "corr"}

    # Only unique off-diagonal pairs expected (3 vars -> 3 pairs)
    assert len(melted) == 3
    # One of the pairs should be (A,B) with high corr
    ab_rows = melted[
        (melted["var1"].isin(["A", "B"])) & (melted["var2"].isin(["A", "B"])) & (melted["var1"] != melted["var2"])
    ]
    assert not ab_rows.empty
    assert ab_rows["corr"].abs().max() > 0.95


# ----------------------------
# REGRESSIONS: Fama–MacBeth
# ----------------------------
def test_fama_macbeth_beta_close_to_truth():
    scores, fwd, beta_true = _mk_fmb_inputs(T=36, N=80, beta_true=0.6, noise=0.03)
    out = fama_macbeth(scores, fwd)

    # Contract: returns a dict with aggregated stats and by-date panel
    assert isinstance(out, dict)
    for key in ["beta_mean", "alpha_mean", "beta_t", "alpha_t", "by_date"]:
        assert key in out

    assert isinstance(out["by_date"], pd.DataFrame)
    assert set(["alpha", "beta"]).issubset(out["by_date"].columns)

    # Beta should be close to truth and significantly > 0
    assert pytest.approx(float(out["beta_mean"]), abs=0.05) == beta_true
    assert float(out["beta_t"]) > 5.0  # strong signal in synthetic data


# ----------------------------
# REGRESSIONS: Time-series (CAPM-like)
# ----------------------------
def test_ts_regression_alpha_beta_estimates_and_diagnostics():
    y, X, alpha_true, beta_true = _mk_ts_reg_inputs(T=400, alpha=0.0015, beta=1.1, noise=0.005)
    out = ts_regression(y, X, add_const=True)

    # Contract: dict with params, tstats, resid, rsq
    assert isinstance(out, dict)
    for key in ["params", "tstats", "resid", "rsq"]:
        assert key in out

    assert isinstance(out["params"], pd.Series)
    assert isinstance(out["tstats"], pd.Series)
    assert isinstance(out["resid"], pd.Series)
    assert isinstance(out["rsq"], (float, np.floating))

    # Expect estimates near truth
    # Keys assumed: 'const' for intercept when add_const=True, and 'mkt_excess'
    assert pytest.approx(float(out["params"]["const"]), abs=5e-4) == alpha_true
    assert pytest.approx(float(out["params"]["mkt_excess"]), abs=0.05) == beta_true

    # Reasonable R^2 with low noise
    assert out["rsq"] > 0.6


# ----------------------------
# TEARSHEET
# ----------------------------
def test_build_factor_tearsheet_contract_and_sanity():
    # Reuse synthetic data
    scores, fwd, _ = _mk_fmb_inputs(T=18, N=40, beta_true=0.7, noise=0.04)

    # Build a trivial portfolio: long top decile each date, equal-weight
    ranks = scores.apply(lambda row: pd.qcut(row.rank(method="first"), 10, labels=False) + 1, axis=1)
    long_mask = ranks == 10
    w = (long_mask.T / long_mask.sum(axis=1).replace(0, np.nan)).T.fillna(0.0)

    # Turn weights into realized same-date portfolio returns from fwd
    port_rets = (w * fwd).sum(axis=1).rename("portfolio_return")

    # Expect the tearsheet to summarize IC, returns, and maybe drawdowns
    ts = build_factor_tearsheet(scores=scores, fwd_returns=fwd, portfolio_returns=port_rets)

    assert isinstance(ts, dict)
    assert "ic" in ts and "summary" in ts

    ic = ts["ic"]
    summary = ts["summary"]
    assert isinstance(ic, pd.Series)
    assert isinstance(summary, (pd.DataFrame, dict))

    # With synthetic signal, mean IC should be positive
    assert ic.mean() > 0.2

    # If summary is a DataFrame, check a couple common fields if present
    if isinstance(summary, pd.DataFrame):
        expected_cols = {"mean_return", "vol", "sharpe"}
        assert expected_cols.intersection(summary.columns)  # at least one present
