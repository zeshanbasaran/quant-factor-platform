# tests/test_portfolio.py
import numpy as np
import pandas as pd
import pytest

from src.portfolio.rank_assign import assign_quintiles
from src.portfolio.weights import equal_weight_long_only, long_short_weights
from src.portfolio.performance import (
    portfolio_returns,
    turnover_series,
    drawdown_curve,
    max_drawdown,
    information_coefficient,
)


# ----------------------------
# Helpers
# ----------------------------
def _mk_scores():
    # Two dates, five symbols; include a NaN on the second date
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    cols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    df = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],    # strictly increasing
            [10.0, 8.0, np.nan, 6.0, 4.0] # one NaN, decreasing-ish
        ],
        index=idx,
        columns=cols,
    )
    return df


def _mk_returns():
    # Asset returns aligned to the same dates/symbols as scores
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    cols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    # Keep deterministic numbers
    rets = pd.DataFrame(
        [
            [0.01, -0.02, 0.005, 0.03, -0.01],
            [0.02,  0.01, 0.00, -0.01, 0.005],
        ],
        index=idx,
        columns=cols,
    )
    return rets


# ----------------------------
# RANK ASSIGNMENT
# ----------------------------
def test_assign_quintiles_basic_and_nan_handling():
    scores = _mk_scores()
    q = assign_quintiles(scores)  # expect values in {1,2,3,4,5}; per-date assignment

    assert isinstance(q, pd.DataFrame)
    assert q.index.equals(scores.index)
    assert q.columns.tolist() == scores.columns.tolist()

    # Date 2020-01-02: strictly increasing -> AAA lowest (=1), EEE highest (=5)
    row = q.loc[pd.Timestamp("2020-01-02")]
    assert row["AAA"] == 1
    assert row["BBB"] == 2
    assert row["CCC"] == 3
    assert row["DDD"] == 4
    assert row["EEE"] == 5

    # Date 2020-01-03: one NaN should remain NaN in ranks
    row2 = q.loc[pd.Timestamp("2020-01-03")]
    assert np.isnan(row2["CCC"])

    # Highest score that day should get highest rank; lowest -> lowest rank
    # Remaining names: AAA(10), BBB(8), DDD(6), EEE(4)
    # With 4 valid names, implementation may still map to 1..5 or 1..4; we check monotonicity.
    valid = row2.dropna()
    # Greater score => greater rank
    assert valid["AAA"] > valid["BBB"] > valid["DDD"] > valid["EEE"]


# ----------------------------
# WEIGHTS
# ----------------------------
def test_equal_weight_long_only_top_quintile():
    scores = _mk_scores()
    ranks = assign_quintiles(scores)

    w = equal_weight_long_only(ranks, pick_quintile=5)  # long the highest quintile
    assert isinstance(w, pd.DataFrame)
    assert w.index.equals(scores.index)
    assert w.columns.tolist() == scores.columns.tolist()

    # Each date’s weights should sum to 1.0 (ignoring all-NaN cases)
    row1_sum = np.nansum(w.loc[pd.Timestamp("2020-01-02")].values)
    assert pytest.approx(row1_sum, rel=1e-12) == 1.0

    # On 2020-01-02, only EEE is quintile 5 -> weight should be 1.0 there
    assert pytest.approx(w.loc["2020-01-02", "EEE"], rel=1e-12) == 1.0
    assert np.isclose(w.loc["2020-01-02"].drop("EEE").fillna(0.0).sum(), 0.0)

    # On 2020-01-03, top rank is AAA only → weight should be 1.0 on AAA
    assert pytest.approx(w.loc["2020-01-03", "AAA"], rel=1e-12) == 1.0


def test_long_short_weights_market_neutral():
    scores = _mk_scores()
    ranks = assign_quintiles(scores)

    w = long_short_weights(
        ranks,
        long_quintile=5,
        short_quintile=1,
        gross=1.0,   # |w| sums to 1.0
        net=0.0      # market neutral
    )

    assert isinstance(w, pd.DataFrame)

    # Gross should be ~1.0 each date; Net ~0.0
    gross = w.abs().sum(axis=1)
    net_ = w.sum(axis=1)

    assert pytest.approx(float(gross.loc["2020-01-02"]), rel=1e-12) == 1.0
    assert pytest.approx(float(net_.loc["2020-01-02"]), abs=1e-12) == 0.0

    # On 2020-01-02 with 5 names and clear quintiles:
    # long EEE, short AAA; others zero. Thus longs sum = 0.5, shorts sum = -0.5.
    assert pytest.approx(w.loc["2020-01-02", "EEE"], rel=1e-12) == 0.5
    assert pytest.approx(w.loc["2020-01-02", "AAA"], rel=1e-12) == -0.5
    assert np.isclose(w.loc["2020-01-02", ["BBB","CCC","DDD"]].fillna(0.0).sum(), 0.0)


# ----------------------------
# PERFORMANCE: RETURNS
# ----------------------------
def test_portfolio_returns_align_and_compute():
    rets = _mk_returns()

    # Simple one-date long-only on 2020-01-02: all weight on DDD
    w = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    w.loc["2020-01-02", "DDD"] = 1.0
    w.loc["2020-01-03", "AAA"] = 1.0  # next day, rotate to AAA

    pr = portfolio_returns(w, rets)
    assert isinstance(pr, pd.Series)
    assert pr.index.equals(rets.index)

    # Expected: day1 = 0.03 (DDD), day2 = 0.02 (AAA)
    assert pytest.approx(float(pr.loc["2020-01-02"]), rel=1e-12) == 0.03
    assert pytest.approx(float(pr.loc["2020-01-03"]), rel=1e-12) == 0.02


# ----------------------------
# PERFORMANCE: TURNOVER
# ----------------------------
def test_turnover_series_half_l1_definition():
    # Build weights that flip from DDD to AAA between the two dates
    cols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    w = pd.DataFrame(0.0, index=idx, columns=cols)
    w.loc["2020-01-02", "DDD"] = 1.0
    w.loc["2020-01-03", "AAA"] = 1.0

    # Turnover_t = 0.5 * sum(|w_t - w_{t-1}|)
    to = turnover_series(w)

    # First date has no previous -> expect NaN (or 0.0; we accept either via check)
    assert (np.isnan(to.iloc[0])) or np.isclose(to.iloc[0], 0.0)

    # Second date: sum abs diffs = |1-0| + |0-1| = 2.0 → turnover = 1.0
    assert pytest.approx(float(to.iloc[1]), rel=1e-12) == 1.0


# ----------------------------
# PERFORMANCE: DRAWDOWN
# ----------------------------
def test_drawdown_and_max_drawdown_basic():
    # Returns: +10%, -5%, -5%, 0%
    r = pd.Series(
        [0.10, -0.05, -0.05, 0.0],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
        name="ret",
    )
    dd = drawdown_curve(r)
    mdd = max_drawdown(r)

    # After +10%: peak=1.10; after two -5%: equity=1.1*0.95*0.95=0.99275
    # drawdown = 0.99275/1.10 - 1 = -0.0975...
    assert isinstance(dd, pd.Series)
    assert pytest.approx(float(mdd), rel=1e-12) == -0.0975


# ----------------------------
# PERFORMANCE: INFORMATION COEFFICIENT (Spearman)
# ----------------------------
def test_information_coefficient_spearman_rank_by_date():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    # Build scores per date (higher is better)
    scores = pd.DataFrame(
        [
            [1, 2, 3, 4, 5],   # perfect ascending
            [5, 4, 3, 2, 1],   # perfect descending
        ],
        index=dates,
        columns=syms,
    )

    # Next-period returns aligned by date (forward returns for each row's date)
    fwd = pd.DataFrame(
        [
            [0.01, 0.02, 0.03, 0.04, 0.05],   # perfectly aligned (IC = +1)
            [0.01, 0.02, 0.03, 0.04, 0.05],   # opposite ordering (IC = -1)
        ],
        index=dates,
        columns=syms,
    )

    ic = information_coefficient(scores, fwd, method="spearman")
    assert isinstance(ic, pd.Series)
    assert ic.index.equals(dates)

    assert pytest.approx(float(ic.iloc[0]), abs=1e-12) == 1.0
    assert pytest.approx(float(ic.iloc[1]), abs=1e-12) == -1.0
