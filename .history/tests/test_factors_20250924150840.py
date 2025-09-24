# tests/test_factors.py
import numpy as np
import pandas as pd
import pytest

# Value
from src.factors.value import earnings_yield  # expected: invert P/E → EY = 1/PE

# Momentum
from src.factors.momentum import momentum_6m  # expected: 6m return, optionally skip last 1m

# Quality
from src.factors.quality import quality_score  # expected: ↑ with ROE, ↓ with D/E


# ----------------------------
# Helpers
# ----------------------------
def _mk_price_series(length=200, start=100.0, step=0.5, freq="B"):
    """Deterministic upward-sloping price series for momentum tests."""
    idx = pd.bdate_range("2020-01-01", periods=length, freq=freq)
    s = pd.Series(start + step * np.arange(length), index=idx, name="adj_close")
    return s


# ----------------------------
# VALUE: earnings_yield
# ----------------------------
def test_value_earnings_yield_basic():
    pe = pd.Series([10.0, 20.0, np.nan, 0.0, -5.0], index=list("ABCDE"))
    ey = earnings_yield(pe)

    # shape/index preserved
    assert isinstance(ey, pd.Series)
    assert ey.index.tolist() == list("ABCDE")

    # core math: EY = 1/PE for positive PE
    assert pytest.approx(ey["A"], rel=1e-12) == 0.1
    assert pytest.approx(ey["B"], rel=1e-12) == 0.05

    # NaN/invalid handling: non-positive or missing PE -> NaN EY
    assert np.isnan(ey["C"])
    assert np.isnan(ey["D"])
    assert np.isnan(ey["E"])

    # monotonic inverse: smaller PE → larger EY
    assert ey["A"] > ey["B"]


# ----------------------------
# MOMENTUM: 6m with optional 1m skip
# ----------------------------
@pytest.mark.parametrize("skip_1m", [True, False])
def test_momentum_6m_skip_one_month(skip_1m):
    # Build a simple deterministic price path
    # Last date = T; we want ret from T-(126+21) → T-21 when skip_1m=True
    # or T-126 → T when skip_1m=False.
    px = _mk_price_series(length=200)  # ~200 business days
    df_prices = pd.DataFrame({"adj_close": px})

    mom = momentum_6m(df_prices, skip_1m=skip_1m)

    assert isinstance(mom, pd.Series)
    assert mom.index.equals(df_prices.index)

    T = df_prices.index[-1]
    if skip_1m:
        end = df_prices.index[-22]  # T-21 (0-based)
        start = df_prices.index[-(126 + 22)]  # T-(126+21)
    else:
        end = df_prices.index[-1]   # T
        start = df_prices.index[-127]  # T-126

    expected = df_prices.loc[end, "adj_close"] / df_prices.loc[start, "adj_close"] - 1.0

    # We only assert on the last computed value (others depend on window fill)
    assert pytest.approx(mom.loc[T], rel=1e-12) == float(expected)

    # Momentum should be increasing on a linearly increasing price series
    # (last value ≥ median of recent window)
    recent = mom.dropna().tail(10)
    if not recent.empty:
        assert recent.iloc[-1] >= recent.median() - 1e-12


def test_momentum_6m_handles_short_series_gracefully():
    # Too short for a full 6m window: expect all-NaN (no crashes)
    df_short = pd.DataFrame({"adj_close": _mk_price_series(length=50)})
    mom = momentum_6m(df_short, skip_1m=True)
    assert mom.isna().all()


# ----------------------------
# QUALITY: composite (↑ with ROE, ↓ with D/E)
# ----------------------------
def test_quality_score_directionality_and_scaling():
    # Construct 3 names with clear ordering:
    # X: high ROE, low D/E  -> best quality
    # Y: medium ROE, medium D/E
    # Z: low ROE, high D/E  -> worst quality
    idx = list("XYZ")
    funda = pd.DataFrame(
        {
            "roe": pd.Series([0.25, 0.12, 0.03], index=idx),
            "de": pd.Series([0.2, 0.8, 2.0], index=idx),
        }
    )

    q = quality_score(funda)
    assert isinstance(q, pd.Series)
    assert q.index.tolist() == idx

    # Ordering check (not assuming exact formula, just monotonic behavior)
    assert q["X"] > q["Y"] > q["Z"]

    # Reasonable scale: z-scored or similar (don’t enforce exact, just sanity)
    assert np.isfinite(q).all()


def test_quality_score_nan_handling():
    # Missing ROE/D/E should not crash; score computed from available info
    funda = pd.DataFrame(
        {
            "roe": pd.Series([0.15, np.nan, 0.10], index=list("ABC")),
            "de": pd.Series([np.nan, 0.5, 1.5], index=list("ABC")),
        }
    )
    q = quality_score(funda)

    # A: only ROE → still finite
    assert np.isfinite(q["A"])
    # B: only D/E → still finite
    assert np.isfinite(q["B"])
    # C: both present → finite
    assert np.isfinite(q["C"])

    # Better fundamentals should give higher score
    assert q["A"] > q["C"]  # A better DE (NaN treated neutrally) and higher ROE than C
