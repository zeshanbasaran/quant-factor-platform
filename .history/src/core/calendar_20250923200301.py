"""
calendar.py
-----------
Trading calendar utilities.

Responsibilities
---------------
- Generate trading sessions (daily/hourly).
- Skip weekends and known holidays.
- Provide rebalancing dates given a frequency (daily, weekly, monthly, quarterly).
- Helpers to align arbitrary dates to the next valid trading day.

Notes
-----
This is a lightweight calendar. For production-grade use,
consider `exchange_calendars` or `pandas_market_calendars`.

Functions
---------
- get_trading_days(start, end, bar="1d", holidays=None)
- get_rebalance_dates(start, end, freq="M", bar="1d", holidays=None)
- is_trading_day(date, holidays=None)
- next_trading_day(date, holidays=None)
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Sequence

# --- Default U.S. holidays (extend as needed) ---
# For simplicity: New Year's Day, Independence Day, Christmas
DEFAULT_HOLIDAYS = {
    "01-01",  # New Year
    "07-04",  # Independence Day
    "12-25",  # Christmas
}


# ----------------------------
# Core utilities
# ----------------------------

def _is_holiday(date: pd.Timestamp, holidays: Optional[Sequence[pd.Timestamp]] = None) -> bool:
    if holidays is None:
        holidays = []
    holidays = set(pd.to_datetime(holidays))
    if date.normalize() in holidays:
        return True
    # Fallback: check mm-dd code against DEFAULT_HOLIDAYS
    if date.strftime("%m-%d") in DEFAULT_HOLIDAYS:
        return True
    return False


def is_trading_day(date: datetime | pd.Timestamp, holidays: Optional[Sequence[pd.Timestamp]] = None) -> bool:
    """
    Return True if given date is a trading day (Mon–Fri, not holiday).
    """
    d = pd.to_datetime(date)
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if _is_holiday(d, holidays):
        return False
    return True


def next_trading_day(date: datetime | pd.Timestamp, holidays: Optional[Sequence[pd.Timestamp]] = None) -> pd.Timestamp:
    """
    Return the next valid trading day >= given date.
    """
    d = pd.to_datetime(date)
    while not is_trading_day(d, holidays):
        d += pd.Timedelta(days=1)
    return d.normalize()


def get_trading_days(
    start: str | datetime,
    end: str | datetime,
    bar: str = "1d",
    holidays: Optional[Sequence[pd.Timestamp]] = None,
) -> pd.DatetimeIndex:
    """
    Generate trading sessions between start and end.

    Parameters
    ----------
    start, end : str | datetime
        Range boundaries.
    bar : str, default "1d"
        "1d" for daily bars, "1h" for hourly bars.
    holidays : list[pd.Timestamp], optional
        Custom holiday dates.

    Returns
    -------
    pd.DatetimeIndex
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    if bar == "1d":
        # Business days Mon-Fri, excluding holidays
        days = pd.date_range(start, end, freq="B")
        days = pd.DatetimeIndex([d for d in days if is_trading_day(d, holidays)])
        return days
    elif bar == "1h":
        # Approximate: 6.5 trading hours per day (9:30–16:00 ET)
        days = get_trading_days(start, end, bar="1d", holidays=holidays)
        hours = []
        for d in days:
            for h in range(9, 16):  # 9–15 full hours
                hours.append(d + pd.Timedelta(hours=h))
            # add last half hour (15:30)
            hours.append(d + pd.Timedelta(hours=15, minutes=30))
        return pd.DatetimeIndex([h for h in hours if h <= end])
    else:
        raise ValueError(f"Unsupported bar '{bar}'. Use '1d' or '1h'.")


def get_rebalance_dates(
    start: str | datetime,
    end: str | datetime,
    freq: str = "M",
    bar: str = "1d",
    holidays: Optional[Sequence[pd.Timestamp]] = None,
) -> pd.DatetimeIndex:
    """
    Return rebalancing dates (aligned to trading days).

    Parameters
    ----------
    start, end : str | datetime
        Date range.
    freq : str, default "M"
        Pandas offset alias: "M"=month-end, "Q"=quarter-end, "W"=weekly.
    bar : str, default "1d"
        "1d" or "1h". For intraday, rebalance dates are expanded into that day’s sessions.
    holidays : list[pd.Timestamp], optional
        Custom holiday list.

    Returns
    -------
    pd.DatetimeIndex
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # anchor dates from pandas offsets
    anchors = pd.date_range(start, end, freq=freq)
    # shift to next valid trading day if needed
    anchors = pd.DatetimeIndex([next_trading_day(d, holidays) for d in anchors])

    if bar == "1d":
        return anchors.unique()
    elif bar == "1h":
        # expand each anchor day into its intraday bars
        all_hours = get_trading_days(start, end, bar="1h", holidays=holidays)
        expanded = []
        for d in anchors:
            expanded.extend([h for h in all_hours if h.normalize() == d.normalize()])
        return pd.DatetimeIndex(expanded)
    else:
        raise ValueError(f"Unsupported bar '{bar}'. Use '1d' or '1h'.")
