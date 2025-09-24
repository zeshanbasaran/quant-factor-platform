"""
models.py
---------
SQLAlchemy ORM models for logging backtests, trades, PnL, and risk events.

Tables
------
- runs: one row per backtest run (strategy/symbol/bar/date range).
- trades: discrete position changes with costs.
- daily_pnl: per-period returns/equity snapshots.
- risk_events: threshold breaches (e.g., MAX_DD, VAR_95, VOL).

Usage
-----
from src.db.models import Base, Run, Trade, DailyPnl, RiskEvent
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ----------------------------
# Run
# ----------------------------
class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # What was run
    strategy = Column(String(64), nullable=False)  # e.g., "SMA", "Bollinger"
    symbol = Column(String(32), nullable=False)    # e.g., "SPY"
    bar = Column(String(8), nullable=False)        # "1d" or "1h"

    # Time window
    start = Column(DateTime, nullable=False)
    end = Column(DateTime, nullable=False)

    # Metadata
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    # Relationships
    trades: List["Trade"] = relationship("Trade", back_populates="run", cascade="all, delete-orphan")
    daily_pnl: List["DailyPnl"] = relationship("DailyPnl", back_populates="run", cascade="all, delete-orphan")
    risk_events: List["RiskEvent"] = relationship("RiskEvent", back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (
        # Helpful for quick lookups of similar runs
        Index("ix_runs_symbol_bar", "symbol", "bar"),
        Index("ix_runs_strategy_symbol", "strategy", "symbol"),
    )

    def __repr__(self) -> str:
        return f"<Run id={self.id} {self.strategy} {self.symbol} {self.bar} {self.start.date()}→{self.end.date()}>"

# ----------------------------
# Trade
# ----------------------------
class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)

    timestamp = Column(DateTime, nullable=False)
    side = Column(String(8), nullable=False)  # "BUY" or "SELL"
    target_weight = Column(Float, nullable=False)  # resulting target w in [-1, 1]
    delta_weight = Column(Float, nullable=False)   # change in weight this trade
    price = Column(Float, nullable=False)          # execution reference price
    cost = Column(Float, nullable=False, default=0.0)  # trading cost in currency

    run: Run = relationship("Run", back_populates="trades")

    __table_args__ = (
        Index("ix_trades_run_ts", "run_id", "timestamp"),
    )

    def __repr__(self) -> str:
        ts = self.timestamp.isoformat(sep=" ") if isinstance(self.timestamp, datetime) else str(self.timestamp)
        return f"<Trade run={self.run_id} {ts} {self.side} dw={self.delta_weight:.4f} px={self.price:.4f}>"


# ----------------------------
# Daily PnL
# ----------------------------
class DailyPnl(Base):
    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)

    timestamp = Column(DateTime, nullable=False)
    ret = Column(Float, nullable=False)     # per-period return
    equity = Column(Float, nullable=False)  # equity level after this period

    run: Run = relationship("Run", back_populates="daily_pnl")

    __table_args__ = (
        UniqueConstraint("run_id", "timestamp", name="uq_daily_pnl_run_ts"),
        Index("ix_daily_pnl_run_ts", "run_id", "timestamp"),
    )

    def __repr__(self) -> str:
        ts = self.timestamp.isoformat(sep=" ") if isinstance(self.timestamp, datetime) else str(self.timestamp)
        return f"<DailyPnl run={self.run_id} {ts} ret={self.ret:.6f} equity={self.equity:.2f}>"


# ----------------------------
# Risk Event
# ----------------------------
class RiskEvent(Base):
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)

    timestamp = Column(DateTime, nullable=False)
    metric = Column(String(16), nullable=False)  # e.g., "MAX_DD", "VAR_95", "VOL"
    value = Column(Float, nullable=False)        # observed metric value
    threshold = Column(Float, nullable=False)    # configured threshold

    run: Run = relationship("Run", back_populates="risk_events")

    __table_args__ = (
        Index("ix_risk_events_run_ts", "run_id", "timestamp"),
        Index("ix_risk_events_metric", "metric"),
    )

    def __repr__(self) -> str:
        ts = self.timestamp.isoformat(sep=" ") if isinstance(self.timestamp, datetime) else str(self.timestamp)
        return f"<RiskEvent run={self.run_id} {ts} {self.metric} value={self.value:.6f} thr={self.threshold:.6f}>"
