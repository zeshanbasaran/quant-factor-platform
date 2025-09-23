-- ================================
-- Quantitative Factor Platform DB
-- ================================

-- Run order: securities → prices → fundamentals → factor_* → portfolios → portfolio_returns → metadata
-- Note: avoid reserved keywords; use trade_date, px_close, etc.

-- ---------- Core Reference ----------
CREATE TABLE securities (
  secid        INTEGER PRIMARY KEY,     -- stable internal id
  symbol       TEXT    NOT NULL,
  name         TEXT,
  exchange     TEXT,
  is_active    INTEGER DEFAULT 1
);

-- ---------- Market Data ----------
CREATE TABLE  prices (
  secid        INTEGER NOT NULL,
  trade_date   DATE    NOT NULL,
  px_close     REAL,
  px_adj_close REAL,
  px_volume    REAL,
  PRIMARY KEY (secid, trade_date)
  -- FOREIGN KEY (secid) REFERENCES securities(secid)
);

-- ---------- Fundamentals ----------
CREATE TABLE fundamentals (
  secid            INTEGER NOT NULL,
  period_end_date  DATE    NOT NULL,
  pe               REAL,
  roe              REAL,
  debt_to_equity   REAL,
  PRIMARY KEY (secid, period_end_date)
  -- FOREIGN KEY (secid) REFERENCES securities(secid)
);

-- ---------- Factor Levels (raw signals per day) ----------
CREATE TABLE factor_levels (
  secid        INTEGER NOT NULL,
  trade_date   DATE    NOT NULL,
  value_raw    REAL,            -- e.g., earnings yield (1/PE)
  mom6_raw     REAL,            -- 6m return (optionally skip last month)
  qual_raw     REAL,            -- composite from ROE & D/E
  PRIMARY KEY (secid, trade_date)
  -- FOREIGN KEY (secid) REFERENCES securities(secid)
);

-- ---------- Factor Scores (winsorized, z-scored, composites) ----------
CREATE TABLE factor_scores (
  secid        INTEGER NOT NULL,
  trade_date   DATE    NOT NULL,
  value_z      REAL,
  mom6_z       REAL,
  qual_z       REAL,
  value_score  REAL,
  mom6_score   REAL,
  qual_score   REAL,
  PRIMARY KEY (secid, trade_date)
  -- FOREIGN KEY (secid) REFERENCES securities(secid)
);

-- ---------- Portfolio Constituents (quintile memberships & weights) ----------
CREATE TABLE portfolios (
  rebalance_date DATE    NOT NULL,
  factor_name    TEXT    NOT NULL,      -- 'value' | 'momentum' | 'quality'
  quintile       INTEGER NOT NULL,      -- 1..5
  secid          INTEGER NOT NULL,
  weight         REAL,
  PRIMARY KEY (rebalance_date, factor_name, quintile, secid)
  -- FOREIGN KEY (secid) REFERENCES securities(secid)
);

-- ---------- Portfolio Performance (per day) ----------
CREATE TABLE portfolio_returns (
  trade_date   DATE    NOT NULL,
  factor_name  TEXT    NOT NULL,
  quintile     INTEGER NOT NULL,        -- 1..5 or 0 for long-short if you store it here
  ret          REAL,                    -- daily return
  cumret       REAL,                    -- cumulative return since start
  PRIMARY KEY (trade_date, factor_name, quintile)
);

-- ---------- Metadata / Key-Value ----------
CREATE TABLE metadata (
  meta_key     TEXT PRIMARY KEY,
  meta_value   TEXT
);

