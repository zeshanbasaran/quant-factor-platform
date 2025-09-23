-- sql/schema.sql
CREATE TABLE securities (
  secid        INTEGER PRIMARY KEY,
  symbol       TEXT NOT NULL,
  name         TEXT,
  exchange     TEXT,
  is_active    INTEGER DEFAULT 1
);

CREATE TABLE prices (
  secid           INTEGER,
  trade_date      DATE,
  px_close        REAL,
  px_adj_close    REAL,
  px_volume       REAL,
  PRIMARY KEY (secid, trade_date)
);

CREATE TABLE fundamentals (
  secid             INTEGER,
  period_end_date   DATE,
  pe                REAL,
  roe               REAL,
  debt_to_equity    REAL,
  PRIMARY KEY (secid, period_end_date)
);

CREATE TABLE factor_levels (         -- raw building blocks
  secid        INTEGER,
  date         DATE,
  value_raw    REAL,                 -- e.g., earnings yield
  mom6_raw     REAL,                 -- 6m return (skip 1m optional)
  qual_raw     REAL,                 -- composite from ROE & D/E
  PRIMARY KEY (secid, date)
);

CREATE TABLE factor_scores (         -- winsorized, z-scored, composite
  secid        INTEGER,
  date         DATE,
  value_z      REAL,
  mom6_z       REAL,
  qual_z       REAL,
  value_score  REAL,
  mom6_score   REAL,
  qual_score   REAL,
  PRIMARY KEY (secid, date)
);

CREATE TABLE portfolios (
  date         DATE,
  factor       TEXT,                 -- 'value' | 'momentum' | 'quality'
  quintile     INTEGER,              -- 1..5
  secid        INTEGER,
  weight       REAL,
  PRIMARY KEY (date, factor, quintile, secid)
);

CREATE TABLE portfolio_returns (
  date         DATE,
  factor       TEXT,
  quintile     INTEGER,
  ret          REAL,
  cumret       REAL,                 -- cumulative
  PRIMARY KEY (date, factor, quintile)
);

CREATE TABLE metadata (
  key          TEXT PRIMARY KEY,
  value        TEXT
);
