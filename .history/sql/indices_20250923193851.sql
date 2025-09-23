-- ======================
-- Helpful Indices
-- ======================
CREATE INDEX idx_prices_trade_date               ON prices(trade_date);
CREATE INDEX idx_prices_secid_trade_date         ON prices(secid, trade_date);

CREATE INDEX idx_fundamentals_period_end_date    ON fundamentals(period_end_date);
CREATE INDEX idx_fundamentals_secid_period_end   ON fundamentals(secid, period_end_date);

CREATE INDEX idx_factor_levels_trade_date        ON factor_levels(trade_date);
CREATE INDEX idx_factor_scores_trade_date        ON factor_scores(trade_date);

CREATE INDEX idx_portfolios_rebalance_date       ON portfolios(rebalance_date);
CREATE INDEX idx_portfolios_factor_quintile_date ON portfolios(factor_name, quintile, rebalance_date);

CREATE INDEX idx_portfolio_returns_trade_date    ON portfolio_returns(trade_date);
CREATE INDEX idx_portfolio_returns_factor_q      ON portfolio_returns(factor_name, quintile);
