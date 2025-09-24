"""
dash_app.py
-----------
Plotly Dash dashboard for the Quant Factor Platform.

Features
--------
- Sidebar controls:
    * Strategy (SMA crossover, Bollinger mean-reversion)
    * Symbol, date range, bar size
    * Strategy params (short/long windows, k/band_exit)
    * Trading costs (slippage bps, $/share commission)
    * Risk thresholds (Max DD, VaR 95, Ann Vol)

- Tabs:
    1) Overview: KPI cards + Equity & Drawdown charts
    2) Risk: VaR / Vol KPIs, breach badges
    3) Trades: trade table + CSV download
    4) PnL: returns histogram + rolling Sharpe
    5) Data: last rows preview

Notes
-----
- Tries to import platform modules first (src/...).
  Falls back to inline helpers (yfinance, basic metrics) if imports fail.
- Run from project root:
      python -m dashboards.dash_app
  or  python dashboards/dash_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- Make "src" importable when run directly ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ----------------------------
# Optional platform imports
# ----------------------------
get_price_data = None
ann_return = ann_vol = sharpe = max_drawdown = None
var_historical = None
simulate = None
sma_crossover = None
bollinger_meanrev = None

# Config fallbacks
INIT_CASH = 100_000
SLIPPAGE_BPS = 1.0
COMM_PER_TRADE = 0.005
START = "2013-01-01"
END = "2025-01-01"
BAR = "1d"
RISK = {"max_dd": 0.20, "var_95": 0.025, "vol_ann": 0.25}

# Attempt multiple import paths used across your repos
try:
    # Systematic backtester layout
    from src.data.loaders import get_price_data as _get_px
    from src.engine.metrics import ann_return as _ar, ann_vol as _av, sharpe as _sh, max_drawdown as _mdd
    from src.risk.risk_metrics import var_historical as _vh
    from src.backtester import simulate as _simulate  # if available in your layout
    from src.strategies.sma_crossover import sma_crossover as _sma
    from src.strategies.bollinger_meanrev import bollinger_meanrev as _boll
    get_price_data = _get_px
    ann_return, ann_vol, sharpe, max_drawdown = _ar, _av, _sh, _mdd
    var_historical = _vh
    simulate = _simulate
    sma_crossover, bollinger_meanrev = _sma, _boll
except Exception:
    pass

# Try quant-factor-platform style paths
if get_price_data is None:
    try:
        from src.data.ingest_prices import get_price_data as _get_px  # type: ignore
        get_price_data = _get_px
    except Exception:
        pass

# Try to import config
try:
    from src.configs.default import (  # type: ignore
        INIT_CASH as _CASH,
        SLIPPAGE_BPS as _SLP,
        COMM_PER_TRADE as _COMM,
        START as _START,
        END as _END,
        BAR as _BAR,
        RISK as _RISK,
    )
    INIT_CASH, SLIPPAGE_BPS, COMM_PER_TRADE = _CASH, _SLP, _COMM
    START, END, BAR, RISK = _START, _END, _BAR, _RISK
except Exception:
    try:
        from src.config import (  # systematic-backtester config.py
            INIT_CASH as _CASH,
            SLIPPAGE_BPS as _SLP,
            COMM_PER_TRADE as _COMM,
            START as _START,
            END as _END,
            BAR as _BAR,
            RISK as _RISK,
        )
        INIT_CASH, SLIPPAGE_BPS, COMM_PER_TRADE = _CASH, _SLP, _COMM
        START, END, BAR, RISK = _START, _END, _BAR, _RISK
    except Exception:
        pass


# ----------------------------
# Fallback helpers if imports fail
# ----------------------------
if get_price_data is None:
    import yfinance as yf

    def get_price_data(symbol: str, start: str, end: str, bar: str = "1d") -> pd.DataFrame:
        interval = bar
        if interval not in ("1d", "1h"):
            interval = "1d"
        df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"])
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        return df[["open", "high", "low", "close", "adj_close", "volume"]].sort_index()

if sma_crossover is None:
    def sma_crossover(df: pd.DataFrame, short: int = 50, long: int = 200, allow_short: bool = False) -> pd.Series:
        s = df["adj_close"].rolling(short).mean()
        l = df["adj_close"].rolling(long).mean()
        pos = (s > l).astype(int)
        if allow_short:
            pos = pos.replace({0: -1})
        return pos.shift(1).fillna(0.0)

if bollinger_meanrev is None:
    def bollinger_meanrev(df: pd.DataFrame, lookback: int = 20, k: float = 2.0, band_exit: float = 0.5, allow_short: bool = True) -> pd.Series:
        m = df["adj_close"].rolling(lookback).mean()
        sd = df["adj_close"].rolling(lookback).std(ddof=0)
        z = (df["adj_close"] - m) / sd
        pos = pd.Series(0.0, index=df.index)
        pos = pos.where(~(z < -k), 1.0)
        if allow_short:
            pos = pos.where(~(z > k), -1.0)
        pos = pos.where(~(z.between(-band_exit, band_exit)), 0.0)
        return pos.ffill().shift(1).fillna(0.0)

if ann_return is None:
    def ann_return(returns: pd.Series, ppy: int) -> float:
        r = pd.Series(returns).dropna()
        if r.empty:
            return np.nan
        growth = (1 + r).prod()
        if growth <= 0:
            return np.nan
        return growth ** (ppy / len(r)) - 1.0

if ann_vol is None:
    def ann_vol(returns: pd.Series, ppy: int) -> float:
        r = pd.Series(returns).dropna()
        return float(r.std(ddof=0) * np.sqrt(ppy)) if not r.empty else np.nan

if sharpe is None:
    def sharpe(returns: pd.Series, ppy: int, rf: float = 0.0) -> float:
        r = pd.Series(returns).dropna()
        if r.empty:
            return np.nan
        excess = r - (rf / ppy)
        sd = excess.std(ddof=0)
        return float(excess.mean() / sd * np.sqrt(ppy)) if sd and not np.isnan(sd) else np.nan

if max_drawdown is None:
    def max_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
        e = pd.Series(equity).astype(float).dropna()
        if e.empty:
            return 0.0, pd.Series(dtype=float, index=equity.index if hasattr(equity, "index") else None)
        roll_max = e.cummax()
        dd = e / roll_max - 1.0
        return float(dd.min()), dd.reindex(equity.index)

if var_historical is None:
    def var_historical(returns: pd.Series, alpha: float = 0.95) -> float:
        return -np.nanpercentile(pd.Series(returns).dropna(), 100 * (1 - alpha))

def periods_per_year_from_bar(bar: str) -> int:
    return 252 if (bar or "").lower() == "1d" else int(252 * 6.5)

def run_backtest(df: pd.DataFrame, pos: pd.Series, init_cash: float, slippage_bps: float = 1.0, comm_per_sh: float = 0.0):
    # Inline simulate if platform engine not present
    px = df["adj_close"].astype(float)
    ret = px.pct_change().fillna(0.0)
    w = pd.Series(pos).clip(-1, 1).astype(float).reindex(ret.index).fillna(0.0)
    dw = w.diff().fillna(w)
    trading_cost = np.abs(dw) * (slippage_bps / 1e4)
    strat_ret = w * ret - trading_cost
    equity = (1 + strat_ret).cumprod() * init_cash
    pnl = equity.diff().fillna(equity - init_cash)
    trades = pd.DataFrame(
        {
            "timestamp": df.index,
            "target_w": w.values,
            "delta_w": dw.values,
            "price": px.values,
            "cost": (trading_cost.values * equity.shift(1).fillna(init_cash).values),
        }
    )
    trades = trades.query("delta_w != 0")
    return {"equity": equity, "returns": strat_ret, "pnl": pnl, "trades": trades}


# ----------------------------
# Dash App
# ----------------------------
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
import plotly.graph_objects as go
import plotly.express as px

app = Dash(__name__, title="Quant Factor Dashboard", suppress_callback_exceptions=True)
server = app.server  # for gunicorn

def kpi_card(title: str, value: str, id_: str = None):
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value", id=id_ if id_ else None),
        ],
        style={
            "padding": "12px 16px",
            "borderRadius": "16px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
            "background": "#11151a",
            "color": "#eaeef2",
        },
    )

def make_equity_figure(equity: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=40, b=30),
        template="plotly_dark",
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
    )
    return fig

def make_drawdown_figure(dd: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown"))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=30),
        template="plotly_dark",
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
    )
    return fig

def make_hist_figure(returns: pd.Series) -> go.Figure:
    hist = np.asarray(pd.Series(returns).dropna())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=hist, nbinsx=50))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=30),
        template="plotly_dark",
        title="Returns Distribution",
        xaxis_title="Return",
        yaxis_title="Frequency",
    )
    return fig

def make_line_figure(series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines"))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=30),
        template="plotly_dark",
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
    )
    return fig

# --- Layout ---
app.layout = html.Div(
    style={"display": "flex", "minHeight": "100vh", "background": "#0b0f14", "color": "#eaeef2", "fontFamily": "Inter, system-ui, sans-serif"},
    children=[
        # Sidebar
        html.Div(
            style={"width": "320px", "padding": "16px", "borderRight": "1px solid #1b232c", "background": "#0f141b", "position": "sticky", "top": 0},
            children=[
                html.H2("Quant Factor", style={"margin": "4px 0 16px"}),
                html.Label("Strategy", style={"marginTop": "8px"}),
                dcc.Dropdown(
                    id="strategy",
                    options=[
                        {"label": "SMA Crossover", "value": "SMA"},
                        {"label": "Bollinger Mean-Reversion", "value": "BOLL"},
                    ],
                    value="SMA",
                    clearable=False,
                ),
                html.Label("Symbol", style={"marginTop": "12px"}),
                dcc.Input(id="symbol", type="text", value="SPY", debounce=True, style={"width": "100%"}),
                html.Label("Date Range", style={"marginTop": "12px"}),
                dcc.DatePickerRange(
                    id="dates",
                    start_date=pd.to_datetime(START).date(),
                    end_date=pd.to_datetime(END).date(),
                    display_format="YYYY-MM-DD",
                    minimum_nights=0,
                ),
                html.Label("Bar Size", style={"marginTop": "12px"}),
                dcc.Dropdown(
                    id="bar",
                    options=[{"label": "Daily (1d)", "value": "1d"}, {"label": "Hourly (1h)", "value": "1h"}],
                    value=BAR,
                    clearable=False,
                ),
                html.Hr(),
                # Strategy params (conditional UI)
                html.Div(id="param-block"),

                html.Hr(),
                html.Label("Slippage (bps)"),
                dcc.Input(id="slippage", type="number", value=float(SLIPPAGE_BPS), step=0.5, style={"width": "100%"}),
                html.Label("Commission ($/share)", style={"marginTop": "8px"}),
                dcc.Input(id="commission", type="number", value=float(COMM_PER_TRADE), step=0.001, style={"width": "100%"}),
                html.Label("Initial Cash ($)", style={"marginTop": "8px"}),
                dcc.Input(id="init_cash", type="number", value=float(INIT_CASH), step=1000, style={"width": "100%"}),
                html.Hr(),
                html.Label("Risk Thresholds", style={"marginBottom": "4px"}),
                dcc.Input(id="thr_dd", type="number", value=float(RISK["max_dd"]), step=0.01, style={"width": "100%", "marginBottom": "6px"}),
                dcc.Input(id="thr_var", type="number", value=float(RISK["var_95"]), step=0.005, style={"width": "100%", "marginBottom": "6px"}),
                dcc.Input(id="thr_vol", type="number", value=float(RISK["vol_ann"]), step=0.01, style={"width": "100%"}),
                html.Div(style={"height": "12px"}),
                html.Button("Run Backtest", id="run_btn", n_clicks=0, style={"width": "100%", "padding": "10px", "borderRadius": "10px", "background": "#2563eb", "color": "white", "border": "none"}),
                dcc.Store(id="bt-store"),
                dcc.Download(id="dl-trades"),
            ],
        ),
        # Main content
        html.Div(
            style={"flex": 1, "padding": "16px 18px"},
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="tab-overview",
                    children=[
                        dcc.Tab(label="Overview", value="tab-overview"),
                        dcc.Tab(label="Risk", value="tab-risk"),
                        dcc.Tab(label="Trades", value="tab-trades"),
                        dcc.Tab(label="PnL", value="tab-pnl"),
                        dcc.Tab(label="Data", value="tab-data"),
                    ],
                    style={"background": "#0b0f14", "color": "#eaeef2"},
                ),
                html.Div(id="tab-content", style={"marginTop": "12px"}),
            ],
        ),
    ],
)

# --- Dynamic param block ---
@app.callback(
    Output("param-block", "children"),
    Input("strategy", "value"),
)
def render_param_block(strategy):
    if strategy == "SMA":
        return html.Div(
            children=[
                html.Label("Short Window"),
                dcc.Input(id="sma_short", type="number", value=50, step=1, style={"width": "100%"}),
                html.Label("Long Window", style={"marginTop": "8px"}),
                dcc.Input(id="sma_long", type="number", value=200, step=1, style={"width": "100%"}),
                dcc.Checklist(
                    id="allow_short",
                    options=[{"label": " Allow Shorting", "value": "yes"}],
                    value=[],
                    style={"marginTop": "8px"},
                ),
            ]
        )
    else:
        return html.Div(
            children=[
                html.Label("Lookback"),
                dcc.Input(id="boll_lookback", type="number", value=20, step=1, style={"width": "100%"}),
                html.Label("Band Width (k)", style={"marginTop": "8px"}),
                dcc.Input(id="boll_k", type="number", value=2.0, step=0.1, style={"width": "100%"}),
                html.Label("Exit Band (|z| <)", style={"marginTop": "8px"}),
                dcc.Input(id="boll_exit", type="number", value=0.5, step=0.1, style={"width": "100%"}),
                dcc.Checklist(
                    id="allow_short",
                    options=[{"label": " Allow Shorting", "value": "yes"}],
                    value=["yes"],
                    style={"marginTop": "8px"},
                ),
            ]
        )

# --- Run backtest & store results ---
@app.callback(
    Output("bt-store", "data"),
    Input("run_btn", "n_clicks"),
    State("strategy", "value"),
    State("symbol", "value"),
    State("dates", "start_date"),
    State("dates", "end_date"),
    State("bar", "value"),
    State("slippage", "value"),
    State("commission", "value"),
    State("init_cash", "value"),
    State("thr_dd", "value"),
    State("thr_var", "value"),
    State("thr_vol", "value"),
    State("sma_short", "value"),
    State("sma_long", "value"),
    State("boll_lookback", "value"),
    State("boll_k", "value"),
    State("boll_exit", "value"),
    State("allow_short", "value"),
    prevent_initial_call=True,
)
def run_bt(n, strategy, symbol, start, end, bar, slp, comm, cash, thr_dd, thr_var, thr_vol,
           sma_short, sma_long, boll_lb, boll_k, boll_exit, allow_short_val):
    try:
        df = get_price_data(symbol, str(start), str(end), bar)
        if df is None or df.empty:
            return {"error": f"No data for {symbol} {start}..{end} ({bar})."}
        allow_short_flag = ("yes" in (allow_short_val or []))
        if strategy == "SMA":
            pos = sma_crossover(df, short=int(sma_short or 50), long=int(sma_long or 200), allow_short=allow_short_flag)
        else:
            pos = bollinger_meanrev(
                df,
                lookback=int(boll_lb or 20),
                k=float(boll_k or 2.0),
                band_exit=float(boll_exit or 0.5),
                allow_short=allow_short_flag,
            )
        bt = run_backtest(df, pos, float(cash or INIT_CASH), float(slp or SLIPPAGE_BPS), float(comm or COMM_PER_TRADE))
        equity, returns, trades = bt["equity"], bt["returns"], bt["trades"]
        ppy = periods_per_year_from_bar(bar)
        dd_min, dd_series = max_drawdown(equity)
        vol = ann_vol(returns, ppy)
        var95 = var_historical(returns, 0.95)
        breaches = []
        if dd_min < -float(thr_dd):
            breaches.append(("MAX_DD", dd_min, float(thr_dd)))
        if var95 > float(thr_var):
            breaches.append(("VAR_95", var95, float(thr_var)))
        if vol > float(thr_vol):
            breaches.append(("VOL", vol, float(thr_vol)))

        payload = {
            "symbol": symbol,
            "strategy": strategy,
            "params": {
                "bar": bar,
                "slippage_bps": float(slp or SLIPPAGE_BPS),
                "commission": float(comm or COMM_PER_TRADE),
                "init_cash": float(cash or INIT_CASH),
            },
            "equity": equity.to_json(date_format="iso"),
            "returns": returns.to_json(date_format="iso"),
            "dd_series": dd_series.to_json(date_format="iso"),
            "trades": trades.to_json(orient="records", date_format="iso"),
            "metrics": {
                "ann_return": float(ann_return(returns, ppy)) if len(returns) else np.nan,
                "ann_vol": float(vol),
                "sharpe": float(sharpe(returns, ppy)),
                "max_drawdown": float(dd_min),
                "var95": float(var95),
                "ppy": int(ppy),
            },
            "breaches": breaches,
        }
        return payload
    except Exception as e:
        return {"error": str(e)}

# --- Tabs renderer ---
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("bt-store", "data"),
)
def render_tabs(tab, data):
    if not data or "error" in data:
        msg = data.get("error") if isinstance(data, dict) else "Configure parameters and click Run Backtest."
        return html.Div(msg, style={"padding": "12px", "color": "#94a3b8"})

    # Rehydrate
    equity = pd.read_json(data["equity"], typ="series")
    returns = pd.read_json(data["returns"], typ="series")
    dd_series = pd.read_json(data["dd_series"], typ="series")
    trades = pd.read_json(data["trades"], orient="records")

    m = data["metrics"]
    ar = f"{m['ann_return']:.2%}" if pd.notna(m["ann_return"]) else "n/a"
    av = f"{m['ann_vol']:.2%}" if pd.notna(m["ann_vol"]) else "n/a"
    sh = f"{m['sharpe']:.2f}" if pd.notna(m["sharpe"]) else "n/a"
    dd = f"{m['max_drawdown']:.2%}" if pd.notna(m["max_drawdown"]) else "n/a"
    vr = f"{m['var95']:.2%}" if pd.notna(m["var95"]) else "n/a"

    if tab == "tab-overview":
        return html.Div(
            children=[
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px"},
                    children=[
                        kpi_card("Annual Return", ar),
                        kpi_card("Volatility", av),
                        kpi_card("Sharpe", sh),
                        kpi_card("Max Drawdown", dd),
                    ],
                ),
                html.Div(style={"height": "12px"}),
                dcc.Graph(figure=make_equity_figure(equity)),
                html.Div(style={"height": "8px"}),
                dcc.Graph(figure=make_drawdown_figure(dd_series)),
            ]
        )

    if tab == "tab-risk":
        badges = []
        for code, value, thr in data["breaches"]:
            color = "#ef4444"
            txt = f"{code} breach: {value:.2%} vs {thr:.2%}" if code == "MAX_DD" else f"{code} breach: {value:.2%} > {thr:.2%}"
            badges.append(html.Div(txt, style={"padding": "8px 10px", "borderRadius": "10px", "background": color, "display": "inline-block", "marginRight": "8px"}))
        if not badges:
            badges = [html.Div("No breaches detected.", style={"padding": "8px 10px", "borderRadius": "10px", "background": "#16a34a", "display": "inline-block"})]
        return html.Div(
            children=[
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px", "marginBottom": "10px"},
                    children=[kpi_card("VaR 95%", vr), kpi_card("Annual Vol", av), kpi_card("Max Drawdown", dd)],
                ),
                html.Div(badges, style={"marginTop": "6px"}),
            ]
        )

    if tab == "tab-trades":
        cols = [
            {"name": "timestamp", "id": "timestamp"},
            {"name": "target_w", "id": "target_w", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "delta_w", "id": "delta_w", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "price", "id": "price", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "cost", "id": "cost", "type": "numeric", "format": {"specifier": ".2f"}},
        ]
        return html.Div(
            children=[
                html.Div(
                    children=html.Button("Download CSV", id="download-trades", n_clicks=0, style={"padding": "8px 12px", "borderRadius": "8px", "background": "#334155", "color": "white", "border": "none"}),
                    style={"marginBottom": "8px"},
                ),
                dash_table.DataTable(
                    id="trades-table",
                    columns=cols,
                    data=trades.to_dict("records"),
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "600"},
                    style_cell={"backgroundColor": "#0b0f14", "color": "#e2e8f0", "border": "1px solid #1b232c"},
                ),
            ]
        )

    if tab == "tab-pnl":
        roll_sharpe = returns.rolling(60).mean() / returns.rolling(60).std()
        return html.Div(
            children=[
                dcc.Graph(figure=make_hist_figure(returns)),
                html.Div(style={"height": "8px"}),
                dcc.Graph(figure=make_line_figure(roll_sharpe, "Rolling Sharpe (60)")),
            ]
        )

    if tab == "tab-data":
        tail = pd.DataFrame(
            {"equity": equity.tail(5).values, "returns": returns.tail(5).values},
            index=equity.tail(5).index.strftime("%Y-%m-%d %H:%M"),
        ).reset_index(names="timestamp")
        return dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in tail.columns],
            data=tail.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "600"},
            style_cell={"backgroundColor": "#0b0f14", "color": "#e2e8f0", "border": "1px solid #1b232c"},
        )

    return html.Div("Unknown tab.")

# --- CSV download ---
@app.callback(
    Output("dl-trades", "data"),
    Input("download-trades", "n_clicks"),
    State("bt-store", "data"),
    prevent_initial_call=True,
)
def download_trades(n_clicks, data):
    if not data or "trades" not in data:
        return None
    trades = pd.read_json(data["trades"], orient="records")
    csv = trades.to_csv(index=False).encode("utf-8")
    fname = f"{data.get('symbol','sym')}_{data.get('strategy','strat')}_trades.csv"
    return dict(content=csv, filename=fname, type="text/csv")

# --- Entrypoint ---
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
