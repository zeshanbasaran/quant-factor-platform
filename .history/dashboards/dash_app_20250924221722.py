"""
Dash dashboard for quant-factor-platform
=======================================
Reads pre-materialized artifacts from `data/processed/dashboard/` produced by
`scripts/rebuild_dash_data.py` (or the pipeline runner) and renders an
interactive dashboard.

Run from repo root:
    python -m dashboards.dash_app

Dependencies (add to requirements.txt if missing):
    dash>=2.17, plotly>=5.22, pandas, pyarrow (for parquet), numpy

Tabs
----
1) Overview: KPI cards + Equity curve + Drawdown
2) Risk: Rolling Vol, Rolling Sharpe, Return Histogram
3) Analytics: Factor Correlations (if available), Regressions (if available)
4) Tables: Factors / Ranks / Weights previews (optional)

Design notes
------------
- Defensive file loading; will gracefully render placeholders when data is
  missing.
- One-click "Refresh" triggers a re-read of the parquet/json files.
- Uses only core Dash components (no external themes) to minimize deps.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table

# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
DASH_DIR = ROOT / "data" / "processed" / "dashboard"
REPORTS_DIR = ROOT / "reports"

# ----------------------------
# Helpers
# ----------------------------
def _maybe_read_parquet_or_csv(path_no_ext: Path) -> Optional[pd.DataFrame]:
    """Try <name>.parquet then <name>.csv; return None if neither exists/loads."""
    p_parq = path_no_ext.with_suffix(".parquet")
    p_csv  = path_no_ext.with_suffix(".csv")
    if p_parq.exists():
        try:
            return pd.read_parquet(p_parq)
        except Exception:
            pass
    if p_csv.exists():
        try:
            return pd.read_csv(p_csv)
        except Exception:
            pass
    return None

def _read_json_flex(path_no_ext: Path) -> Dict[str, Any]:
    p_json = path_no_ext.with_suffix(".json")
    if p_json.exists():
        try:
            return json.loads(p_json.read_text())
        except Exception:
            return {}
    # CSV fallback for perf summary written by pilot
    p_csv = path_no_ext.with_suffix(".csv")
    if p_csv.exists():
        try:
            df = pd.read_csv(p_csv)
            if df.shape[0] == 1:
                return {k: (float(v) if pd.api.types.is_number(v) else v) for k, v in df.iloc[0].items()}
        except Exception:
            return {}
    return {}

def _parse_date_index(df: pd.DataFrame) -> pd.DataFrame:
    # If a 'date' column exists, set it as index; else try to parse the current index
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
    else:
        try:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except Exception:
            pass
    return df

def _load_series(name_base: str, *, col: Optional[str] = None) -> pd.Series:
    """
    Load a series from dashboard/ or reports/ (parquet/csv).
    If col is None, take the first column.
    """
    for base in (DASH_DIR, REPORTS_DIR):
        df = _maybe_read_parquet_or_csv(base / name_base)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _parse_date_index(df)
            if df.empty:
                continue
            c = col or (df.columns[0] if len(df.columns) else None)
            if c and c in df.columns:
                return df[c].astype(float)
            if c is None and len(df.columns):
                return df.iloc[:, 0].astype(float)
    return pd.Series(dtype=float)

def _load_df(name_base: str) -> pd.DataFrame:
    for base in (DASH_DIR, REPORTS_DIR):
        df = _maybe_read_parquet_or_csv(base / name_base)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return _parse_date_index(df)
    return pd.DataFrame()

def _compute_drawdown_from_returns(returns: pd.Series) -> pd.DataFrame:
    s = returns.astype(float).fillna(0.0)
    wealth = (1.0 + s).cumprod()
    peak = wealth.cummax()
    draw = (wealth / peak) - 1.0
    return pd.DataFrame({"date": wealth.index, "wealth": wealth.values, "peak": peak.values, "drawdown": draw.values}).set_index("date")

def _load_equity_and_drawdown() -> tuple[pd.Series, pd.DataFrame]:
    # 1) Try dashboard-native files
    eq = _load_series("equity")  # date-indexed, first col
    dd = _load_df("drawdown")    # expects drawdown column

    # 2) Try reports/ pilot outputs
    if eq.empty:
        eq_df = _load_df("equity")  # reports/equity.csv (date, wealth)
        if not eq_df.empty and "wealth" in eq_df.columns:
            eq = eq_df["wealth"]
    if dd.empty:
        dd_df = _load_df("drawdown_curve")  # reports/drawdown_curve.csv
        if not dd_df.empty:
            dd = dd_df

    # 3) Derive from returns if still missing
    if (eq.empty or dd.empty):
        # pilot file: reports/portfolio_returns.csv with 'port_ret'
        rets_df = _load_df("portfolio_returns")
        if not rets_df.empty:
            col = "port_ret" if "port_ret" in rets_df.columns else rets_df.columns[0]
            rets = rets_df[col]
            dd_from = _compute_drawdown_from_returns(rets)
            if eq.empty:
                eq = dd_from["wealth"]
            if dd.empty:
                dd = dd_from
    return eq, dd

def _load_perf_summary() -> Dict[str, Any]:
    # dashboard json first
    j = _read_json_flex(DASH_DIR / "perf_summary")
    if j:
        return j
    # reports CSV fallback (pilot): performance_summary.csv
    j = _read_json_flex(REPORTS_DIR / "performance_summary")
    return j or {}

# ----------------------------
# Figures
# ----------------------------
def fig_equity(equity: pd.Series) -> go.Figure:
    fig = go.Figure()
    if equity is None or equity.empty:
        fig.add_annotation(text="No equity data found", showarrow=False)
        fig.update_layout(height=340, margin=dict(l=30, r=10, t=30, b=30))
        return fig
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(title="Equity Curve", height=340, margin=dict(l=30, r=10, t=30, b=30),
                      xaxis_title="Date", yaxis_title="Portfolio Value")
    return fig

def fig_drawdown(dd: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if dd is None or dd.empty or "drawdown" not in dd.columns:
        fig.add_annotation(text="No drawdown data found", showarrow=False)
        fig.update_layout(height=260, margin=dict(l=30, r=10, t=30, b=30))
        return fig
    fig.add_trace(go.Scatter(x=dd.index, y=dd["drawdown"], mode="lines", name="Drawdown", fill="tozeroy"))
    fig.update_layout(title="Drawdown", height=260, margin=dict(l=30, r=10, t=30, b=30),
                      xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".0%")
    return fig

def fig_rolling(series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    if series is None or series.empty:
        fig.add_annotation(text=f"No {title.lower()} data found", showarrow=False)
        fig.update_layout(height=280, margin=dict(l=30, r=10, t=30, b=30))
        return fig
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=title))
    yfmt = ".2f" if "Sharpe" in title else ".2%"
    fig.update_layout(title=title, height=280, margin=dict(l=30, r=10, t=30, b=30),
                      xaxis_title="Date", yaxis_tickformat=yfmt)
    return fig

def fig_histogram(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if hist is None or hist.empty or not {"bin_left","bin_right","freq"}.issubset(hist.columns):
        fig.add_annotation(text="No histogram data found", showarrow=False)
        fig.update_layout(height=280, margin=dict(l=30, r=10, t=30, b=30))
        return fig
    x = (hist["bin_left"].values + hist["bin_right"].values) / 2
    fig.add_trace(go.Bar(x=x, y=hist["freq"], name="Frequency"))
    fig.update_layout(title="Daily Returns Histogram", height=280, margin=dict(l=30, r=10, t=30, b=30),
                      xaxis_title="Return", yaxis_title="Count")
    return fig

def fig_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text=f"No {title.lower()} data found", showarrow=False)
        fig.update_layout(height=400, margin=dict(l=30, r=10, t=30, b=30))
        return fig
    z = df.values.astype(float)
    fig.add_trace(go.Heatmap(z=z, x=list(df.columns), y=list(df.index), coloraxis="coloraxis"))
    fig.update_layout(title=title, height=400, margin=dict(l=30, r=10, t=30, b=30),
                      coloraxis_colorscale="RdBu", coloraxis_cmin=-1, coloraxis_cmax=1)
    return fig

# ----------------------------
# Layout utilities
# ----------------------------
def kpi_card(label: str, value: str, sub: Optional[str] = None):
    return html.Div(
        className="kpi-card",
        children=[html.Div(label, className="kpi-label"),
                  html.Div(value, className="kpi-value"),
                  html.Div(sub or "", className="kpi-sub")],
    )

def navbar():
    return html.Div(
        className="nav",
        children=[
            html.Div("Quant Factor Dashboard", className="brand"),
            html.Div([html.Button("Refresh", id="btn-refresh", n_clicks=0, className="btn"),
                      html.Span(id="status", className="status")], className="nav-actions"),
        ],
    )

# ----------------------------
# App
# ----------------------------
app = Dash(__name__)
server = app.server

STYLE_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', Arial, sans-serif; background:#0f1115; color:#e6e6e6; }
.container { max-width: 1200px; margin: 0 auto; padding: 12px 16px 40px; }
.nav { display:flex; justify-content:space-between; align-items:center; padding:12px 16px; background:#141821; border-bottom:1px solid #272b36; position:sticky; top:0; z-index:5; }
.brand { font-weight:700; letter-spacing:.3px; }
.btn { background:#2b6cb0; color:white; border:none; padding:8px 12px; border-radius:8px; cursor:pointer; }
.btn:hover { filter:brightness(1.05); }
.status { margin-left:12px; font-size:12px; color:#9aa4b2; }
.kpi-row { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin:16px 0; }
.kpi-card { background:#141821; padding:12px 16px; border:1px solid #272b36; border-radius:12px; }
.kpi-label { font-size:12px; color:#9aa4b2; }
.kpi-value { font-size:22px; font-weight:700; margin-top:2px; }
.kpi-sub { font-size:12px; color:#9aa4b2; margin-top:2px; }
.grid-2 { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
.panel { background:#141821; padding:10px 12px; border:1px solid #272b36; border-radius:12px; margin-top:14px; }
.dash-table-container .row { color:#e6e6e6; }
.dash-spreadsheet.dash-freeze-top, .dash-spreadsheet-container { background:#0f1115; }
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-menu { background:#141821; }
"""

app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>Quant Factor Dashboard</title>
        {{%favicon%}}
        {{%css%}}
        <style>{STYLE_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""

app.layout = html.Div([
    dcc.Store(id="ts-refresh", data=str(datetime.utcnow())),
    navbar(),
    html.Div(className="container", children=[
        dcc.Tabs(id="tabs", value="tab-overview", children=[
            dcc.Tab(label="Overview", value="tab-overview"),
            dcc.Tab(label="Risk", value="tab-risk"),
            dcc.Tab(label="Analytics", value="tab-analytics"),
            dcc.Tab(label="Tables", value="tab-tables"),
        ]),
        html.Div(id="tab-body"),
    ]),
])

# ----------------------------
# Tab renderer
# ----------------------------
@app.callback(
    Output("tab-body", "children"),
    Input("tabs", "value"),
    Input("ts-refresh", "data"),
)
def render_tab(tab: str, _ts: str):
    # Load common items each time
    equity, dd = _load_equity_and_drawdown()
    summary = _load_perf_summary()

    # KPI value keys differ between dashboard vs pilot; accept both
    cagr   = summary.get("cagr") or summary.get("CAGR") or 0.0
    vol    = summary.get("vol_annual") or summary.get("vol_ann") or 0.0
    sharpe = summary.get("sharpe") or 0.0
    maxdd  = summary.get("max_drawdown") or summary.get("max_dd") or 0.0

    if tab == "tab-overview":
        kpi = [
            kpi_card("CAGR", f"{float(cagr):.2%}"),
            kpi_card("Vol (ann)", f"{float(vol):.2%}"),
            kpi_card("Sharpe", f"{float(sharpe):.2f}"),
            kpi_card("Max DD", f"{float(maxdd):.2%}"),
        ]
        body = html.Div([
            html.Div(className="kpi-row", children=kpi),
            html.Div(className="panel", children=dcc.Graph(figure=fig_equity(equity), config={"displayModeBar": False})),
            html.Div(className="panel", children=dcc.Graph(figure=fig_drawdown(dd), config={"displayModeBar": False})),
        ])
        return body

    if tab == "tab-risk":
        rvol  = _load_series("rolling_vol")
        rshp  = _load_series("rolling_sharpe")
        hist  = _load_df("returns_hist")
        body = html.Div([
            html.Div(className="grid-2", children=[
                html.Div(className="panel", children=dcc.Graph(figure=fig_rolling(rvol, "Rolling Volatility"), config={"displayModeBar": False})),
                html.Div(className="panel", children=dcc.Graph(figure=fig_rolling(rshp, "Rolling Sharpe"), config={"displayModeBar": False})),
            ]),
            html.Div(className="panel", children=dcc.Graph(figure=fig_histogram(hist), config={"displayModeBar": False})),
        ])
        return body

    if tab == "tab-analytics":
        corr = _load_df("correlations")
        regs = _load_df("regressions")
        kids = [html.Div(className="panel", children=dcc.Graph(figure=fig_heatmap(corr, "Factor Correlations"), config={"displayModeBar": False}))]
        if not regs.empty:
            kids.append(html.Div(className="panel", children=[
                html.H4("Regressions (preview)"),
                dash_table.DataTable(
                    data=regs.head(10).to_dict("records"),
                    columns=[{"name": c, "id": c} for c in regs.columns],
                    page_size=10, style_table={"overflowX": "auto"},
                    style_cell={"backgroundColor": "#0f1115", "color": "#e6e6e6"},
                ),
            ]))
        return html.Div(kids)

    # Tables
    factors = _load_df("factors")
    ranks   = _load_df("ranks")
    weights = _load_df("weights")
    tables = []
    if not factors.empty:
        tables.append(html.Div(className="panel", children=[
            html.H4("Factors (preview)"),
            dash_table.DataTable(
                data=factors.head(200).to_dict("records"),
                columns=[{"name": c, "id": c} for c in factors.columns],
                page_size=20, filter_action="native", sort_action="native",
                style_table={"overflowX": "auto"}, style_cell={"backgroundColor": "#0f1115", "color": "#e6e6e6"},
            ),
        ]))
    if not ranks.empty:
        tables.append(html.Div(className="panel", children=[
            html.H4("Ranks (preview)"),
            dash_table.DataTable(
                data=ranks.head(200).to_dict("records"),
                columns=[{"name": c, "id": c} for c in ranks.columns],
                page_size=20, filter_action="native", sort_action="native",
                style_table={"overflowX": "auto"}, style_cell={"backgroundColor": "#0f1115", "color": "#e6e6e6"},
            ),
        ]))
    if not weights.empty:
        wt = weights.reset_index() if "date" not in weights.columns else weights
        tables.append(html.Div(className="panel", children=[
            html.H4("Weights (preview)"),
            dash_table.DataTable(
                data=wt.head(200).to_dict("records"),
                columns=[{"name": c, "id": c} for c in wt.columns],
                page_size=20, filter_action="native", sort_action="native",
                style_table={"overflowX": "auto"}, style_cell={"backgroundColor": "#0f1115", "color": "#e6e6e6"},
            ),
        ]))
    if not tables:
        tables = [html.Div(className="panel", children=html.Div("No tables available yet."))]
    return html.Div(tables)

# Refresh → re-render
@app.callback(
    Output("ts-refresh", "data"),
    Output("status", "children"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=True,
)
def do_refresh(_):
    ts = datetime.utcnow().strftime("%H:%M:%S UTC")
    return str(datetime.utcnow()), f"Reloaded at {ts}"

if __name__ == "__main__":
    app.run(debug=True)
