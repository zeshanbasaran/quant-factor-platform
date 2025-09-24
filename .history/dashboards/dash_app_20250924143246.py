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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html

ROOT = Path(__file__).resolve().parents[1]
DASH_DIR = ROOT / "data" / "processed" / "dashboard"

# ----------------------------
# Loading helpers (defensive)
# ----------------------------

def _read_series(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    if path.suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path.with_suffix(".csv")) if path.with_suffix(".csv").exists() else pd.DataFrame()
    else:
        df = pd.read_parquet(path)
    if isinstance(df, pd.DataFrame) and not df.empty:
        s = df.iloc[:, 0]
        # parse index as datetime if possible
        try:
            s.index = pd.to_datetime(df.index)
        except Exception:
            pass
        return s
    return pd.Series(dtype=float)


def _read_df(name: str) -> pd.DataFrame:
    p = DASH_DIR / f"{name}.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            csvp = p.with_suffix(".csv")
            if csvp.exists():
                return pd.read_csv(csvp)
    return pd.DataFrame()


def _read_json(name: str) -> Dict[str, Any]:
    p = DASH_DIR / f"{name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


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
    fig.update_layout(title="Equity Curve", height=340, margin=dict(l=30, r=10, t=30, b=30), xaxis_title="Date", yaxis_title="Portfolio Value")
    return fig


def fig_drawdown(dd: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if dd is None or dd.empty or "drawdown" not in dd.columns:
        fig.add_annotation(text="No drawdown data found", showarrow=False)
        fig.update_layout(height=260, margin=dict(l=30, r=10, t=30, b=30))
        return fig
    fig.add_trace(go.Scatter(x=dd.index, y=dd["drawdown"], mode="lines", name="Drawdown", fill="tozeroy"))
    fig.update_layout(title="Drawdown", height=260, margin=dict(l=30, r=10, t=30, b=30), xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".0%")
    return fig


def fig_rolling(series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    if series is None or series.empty:
        fig.add_annotation(text=f"No {title.lower()} data found", showarrow=False)
        fig.update_layout(height=280, margin=dict(l=30, r=10, t=30, b=30))
        return fig
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=title))
    yfmt = ".2f" if "Sharpe" in title else ".2%"
    fig.update_layout(title=title, height=280, margin=dict(l=30, r=10, t=30, b=30), xaxis_title="Date", yaxis_tickformat=yfmt)
    return fig


def fig_histogram(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if hist is None or hist.empty:
        fig.add_annotation(text="No histogram data found", showarrow=False)
        fig.update_layout(height=280, margin=dict(l=30, r=10, t=30, b=30))
        return fig
    x = (hist["bin_left"].values + hist["bin_right"].values) / 2
    fig.add_trace(go.Bar(x=x, y=hist["freq"], name="Frequency"))
    fig.update_layout(title="Daily Returns Histogram", height=280, margin=dict(l=30, r=10, t=30, b=30), xaxis_title="Return", yaxis_title="Count")
    return fig


def fig_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text=f"No {title.lower()} data found", showarrow=False)
        fig.update_layout(height=400, margin=dict(l=30, r=10, t=30, b=30))
        return fig
    z = df.values.astype(float)
    fig.add_trace(go.Heatmap(z=z, x=list(df.columns), y=list(df.index), coloraxis="coloraxis"))
    fig.update_layout(title=title, height=400, margin=dict(l=30, r=10, t=30, b=30), coloraxis_colorscale="RdBu", coloraxis_cmin=-1, coloraxis_cmax=1)
    return fig


# ----------------------------
# Layout helpers
# ----------------------------

def kpi_card(label: str, value: str, sub: Optional[str] = None):
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub or "", className="kpi-sub"),
        ],
    )


def navbar():
    return html.Div(
        className="nav",
        children=[
            html.Div("Quant Factor Dashboard", className="brand"),
            html.Div([
                html.Button("Refresh", id="btn-refresh", n_clicks=0, className="btn"),
                html.Span(id="status", className="status"),
            ], className="nav-actions"),
        ],
    )


# ----------------------------
# Build App
# ----------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    dcc.Store(id="ts-refresh", data=str(datetime.utcnow())),  # used to trigger reload
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
    # Load all at render time to reflect new files
    equity = _read_series(DASH_DIR / "equity.parquet")
    dd = _read_df("drawdown")
    returns = _read_series(DASH_DIR / "returns.parquet")
    summary = _read_json("perf_summary")
    rvol = _read_series(DASH_DIR / "rolling_vol.parquet")
    rshp = _read_series(DASH_DIR / "rolling_sharpe.parquet")
    hist = _read_df("returns_hist")

    if tab == "tab-overview":
        # KPIs
        kpi = [
            kpi_card("CAGR", f"{summary.get('cagr', 0.0):.2%}"),
            kpi_card("Vol (ann)", f"{summary.get('vol_ann', 0.0):.2%}"),
            kpi_card("Sharpe", f"{summary.get('sharpe', 0.0):.2f}"),
            kpi_card("Max DD", f"{summary.get('max_dd', 0.0):.2%}"),
        ]
        body = html.Div([
            html.Div(className="kpi-row", children=kpi),
            html.Div(className="panel", children=dcc.Graph(figure=fig_equity(equity), config={"displayModeBar": False})),
            html.Div(className="panel", children=dcc.Graph(figure=fig_drawdown(dd), config={"displayModeBar": False})),
        ])
        return body

    if tab == "tab-risk":
        body = html.Div([
            html.Div(className="grid-2", children=[
                html.Div(className="panel", children=dcc.Graph(figure=fig_rolling(rvol, "Rolling Volatility"), config={"displayModeBar": False})),
                html.Div(className="panel", children=dcc.Graph(figure=fig_rolling(rshp, "Rolling Sharpe"), config={"displayModeBar": False})),
            ]),
            html.Div(className="panel", children=dcc.Graph(figure=fig_histogram(hist), config={"displayModeBar": False})),
        ])
        return body

    if tab == "tab-analytics":
        corr = _read_df("correlations")
        regs = _read_df("regressions")
        kids = []
        kids.append(html.Div(className="panel", children=dcc.Graph(figure=fig_heatmap(corr, "Factor Correlations"), config={"displayModeBar": False})))
        if not regs.empty:
            # Show head(10) as a table preview
            kids.append(html.Div(className="panel", children=[
                html.H4("Regressions (preview)"),
                dash_table.DataTable(
                    data=regs.head(10).to_dict("records"),
                    columns=[{"name": c, "id": c} for c in regs.columns],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"backgroundColor": "#0f1115", "color": "#e6e6e6"},
                ),
            ]))
        return html.Div(kids)

    # Tables tab
    factors = _read_df("factors")
    ranks = _read_df("ranks")
    weights = _read_df("weights")
    tables = []
    if not factors.empty:
        tables.append(html.Div(className="panel", children=[
            html.H4("Factors (preview)"),
            dash_table.DataTable(
                data=factors.head(200).to_dict("records"),
                columns=[{"name": c, "id": c} for c in factors.columns],
                page_size=20,
                filter_action="native",
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#0f1115", "color": "#e6e6e6"},
            ),
        ]))
    if not ranks.empty:
        tables.append(html.Div(className="panel", children=[
            html.H4("Ranks (preview)"),
            dash_table.DataTable(
                data=ranks.head(200).to_dict("records"),
                columns=[{"name": c, "id": c} for c in ranks.columns],
                page_size=20,
                filter_action="native",
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#0f1115", "color": "#e6e6e6"},
            ),
        ]))
    if not weights.empty:
        tables.append(html.Div(className="panel", children=[
            html.H4("Weights (preview)"),
            dash_table.DataTable(
                data=weights.reset_index().head(200).to_dict("records"),
                columns=[{"name": c, "id": c} for c in weights.reset_index().columns],
                page_size=20,
                filter_action="native",
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#0f1115", "color": "#e6e6e6"},
            ),
        ]))

    if not tables:
        tables = [html.Div(className="panel", children=html.Div("No tables available yet."))]

    return html.Div(tables)


# Refresh button simply stamps a new time into the Store so the tab callback re-runs
@app.callback(
    Output("ts-refresh", "data"),
    Output("status", "children"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=True,
)
def do_refresh(n: int):
    ts = datetime.utcnow().strftime("%H:%M:%S UTC")
    return str(datetime.utcnow()), f"Reloaded at {ts}"


if __name__ == "__main__":
    app.run_server(debug=True)
