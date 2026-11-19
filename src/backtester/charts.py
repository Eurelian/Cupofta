"""Plotly figures for the UI.

Colour follows the entity and is assigned in a fixed order (never cycled by rank):
slot 1 = the strategy / first run, slot 2 = buy & hold / second run, and so on.
Single axis everywhere; runs on different capital are compared indexed to 100.
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

# Validated categorical order (light-surface steps).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Single-hue sequential ramp (blue 100 -> 700) for magnitude.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging: orange pole, neutral gray midpoint, blue pole.
DIVERGING = [[0.0, "#d95926"], [0.5, "#e7e6e2"], [1.0, "#2a78d6"]]

MAX_SERIES = len(SERIES)


def _layout(fig: go.Figure, title: str, y_title: str, height: int = 380, pct: bool = False):
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        height=height,
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
        yaxis={"title": y_title, "tickformat": ".0%" if pct else ",.0f", "zeroline": False},
        xaxis={"showgrid": False, "showspikes": True, "spikemode": "across", "spikethickness": 1},
    )
    return fig


def equity_chart(curves: dict[str, pl.DataFrame], indexed: bool = False, title: str = "Equity"):
    """curves: name -> frame with date, equity. Order of the dict fixes the colours."""
    fig = go.Figure()
    for i, (name, df) in enumerate(list(curves.items())[:MAX_SERIES]):
        y = df["equity"]
        if indexed:
            y = y / y[0] * 100
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=y,
                name=name,
                mode="lines",
                line={"color": SERIES[i], "width": 2},
                hovertemplate="%{y:,.2f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(showlegend=len(curves) > 1)
    return _layout(fig, title, "Indexed (start = 100)" if indexed else "Equity")


def drawdown_chart(curves: dict[str, pl.DataFrame], title: str = "Drawdown"):
    fig = go.Figure()
    for i, (name, df) in enumerate(list(curves.items())[:MAX_SERIES]):
        eq = df["equity"]
        dd = eq / eq.cum_max() - 1
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=dd,
                name=name,
                mode="lines",
                line={"color": SERIES[i], "width": 2},
                fill="tozeroy" if len(curves) == 1 else None,
                fillcolor="rgba(42,120,214,0.15)",
                hovertemplate="%{y:.1%}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(showlegend=len(curves) > 1)
    return _layout(fig, title, "From peak", height=260, pct=True)


def price_with_position(df: pl.DataFrame, title: str = "Price & exposure"):
    """Close price, with bars while a position is held highlighted."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            name="Close",
            mode="lines",
            line={"color": "#8a8984", "width": 1.5},
            hovertemplate="%{y:,.2f}<extra>Close</extra>",
        )
    )
    in_mkt = df.with_columns(
        held=pl.when(pl.col("position") != 0).then(pl.col("close")).otherwise(None)
    )
    fig.add_trace(
        go.Scatter(
            x=in_mkt["date"],
            y=in_mkt["held"],
            name="In position",
            mode="lines",
            line={"color": SERIES[0], "width": 2},
            connectgaps=False,
            hovertemplate="%{y:,.2f}<extra>In position</extra>",
        )
    )
    return _layout(fig, title, "Price", height=300)


def trade_returns_chart(trades: pl.DataFrame, title: str = "Trade returns"):
    if trades.is_empty():
        return None
    color = [SERIES[0] if r > 0 else SERIES[1] for r in trades["return"]]
    fig = go.Figure(
        go.Bar(
            x=trades["exit_date"],
            y=trades["return"],
            marker={"color": color, "cornerradius": 4},
            customdata=trades["entry_date"],
            hovertemplate="Entry %{customdata}<br>Exit %{x}<br>%{y:.2%}<extra></extra>",
            name="Trade",
        )
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    return _layout(fig, title, "Return", height=260, pct=True)


def sweep_heatmap(table: pl.DataFrame, x: str, y: str, metric: str = "sharpe"):
    """2-parameter grid as a heatmap of `metric` (diverging around 0 for signed metrics)."""
    piv = table.group_by(y, x).agg(pl.col(metric).max()).pivot(on=x, index=y, values=metric).sort(y)
    xs = sorted(c for c in piv.columns if c != y)
    xs_sorted = sorted(
        xs, key=lambda v: float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v
    )
    z = piv.select(xs_sorted).to_numpy()
    signed = metric in {"sharpe", "sortino", "calmar", "total_return", "cagr", "avg_trade"}
    zmax = float(abs(table[metric]).max() or 1)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(v) for v in xs_sorted],
            y=[str(v) for v in piv[y]],
            colorscale=DIVERGING if signed else SEQUENTIAL,
            zmid=0 if signed else None,
            zmin=-zmax if signed else None,
            zmax=zmax if signed else None,
            xgap=2,
            ygap=2,
            hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y}}<br>{metric}=%{{z:.3f}}<extra></extra>",
            colorbar={"title": metric},
        )
    )
    fig.update_layout(hovermode="closest")
    fig = _layout(fig, f"{metric} by {x} × {y}", y, height=420)
    fig.update_xaxes(title=x, type="category", showspikes=False)
    fig.update_yaxes(type="category", tickformat=None)
    return fig


def folds_chart(folds: pl.DataFrame, title: str = "Out-of-sample return per fold"):
    vals = folds["oos_total_return"]
    fig = go.Figure(
        go.Bar(
            x=[f"Fold {f}" for f in folds["fold"]],
            y=vals,
            marker={"color": [SERIES[0] if v > 0 else SERIES[1] for v in vals], "cornerradius": 4},
            hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    return _layout(fig, title, "Return", height=280, pct=True)
