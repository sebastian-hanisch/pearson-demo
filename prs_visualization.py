"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import prs_constants as C

METHOD_COLORS = {"Fisher-z": C.COLOR_FISHER, "Perzentil-Bootstrap": C.COLOR_PERCENTILE, "BCa-Bootstrap": C.COLOR_BCA}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_scatter(x, y, lever=(), slope=None, intercept=None, height=380):
    """Streudiagramm mit Regressionsgerade; Hebelpunkte in Orange."""
    x, y = np.asarray(x), np.asarray(y)
    mask = np.zeros(len(x), dtype=bool)
    mask[list(lever)] = True
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[~mask], y=y[~mask], mode="markers", name="Beobachtungen", marker=dict(color=C.COLOR_POINTS, size=7, opacity=0.8),
                             hovertemplate="x = %{x:.2f}, y = %{y:.2f}<extra></extra>"))
    if mask.any():
        fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers", name="Hebelpunkt", marker=dict(color=C.COLOR_LEVER, size=11, symbol="diamond"),
                                 hovertemplate="x = %{x:.2f}, y = %{y:.2f}<extra></extra>"))
    if slope is not None:
        xs = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(x=xs, y=intercept + slope * xs, mode="lines", name="Regressionsgerade", line=dict(color=C.COLOR_LINE, width=2.5), hoverinfo="skip"))
    fig.update_xaxes(title_text="x: Abweichung des Auftragsvolumens vom Plan")
    fig.update_yaxes(title_text="y: Überstunden")
    return _base(fig, height)


def build_intervals(intervals, r, true_r):
    """Die drei Intervalle als waagerechte Balken, die gemessene Korrelation als Punkt, die wahre als senkrechte Linie."""
    names = list(intervals)
    fig = go.Figure()
    for name in names:
        lo, hi = intervals[name]
        fig.add_trace(go.Scatter(x=[lo, hi], y=[name, name], mode="lines+markers", name=name, line=dict(color=METHOD_COLORS[name], width=5),
                                 marker=dict(symbol="line-ns-open", size=14, color=METHOD_COLORS[name]),
                                 hovertemplate=f"{name}: %{{x:.3f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r] * len(names), y=names, mode="markers", name="gemessenes r", marker=dict(color=C.COLOR_TRUE, size=9), hovertemplate="r = %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[true_r, true_r], y=[names[0], names[-1]], mode="lines", name=f"wahr: {true_r:.2f}", line=dict(color=C.COLOR_TRUE, dash="dash", width=2),
                             hovertemplate="wahre Korrelation %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Korrelation", range=[-1.05, 1.05])
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=names)
    fig.update_layout(showlegend=False)
    return _base(fig, 260)


def build_influence(infl, top):
    """r − r_(−i) je Beobachtung: wie viel r sich ändert, wenn der Punkt fehlt."""
    idx = list(range(1, len(infl) + 1))
    colors = [C.COLOR_LEVER if i == top else C.COLOR_POINTS for i in range(len(infl))]
    fig = go.Figure(go.Bar(x=idx, y=list(infl), marker_color=colors, hovertemplate="Beobachtung %{x}: Änderung %{y:+.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Beobachtung", type="category" if len(infl) <= 40 else "linear")
    fig.update_yaxes(title_text="r − r ohne diesen Punkt")
    return _base(fig, 280)


def build_shapes(results, target):
    """Sechs Punktwolken mit demselben r (2 × 3)."""
    labels = [f"{lab}: r = {target:.2f}" for _, lab, _ in results]
    fig = make_subplots(rows=2, cols=3, subplot_titles=labels, horizontal_spacing=0.06, vertical_spacing=0.18)
    for i, (key, lab, res) in enumerate(results):
        row, col = divmod(i, 3)
        if res is None:
            fig.add_annotation(text="nicht erreichbar", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=row + 1, col=col + 1)
            continue
        fig.add_trace(go.Scatter(x=list(res.x), y=list(res.y), mode="markers", marker=dict(color=C.COLOR_POINTS, size=5, opacity=0.85), showlegend=False,
                                 hovertemplate="x = %{x:.2f}, y = %{y:.2f}<extra></extra>"), row=row + 1, col=col + 1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)")
    for ann in fig.layout.annotations:
        ann.font.size = 12
    return lock_axes(fig)


def build_coverage(rows, family_label):
    """Abdeckung der drei Intervalle gegen n (Raster), mit der 95-%-Linie."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    for name, key, color in (("Fisher-z", "fisher", C.COLOR_FISHER), ("Perzentil-Bootstrap", "perc", C.COLOR_PERCENTILE), ("BCa-Bootstrap", "bca", C.COLOR_BCA)):
        fig.add_trace(go.Scatter(x=ns, y=[100 * r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=3),
                                 error_y=dict(type="data", array=[100 * r[key + "_sd"] for r in rows], visible=True, thickness=1),
                                 hovertemplate="n = %{x}: %{y:.1f} %<extra>" + name + "</extra>"))
    fig.add_hline(y=95, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_xaxes(title_text=f"Stichprobengröße n ({family_label})", type="log", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="Abdeckung (%)", range=[40, 100])
    return _base(fig)


def build_law(rows):
    """Vorhergesagte und gemessene Fisher-Abdeckung je ν (Balken)."""
    rows = [r for r in rows if r["rho"] == rows[0]["rho"]]
    labels = ["∞ (Normal)" if r["nu"] == "inf" else f"ν = {r['nu']}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * r["cov_predicted"] for r in rows], name="vorhergesagt: 2Φ(1,96/√(1+κ)) − 1", marker_color="rgba(150,150,150,0.7)"))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["cov_measured"] for r in rows], name="gemessen", marker_color=C.COLOR_FISHER))
    fig.add_hline(y=95, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Abdeckung des Fisher-Intervalls (%)", range=[60, 100])
    return _base(fig, 340)


def build_leverage(rows, ns=C.LEVERAGE_NS):
    """Anteil „signifikant“ gegen die Lage k des Hebelpunkts, eine Linie je n."""
    palette = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    fig = go.Figure()
    for i, n in enumerate(ns):
        sub = [r for r in rows if r["n"] == n]
        fig.add_trace(go.Scatter(x=[r["k"] for r in sub], y=[100 * r["share"] for r in sub], mode="lines+markers", name=f"n = {n}", line=dict(color=palette[i % 5], width=3),
                                 hovertemplate="k = %{x}: %{y:.1f} %<extra>n = " + str(n) + "</extra>"))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_xaxes(title_text="Lage des Punktes k (Standardabweichungen)")
    fig.update_yaxes(title_text="Anteil „signifikant“ (%)", range=[0, 102])
    return _base(fig)


def build_size(rows):
    """Fehlalarmrate je Familie für t-Test, Bootstrap und Permutation bei einem n."""
    labels = [C.ALL_FAMILIES[r["family"]].split(",")[0].split(" (")[0] if r["family"] != "tindep3" else "t(3) unabhängig" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * r["t"] for r in rows], name="t-Test", marker_color=C.COLOR_FISHER))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["boot"] for r in rows], name="Bootstrap-Intervall", marker_color=C.COLOR_PERCENTILE))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["perm"] for r in rows], name="Permutationstest", marker_color="rgba(150,150,150,0.8)"))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Fehlalarm bei wahrem ρ = 0 (%)")
    return _base(fig)


def build_utility(rows):
    """Vorhersagefehler (RMSE geteilt durch Streuung von y) und Vorzeichen-Trefferquote gegen r."""
    rs = [r["r"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs, y=[100 * r["rmse_measured"] for r in rows], mode="lines+markers", name="verbleibender Fehler (% der Streuung von y)", line=dict(color=C.COLOR_LINE, width=3)))
    fig.add_trace(go.Scatter(x=rs, y=[100 * r["sign_measured"] for r in rows], mode="lines+markers", name="Vorzeichen stimmt (% der Paare)", line=dict(color=C.COLOR_PERCENTILE, width=3)))
    fig.update_xaxes(title_text="Korrelation r")
    fig.update_yaxes(title_text="Prozent", range=[0, 102])
    return _base(fig, 340)
