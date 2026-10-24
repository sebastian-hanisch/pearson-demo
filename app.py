"""Pearson: eine Zahl, viele Daten - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Erstes Stück der Korrelations-Reihe (Wurzel). Simulierte Standort-Wochen mit bekannter wahrer Korrelation: r als Zahl mit sechs möglichen Bildern, drei Intervallverfahren
gegen die Wahrheit, das Gesetz hinter dem Versagen, der Einfluss eines Punktes, der Fehlalarm bei Unkorreliertheit und der Nutzen von r für Vorhersage und Planung.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import prs_algorithm as A
import prs_constants as C
import prs_evaluation as ev
import prs_shapes as SH
from prs_layout import dunkles_design, kopf_css
from prs_pdf_export import build_pdf
from prs_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from prs_visualization import (
    build_coverage,
    build_influence,
    build_intervals,
    build_law,
    build_leverage,
    build_scatter,
    build_shapes,
    build_size,
    build_utility,
)

st.set_page_config(page_title="Pearson – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


@st.cache_data(show_spinner=False)
def _leverage():
    rows = ev.leverage_grid()
    thresholds = {n: (ev.leverage_threshold(n, 0.5), ev.leverage_threshold(n, 0.95)) for n in C.LEVERAGE_NS}
    return rows, thresholds


@st.cache_data(show_spinner=False)
def _shapes(target):
    return ev.shapes_table(target)


@st.cache_data(show_spinner=False)
def _utility():
    return ev.utility_rows()


@st.cache_data(show_spinner=False)
def _plan_power():
    return [ev.plan_power(rho, fam) for rho in (0.1, 0.3, 0.5) for fam in ("normal", "t5")]


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in prs_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Pearson: eine Zahl, viele Daten")
st.markdown(
    """
**Erstes Stück der Korrelations-Reihe** (Wurzel): Ein Disponent legt zwei Kennzahlen je Standort-Woche nebeneinander, das **Auftragsvolumen** gegen den Plan und die **Überstunden**, und
liest daraus **eine** Zahl, die **Pearson-Korrelation r**. Die Daten sind simuliert, die **wahre Korrelation ist bekannt**; so lässt sich messen, was die Zahl wert ist. Drei Dinge stehen im Mittelpunkt:
**dasselbe r kann ganz verschiedene Bilder tragen**, das übliche **95 %-Intervall trifft bei schweren Rändern nicht** (und wird mit mehr Daten schlechter), und **„unkorreliert“ heißt nicht „unabhängig“**.

Weiter unten: sechs Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Ursachen und Störgrößen gehören nicht hierher
(Linie *Kausale Diagramme und Adjustierung*), Rangmaße kommen im nächsten Stück.
"""
)

with st.expander("So funktioniert die Pearson-Korrelation", expanded=True):
    st.markdown(
        """
1. **Zentrieren:** von x und y den Mittelwert abziehen.
2. **Kreuzprodukt:** r = Summe der Produkte (x − Mittel)(y − Mittel), geteilt durch die Wurzel aus dem Produkt der beiden Quadratsummen. Das Ergebnis liegt zwischen −1 und +1 und misst die **Geradlinigkeit**, nicht die Stärke eines beliebigen Zusammenhangs.
3. **Zusammenhang mit der Geraden:** Steigung = r · s_y / s_x, und r² ist der Anteil der Varianz von y, den die Gerade erklärt.
4. **Wie sicher ist r?** Drei Intervalle stehen zur Wahl:

| Verfahren | Idee | Setzt voraus |
|---|---|---|
| **Fisher-z** | atanh(r) ist ungefähr normalverteilt mit Standardabweichung 1/√(n − 3); Intervall bilden, zurücktransformieren | normalverteilte Paare (genauer: Kurtosis wie die Normalverteilung) |
| **Perzentil-Bootstrap** | Paare mit Zurücklegen neu ziehen, r jedes Mal neu rechnen, die mittleren 95 % der Werte nehmen | unabhängige Beobachtungspaare; bei sehr kleinem n zu eng |
| **BCa-Bootstrap** | wie Perzentil, aber die Quantile werden um eine Verzerrung z₀ und eine Beschleunigung a verschoben (aus Bootstrap- und Leave-one-out-Werten) | wie Perzentil |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten**")
    family = st.selectbox("Verteilung", list(C.FAMILIES), format_func=C.FAMILIES.get, key="family_select",
                          help="Die Familie der Paare. t(ν) und kontaminiert: beide Kennzahlen teilen sich einen Skalenfaktor (Spitzenwochen heben beide zugleich). Lognormal: Mengen, schief.")
    rho = st.slider("Wahre Korrelation ρ (Parameter)", *bounds("rho_slider"), step=0.05, key="rho_slider",
                    help="Parameter der Normalfelder. Bei Lognormal weicht die wahre Korrelation davon ab (die App nennt beide Werte).")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), key="n_slider", help="Standort-Wochen im Datensatz.")
    st.markdown("**Hebelpunkte**")
    lev_count = st.slider("Zahl der Hebelpunkte", *bounds("lev_count_slider"), key="lev_count_slider",
                          help="Ersetzt so viele der ersten Beobachtungen durch weit entfernte Punkte (mindestens drei Beobachtungen bleiben unverändert).")
    lev_k = st.slider("Lage k (Standardabweichungen)", *bounds("lev_k_slider"), step=0.5, key="lev_k_slider", disabled=lev_count == 0,
                      help="Wie weit der Punkt vom Rest liegt, in Standardabweichungen der übrigen Daten. Wirkt nur mit mindestens einem Hebelpunkt.")
    lev_dir = st.selectbox("Richtung", list(C.LEV_DIRS), key="lev_dir_select", disabled=lev_count == 0,
                           help="gleichläufig: Punkt bei (k, k), er hebt r; gegenläufig: (k, −k), er senkt r.")
    st.markdown("**Verfahren**")
    boot_b = st.slider("Bootstrap-Ziehungen B", *bounds("boot_slider"), step=100, key="boot_slider", help="Wie oft die Paare neu gezogen werden (Einzeldatensatz und Wiederholungsexperiment).")
    reps = st.slider("Wiederholungen (Experiment 2)", *bounds("reps_slider"), step=100, key="reps_slider",
                     help=f"Wie viele Datensätze das Abdeckungsexperiment zieht; begrenzt durch das Budget Wiederholungen · B · n ≤ {C.REPS_BUDGET:,}.".replace(",", "."))
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Datensatz ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"family_select": family, "rho_slider": rho, "n_slider": int(n), "lev_count_slider": int(lev_count), "lev_k_slider": lev_k, "lev_dir_select": lev_dir,
                   "boot_slider": int(boot_b), "reps_slider": int(reps), "seed_input": int(seed)})

settings = ev.Settings(family, round(float(rho), 2), int(n), int(lev_count), float(lev_k), lev_dir, int(boot_b), int(reps), int(seed))
a = _analysis(settings)
x_arr, y_arr = np.array(a.x), np.array(a.y)
intercept = float(y_arr.mean() - a.slope * x_arr.mean())

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m1.metric("Gemessenes r", f"{a.r:.2f}", delta=f"{a.r - a.true_r:+.2f} gegen wahr", delta_color="off")
m2.metric("Wahre Korrelation", f"{a.true_r:.2f}", delta="Referenz", delta_color="off")
m3.metric("p-Wert (ρ = 0)", f"{a.p:.3f}" if a.p >= 0.001 else "< 0,001", delta="t-Test", delta_color="off")
st.caption(f"r² = {a.r2:.2f} · Steigung = {a.slope:.2f} · Spearman = {a.spearman:.2f} (Rangmaß, erklärt im nächsten Stück). Der t-Test setzt unabhängige Ränder voraus (Experiment 5).")
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(build_scatter(a.x, a.y, a.lever, a.slope, intercept), width="stretch", key="main_scatter")
    st.caption("Blau: Beobachtungen, Orange: Hebelpunkte, grün: Regressionsgerade. Alle Achsen fest.")
with right:
    st.plotly_chart(build_intervals(a.intervals, a.r, a.true_r), width="stretch", key="main_intervals")
    st.caption("Drei 95 %-Intervalle für ρ, schwarzer Punkt: gemessenes r, gestrichelte Linie: wahre Korrelation der erzeugenden Verteilung.")
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="pearson-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Wo liegt die wahre Korrelation, und wer fängt sie?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)
st.plotly_chart(build_influence(a.influence, a.top), width="stretch", key="influence_chart")
st.caption(f"Einfluss jeder Beobachtung: um wie viel sich r ändert, wenn sie fehlt. Die einflussreichste ist Nummer {a.top + 1} "
           f"(r ändert sich um {a.influence[a.top]:+.3f}); ohne sie läge r bei {a.r - a.influence[a.top]:.2f}.")

st.markdown("---")

# --- E1 Gleiches r, andere Bilder ----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Gleiches r, andere Bilder")
st.caption(f"Sechs Punktwolken mit n = {C.SHAPE_N} und **genau** dem gewählten r (Abweichung unter 10⁻¹⁴). Die Tabelle mittelt über {len(C.SHAPE_SEEDS)} feste Zufallsstände.")
target = st.slider("Zielwert r", C.SHAPE_TARGET_MIN, C.SHAPE_TARGET_MAX, C.DEFAULT_SHAPE_TARGET, step=0.05, key="shape_target")
results = [(key, SH.SHAPES[key].label, SH.solve_shape(key, target, 1)) for key in SH.SHAPES]
st.plotly_chart(build_shapes(results, target), width="stretch", key="shapes_chart")
rows1 = _shapes(round(float(target), 2))
table = ["| Form | lösbar | Spearman (Mittel) | Steigung (Mittel) | größter Einfluss einer Beobachtung |", "|---|---|---|---|---|"]
for r in rows1:
    if "spearman" in r:
        table.append(f"| {r['label']} | {r['solved']} von {r['of']} | {r['spearman'][0]:.2f} [{r['spearman'][1]:.2f}; {r['spearman'][2]:.2f}] | {r['slope'][0]:.2f} | {r['influence'][0]:.2f} |")
    else:
        table.append(f"| {r['label']} | 0 von {r['of']} | nicht erreichbar (erreichbar: {r['range'][0]:.2f} bis {r['range'][1]:.2f}) | | |")
st.markdown("\n".join(table))
kind, text = ev.verdict_shapes(rows1, target)
getattr(st, kind)(text)

st.markdown("---")

# --- E2 Wie sicher ist die Zahl? -----------------------------------------------------------------------------------------------------------

grid = _grid()
st.subheader("🔬 Experiment 2: Wie sicher ist die Zahl? Abdeckung der 95 %-Intervalle")
st.caption("Wie oft enthält ein 95 %-Intervall die wahre Korrelation? Oben: das vorgerechnete Raster (vier Zufalls-Seeds, Bootstrap mit 400 Ziehungen, Streuung über die Seeds als Fehlerbalken). Darunter: Wiederholung bei Ihren Einstellungen.")
if grid is None:
    st.warning("Das vorgerechnete Raster (data/pearson_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`.")
else:
    g1, g2 = st.columns(2)
    gfam = g1.selectbox("Verteilung im Raster", list(C.FAMILIES), format_func=C.FAMILIES.get, index=list(C.FAMILIES).index("t3"), key="grid_family")
    grho = g2.selectbox("ρ im Raster", list(C.GRID_RHOS), index=1, key="grid_rho")
    series = ev.coverage_series(grid, gfam, grho)
    st.plotly_chart(build_coverage(series, C.FAMILIES[gfam]), width="stretch", key="coverage_chart")
    kind, text = ev.verdict_grid_series(series, C.FAMILIES[gfam])
    getattr(st, kind)(text)
st.caption("Die Wiederholung unten nutzt Verteilung, ρ, n, B und Seed der Seitenleiste, aber keine Hebelpunkte.")
if st.button("🔁 Wiederholung bei meinen Einstellungen rechnen", key="btn_cov"):
    with st.spinner(f"Rechne {ev.affordable_reps(settings.reps, settings.boot_b, settings.n)} Wiederholungen …"):
        st.session_state["cov_result"] = (settings, ev.coverage_experiment(settings))
cov = st.session_state.get("cov_result")
if cov and cov[0] == settings:
    res = cov[1]
    if res["runs"] < res["requested"]:
        st.caption(f"Das Budget erlaubt bei B = {res['boot_b']} und n = {settings.n} nur {res['runs']} von {res['requested']} Wiederholungen.")
    kind, text = ev.verdict_coverage(res)
    getattr(st, kind)(text)
    st.caption(f"Mittlere Breite: Fisher-z {res['wf']:.2f}, Perzentil {res['wp']:.2f}, BCa {res['wb']:.2f}.")

st.markdown("---")

# --- E3 Das Gesetz dahinter ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Das Gesetz hinter dem Versagen")
st.markdown(
    "Bei t-verteilten Paaren mit gemeinsamem Skalenfaktor ist die Streuung von r um den Faktor **(ν − 2)/(ν − 4)** größer, als die Fisher-Formel annimmt "
    "(bei ν ≤ 4 ist sie unendlich). Daraus folgt die erwartete Abdeckung 2Φ(1,96/√(1 + κ)) − 1. Die Tabelle stellt Vorhersage und Messung nebeneinander."
)
if grid is not None and grid.get("law"):
    lrho = st.selectbox("ρ", list(C.GRID_RHOS), index=1, key="law_rho")
    law_rows = [r for r in grid["law"] if abs(r["rho"] - lrho) < 1e-9]
    t3 = ["| ν | Faktor vorhergesagt | Faktor gemessen (± Seeds) | Abdeckung vorhergesagt | Abdeckung gemessen |", "|---|---|---|---|---|"]
    for r in law_rows:
        nu = "∞ (Normal)" if r["nu"] == "inf" else r["nu"]
        t3.append(f"| {nu} | {r['predicted']:.3f} | {r['measured']:.3f} (± {r['measured_sd']:.3f}) | {100 * r['cov_predicted']:.1f} % | {100 * r['cov_measured']:.1f} % |")
    st.markdown("\n".join(t3))
    st.plotly_chart(build_law(law_rows), width="stretch", key="law_chart")
    st.caption(f"Vorgerechnet: n = {law_rows[0]['n']}, {law_rows[0]['runs']} Läufe je Zeile (Streuung des Faktors über die vier Seeds in Klammern). ν = 3 hat unendliche Varianz, dort geht die Abdeckung mit n gegen 0 (Experiment 2). "
               "Bei ν = 5 ist der 8. Moment unendlich: der Faktor schwankt von Seed zu Seed stark, das Verdikt zählt nur ν ≥ 6.")
    kind, text = ev.verdict_law(law_rows)
    getattr(st, kind)(text)
if st.button("🔁 Gesetz live nachrechnen (n = 1 000, 400 Läufe)", key="btn_law"):
    with st.spinner("Rechne …"):
        st.session_state["law_live"] = ev.variance_law()
live = st.session_state.get("law_live")
if live:
    st.caption("Live (n = 1 000, ρ = 0,5, 400 Läufe je ν): Der Varianzfaktor streut bei so wenigen Läufen um etwa ±10 %.")
    st.markdown("\n".join(["| ν | vorhergesagt | gemessen | Abdeckung gemessen |", "|---|---|---|---|"]
                          + [f"| {r['nu']} | {r['predicted']:.2f} | {r['measured']:.2f} | {100 * r['cov_measured']:.1f} % |" for r in live]))

st.markdown("---")

# --- E4 Ein einzelner Punkt ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Ein einzelner Punkt")
st.caption("n unabhängige Standardnormalpaare, ein Punkt bei (k, k). Anteil der Läufe, in denen der t-Test „signifikant“ meldet (1 000 Läufe je Zelle).")
lev_rows, lev_thr = _leverage()
st.plotly_chart(build_leverage(lev_rows), width="stretch", key="leverage_chart")
c1, c2 = st.columns(2)
sel_n = c1.selectbox("n", list(C.LEVERAGE_NS), index=1, key="lev_n")
sel_k = c2.selectbox("k", list(C.LEVERAGE_KS), index=C.LEVERAGE_KS.index(5), key="lev_k_sel")
kind, text = ev.verdict_leverage(lev_rows, sel_n, sel_k)
getattr(st, kind)(text)
st.markdown("\n".join(["| n | k für 50 % „signifikant“ | k für 95 % |", "|---|---|---|"]
                      + [f"| {n} | {lo:g} | {hi:g} |" if lo is not None and hi is not None else f"| {n} | – | – |" for n, (lo, hi) in lev_thr.items()]))
st.caption("Schwellen auf dem Raster k = 0, 0,25, 0,5, … mit 500 Läufen; mit wachsendem n muss der Punkt weiter weg liegen.")

st.markdown("---")

# --- E5 Unkorreliert ist nicht unabhängig --------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Unkorreliert ist nicht unabhängig")
st.markdown(
    "Zwei Kennzahlen können unkorreliert sein und sich trotzdem eine **gemeinsame Streuung** teilen (in Spitzenwochen sind beide groß). Der übliche Test auf ρ = 0 geht von "
    "unabhängigen Rändern aus und schlägt dann zu oft an. Der Permutationstest prüft **Unabhängigkeit**, nicht Unkorreliertheit, und teilt das Problem."
)
if grid is not None:
    sn = st.selectbox("n", list(C.SIZE_NS), index=1, key="size_n")
    size_rows = [r for f in C.SIZE_FAMILIES if (r := ev.grid_size_row(grid, f, sn))]
    st.plotly_chart(build_size(size_rows), width="stretch", key="size_chart")
    sfam = st.selectbox("Verteilung für das Verdikt", list(C.SIZE_FAMILIES), format_func=C.ALL_FAMILIES.get, index=2, key="size_family")
    srow = ev.grid_size_row(grid, sfam, sn)
    kind, text = ev.verdict_size(sfam, sn, srow["t"], srow["t_runs"])
    getattr(st, kind)(text)
    st.caption(f"t-Test: {srow['t_runs']} Läufe (vier Seeds), Bootstrap und Permutation: {srow['bp_runs']} Läufe (Monte-Carlo-Fehler etwa ±1 Punkt). Bootstrap-Test: liegt 0 außerhalb des Perzentil-Intervalls?")
if st.button("🔁 Fehlalarm live nachrechnen (meine Verteilung, mein n)", key="btn_size"):
    with st.spinner("Rechne 2 000 Wiederholungen …"):
        st.session_state["size_live"] = (settings.family, settings.n, ev.size_runs(settings.family, max(settings.n, 5), 2000, settings.seed + 3))
live = st.session_state.get("size_live")
if live and live[0] == settings.family and live[1] == settings.n:
    kind, text = ev.verdict_size(live[0], live[1], live[2], 2000)
    getattr(st, kind)(text)

st.markdown("---")

# --- E6 Was bringt r? ----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Was bringt r für die Vorhersage, und wie viele Daten braucht man?")
util = _utility()
st.plotly_chart(build_utility(util), width="stretch", key="utility_chart")
st.markdown("\n".join(["| r | r² | Fehler der Regression (% der Streuung von y) | Formel √(1 − r²) | Vorzeichen stimmt |", "|---|---|---|---|---|"]
                      + [f"| {u['r']:.2f} | {u['r2']:.2f} | {100 * u['rmse_measured']:.1f} % | {100 * u['rmse_formula']:.1f} % | {100 * u['sign_measured']:.1f} % |" for u in util]))
kind, text = ev.verdict_utility(util)
getattr(st, kind)(text)
plan = _plan_power()
st.markdown("**Wie viele Beobachtungen für 80 % Teststärke?** Planwert n = ((1,96 + 0,84)/atanh ρ)² + 3. Gemessene Teststärke (2 000 Läufe, zweiseitig 5 %):")
st.markdown("\n".join(["| wahres ρ | Planwert n | Teststärke Normal | Teststärke t(5), gemeinsame Skala |", "|---|---|---|---|"]
                      + [f"| {rho_} | {pn['n']} | {100 * pn['power']:.1f} % | {100 * pt['power']:.1f} % |" for rho_, pn, pt in zip((0.1, 0.3, 0.5), plan[0::2], plan[1::2])]))
st.caption("Die Teststärke bei t(5) ist nicht um die höhere Fehlalarmrate bereinigt: sie ist nur zusammen mit Experiment 5 lesbar.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Geradlinigkeit:** r misst nur die Geradlinigkeit | dasselbe r = 0,8 trägt Wolke, Kurve und einen einzelnen Punkt (Experiment 1); Parabel oder Kreis ergeben r ≈ 0 trotz Abhängigkeit | Rangmaße (nächstes Stück), nichtlineare Abhängigkeit (Stück 3) |
| **Kein einzelner Punkt dominiert** | ein Punkt bei fünf Standardabweichungen macht bei n = 30 meist aus Unabhängigkeit einen „signifikanten“ Befund (Experiment 4) | Rangmaße (nächstes Stück) |
| **Schwache Ränder:** Fisher-z setzt Kurtosis wie die Normalverteilung voraus | die Abdeckung fällt bei t(3) mit n von 0,85 auf rund 0,5 (Experiment 2 und 3) | Bootstrap hilft nur zum Teil; robuste Maße (Stück 2 und 3) |
| **Unabhängige Beobachtungen:** jede Zeile ein unabhängiges Paar | Reihen in der Zeit verletzen das; dann sind r und der Test irreführend | Scheinkorrelation in Zeitreihen (Stück 4) |
| **Die wahre Korrelation ist bekannt** | nur in der Simulation; bei echten Daten fehlt der Maßstab, die Abdeckung lässt sich nicht messen | – |
| **Eine Zahl für viele Variablen** | bei vielen Variablen entstehen viele Paare, und Zufallstreffer häufen sich | Korrelationsmatrix (Stück 5) |
| **Zusammenhang ist keine Ursache** | r sagt nichts darüber, welche Kennzahl die andere bewegt | Linie *Kausale Diagramme und Adjustierung* |
"""
)
st.caption("Familien und Zahlen sind simuliert; der gemeinsame Skalenfaktor ist eine Modellwahl für Spitzenwochen, kein Befund über echte Standorte. Bootstrap-Raster mit 400 Ziehungen.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Paare $(x_i, y_i)$; gesucht ist die Stärke des geradlinigen Zusammenhangs und ein Intervall für die wahre Korrelation $\rho$.

| Symbol | Bedeutung |
|---|---|
| $x_i, y_i$ | Beobachtungspaar (Auftragsabweichung, Überstunden) |
| $r$ | gemessene Korrelation |
| $\rho$ | wahre Korrelation der erzeugenden Verteilung |
| $n$ | Zahl der Paare |
| $\nu$ | Freiheitsgrade der $t$-Verteilung |
| $\kappa$ | Kurtosis-Größe, $1+\kappa = (\nu-2)/(\nu-4)$ |

**Korrelation.**

$$ r = \frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i (x_i-\bar x)^2 \,\sum_i (y_i-\bar y)^2}} $$

**Steigung und Varianzanteil.**

$$ b = r\,\frac{s_y}{s_x}, \qquad r^2 = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2} $$

**Fisher-Intervall.** Mit $z = \operatorname{artanh} r$ und dem Normalquantil $z_{0{,}975} = 1{,}96$:

$$ \left[\tanh\!\left(z - \frac{1{,}96}{\sqrt{n-3}}\right),\; \tanh\!\left(z + \frac{1{,}96}{\sqrt{n-3}}\right)\right] $$

**Streuung bei schweren Rändern** (elliptische Verteilung, gemessen in Experiment 3, nicht hergeleitet):

$$ n\operatorname{Var}(r) \to (1+\kappa)(1-\rho^2)^2, \qquad \text{Abdeckung} \to 2\Phi\!\left(\frac{1{,}96}{\sqrt{1+\kappa}}\right) - 1 $$

**Vorhersagefehler.** Bei Normalverteilung gilt für die Regression:

$$ \frac{\text{RMSE}}{s_y} = \sqrt{1-r^2} $$

**Planwert** für einen zweiseitigen Test von $\rho = 0$ mit Teststärke 80 %:

$$ n = \left(\frac{z_{0{,}975} + z_{0{,}80}}{\operatorname{artanh}\rho}\right)^2 + 3 $$

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Formeln stützen die Messungen der Experimente.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
