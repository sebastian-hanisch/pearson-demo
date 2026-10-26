# Pearson: eine Zahl, viele Daten – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-pearson-demo.streamlit.app/)** -->

Erstes Stück (Wurzel) der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es ist zugleich die **erste Demo des Portfolios, die die Korrelation selbst zum Gegenstand macht**: In den übrigen Demos ist sie nur ein Hilfsmaß. Vehikel sind simulierte Standort-Wochen mit zwei Kennzahlen (x: Abweichung des Auftragsvolumens vom Plan, y: Überstunden) und **bekannter wahrer Korrelation**; so lässt sich messen, was die Zahl r und ihre 95 %-Intervalle wert sind. Ursachen und Störgrößen gehören der Linie *Kausale Diagramme und Adjustierung*, Rangmaße dem nächsten Stück (nicht `korreliert-demo`: das ist die Spieltheorie-Demo zu korrelierten Gleichgewichten).

## Warum dieses Problem

Die Korrelation ist die Zahl, zu der man als Erstes greift. Die Demo fragt, was sie sagt und wann die Sicherheitsangabe nicht trägt, und kommt auf drei Befunde: **dasselbe r trägt ganz verschiedene Bilder**, das übliche **Fisher-Intervall verfehlt bei schweren Rändern die 95 % und wird bei t(3), t(5) und Lognormal mit mehr Daten schlechter** (bei kontaminierten Daten bleibt es ab n = 100 bei 55,5 bis 58,6 %), und **„unkorreliert“ heißt nicht „unabhängig“**: gemeinsame Streuung (Spitzenwochen) lässt den Test auf ρ = 0 bei wahrem ρ = 0 bei t(3) und kontaminierten Daten in 23 bis 45 % der Läufe anschlagen (t(5): 13 bis 19 %, t(10): 8 bis 9 %).

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-1-pearson.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit dem Zufallsgenerator von numpy. Gebaut ist alles mit SplitMix64 (plattformunabhängig); die Zahlen weichen daher leicht ab, die Befunde bleiben:

- **t(3), ρ = 0,5:** Fisher-z deckt 85,0 / 76,6 / 68,3 / 50,5 % bei n = 10 / 30 / 100 / 1 000 (Plan: 85,0 / 76,2 / 67,3 / 51,9 %).
- **Formen:** bei r = 0,8 liegt das mittlere Spearman zwischen 0,08 und 0,83 und die mittlere Steigung zwischen 0,57 und 1,73 (Plan: 0,06 bis 0,83 und 0,56 bis 1,78).
- **Erreichbarkeit:** die gekrümmte Kurve ist bei r = 0,95 in 33 von 50 Seeds erreichbar, die U-Form bei r = 0,3 in 42 von 50 (Plan: 11 von 20 bzw. 18 von 20).
- **Hebelpunkt-Schwellen:** 50 % „signifikant“ ab k = 4,0 / 4,0 / 5,0 / 6,25 für n = 10 / 30 / 100 / 300 (Plan: 3,75 / 4,0 / 5,0 / 6,25), 95 % ab 6,25 / 5,75 / 6,75 / 8,5 (Plan: 5,75 / 5,75 / 6,75 / 8,5).
- **Varianzgesetz bei ν = 5:** der gemessene Faktor ist **nicht stabil**: bei ρ = 0,5 im Mittel 4,27 mit einer Standardabweichung von 3,11 über die vier Seeds (Vorhersage 3,0); die Messung von ν = 5 gilt deshalb nur grob (der 8. Moment ist unendlich).
- **Laufzeit der Messreihe:** das vorgerechnete Raster braucht 486 Sekunden auf 14 Prozessen (Plan: unter 3 Minuten, gemessen mit weniger Läufen).
- **Nicht aus dem Plan umgesetzt:** das Preset „Gleiches r, andere Bilder“ (Experiment 1 steht mit eigenem Regler im Hauptbereich); Rangintervalle (Stück 2).

## Modell

Paare (x, y) aus sieben Familien, alle mit bekannter wahrer Korrelation: **Normal**; **t(10), t(5), t(3)** und **kontaminiert** (5 % der Wochen mit 5-facher Streuung) mit **gemeinsamem Skalenfaktor** für beide Kennzahlen (eine Spitzenwoche hebt beide zugleich; die Korrelation bleibt ρ, die Kennzahlen sind aber nicht unabhängig); **Lognormal σ = 0,5 und 1** (Mengen; das wahre r weicht von ρ ab: (e^(σ²ρ) − 1)/(e^(σ²) − 1)). Dazu gibt es für Experiment 5 t(3) mit unabhängigen Rändern. Zufall nur aus SplitMix64 (`prs_rng.py` ist eine wörtliche Kopie aus `reduktionskette-demo`; `prs_scenario.py` rechnet denselben Strom vektorisiert). Formeln im 📐-Expander der App.

## Methodik

- **Fisher-z** (artanh r ± 1,96/√(n − 3)), **Perzentil-Bootstrap** und **BCa-Bootstrap** (Verzerrungskorrektur z₀ und Beschleunigung a aus Leave-one-out-Werten), t-Test auf ρ = 0 und Permutationstest, alle in `prs_algorithm.py` ohne scipy (Fehlerfunktion, Beta-Kettenbruch für die Student-Verteilung).
- **Sechs Formen mit exakt gleichem r** (`prs_shapes.py`): Wolke, Hebelpunkt, Gegenausreißer, zwei Gruppen, U-Form mit Schräge, gekrümmte Kurve; ein Formparameter, Zielwert per Bisektion (Abweichung unter 10⁻¹²).
- **Experimente:** (1) Formen, (2) Abdeckung der Intervalle (Raster vorgerechnet, Wiederholung live), (3) Varianzgesetz, (4) Hebelpunkt, (5) Fehlalarm bei Unkorreliertheit, (6) Nutzen von r und Planwert.
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/pearson_grid.json`): 7 Familien · 7 Größen (10 bis 1 000) · 3 ρ (0; 0,5; 0,9) mit je vier Zufalls-Seeds zu 500 Läufen (n = 1 000: 1 000), Bootstrap mit 400 Ziehungen; dazu Fehlalarmrate (t-Test 4 · 2 000 Läufe, Bootstrap und Permutation je 500) und Varianzgesetz (n = 4 000, 4 · 500 Läufe).

## Befunde (gemessen, keine Behauptungen)

Alle Messzahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, Live-Werte über dieselben Funktionen wie in der App); Laufzeit- und Mutationsangaben sind dort nicht erfasst.

| Frage | Befund |
|---|---|
| Wie verschieden kann dasselbe r aussehen? (Experiment 1, n = 40, 50 Seeds) | Bei **r = 0,8** haben die sechs konstruierten Formen ein mittleres Spearman von **0,08 bis 0,83** und eine mittlere Steigung von **0,57 bis 1,73** (die Spannweite gehört zu diesen sechs Formen, nicht zu allem, was dasselbe r tragen kann). Beim Hebelpunkt ist Spearman 0,08, und sein einziger Punkt trägt das ganze r (Weglassen: Änderung 0,80); beim Gegenausreißer senkt ein Punkt r um 0,17. |
| Hält das Fisher-Intervall bei Normalverteilung? | Ja: **93,6 bis 95,3 %** in allen 21 Zellen (7 Größen · 3 ρ). Der Perzentil-Bootstrap deckt bei n = 10 nur 90,1 % (ρ = 0,5). |
| Und bei schweren Rändern? (ρ = 0,5, n = 10 / 20 / 30 / 50 / 100 / 300 / 1 000) | **t(3):** Fisher-z 85,0 / 79,7 / 76,6 / 73,4 / 68,3 / 59,2 / **50,5 %**, Perzentil-Bootstrap 89,2 / 89,5 / 90,4 / 91,5 / 91,6 / 92,4 / 92,4 %, BCa 90,8 / 88,8 / 87,4 / 87,0 / 87,0 / 87,7 / 85,9 %. **t(5):** Fisher-z 90,0 → 79,7 %. **Lognormal σ = 1:** 86,0 → 65,6 %. **Kontaminiert:** 82,4 % bei n = 10, ab n = 100 zwischen 55,5 und 58,6 %. Bei t(3), t(5) und Lognormal **sinkt die Abdeckung mit n**; bei kontaminierten Daten bleibt sie ab n = 100 zwischen 55,5 und 58,6 %. |
| Hängt das an ρ? | Bei t(3), n = 1 000 kaum (52,0 / 50,5 / 50,9 % für ρ = 0 / 0,5 / 0,9). Bei **Lognormal σ = 1** stark, weil das wahre r mit ρ wandert: 95,4 % (ρ = 0, wahres r 0), 65,6 % (ρ = 0,5, wahres r 0,378), 48,2 % (ρ = 0,9, wahres r 0,849). |
| Wie breit sind die Intervalle? | t(3), n = 1 000, ρ = 0,5: Fisher 0,092, Perzentil-Bootstrap 0,241. Bei Normalverteilung 0,093 und 0,092. Das Fisher-Intervall ist zu schmal, nicht verzerrt. |
| Rettet BCa den Bootstrap? | **Nein, bei schweren Rändern ist BCa schlechter als das Perzentil-Intervall:** bei t(3) liegt es in 18 von 21 Zellen darunter, bei „kontaminiert“ in 19 von 21. Die Ursache ist nicht untersucht. Die Implementierung stimmt mit `scipy.stats.bootstrap(method="BCa")` überein (Orakel-Test, Toleranz 0,015). |
| Wie stabil sind diese Zahlen? | Die Standardabweichung der Fisher-Abdeckung über die vier Seeds beträgt höchstens 4,4 Punkte (n ≤ 300) bzw. 2,8 Punkte (n = 1 000). |
| Gilt das Varianzgesetz? (Experiment 3, n = 4 000, 4 · 500 Läufe) | Der gemessene Faktor n·Var(r)/(1 − ρ²)² weicht für ν ≥ 6 im Mittel der vier Seeds um höchstens **2 % (ρ = 0), 4 % (ρ = 0,5) bzw. 6 % (ρ = 0,9)** von (ν − 2)/(ν − 4) ab; dieser Mittelwert hat bei ν = 6 selbst einen Standardfehler von rund 3 bis 5 %, die kleinen Abweichungen sind also teils Glück und kein Maß für die Genauigkeit des Gesetzes. Die vorhergesagte Abdeckung ist 74,2 / 83,4 / 89,0 / 91,0 / 93,5 / 95,0 % für ν = 5 / 6 / 8 / 10 / 20 / ∞; gemessen (ρ = 0,5) 83,3 / 89,5 / 91,1 / 93,8 % für ν = 6 / 8 / 10 / 20. Bei ν = 5 streut der Faktor stark (ρ = 0,5: **4,27** bei Standardabweichung 3,11 über die Seeds). |
| Was macht ein einzelner Punkt? (Experiment 4, 1 000 Läufe, k = 5) | Der t-Test schlägt bei n = 10 / 30 / 100 / 300 in **84,2 / 83,9 / 52,0 / 23,5 %** der Läufe an (ohne Punkt 4,2 / 4,3 / 5,7 / 4,4 %). Bei n = 30: k = 2 / 3 / 4 / 6 / 8 geben 8,0 / 22,9 / 52,6 / 98,0 / 100 %. Für 50 % bzw. 95 % muss der Punkt bei n = 10 / 30 / 100 / 300 mindestens bei k = 4,0 / 4,0 / 5,0 / 6,25 bzw. 6,25 / 5,75 / 6,75 / 8,5 liegen (Raster mit Schritt 0,25 und 500 Läufen je Zelle: Abweichungen um einen Schritt sind Messrauschen). Im Preset „Ein einzelner Punkt“ (Seed 35, ρ = 0) ist p = 0,0065. |
| Schlägt der Test bei wahrem ρ = 0 zu oft an? (Experiment 5, n = 30 / 100 / 300) | **Unabhängige Ränder stören nicht:** Normal 5,3 / 5,0 / 4,8 %, t(3) unabhängig 5,6 / 5,0 / 5,0 %, Lognormal σ = 1 5,1 / 4,8 / 4,1 %. **Gemeinsame Streuung schon:** t(3) **22,9 / 32,6 / 40,1 %**, kontaminiert **30,7 / 42,0 / 44,7 %**, t(5) 13,5 / 16,4 / 19,2 %, t(10) 7,7 / 8,1 / 9,3 %. |
| Hilft ein anderer Test? | Der **Permutationstest teilt das Problem** (t(3): 22,4 / 27,6 / 39,6 %, kontaminiert 26,4 / 37,0 / 41,0 %), denn er prüft Unabhängigkeit, nicht Unkorreliertheit. Der Bootstrap-Test (liegt 0 im Intervall?) bleibt bei 4,6 bis 11,2 % (500 Läufe, Monte-Carlo-Fehler nahe 5 % etwa ±1 Punkt, bei Raten um 30 bis 40 % bis ±2 Punkte); er liegt selbst bei Normalverteilung bei 5,6 bis 8,0 % und ist damit leicht zu liberal. |
| Was bringt r für die Vorhersage? (Experiment 6) | Bei **r = 0,5** erklärt die Gerade 25 % der Varianz, verkleinert den Vorhersagefehler aber nur um **13 %** (RMSE-Verhältnis 0,866); das Vorzeichen stimmt in 66,7 % der Paare. Bei r = 0,9: 0,436 und 85,6 %. |
| Wie viele Beobachtungen für 80 % Teststärke? | Planwert n = 783 / 85 / 30 für ρ = 0,1 / 0,3 / 0,5 (n = ((1,96 + 0,8416)/atanh ρ)² + 3, aufgerundet). Gemessene Teststärke bei Normalverteilung 81,0 / 82,4 / 84,3 %, bei t(5) mit gemeinsamer Skala 70,4 / 74,4 / 77,6 % (2 000 Läufe, Standardfehler rund 0,9 Punkte; nicht um die höhere Fehlalarmrate bereinigt). |
| Handwerte | r = 0,982 für (0,0), (1,1), (2,3); r = 0,8 für (1,1), (2,3), (3,2), (4,4), mit Leave-one-out 0,5 / 0,929 / 0,929 / 0,5; Fisher-Intervall (0,113; 0,466) bei r = 0,3, n = 103. |

## Ehrliche Grenzen

- Alles ist **simuliert**; die Familien sind Beispiele, nicht „alle“ Verteilungen. Der gemeinsame Skalenfaktor ist eine Modellwahl für Spitzenwochen, kein Befund über echte Standorte.
- Die **wahre Korrelation ist nur in der Simulation bekannt**. Bei echten Daten lässt sich die Abdeckung nicht messen.
- Die Raster-Zahlen haben vier Seeds und Bootstrap mit 400 Ziehungen; die angegebene Streuung über die Seeds ist die Unsicherheit der Messung. Bei n = 1 000 und der Bootstrap-Messung der Fehlalarmrate (500 Läufe) ist sie größer.
- Beim Varianzgesetz ist die Formel **nur durch die Messung gestützt**, nicht aus einer Quelle belegt; bei ν = 5 ist die Messung selbst unstabil.
- **Warum BCa schlechter abschneidet**, ist nicht untersucht. Die Aussage gilt für diese Familien, B = 400 und diese Stichprobengrößen.
- Die **Teststärke bei t(5)** ist nicht um die höhere Fehlalarmrate bereinigt, Lognormal fehlt dort, weil sein wahres r von ρ abweicht.
- Die gekrümmte Kurve und die U-Form sind nicht für jedes Ziel-r in jedem Seed erreichbar (Experiment 1 zeigt „nicht erreichbar“ bzw. die Zahl der lösbaren Seeds).
- Es gibt **keine Literaturangaben**: Namen der Verfahren sind allgemein bekannt, Zahlen stammen ausschließlich aus den eigenen Messungen.

## Verifikation

- **Kern:** Handrechnung an drei Mini-Instanzen (`tests/conftest.py`), Fehlerfälle (konstante Spalte, zu kurze Eingabe), jede Zweigentscheidung (BCa mit z₀ = 0 und a = 0 gleich Perzentil), `tests/test_algorithm.py`.
- **Orakel** (`tests/test_oracle_pearson.py`): Pearson, Spearman (auch mit Gleichständen), p-Wert und Fisher-Intervall gegen scipy (40 Datensätze), Normal-, Beta- und Student-Verteilung gegen scipy, Leave-one-out und Bootstrap gegen Schleifen mit denselben Indizes, Permutationstest gegen die vollständige Aufzählung (n = 3 bis 6), BCa gegen `scipy.stats.bootstrap`, RMSE-Formel gegen eine große Regression.
- **Generator** (`tests/test_scenario.py`, `tests/test_copies.py`): der vektorisierte Strom gleich der skalaren Klasse, Verteilungsmomente je Familie, Hebelpunkte, Kopie-Test gegen das Original.
- **Formen** (`tests/test_shapes.py`): jedes Ziel auf 10⁻¹² getroffen, Erreichbarkeit, Eigenheiten je Form.
- **Messreihen und Verdikte:** `tests/test_evaluation.py`, `tests/test_verdicts.py` (jede Schwelle und jede Zahl im Text an handgesetzten Werten), `tests/test_grid.py` (Raster vollständig, Plausibilität, **Probezelle gegen den Erzeuger**), `tests/test_eigenschaften_pearson.py` (affine Invarianz, Spiegelung, Monotonie), `tests/test_edge_cases.py` (alle Familien bei n = 5 und n = 1 000, Eckfälle, Permalink).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum), jedes Preset, Permalink, jeder Experiment-Knopf einzeln, Layout (Hervorhebung des aktiven Presets, jede Reglerwirkung), Footer, Laufzeit-Importe.
- **Zahlen:** `tests/test_claims.py` rechnet die Messzahlen dieser README gegen das Raster bzw. die Live-Funktionen nach (nicht erfasst: Laufzeit, Mutationszahlen).
- **Mutationsprüfung** (`tools/mutate_demo.py`, `prs_rng.py` ausgenommen; Stand in `tests/mutation_state.json`): 222 Einträge in der Zustandsdatei, **alle gesichert** (206 gefangen, 16 als rechnerisch gleichwertig festgehalten: Vergleiche mit `tiny` im Kettenbruch, Grenzen der Näherungsbereiche von `norm_ppf`, Iterationsgrenzen, `y − ȳ` gegen `y + ȳ` in der Steigung, weil Σ(x − x̄) = 0), dazu **1 offen** (Zeitüberschreitung in `rank_average`: ein Endlosschleifen-Mutant, steht nicht in der Zustandsdatei). Der erste Lauf fand fünf echte Lücken (exakter Permutations-p-Wert, Fehlermeldungen der Grenzfälle, Gleichstand in der BCa-Korrektur, kleinster Fall des t-Tests), die jetzt Tests haben.
- **Smoke-Test:** `tests/test_app.py` ist die Portfolio-Vorlage mit einer Abweichung: gesperrte Regler (Lage und Richtung der Hebelpunkte ohne Hebelpunkt) lässt er aus, `tests/test_app_experiments.py` prüft sie mit einem und zwei Hebelpunkten.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Hauptansicht, sechs Experimente, Grenzen, Formeln |
| `data/pearson_grid.json` | vorgerechnetes Raster (Abdeckung, Fehlalarm, Varianzgesetz, Probezelle) |
| `prs_algorithm.py` | Kern: Pearson, Intervalle, Tests, Verteilungen, Formeln |
| `prs_constants.py` | Regler-Grenzen, Familien, Raster, Farben, Presets mit Hilfetexten |
| `prs_evaluation.py` | Einzeldatensatz, Wiederholungen, Messreihen, Raster-Zellen, Verdikte |
| `prs_layout.py` | Fester Kopf (Stylesheet) |
| `prs_pdf_export.py` | PDF-Export der Hauptansicht (fpdf2) |
| `prs_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, Hervorhebung), Zufalls-Seed-Button |
| `prs_rng.py` | SplitMix64, wörtliche Kopie aus `reduktionskette-demo/red_scenario.py` |
| `prs_scenario.py` | vektorisierter SplitMix64, Familien, Hebelpunkte |
| `prs_shapes.py` | sechs Formen mit exakt gleichem r |
| `prs_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/make_grid.py` | rechnet das Raster vor (`python tools/make_grid.py`, rund 8 Minuten) |
| `tests/` | siehe Verifikation |

## Bewusst nicht umgesetzt

- **Rangmaße und rangbasierte Intervalle** (nächstes Stück), **nichtlineare Abhängigkeitsmaße** (Stück 3), **Zeitreihen** (Stück 4), **viele Variablen** (Stück 5), **Kausalität, Simpson, Kollider** (Linie *Kausale Diagramme und Adjustierung*).
- Robuste Schätzer (Winsorisieren, MCD), Bayes-Intervalle, Transformationen zur Normalisierung der Ränder.
- Eine Erklärung, **warum** BCa hier schlechter abschneidet, und ein Beweis des Varianzgesetzes.
- Echte Daten: die Demo bleibt bei Simulation, weil nur dort die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Das vorgerechnete Raster liegt im Repo; neu erzeugen mit `python tools/make_grid.py`. Die Demo nutzt nur Sprachmerkmale, die auch in der CI-Python-Version laufen; die Version der Streamlit Cloud steht in den App-Einstellungen.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
