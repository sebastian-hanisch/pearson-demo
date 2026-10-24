"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import math

import pytest

import prs_evaluation as ev


def _analysis(covers, lever=(), true_r=0.5, r=0.62, n=30, family="normal"):
    names = ev.interval_names()
    return ev.Analysis(ev.Settings(family=family, n=n, lev_k=5.0), (0.0,), (0.0,), lever, true_r, r, r * r, 1.0, 0.5, 0.01,
                       {k: (0.0, 1.0) for k in names}, (0.0,), 0, dict(zip(names, covers)))


def test_main_all_hit():
    kind, text = ev.verdict_main(_analysis((True, True, True)))
    assert kind == "success" and "r = **0.62**" in text and "wahr ist **0.50**" in text and "Alle drei Intervalle" in text and "n = 30" in text


def test_main_none_hits_is_a_warning():
    kind, text = ev.verdict_main(_analysis((False, False, False)))
    assert kind == "warning" and "Kein Intervall" in text


def test_main_partial_names_hitters_and_missers():
    kind, text = ev.verdict_main(_analysis((False, True, True)))
    assert kind == "info" and "enthalten ihn Perzentil-Bootstrap, BCa-Bootstrap" in text and "verfehlen ihn Fisher-z" in text
    kind, text = ev.verdict_main(_analysis((True, False, False)))
    assert "enthalten ihn Fisher-z;" in text and "verfehlen ihn Perzentil-Bootstrap, BCa-Bootstrap" in text


def test_main_mentions_leverage_points():
    _, text = ev.verdict_main(_analysis((True, True, True), lever=(0, 1)))
    assert "2 Hebelpunkt(e) bei k = 5" in text
    assert "Hebelpunkt" not in ev.verdict_main(_analysis((True, True, True)))[1]


def test_main_names_the_family():
    assert "t(3), gemeinsame Skala" in ev.verdict_main(_analysis((True, True, True), family="t3"))[1]


def test_coverage_verdict_threshold_is_three_standard_errors():
    runs = 400
    se = math.sqrt(0.95 * 0.05 / runs)
    inside = {"runs": runs, "f": 0.95 + 2.9 * se, "p": 0.95 - 2.9 * se, "b": 0.95}
    kind, text = ev.verdict_coverage(inside)
    assert kind == "success" and "400 Wiederholungen" in text and f"±{100 * 3 * se:.1f}" in text
    outside = {"runs": runs, "f": 0.95 - 3.1 * se, "p": 0.95, "b": 0.95 + 3.1 * se}
    kind, text = ev.verdict_coverage(outside)
    assert kind == "warning" and "Fisher-z, BCa-Bootstrap" in text and "Perzentil-Bootstrap **95.0 %**" in text


def test_coverage_verdict_prints_the_rates():
    _, text = ev.verdict_coverage({"runs": 100, "f": 0.5, "p": 0.9, "b": 0.8})
    assert "Fisher-z **50.0 %**" in text and "Perzentil-Bootstrap **90.0 %**" in text and "BCa-Bootstrap **80.0 %**" in text


def _grid_row(n, f, p, b):
    return {"n": n, "rho": 0.5, "fisher": f, "perc": p, "bca": b}


def test_grid_series_verdict_threshold_is_90_percent_at_the_largest_n():
    kind, text = ev.verdict_grid_series([_grid_row(10, 0.85, 0.88, 0.9), _grid_row(1000, 0.519, 0.934, 0.87)], "t(3)")
    assert kind == "warning" and "n = 10 zu **85.0 %**" in text and "n = 1000 zu **51.9 %**" in text and "Perzentil-Bootstrap 88.0 % und 93.4 %" in text and "BCa 90.0 % und 87.0 %" in text
    kind, _ = ev.verdict_grid_series([_grid_row(10, 0.95, 0.9, 0.9), _grid_row(1000, 0.90, 0.9, 0.9)], "Normal")
    assert kind == "success"
    kind, _ = ev.verdict_grid_series([_grid_row(10, 0.95, 0.9, 0.9), _grid_row(1000, 0.8999, 0.9, 0.9)], "Normal")
    assert kind == "warning"


def _law(nu, predicted, measured):
    return {"nu": nu, "predicted": predicted, "measured": measured, "rho": 0.5}


def test_law_verdict_names_the_worst_deviation_among_nu_at_least_6():
    rows = [_law(5, 3.0, 2.0), _law(6, 2.0, 2.1), _law(8, 1.5, 1.2), _law("inf", 1.0, 1.0)]
    kind, text = ev.verdict_law(rows)
    assert kind == "info" and "höchstens **20 %**" in text and "bei ν = 8: 1.20 gegen 1.50" in text        # ν = 5 und ∞ zählen nicht
    kind, text = ev.verdict_law([_law(6, 2.0, 2.1), _law(10, 4 / 3, 1.3)])
    assert kind == "success" and "höchstens **5 %**" in text
    assert ev.verdict_law([_law(6, 2.0, 2.29)])[0] == "success"                                          # knapp unter der Grenze von 15 %
    kind, text = ev.verdict_law([_law(6, 2.0, 2.34)])
    assert kind == "info" and "**17 %**" in text


def test_leverage_verdict_branches():
    rows = [{"n": 30, "k": 0, "share": 0.052, "mean_r": 0.0, "runs": 1000}, {"n": 30, "k": 3, "share": 0.223, "mean_r": 0.24, "runs": 1000}, {"n": 30, "k": 5, "share": 0.841, "mean_r": 0.47, "runs": 1000}]
    kind, text = ev.verdict_leverage(rows, 30, 0)
    assert kind == "info" and "5.2 %" in text and "erwartet 5 %" in text
    kind, text = ev.verdict_leverage(rows, 30, 3)
    assert kind == "info" and "22.3 %" in text
    kind, text = ev.verdict_leverage(rows, 30, 5)
    assert kind == "warning" and "84.1 %" in text and "5 Standardabweichungen" in text and "mittleres r = 0.47" in text and "1000 Läufe" in text
    assert ev.verdict_leverage([{**rows[2], "share": 0.5}], 30, 5)[0] == "warning"                       # genau 50 % zählt als Warnung
    assert ev.verdict_leverage([{**rows[2], "share": 0.4999}], 30, 5)[0] == "info"


def test_size_verdict_threshold_is_three_standard_errors_above_5_percent():
    runs = 8000
    se = math.sqrt(0.05 * 0.95 / runs)
    kind, text = ev.verdict_size("t3", 100, 0.05 + 2.9 * se, runs)
    assert kind == "success" and "t(3), gemeinsame Skala" in text and "n = 100" in text
    kind, text = ev.verdict_size("t3", 100, 0.33, runs)
    assert kind == "warning" and "**33.0 %**" in text and "8000 Läufe" in text
    assert ev.verdict_size("t3", 100, 0.05 + 3.1 * se, runs)[0] == "warning"
    assert ev.verdict_size("normal", 30, 0.01, runs)[0] == "success"                                   # zu wenige Alarme sind kein Warnfall


def test_shapes_verdict():
    rows = [{"spearman": (0.06, 0, 0), "slope": (0.8, 0, 0)}, {"spearman": (0.83, 0, 0), "slope": (1.78, 0, 0)}, {"spearman": (0.5, 0, 0), "slope": (0.56, 0, 0)}, {"label": "x"}]
    kind, text = ev.verdict_shapes(rows, 0.8)
    assert kind == "info" and "Alle 3 erreichbaren Formen" in text and "r = 0.80" in text and "**0.06 bis 0.83**" in text and "**0.56 bis 1.78**" in text
    kind, text = ev.verdict_shapes(rows[:1] + rows[3:], 0.95)
    assert kind == "warning" and "0.95" in text
    assert ev.verdict_shapes(rows[:2], 0.8)[0] == "info"                                                 # genau zwei Formen genügen


def test_utility_verdict_picks_the_row_nearest_to_one_half():
    rows = [{"r": 0.3, "rmse_measured": 0.954, "rmse_formula": 0.954, "r2": 0.09, "sign_measured": 0.597},
            {"r": 0.5, "rmse_measured": 0.866, "rmse_formula": 0.866, "r2": 0.25, "sign_measured": 0.667},
            {"r": 0.7, "rmse_measured": 0.714, "rmse_formula": 0.714, "r2": 0.49, "sign_measured": 0.748}]
    kind, text = ev.verdict_utility(rows)
    assert kind == "info" and "r = 0.50" in text and "25 % der Varianz" in text and "um **13 %**" in text and "(Formel: 13 %)" in text and "in 67 % der Paare" in text
