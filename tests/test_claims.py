"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/pearson_grid.json) oder über dieselben Auswertungsfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Raster-Werte stehen gerundet auf drei Stellen in der README; die Vergleiche nutzen eine Toleranz von einem halben Tausendstel.
Live nachgerechnete Werte (Hebelpunkte, Formen, Planwerte) lassen eine Toleranz von 0,0015 zu (eine Zählung mehr oder weniger in 1 000 bzw. 2 000 Läufen)."""

import re
from pathlib import Path

import pytest

import prs_constants as C
import prs_evaluation as ev

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
GRID_TOL = 0.0006
LIVE_TOL = 0.0015


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


def _series(grid, family, rho=0.5):
    return ev.coverage_series(grid, family, rho)


def _vals(series, key):
    return [r[key] for r in series]


# --- Befund 1: gleiches r, andere Bilder ----------------------------------------------------------------------------------------------------

def test_claim_shapes_at_r_0_8():
    """README: bei r = 0,8 haben die sechs Formen ein mittleres Spearman von 0,08 bis 0,83, eine mittlere Steigung von 0,57 bis 1,73; der Hebelpunkt hat Einfluss 0,80."""
    rows = {r["key"]: r for r in ev.shapes_table(0.8)}
    sp = [r["spearman"][0] for r in rows.values()]
    sl = [r["slope"][0] for r in rows.values()]
    assert min(sp) == pytest.approx(0.08, abs=0.006) and max(sp) == pytest.approx(0.83, abs=0.006)
    assert min(sl) == pytest.approx(0.57, abs=0.006) and max(sl) == pytest.approx(1.73, abs=0.006)
    assert rows["hebelpunkt"]["spearman"][0] == pytest.approx(0.080, abs=0.0006) and rows["hebelpunkt"]["influence"][0] == pytest.approx(0.8, abs=1e-9)
    assert rows["gegenausreisser"]["spearman"][0] == pytest.approx(0.828, abs=0.0006) and rows["gegenausreisser"]["influence"][0] == pytest.approx(0.17, abs=1e-9)
    assert all(r["solved"] == 50 for r in rows.values())


def test_claim_shapes_reachability():
    """README: bei r = 0,3 fehlt die U-Form in 8 von 50 Seeds (42 gelöst), bei r = 0,95 die gekrümmte Kurve in 17 von 50 (33 gelöst)."""
    assert {r["key"]: r["solved"] for r in ev.shapes_table(0.3)}["parabel"] == 42
    assert {r["key"]: r["solved"] for r in ev.shapes_table(0.95)}["kurve"] == 33


# --- Befund 2: Abdeckung ---------------------------------------------------------------------------------------------------------------------

def test_claim_normal_data_hold_the_promise(grid):
    """README: bei Normalverteilung liegt die Abdeckung von Fisher-z in allen 21 Zellen zwischen 93,6 und 95,3 %."""
    vals = [r["fisher"] for r in grid["coverage"] if r["family"] == "normal"]
    assert len(vals) == 21 and min(vals) == pytest.approx(0.9365, abs=GRID_TOL) and max(vals) == pytest.approx(0.953, abs=GRID_TOL)


def test_claim_percentile_bootstrap_is_too_narrow_for_tiny_samples(grid):
    """README: der Perzentil-Bootstrap deckt bei Normalverteilung und n = 10 nur 90,1 % (ρ = 0,5)."""
    assert _series(grid, "normal")[0]["perc"] == pytest.approx(0.901, abs=GRID_TOL)


T3 = {"fisher": (0.850, 0.797, 0.766, 0.734, 0.683, 0.592, 0.505), "perc": (0.892, 0.895, 0.904, 0.915, 0.916, 0.924, 0.924), "bca": (0.908, 0.888, 0.874, 0.870, 0.870, 0.877, 0.859)}
T5_FISHER = (0.900, 0.880, 0.866, 0.853, 0.853, 0.816, 0.797)
KONT_FISHER = (0.824, 0.761, 0.692, 0.636, 0.575, 0.586, 0.555)
LOGN1_FISHER = (0.860, 0.841, 0.829, 0.795, 0.786, 0.726, 0.656)


@pytest.mark.parametrize("key", ["fisher", "perc", "bca"])
def test_claim_t3_series(grid, key):
    """README, Tabelle t(3): Abdeckung bei n = 10 / 20 / 30 / 50 / 100 / 300 / 1 000 (ρ = 0,5)."""
    assert _vals(_series(grid, "t3"), key) == pytest.approx(T3[key], abs=GRID_TOL)


def test_claim_other_heavy_tails_fisher(grid):
    assert _vals(_series(grid, "t5"), "fisher") == pytest.approx(T5_FISHER, abs=GRID_TOL)
    assert _vals(_series(grid, "kontaminiert"), "fisher") == pytest.approx(KONT_FISHER, abs=GRID_TOL)
    assert _vals(_series(grid, "lognormal1"), "fisher") == pytest.approx(LOGN1_FISHER, abs=GRID_TOL)


def test_claim_fisher_falls_with_n_on_heavy_tails(grid):
    """README: bei t(3) fällt Fisher-z von 85,0 % (n = 10) auf 50,5 % (n = 1 000); bei t(5) von 90,0 auf 79,7 %; bei Lognormal σ = 1 von 86,0 auf 65,6 %; bei kontaminiert bleibt sie ab n = 100 zwischen 55,5 und 58,6 %."""
    cont = _vals(_series(grid, "kontaminiert"), "fisher")
    assert min(cont[4:]) == pytest.approx(0.555, abs=GRID_TOL) and max(cont[4:]) == pytest.approx(0.586, abs=GRID_TOL)


def test_claim_rho_hardly_matters_for_t3(grid):
    """README: bei t(3), n = 1 000 ändert ρ die Fisher-Abdeckung kaum (0,520 / 0,505 / 0,509 für ρ = 0 / 0,5 / 0,9)."""
    got = [ev.grid_cell(grid, "t3", 1000, rho)["fisher"] for rho in C.GRID_RHOS]
    assert got == pytest.approx([0.520, 0.505, 0.509], abs=GRID_TOL)


def test_claim_lognormal_depends_on_rho(grid):
    """README: bei Lognormal σ = 1, n = 1 000 fällt Fisher von 95,4 % (ρ = 0, wahres r 0) auf 65,6 % (ρ = 0,5, wahres r 0,378) und 48,2 % (ρ = 0,9, wahres r 0,849)."""
    cells = [ev.grid_cell(grid, "lognormal1", 1000, rho) for rho in C.GRID_RHOS]
    assert [c["fisher"] for c in cells] == pytest.approx([0.9535, 0.656, 0.482], abs=GRID_TOL)
    assert [c["true_r"] for c in cells] == pytest.approx([0.0, 0.378, 0.849], abs=0.0006)


def test_claim_interval_widths(grid):
    """README: bei t(3), n = 1 000, ρ = 0,5 ist das Fisher-Intervall im Mittel 0,092, das Perzentil-Intervall 0,241 breit; bei Normalverteilung 0,093 und 0,092."""
    t3, normal = ev.grid_cell(grid, "t3", 1000, 0.5), ev.grid_cell(grid, "normal", 1000, 0.5)
    assert (t3["w_fisher"], t3["w_perc"], normal["w_fisher"], normal["w_perc"]) == pytest.approx((0.092, 0.241, 0.093, 0.092), abs=0.0006)


def test_claim_bca_is_worse_than_percentile_on_heavy_tails(grid):
    """README: BCa liegt bei t(3) in 18 von 21 Zellen unter dem Perzentil-Intervall, bei „kontaminiert“ in 19 von 21; bei t(3) und ρ = 0,5 deckt BCa zu 85,9 bis 90,8 %, das Perzentil-Intervall zu 89,2 bis 92,4 %."""
    for fam, expected in (("t3", 18), ("kontaminiert", 19)):
        cells = [r for r in grid["coverage"] if r["family"] == fam]
        assert len(cells) == 21 and sum(r["bca"] < r["perc"] for r in cells) == expected


def test_claim_seed_spread(grid):
    """README: die Standardabweichung der Fisher-Abdeckung über die vier Seeds beträgt höchstens 4,4 Punkte (n ≤ 300) bzw. 2,8 Punkte (n = 1 000)."""
    assert max(r["fisher_sd"] for r in grid["coverage"] if r["n"] <= 300) == pytest.approx(0.0436, abs=0.0006)
    assert max(r["fisher_sd"] for r in grid["coverage"] if r["n"] == 1000) == pytest.approx(0.0282, abs=0.0006)


# --- Befund 3: das Gesetz ----------------------------------------------------------------------------------------------------------------------

def test_claim_variance_law(grid):
    """README: für ν ≥ 6 weicht der gemessene Varianzfaktor um höchstens 2 % (ρ = 0), 4 % (ρ = 0,5) bzw. 6 % (ρ = 0,9) von (ν − 2)/(ν − 4) ab; bei ν = 5 streut er stark (ρ = 0,5: 4,27 mit Standardabweichung 3,11 über die Seeds)."""
    for rho, pct in ((0.0, 2), (0.5, 4), (0.9, 6)):
        rows = [r for r in grid["law"] if r["rho"] == rho and r["nu"] != "inf" and r["nu"] >= 6]
        worst = max(abs(r["measured"] / r["predicted"] - 1) for r in rows)
        assert round(100 * worst) == pct
    v5 = next(r for r in grid["law"] if r["nu"] == 5 and r["rho"] == 0.5)
    assert v5["measured"] == pytest.approx(4.27, abs=0.006) and v5["measured_sd"] == pytest.approx(3.11, abs=0.006)


def test_claim_predicted_coverage(grid):
    """README: Vorhersage der Abdeckung 74,2 / 83,4 / 89,0 / 91,0 / 93,5 / 95,0 % für ν = 5 / 6 / 8 / 10 / 20 / ∞."""
    got = {r["nu"]: r["cov_predicted"] for r in grid["law"] if r["rho"] == 0.5}
    assert [got[k] for k in (5, 6, 8, 10, 20, "inf")] == pytest.approx([0.742, 0.834, 0.890, 0.910, 0.935, 0.950], abs=0.0006)
    measured = {r["nu"]: r["cov_measured"] for r in grid["law"] if r["rho"] == 0.5}
    assert [measured[k] for k in (6, 8, 10, 20)] == pytest.approx([0.833, 0.895, 0.911, 0.938], abs=GRID_TOL)


# --- Befund 4: ein einzelner Punkt -------------------------------------------------------------------------------------------------------------

def test_claim_leverage_grid():
    """README: ein Punkt bei k = 5 macht in 84,2 / 83,9 / 52,0 / 23,5 % der Läufe (n = 10 / 30 / 100 / 300) aus Unabhängigkeit einen „signifikanten“ Befund; ohne Punkt 4,2 / 4,3 / 5,7 / 4,4 %."""
    rows = ev.leverage_grid()
    at5 = {r["n"]: r for r in rows if r["k"] == 5}
    assert [at5[n]["share"] for n in C.LEVERAGE_NS] == pytest.approx([0.842, 0.839, 0.520, 0.235], abs=LIVE_TOL)
    assert [at5[n]["mean_r"] for n in C.LEVERAGE_NS] == pytest.approx([0.75, 0.47, 0.20, 0.07], abs=0.006)
    at0 = {r["n"]: r["share"] for r in rows if r["k"] == 0}
    assert [at0[n] for n in C.LEVERAGE_NS] == pytest.approx([0.042, 0.043, 0.057, 0.044], abs=LIVE_TOL)
    n30 = {r["k"]: r["share"] for r in rows if r["n"] == 30}
    assert [n30[k] for k in (2, 3, 4, 6, 8)] == pytest.approx([0.080, 0.229, 0.526, 0.980, 1.0], abs=LIVE_TOL)


def test_claim_leverage_thresholds():
    """README: der Punkt muss für 50 % bzw. 95 % „signifikant“ bei n = 10 / 30 / 100 / 300 mindestens bei k = 4,0 / 4,0 / 5,0 / 6,25 bzw. 6,25 / 5,75 / 6,75 / 8,5 liegen."""
    got = {n: (ev.leverage_threshold(n, 0.5), ev.leverage_threshold(n, 0.95)) for n in C.LEVERAGE_NS}
    assert got == {10: (4.0, 6.25), 30: (4.0, 5.75), 100: (5.0, 6.75), 300: (6.25, 8.5)}


def test_claim_preset_single_point():
    """README und Preset-Hilfe: im Preset „Ein einzelner Punkt“ (Seed 35) liegt der p-Wert des t-Tests bei 0,0065; die Daten sind unabhängig (ρ = 0)."""
    a = ev.analyse(ev.Settings(**C.PRESETS["Ein einzelner Punkt (k = 5, n = 30)"]))
    assert a.p == pytest.approx(0.00646, abs=0.0001) and a.true_r == 0.0 and a.lever == (0,)


# --- Befund 5: unkorreliert ist nicht unabhängig -----------------------------------------------------------------------------------------------

SIZE_T = {
    ("normal", 30): 0.053, ("normal", 100): 0.050, ("normal", 300): 0.048, ("tindep3", 30): 0.056, ("tindep3", 100): 0.050, ("tindep3", 300): 0.050,
    ("t3", 30): 0.229, ("t3", 100): 0.326, ("t3", 300): 0.401, ("t5", 30): 0.135, ("t5", 100): 0.164, ("t5", 300): 0.192,
    ("t10", 30): 0.077, ("t10", 100): 0.081, ("t10", 300): 0.093, ("kontaminiert", 30): 0.307, ("kontaminiert", 100): 0.420, ("kontaminiert", 300): 0.447,
    ("lognormal1", 30): 0.051, ("lognormal1", 100): 0.048, ("lognormal1", 300): 0.041,
}


def test_claim_size_of_the_t_test(grid):
    """README, Tabelle Fehlalarm bei wahrem ρ = 0 (t-Test, 8 000 Läufe)."""
    for (fam, n), expected in SIZE_T.items():
        assert ev.grid_size_row(grid, fam, n)["t"] == pytest.approx(expected, abs=GRID_TOL), (fam, n)


def test_claim_size_of_permutation_and_bootstrap(grid):
    """README: der Permutationstest teilt das Problem (t(3): 22,4 / 27,6 / 39,6 %, kontaminiert 26,4 / 37,0 / 41,0 %); der Bootstrap-Test bleibt bei 4,6 bis 11,2 % (500 Läufe)."""
    assert [ev.grid_size_row(grid, "t3", n)["perm"] for n in C.SIZE_NS] == pytest.approx([0.224, 0.276, 0.396], abs=GRID_TOL)
    assert [ev.grid_size_row(grid, "kontaminiert", n)["perm"] for n in C.SIZE_NS] == pytest.approx([0.264, 0.370, 0.410], abs=GRID_TOL)
    boots = [r["boot"] for r in grid["size"]]
    assert min(boots) == pytest.approx(0.046, abs=GRID_TOL) and max(boots) == pytest.approx(0.112, abs=GRID_TOL)


def test_claim_preset_help_numbers(grid):
    """Hilfetexte der Presets: 95,3 % (Standard), 59,2 % und 92,4 % (schwerer Rand), 57,5 % und 91,2 % (Spitzenwochen), 32,6 % (unkorreliert, aber nicht unabhängig)."""
    assert ev.grid_cell(grid, "normal", 30, 0.5)["fisher"] == pytest.approx(0.953, abs=GRID_TOL)
    heavy = ev.grid_cell(grid, "t3", 300, 0.5)
    assert (heavy["fisher"], heavy["perc"]) == pytest.approx((0.592, 0.924), abs=GRID_TOL)
    spikes = ev.grid_cell(grid, "kontaminiert", 100, 0.5)
    assert (spikes["fisher"], spikes["perc"]) == pytest.approx((0.575, 0.912), abs=GRID_TOL)
    assert ev.grid_size_row(grid, "t3", 100)["t"] == pytest.approx(0.326, abs=GRID_TOL)
    for name, help_text in C.PRESET_HELP.items():
        assert name in C.PRESETS and help_text


# --- Befund 6: Nutzen von r und Planwerte ------------------------------------------------------------------------------------------------------

def test_claim_utility():
    """README: bei r = 0,5 erklärt die Gerade 25 % der Varianz, verkleinert den Fehler aber nur um 13 % (RMSE-Verhältnis 0,866); das Vorzeichen stimmt in 66,7 % der Paare (Formel 66,7 %); bei r = 0,9: 0,436 und 85,6 %."""
    rows = {r["r"]: r for r in ev.utility_rows()}
    assert rows[0.5]["rmse_formula"] == pytest.approx(0.866, abs=0.0006) and rows[0.5]["rmse_measured"] == pytest.approx(0.865, abs=0.002)
    assert rows[0.5]["sign_measured"] == pytest.approx(0.668, abs=0.002) and rows[0.5]["sign_formula"] == pytest.approx(0.6667, abs=0.0001)
    assert rows[0.9]["rmse_formula"] == pytest.approx(0.436, abs=0.0006) and rows[0.9]["sign_measured"] == pytest.approx(0.856, abs=0.002)


def test_claim_plan_values():
    """README: Planwert n = 783 / 85 / 30 für ρ = 0,1 / 0,3 / 0,5; gemessene Teststärke Normal 81,0 / 82,4 / 84,3 %, t(5) 70,4 / 74,4 / 77,6 % (2 000 Läufe)."""
    assert [ev.plan_power(rho, "normal")["n"] for rho in (0.1, 0.3, 0.5)] == [783, 85, 30]
    assert [ev.plan_power(rho, "normal")["power"] for rho in (0.1, 0.3, 0.5)] == pytest.approx([0.8095, 0.824, 0.8425], abs=LIVE_TOL)
    assert [ev.plan_power(rho, "t5")["power"] for rho in (0.1, 0.3, 0.5)] == pytest.approx([0.7035, 0.7435, 0.776], abs=LIVE_TOL)


# --- Handwerte der README ------------------------------------------------------------------------------------------------------------------------

def test_claim_hand_values():
    """README: r = 0,982 für (0,0), (1,1), (2,3); r = 0,8 für (1,1), (2,3), (3,2), (4,4) mit Leave-one-out 0,5 / 0,929 / 0,929 / 0,5; Fisher (0,113; 0,466) bei r = 0,3, n = 103."""
    import prs_algorithm as A
    assert A.pearson_r([0, 1, 2], [0, 1, 3]) == pytest.approx(0.982, abs=0.0006)
    four = ([1, 2, 3, 4], [1, 3, 2, 4])
    assert A.pearson_r(*four) == pytest.approx(0.8) and list(A.jackknife_r(*four)) == pytest.approx([0.5, 0.929, 0.929, 0.5], abs=0.0006)
    assert A.fisher_interval(0.3, 103) == pytest.approx((0.113, 0.466), abs=0.0006)


def test_claim_readme_mentions_its_key_numbers():
    """Verbindet die README mit den Tests: die Zahlen, die dort als Kernbefunde stehen, kommen im Text vor."""
    for needle in ("50,5", "85,0 / 79,7", "22,9 / 32,6 / 40,1", "0,08 bis 0,83", "0,57 bis 1,73", "84,2 / 83,9", "um **13 %**", "4,27", "0,0065"):
        assert needle in README, needle
