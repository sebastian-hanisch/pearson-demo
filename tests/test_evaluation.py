"""Auswertung: Einzeldatensatz, Wiederholungsexperimente, Messreihen. Toleranzen aus dem Monte-Carlo-Fehler, keine exakten Zufallswerte (DEMO-PLAYBOOK §4)."""

import math

import numpy as np
import pytest

import prs_algorithm as A
import prs_constants as C
import prs_evaluation as ev
import prs_shapes as SH


def test_analyse_is_consistent():
    a = ev.analyse(ev.Settings())
    x, y = np.array(a.x), np.array(a.y)
    assert a.r == pytest.approx(A.pearson_r(x, y)) and a.r2 == pytest.approx(a.r**2)
    assert a.slope == pytest.approx(np.polyfit(x, y, 1)[0])
    assert a.influence == pytest.approx(tuple(A.influence(x, y)))
    assert a.top == int(np.argmax(np.abs(a.influence)))
    assert set(a.intervals) == set(ev.interval_names()) == set(a.covers)
    for name, (lo, hi) in a.intervals.items():
        assert lo < hi and a.covers[name] == (lo <= a.true_r <= hi)
    assert a.intervals["Fisher-z"][0] < a.r < a.intervals["Fisher-z"][1]


def test_analyse_is_deterministic_and_seed_dependent():
    s = ev.Settings(n=40, seed=3)
    assert ev.analyse(s) == ev.analyse(s) and ev.analyse(s) != ev.analyse(ev.Settings(n=40, seed=4))


def test_analyse_marks_the_leverage_points_and_caps_their_number():
    a = ev.analyse(ev.Settings(n=30, lev_count=2, lev_k=6.0))
    assert a.lever == (0, 1) and a.top in (0, 1)
    assert ev.analyse(ev.Settings(n=5, lev_count=2, lev_k=6.0)).lever == (0, 1)
    assert ev.analyse(ev.Settings(n=5, lev_count=2, lev_k=0.0)).lever == ()


def test_the_leverage_direction_changes_the_result():
    up = ev.analyse(ev.Settings(n=30, rho=0.0, lev_count=1, lev_k=6.0, lev_dir="gleichläufig"))
    down = ev.analyse(ev.Settings(n=30, rho=0.0, lev_count=1, lev_k=6.0, lev_dir="gegenläufig"))
    assert up.r > 0.3 and down.r < -0.3 and up.influence[0] > 0 > down.influence[0]


def test_the_bootstrap_count_changes_the_bootstrap_but_not_the_fisher_interval():
    a, b = ev.analyse(ev.Settings(boot_b=200)), ev.analyse(ev.Settings(boot_b=1000))
    assert a.intervals["Fisher-z"] == b.intervals["Fisher-z"] and a.intervals["Perzentil-Bootstrap"] != b.intervals["Perzentil-Bootstrap"]


def test_affordable_reps_respects_the_budget():
    assert ev.affordable_reps(500, 1000, 30) == 500
    assert ev.affordable_reps(2000, 2000, 1000) == C.REPS_BUDGET // (2000 * 1000) == 50
    assert ev.affordable_reps(100, 10**9, 10**9) == 1                       # mindestens eine Wiederholung
    assert ev.affordable_reps(10, 100, 10, budget=5000) == 5


def test_coverage_runs_on_normal_data_is_close_to_95_percent():
    res = ev.coverage_runs("normal", 50, 0.5, 600, 200, 3)
    se = math.sqrt(0.95 * 0.05 / 600)
    assert abs(res["f"] - 0.95) < 4 * se and res["runs"] == 600 and res["true_r"] == 0.5
    assert res["wf"] > 0 and res["wp"] > 0 and res["wb"] > 0


def test_coverage_runs_separates_fisher_from_bootstrap_on_heavy_tails():
    """Zweig-Test: bei t(3), n = 300 weicht Fisher von den Bootstrap-Intervallen ab."""
    res = ev.coverage_runs("t3", 300, 0.5, 200, 200, 4)
    assert res["f"] < 0.75 and res["p"] > res["f"] + 0.1 and res["b"] > res["f"] + 0.05


def test_coverage_runs_uses_the_lognormal_true_correlation():
    res = ev.coverage_runs("lognormal1", 100, 0.9, 20, 100, 5)
    assert res["true_r"] == pytest.approx((math.exp(0.9) - 1) / (math.e - 1))


def test_coverage_experiment_reports_requested_and_actual_runs():
    out = ev.coverage_experiment(ev.Settings(n=1000, boot_b=2000, reps=2000))
    assert out["requested"] == 2000 and out["runs"] == 50 and out["boot_b"] == 2000


def test_variance_law_rows_are_complete_and_near_the_formula():
    rows = ev.variance_law(nus=(8, 20), n=500, runs=300, seed=3)
    assert [r["nu"] for r in rows] == [8, 20]
    assert rows[0]["predicted"] == pytest.approx(1.5) and rows[1]["predicted"] == pytest.approx(1.125)
    for r in rows:
        assert 0.5 * r["predicted"] < r["measured"] < 1.6 * r["predicted"]
        assert 0.8 < r["cov_measured"] <= 1.0


def test_law_correlations_handles_normal_and_chunks():
    rs = ev.law_correlations(math.inf, 0.5, 100, 120, 1, chunk=50)
    assert len(rs) == 120 and abs(rs.mean() - 0.5) < 0.03
    assert ev.law_correlations(8, 0.5, 100, 7, 1, chunk=3).shape == (7,)


def test_law_row_for_normal_data_has_factor_one():
    rs = ev.law_correlations(math.inf, 0.5, 400, 800, 2)
    row = ev.law_row(rs, math.inf, 0.5, 400)
    assert row["predicted"] == 1.0 and row["measured"] == pytest.approx(1.0, abs=0.15) and row["cov_predicted"] == pytest.approx(0.95, abs=1e-9)


def test_grid_law_cell_small():
    row = ev.grid_law_cell(8, 0.5, n=300, seeds=(101, 102), runs_per_seed=100)
    assert row["nu"] == 8 and row["runs"] == 200 and row["measured_sd"] >= 0 and row["predicted"] == pytest.approx(1.5)
    assert ev.grid_law_cell(math.inf, 0.0, n=200, seeds=(101, 102), runs_per_seed=50)["nu"] == "inf"


def test_leverage_grid_is_monotone_in_k_and_reaches_one():
    rows = ev.leverage_grid(ns=(30,), ks=(0, 3, 5, 8, 15), runs=600, seed=2)
    shares = [r["share"] for r in rows]
    assert shares[0] == pytest.approx(0.05, abs=0.03) and shares == sorted(shares) and shares[-1] == 1.0
    assert [r["mean_r"] for r in rows] == sorted(r["mean_r"] for r in rows)


def test_leverage_needs_a_larger_k_when_n_is_larger():
    k_small = ev.leverage_threshold(10, 0.5, runs=300)
    k_large = ev.leverage_threshold(300, 0.5, runs=300)
    assert k_small < k_large


def test_leverage_threshold_rises_with_the_quantile_and_can_be_none():
    assert ev.leverage_threshold(30, 0.5, runs=300) < ev.leverage_threshold(30, 0.95, runs=300)
    assert ev.leverage_threshold(30, 0.99999, runs=50, kmax=1.0) is None


def test_size_of_the_t_test_by_family():
    assert ev.size_runs("normal", 100, 2000, 1) == pytest.approx(0.05, abs=0.02)
    assert ev.size_runs("tindep3", 100, 2000, 2) == pytest.approx(0.05, abs=0.02)        # unabhängige schwere Ränder stören nicht
    assert ev.size_runs("t3", 100, 2000, 3) > 0.2                                          # gemeinsame Skala schon
    assert ev.size_runs("kontaminiert", 100, 2000, 4) > 0.2


def test_plan_power_normal_is_near_80_percent_and_t5_is_lower():
    n = ev.plan_power(0.3, "normal", runs=3000, seed=9)
    t = ev.plan_power(0.3, "t5", runs=3000, seed=9)
    assert n["n"] == t["n"] == 85 and n["power"] == pytest.approx(0.8, abs=0.04) and t["power"] < n["power"]


def test_plan_power_rejects_lognormal_where_the_true_r_differs():
    with pytest.raises(ValueError):
        ev.plan_power(0.3, "lognormal05")


def test_shapes_table_rows():
    rows = ev.shapes_table(0.8, seeds=tuple(range(1, 6)))
    assert [r["key"] for r in rows] == list(SH.SHAPES)
    for r in rows:
        assert r["of"] == 5 and r["solved"] == 5
        lo, hi = r["spearman"][1], r["spearman"][2]
        assert lo <= r["spearman"][0] <= hi and r["influence"][0] <= 1.0
    by = {r["key"]: r for r in rows}
    assert by["hebelpunkt"]["spearman"][0] < 0.35 < by["gegenausreisser"]["spearman"][0]
    assert by["hebelpunkt"]["influence"][0] == pytest.approx(0.8, abs=1e-9)


def test_shapes_table_reports_unreachable_forms():
    rows = ev.shapes_table(0.99, seeds=(1, 2, 3))
    kurve = next(r for r in rows if r["key"] == "kurve")
    assert kurve["solved"] == 0 and kurve["range"][1] < 0.99 and "spearman" not in kurve
    assert all("spearman" in r for r in rows if r["solved"])


def test_utility_rows_match_the_formulas():
    for r in ev.utility_rows(rs=(0.3, 0.7), n=100_000):
        assert r["rmse_measured"] == pytest.approx(r["rmse_formula"], abs=0.005) and r["sign_measured"] == pytest.approx(r["sign_formula"], abs=0.005)


def test_cell_seeds_are_distinct_across_cells():
    seeds = {ev.cell_seed(f, n, rho, s) for f in C.ALL_FAMILIES for n in C.GRID_NS for rho in C.GRID_RHOS for s in C.GRID_SEEDS}
    assert len(seeds) == len(C.ALL_FAMILIES) * len(C.GRID_NS) * len(C.GRID_RHOS) * len(C.GRID_SEEDS)


def test_grid_coverage_cell_small():
    row = ev.grid_coverage_cell("normal", 30, 0.5, seeds=(101, 102), runs_per_seed=40, boot_b=100)
    assert row["runs"] == 80 and row["seeds"] == 2 and 0.8 < row["fisher"] <= 1.0 and row["fisher_sd"] >= 0
    for key in ("w_fisher", "w_perc", "w_bca"):
        assert row[key] > 0


def test_grid_size_cell_small():
    row = ev.grid_size_cell("t3", 30, seeds=(101, 102), t_runs=300, bp_runs=30, boot_b=100, perm_b=49)
    assert row["t_runs"] == 600 and row["bp_runs"] == 30 and 0 <= row["boot"] <= 1 and 0 <= row["perm"] <= 1 and row["t"] > 0.1


def test_grid_probe_is_a_small_deterministic_cell():
    assert ev.grid_probe() == ev.grid_probe() and ev.grid_probe()["runs"] == 25


def test_load_grid_handles_a_missing_file(tmp_path):
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    bad = tmp_path / "kaputt.json"
    bad.write_text("{", encoding="utf-8")
    assert ev.load_grid(bad) is None


def test_grid_lookups_on_a_hand_built_grid():
    grid = {"coverage": [{"family": "t3", "n": 30, "rho": 0.5, "fisher": 0.7}, {"family": "t3", "n": 10, "rho": 0.5, "fisher": 0.8}, {"family": "t3", "n": 10, "rho": 0.0, "fisher": 0.9}],
            "size": [{"family": "t3", "n": 30, "t": 0.2}]}
    assert ev.grid_cell(grid, "t3", 30, 0.5)["fisher"] == 0.7 and ev.grid_cell(grid, "t3", 30, 0.9) is None
    assert [r["n"] for r in ev.coverage_series(grid, "t3", 0.5)] == [10, 30]
    assert ev.grid_size_row(grid, "t3", 30)["t"] == 0.2 and ev.grid_size_row(grid, "normal", 30) is None
