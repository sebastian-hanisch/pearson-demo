"""Das vorgerechnete Raster (data/pearson_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probezelle)."""

import pytest

import prs_constants as C
import prs_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/pearson_grid.json fehlt: python tools/make_grid.py"
    return g


def test_probe_cell_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probezelle nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    probe = ev.grid_probe()
    assert probe["runs"] == grid["probe"]["runs"] and all(probe[k] == grid["probe"][k] for k in ("true_r", "f", "p", "b"))      # Trefferzahlen exakt
    assert probe == pytest.approx(grid["probe"], rel=1e-9)                                                                      # Breiten: Summationsreihenfolge weicht je Plattform in der 16. Stelle ab (CI-Linux, DEMO-PLAYBOOK §4)


def test_grid_has_every_cell_exactly_once(grid):
    cov = [(r["family"], r["n"], r["rho"]) for r in grid["coverage"]]
    assert len(cov) == len(set(cov)) == len(C.FAMILIES) * len(C.GRID_NS) * len(C.GRID_RHOS)
    assert set(cov) == {(f, n, rho) for f in C.FAMILIES for n in C.GRID_NS for rho in C.GRID_RHOS}
    size = [(r["family"], r["n"]) for r in grid["size"]]
    assert set(size) == {(f, n) for f in C.SIZE_FAMILIES for n in C.SIZE_NS} and len(size) == len(set(size))
    law = {(r["nu"], r["rho"]) for r in grid["law"]}
    assert law == {(nu, rho) for nu in (*C.LAW_NUS, "inf") for rho in C.GRID_RHOS}


def test_grid_meta_matches_the_constants(grid):
    m = grid["meta"]
    assert m["boot_b"] == C.GRID_BOOT and m["seeds"] == list(C.GRID_SEEDS) and m["ns"] == list(C.GRID_NS) and m["rhos"] == list(C.GRID_RHOS)


def test_every_rate_is_a_probability_and_the_runs_are_documented(grid):
    for r in grid["coverage"]:
        assert all(0.0 <= r[k] <= 1.0 for k in ("fisher", "perc", "bca")) and r["runs"] == (4000 if r["n"] >= 1000 else 2000) and r["seeds"] == 4
        assert r["w_fisher"] > 0 and r["w_perc"] > 0 and r["w_bca"] > 0
    for r in grid["size"]:
        assert all(0.0 <= r[k] <= 1.0 for k in ("t", "boot", "perm")) and r["t_runs"] == 8000 and r["bp_runs"] == 500


def test_normal_data_gives_nominal_coverage_for_fisher(grid):
    for r in grid["coverage"]:
        if r["family"] == "normal":
            assert abs(r["fisher"] - 0.95) < 0.03, (r["n"], r["rho"])


def test_fisher_coverage_on_heavy_tails_falls_with_n(grid):
    for fam in ("t3", "t5", "lognormal1"):
        series = ev.coverage_series(grid, fam, 0.5)
        assert series[0]["fisher"] > series[-1]["fisher"] + 0.1, fam
    s3 = ev.coverage_series(grid, "t3", 0.5)
    assert s3[-1]["fisher"] < 0.6 and s3[-1]["perc"] > 0.9


def test_common_scale_inflates_the_size_but_independent_margins_do_not(grid):
    rows = {(r["family"], r["n"]): r for r in grid["size"]}
    for n in C.SIZE_NS:
        assert abs(rows[("normal", n)]["t"] - 0.05) < 0.015 and abs(rows[("tindep3", n)]["t"] - 0.05) < 0.015 and abs(rows[("lognormal1", n)]["t"] - 0.05) < 0.015
        assert rows[("t3", n)]["t"] > 0.2 and rows[("kontaminiert", n)]["t"] > 0.25
    assert rows[("t3", 300)]["t"] > rows[("t3", 30)]["t"]


def test_the_permutation_test_shares_the_problem_and_the_bootstrap_does_not(grid):
    rows = {(r["family"], r["n"]): r for r in grid["size"]}
    for n in C.SIZE_NS:
        assert abs(rows[("t3", n)]["perm"] - rows[("t3", n)]["t"]) < 0.08 and rows[("t3", n)]["boot"] < 0.15


def test_variance_law_holds_for_nu_at_least_6(grid):
    for r in grid["law"]:
        if r["nu"] != "inf" and r["nu"] >= 6:
            assert abs(r["measured"] / r["predicted"] - 1) < 0.2, r
        assert abs(r["cov_measured"] - r["cov_predicted"]) < 0.04 or r["nu"] == 5, r


def test_verdicts_run_on_every_grid_series(grid):
    for fam in C.FAMILIES:
        for rho in C.GRID_RHOS:
            kind, text = ev.verdict_grid_series(ev.coverage_series(grid, fam, rho), C.FAMILIES[fam])
            assert kind in ("success", "warning") and text
    for fam in C.SIZE_FAMILIES:
        row = ev.grid_size_row(grid, fam, 100)
        assert ev.verdict_size(fam, 100, row["t"], row["t_runs"])[0] in ("success", "warning")
    for rho in C.GRID_RHOS:
        assert ev.verdict_law([r for r in grid["law"] if r["rho"] == rho])[0] in ("success", "info")
