"""Randfälle: kleinstes und größtes n, Hebelpunkte am Rand, extreme ρ, jede Familie, entartete Eingaben (DEMO-PLAYBOOK §1d)."""

import numpy as np
import pytest

import prs_algorithm as A
import prs_constants as C
import prs_evaluation as ev
import prs_presets as P
from prs_scenario import VRng, draw_pairs


@pytest.mark.parametrize("family", list(C.FAMILIES))
@pytest.mark.parametrize("n", [C.N_MIN, 6, C.N_MAX])
def test_analyse_works_for_every_family_at_the_smallest_and_largest_n(family, n):
    a = ev.analyse(ev.Settings(family=family, n=n, boot_b=C.B_MIN))
    assert -1.0 <= a.r <= 1.0 and all(lo <= hi for lo, hi in a.intervals.values())
    assert len(a.x) == n and len(a.influence) == n


@pytest.mark.parametrize("rho", [C.RHO_MIN, -0.5, 0.0, C.RHO_MAX])
def test_extreme_rho(rho):
    a = ev.analyse(ev.Settings(rho=rho, n=50))
    assert a.true_r == rho
    if abs(rho) >= 0.5:
        assert np.sign(a.r) == np.sign(rho)


@pytest.mark.parametrize("count,k", [(0, 0.0), (1, 0.0), (2, C.LEV_K_MAX), (1, C.LEV_K_MAX)])
@pytest.mark.parametrize("n", [C.N_MIN, 30])
def test_leverage_at_the_edges(count, k, n):
    a = ev.analyse(ev.Settings(n=n, lev_count=count, lev_k=k, lev_dir="gegenläufig"))
    assert -1.0 <= a.r <= 1.0 and a.p >= 0.0


def test_all_sliders_at_their_maxima_and_minima_together():
    low = ev.Settings(family="t3", rho=C.RHO_MIN, n=C.N_MIN, lev_count=C.LEV_COUNT_MAX, lev_k=C.LEV_K_MAX, boot_b=C.B_MIN, reps=C.REPS_MIN, seed=0)
    high = ev.Settings(family="lognormal1", rho=C.RHO_MAX, n=C.N_MAX, lev_count=C.LEV_COUNT_MAX, lev_k=C.LEV_K_MAX, boot_b=C.B_MAX, reps=C.REPS_MAX, seed=C.SEED_MAX)
    for s in (low, high):
        a = ev.analyse(s)
        assert -1.0 <= a.r <= 1.0
    assert ev.coverage_experiment(low)["runs"] == C.REPS_MIN
    assert ev.coverage_experiment(high)["runs"] == C.REPS_BUDGET // (C.B_MAX * C.N_MAX)


def test_smallest_n_with_the_worst_case_budget_finishes():
    """Worst-Case am Regler-Maximum (n = 1 000, B = 2 000): ein Datensatz rechnet durch (Timeout statt Zeit-Assertion, DEMO-PLAYBOOK §1c)."""
    a = ev.analyse(ev.Settings(n=C.N_MAX, boot_b=C.B_MAX, family="t3"))
    assert a.intervals["Perzentil-Bootstrap"][0] < a.intervals["Perzentil-Bootstrap"][1]


def test_r_of_one_does_not_break_the_interval_or_the_test():
    x = np.arange(10.0)
    assert A.pearson_r(x, 2 * x) == pytest.approx(1.0)
    lo, hi = A.fisher_interval(A.pearson_r(x, 2 * x), 10)
    assert lo <= hi <= 1.0 and A.t_test_p(A.pearson_r(x, 2 * x), 10) < 1e-12


def test_exactly_four_points_is_the_smallest_fisher_case(four):
    lo, hi = A.fisher_interval(A.pearson_r(*four), 4)
    assert lo < 0 < hi                                  # das Intervall ist riesig und enthält 0
    with pytest.raises(ValueError):
        A.fisher_interval(0.5, 3)


def test_clean_value_for_the_permalink():
    spec = P.SETTING_SPECS
    assert P.clean_value(spec["n_slider"], "99999") == C.N_MAX and P.clean_value(spec["n_slider"], "-4") == C.N_MIN
    assert P.clean_value(spec["n_slider"], "abc") is None and P.clean_value(spec["n_slider"], None) is None
    assert P.clean_value(spec["rho_slider"], "0.1500000001") == 0.15 and P.clean_value(spec["rho_slider"], "7") == C.RHO_MAX
    assert P.clean_value(spec["family_select"], "t3") == "t3" and P.clean_value(spec["family_select"], "tindep3") is None
    assert P.clean_value(spec["lev_dir_select"], "gegenläufig") == "gegenläufig" and P.clean_value(spec["lev_dir_select"], "quer") is None
    assert P.clean_value(spec["seed_input"], "12.0") is None                    # int("12.0") scheitert: ungültig, kein Absturz


def test_every_family_has_a_label_and_a_true_correlation():
    assert set(C.FAMILIES) <= set(C.ALL_FAMILIES) and C.INDEPENDENT_T3 not in C.FAMILIES
    for fam in C.ALL_FAMILIES:
        x, y = draw_pairs(fam, 2, 7, 0.3, VRng(1))
        assert x.shape == y.shape == (2, 7)
