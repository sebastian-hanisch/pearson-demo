"""Eigenschaftstests (metamorphe Relationen, ORACLE-PLAYBOOK §5): was bei Umformungen der Daten gleich bleiben oder sich vorhersagbar ändern muss."""

import numpy as np
import pytest

import prs_algorithm as A
from prs_scenario import VRng, draw_pairs


def _pair(seed, n=40, family="t5"):
    x, y = draw_pairs(family, 1, n, 0.4, VRng(seed))
    return x[0], y[0]


@pytest.mark.parametrize("seed", range(15))
def test_r_is_invariant_under_positive_affine_maps_and_flips_sign_under_negation(seed):
    x, y = _pair(seed)
    r = A.pearson_r(x, y)
    assert A.pearson_r(3.5 * x + 7, 0.2 * y - 4) == pytest.approx(r, abs=1e-12)
    assert A.pearson_r(-x, y) == pytest.approx(-r, abs=1e-12) and A.pearson_r(y, x) == pytest.approx(r, abs=1e-15)
    assert abs(r) <= 1.0


@pytest.mark.parametrize("seed", range(10))
def test_spearman_is_invariant_under_monotone_maps_but_pearson_is_not(seed):
    x, y = _pair(seed, family="normal")
    assert A.spearman_r(np.exp(x), y**3) == pytest.approx(A.spearman_r(x, y), abs=1e-12)
    assert A.pearson_r(np.exp(2 * x), y) != pytest.approx(A.pearson_r(x, y), abs=1e-3)


@pytest.mark.parametrize("seed", range(10))
def test_intervals_mirror_when_y_is_negated(seed):
    x, y = _pair(seed, n=60, family="normal")
    r = A.pearson_r(x, y)
    lo, hi = A.fisher_interval(r, 60)
    lo2, hi2 = A.fisher_interval(-r, 60)
    assert (lo2, hi2) == pytest.approx((-hi, -lo), abs=1e-12)
    rb = A.bootstrap_replicates(x, y, 300, VRng(seed))
    rb_neg = A.bootstrap_replicates(x, -y, 300, VRng(seed))
    assert A.percentile_interval(rb_neg) == pytest.approx((-A.percentile_interval(rb)[1], -A.percentile_interval(rb)[0]), abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_more_data_shortens_the_fisher_interval_and_a_higher_level_widens_it(seed):
    r = 0.3 + 0.05 * seed
    widths = [A.fisher_interval(r, n)[1] - A.fisher_interval(r, n)[0] for n in (10, 30, 100, 1000)]
    assert widths == sorted(widths, reverse=True)
    assert A.fisher_interval(r, 50, 0.99)[1] - A.fisher_interval(r, 50, 0.99)[0] > A.fisher_interval(r, 50, 0.90)[1] - A.fisher_interval(r, 50, 0.90)[0]


def test_p_value_falls_with_larger_abs_r_and_with_larger_n():
    ps = [A.t_test_p(r, 30) for r in (0.0, 0.1, 0.3, 0.5, 0.9)]
    assert ps == sorted(ps, reverse=True) and A.t_test_p(0.3, 100) < A.t_test_p(0.3, 30)
    assert A.t_test_p(-0.3, 30) == pytest.approx(A.t_test_p(0.3, 30))


@pytest.mark.parametrize("seed", range(8))
def test_adding_a_lever_point_far_out_pushes_r_toward_the_sign_of_the_point(seed):
    x, y = _pair(seed, n=30, family="normal")
    r0 = A.pearson_r(x, y)
    x2, y2 = np.append(x, 20.0), np.append(y, 20.0)
    x3, y3 = np.append(x, 20.0), np.append(y, -20.0)
    assert A.pearson_r(x2, y2) > max(r0, 0.5) and A.pearson_r(x3, y3) < min(r0, -0.5)


def test_a_dominant_point_has_the_largest_and_a_raising_influence():
    x = np.append(np.random.default_rng(1).standard_normal(20), 10.0)
    y = np.append(np.random.default_rng(2).standard_normal(20), 10.0)
    infl = A.influence(x, y)
    assert int(np.argmax(np.abs(infl))) == 20 and infl[20] > 0.3


def test_permutation_p_of_an_exact_copy_is_the_smallest_possible():
    x = VRng(1).normal(12)
    p = A.permutation_p(x, x, 499, VRng(2))
    assert p == pytest.approx(1 / 500, abs=1 / 500)


def test_jackknife_mean_is_close_to_r_for_large_n():
    x, y = _pair(3, n=500, family="normal")
    assert A.jackknife_r(x, y).mean() == pytest.approx(A.pearson_r(x, y), abs=2e-3)
