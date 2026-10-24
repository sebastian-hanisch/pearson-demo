"""Orakel (ORACLE-PLAYBOOK, DEMO-PLAYBOOK §12): jede Kernroutine gegen einen unabhängigen Rechenweg.

Pearson, Spearman, Fisher-Intervall, Verteilungsfunktionen und der Student-Test gegen scipy; Leave-one-out und Bootstrap gegen Schleifen mit denselben Indizes;
der Permutationstest gegen die vollständige Aufzählung aller Permutationen; BCa gegen scipy.stats.bootstrap (statistisch, mit Toleranz aus dem Monte-Carlo-Fehler)."""

import itertools
import math

import numpy as np
import pytest

import prs_algorithm as A
import prs_scenario as S

sps = pytest.importorskip("scipy.stats")
spsp = pytest.importorskip("scipy.special")


def _sample(family, n, rho, seed):
    x, y = S.draw_pairs(family, 1, n, rho, S.VRng(seed))
    return x[0], y[0]


@pytest.mark.parametrize("seed", range(40))
def test_pearson_spearman_and_fisher_match_scipy(seed):
    family = ("normal", "t3", "lognormal1", "kontaminiert")[seed % 4]
    n = (5, 8, 20, 60, 200)[seed % 5]
    x, y = _sample(family, n, 0.5, seed)
    assert A.pearson_r(x, y) == pytest.approx(sps.pearsonr(x, y)[0], abs=1e-12)
    assert A.spearman_r(x, y) == pytest.approx(sps.spearmanr(x, y)[0], abs=1e-12)
    assert A.t_test_p(A.pearson_r(x, y), n) == pytest.approx(sps.pearsonr(x, y)[1], rel=1e-8, abs=1e-14)
    if n >= 4:
        ci = sps.pearsonr(x, y).confidence_interval(0.95)
        assert A.fisher_interval(A.pearson_r(x, y), n) == pytest.approx((ci.low, ci.high), abs=1e-12)


def test_spearman_with_ties_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 4, 30).astype(float)
        y = rng.integers(0, 5, 30).astype(float)
        if x.std() and y.std():
            assert A.spearman_r(x, y) == pytest.approx(sps.spearmanr(x, y)[0], abs=1e-12)


def test_normal_functions_match_scipy():
    for p in (1e-12, 1e-6, 0.001, 0.1, 0.5, 0.9, 0.999, 1 - 1e-9):
        assert A.norm_ppf(p) == pytest.approx(sps.norm.ppf(p), abs=1e-9)
    for z in (-8.0, -2.5, 0.0, 0.7, 4.0):
        assert A.norm_cdf(z) == pytest.approx(sps.norm.cdf(z), abs=1e-14)


def test_betainc_and_t_distribution_match_scipy():
    for a, b, x in itertools.product((0.5, 1.0, 2.5, 14.0), (0.5, 3.0, 49.0), (0.001, 0.2, 0.5, 0.9, 0.999)):
        assert A.betainc_reg(a, b, x) == pytest.approx(float(spsp.betainc(a, b, x)), rel=1e-9, abs=1e-14)
    for df in (1, 2, 3, 10, 98, 998):
        for t in (0.1, 1.0, 2.2, 5.0):
            assert A.t_two_sided_p(t, df) == pytest.approx(2 * sps.t.sf(t, df), rel=1e-9, abs=1e-14)
        assert A.t_critical(df) == pytest.approx(sps.t.ppf(0.975, df), rel=1e-9)


def test_plan_n_matches_the_formula_with_scipy_quantiles():
    for rho in (0.1, 0.2, 0.3, 0.5, 0.7):
        n = math.ceil(((sps.norm.ppf(0.975) + sps.norm.ppf(0.8)) / math.atanh(rho)) ** 2 + 3)
        assert A.plan_n(rho) == n


@pytest.mark.parametrize("seed", range(10))
def test_jackknife_matches_the_loop(seed):
    x, y = _sample("t5", 12 + seed, 0.4, seed)
    loop = np.array([np.corrcoef(np.delete(x, i), np.delete(y, i))[0, 1] for i in range(len(x))])
    assert A.jackknife_r(x, y) == pytest.approx(loop, abs=1e-12)
    assert A.influence(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1] - loop, abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_bootstrap_matches_the_loop_with_the_same_indices(seed):
    x, y = _sample("normal", 15, 0.5, seed)
    B = 60
    idx = S.VRng(100 + seed).below(len(x), (B, len(x)))              # dieselben Indizes, die bootstrap_replicates zieht
    loop = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    assert A.bootstrap_replicates(x, y, B, S.VRng(100 + seed)) == pytest.approx(loop, abs=1e-12)


def test_percentile_interval_matches_numpy_on_the_replicates():
    x, y = _sample("normal", 25, 0.5, 1)
    rb = A.bootstrap_replicates(x, y, 500, S.VRng(2))
    lo, hi = A.percentile_interval(rb)
    assert (lo, hi) == pytest.approx(tuple(np.percentile(rb, [2.5, 97.5])))


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_permutation_p_matches_complete_enumeration(n):
    rng = S.VRng(n)
    x = rng.normal(n)
    y = 0.8 * x + 0.6 * rng.normal(n)
    r = abs(A.pearson_r(x, y))
    exact = np.mean([abs(A.pearson_r(x, y[list(p)])) >= r - 1e-12 for p in itertools.permutations(range(n))])
    est = A.permutation_p(x, y, 4000, S.VRng(50 + n))
    se = math.sqrt(max(exact * (1 - exact), 1e-4) / 4000)
    assert abs(est - (4000 * exact + 1) / 4001) < 5 * se + 0.002


def test_bca_matches_scipy_bootstrap_within_monte_carlo_error():
    for family, n, seed in (("normal", 40, 1), ("t3", 60, 2), ("lognormal1", 80, 3)):
        x, y = _sample(family, n, 0.5, seed)
        r = A.pearson_r(x, y)
        mine = A.bca_interval(x, y, A.bootstrap_replicates(x, y, 20000, S.VRng(seed)), r)
        ref = sps.bootstrap((x, y), lambda a, b: np.corrcoef(a, b)[0, 1], paired=True, vectorized=False, n_resamples=9999, method="BCa", random_state=3).confidence_interval
        assert mine == pytest.approx((ref.low, ref.high), abs=0.015), family


def test_predicted_coverage_and_variance_factor_match_scipy_moments():
    for nu in (6, 8, 10, 20):
        m2, m4 = sps.t.moment(2, nu), sps.t.moment(4, nu)
        # E[w⁴]/E[w²]² mit w = √(ν/χ²) ist (ν − 2)/(ν − 4); die Kurtosis der t-Verteilung ist 3(ν − 2)/(ν − 4)
        assert m4 / m2**2 == pytest.approx(3.0 * A.variance_factor(nu), rel=1e-9)
        assert A.predicted_coverage(nu) == pytest.approx(2 * sps.norm.cdf(sps.norm.ppf(0.975) / math.sqrt(A.variance_factor(nu))) - 1, abs=1e-12)


def test_rmse_ratio_matches_a_large_regression():
    x, y = S.draw_pairs("normal", 1, 400_000, 0.6, S.VRng(8))
    x, y = x[0], y[0]
    b, a0 = np.polyfit(x, y, 1)
    assert float((y - (a0 + b * x)).std() / y.std()) == pytest.approx(A.rmse_ratio(0.6), abs=2e-3)
    assert float(np.mean((x > 0) == (y > 0))) == pytest.approx(A.sign_hit(0.6), abs=2e-3)
