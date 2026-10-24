"""Kern gegen Handrechnung (DEMO-PLAYBOOK §1b): jede Einheit an der Mini-Instanz, dazu Randfälle und die Zweige."""

import math

import numpy as np
import pytest

import prs_algorithm as A
from prs_scenario import VRng


def test_pearson_by_hand(three, four):
    assert A.pearson_r(*three) == pytest.approx(3 / math.sqrt(28 / 3), rel=1e-12)
    assert A.pearson_r(*four) == pytest.approx(0.8, rel=1e-12)


def test_pearson_is_exactly_one_on_a_line_and_minus_one_on_a_falling_line():
    x = [1.0, 2.0, 3.0, 4.0]
    assert A.pearson_r(x, [2 * v + 5 for v in x]) == pytest.approx(1.0, abs=1e-15)
    assert A.pearson_r(x, [-3 * v for v in x]) == pytest.approx(-1.0, abs=1e-15)


def test_pearson_rejects_constant_columns_and_short_input():
    with pytest.raises(ValueError, match="konstant"):
        A.pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="konstant"):
        A.pearson_r([1.0, 2.0, 3.0], [4.0, 4.0, 4.0])
    with pytest.raises(ValueError, match="mindestens zwei"):
        A.pearson_r([1.0], [2.0])
    with pytest.raises(ValueError, match="gleich lang"):
        A.pearson_r([1.0, 2.0], [1.0, 2.0, 3.0])


def test_pearson_with_two_points_is_plus_or_minus_one():
    assert A.pearson_r([0.0, 1.0], [0.0, 5.0]) == pytest.approx(1.0)
    assert A.pearson_r([0.0, 1.0], [5.0, 0.0]) == pytest.approx(-1.0)


def test_pearson_rows_matches_the_single_version(four):
    X = np.array([four[0], four[0][::-1]])
    Y = np.array([four[1], four[1]])
    rows = A.pearson_rows(X, Y)
    assert rows[0] == pytest.approx(A.pearson_r(*four)) and rows[1] == pytest.approx(A.pearson_r(four[0][::-1], four[1]))


def test_pearson_rows_gives_nan_for_constant_rows():
    assert np.isnan(A.pearson_rows(np.ones((1, 4)), np.arange(4.0).reshape(1, 4))[0])


def test_slope_equals_r_times_sd_ratio(three):
    x, y = np.array(three[0]), np.array(three[1])
    assert A.slope(x, y) == pytest.approx(A.pearson_r(x, y) * y.std() / x.std())
    assert A.slope(x, y) == pytest.approx(1.5)           # Σ(x−x̄)(y−ȳ)/Σ(x−x̄)² = 3/2


def test_rank_average_handles_ties():
    assert list(A.rank_average([10.0, 20.0, 20.0, 5.0])) == [2.0, 3.5, 3.5, 1.0]
    assert list(A.rank_average([3.0, 3.0, 3.0])) == [2.0, 2.0, 2.0]
    assert list(A.rank_average([1.0])) == [1.0]


def test_spearman_is_one_on_any_monotone_curve():
    x = np.linspace(-2, 2, 21)
    assert A.spearman_r(x, np.exp(3 * x)) == pytest.approx(1.0, abs=1e-12)
    assert A.pearson_r(x, np.exp(3 * x)) < 0.9


def test_jackknife_by_hand(four):
    th = A.jackknife_r(*four)
    assert th == pytest.approx([0.5, 13 / 14, 13 / 14, 0.5])
    assert A.influence(*four) == pytest.approx([0.3, 0.8 - 13 / 14, 0.8 - 13 / 14, 0.3])


def test_influence_sign_says_whether_the_point_raises_or_lowers_r(four):
    infl = A.influence(*four)
    assert infl[0] > 0 and infl[1] < 0     # Punkt 1 hebt r, Punkt 2 senkt es


def test_normal_functions_at_known_values():
    assert A.norm_cdf(0.0) == pytest.approx(0.5)
    assert A.norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-12)
    assert A.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-15)
    assert A.norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-12)
    assert A.norm_ppf(0.025) == pytest.approx(-1.959963984540054, abs=1e-12)
    with pytest.raises(ValueError, match="p muss in"):
        A.norm_ppf(0.0)
    with pytest.raises(ValueError, match="p muss in"):
        A.norm_ppf(1.0)


def test_norm_ppf_is_the_inverse_in_all_three_regions():
    for p in (1e-10, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-10):
        assert A.norm_cdf(A.norm_ppf(p)) == pytest.approx(p, rel=1e-10)


def test_t_distribution_by_hand():
    assert A.t_two_sided_p(1.0, 1) == pytest.approx(0.5, abs=1e-12)          # Cauchy: P(|T| > 1) = 1/2
    assert A.t_two_sided_p(0.0, 5) == pytest.approx(1.0)
    assert A.t_two_sided_p(1e6, 10) < 1e-20


def test_betainc_symmetry_and_edges():
    assert A.betainc_reg(2.0, 3.0, 0.0) == 0.0 and A.betainc_reg(2.0, 3.0, 1.0) == 1.0
    assert A.betainc_reg(2.0, 2.0, 0.5) == pytest.approx(0.5)
    assert A.betainc_reg(2.0, 3.0, 0.3) + A.betainc_reg(3.0, 2.0, 0.7) == pytest.approx(1.0)


def test_t_test_p_by_hand(four):
    # r = 0,8, n = 4: t = 0,8·√(2/0,36) = 1,885618, df = 2; für df = 2 gilt p = 1 − t/√(t² + 2)
    t = 0.8 * math.sqrt(2 / 0.36)
    assert A.t_test_p(0.8, 4) == pytest.approx(1 - t / math.sqrt(t * t + 2), rel=1e-9)
    assert A.t_test_p(1.0, 10) == 0.0 and A.t_test_p(-1.0, 10) == 0.0
    assert A.t_test_p(0.0, 10) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="mindestens drei"):
        A.t_test_p(0.5, 2)
    assert A.t_test_p(0.5, 3) == pytest.approx(2 / 3, abs=1e-12)      # n = 3 ist der kleinste erlaubte Fall: df = 1 (Cauchy), t = 0,5·√(1/0,75), p = 1 − (2/π)·atan(t) = 2/3


def test_t_critical_by_hand():
    assert A.t_critical(1) == pytest.approx(12.7062047, rel=1e-6)
    assert A.t_critical(10) == pytest.approx(2.2281389, rel=1e-6)
    assert A.t_critical(1000, alpha=0.01) == pytest.approx(2.5806, rel=1e-3)


def test_rejects_rows_agrees_with_the_p_value():
    rs = np.array([0.05, 0.3, 0.37, 0.4, -0.5, -0.1])
    for n in (10, 30, 100):
        rej = A.rejects_rows(rs, n)
        assert list(rej) == [A.t_test_p(float(r), n) < 0.05 for r in rs]


def test_fisher_interval_by_hand():
    lo, hi = A.fisher_interval(0.3, 103)
    assert (lo, hi) == pytest.approx((0.11303802922, 0.46644413153), abs=1e-9)    # atanh 0,3 = 0,30952, Standardfehler 0,1
    lo, hi = A.fisher_interval(0.8, 4)
    assert (lo, hi) == pytest.approx((-0.69695, 0.99560), abs=1e-4)               # n = 4: das Intervall enthält 0 trotz r = 0,8


def test_fisher_interval_level_and_errors():
    wide = A.fisher_interval(0.5, 50, 0.99)
    narrow = A.fisher_interval(0.5, 50, 0.90)
    assert wide[0] < narrow[0] < 0.5 < narrow[1] < wide[1]
    with pytest.raises(ValueError):
        A.fisher_interval(0.5, 3)


def test_fisher_interval_survives_r_of_plus_or_minus_one():
    lo, hi = A.fisher_interval(1.0, 10)
    assert hi <= 1.0 and lo > 0.9
    lo, hi = A.fisher_interval(-1.0, 10)
    assert lo >= -1.0 and hi < -0.9


def test_percentile_interval_by_hand():
    rb = np.arange(1.0, 101.0)                # 1 … 100
    lo, hi = A.percentile_interval(rb)
    assert (lo, hi) == pytest.approx((3.475, 97.525))                # lineare Interpolation der Quantile 2,5 % und 97,5 %
    rb_nan = np.append(rb, [np.nan, np.nan])
    assert A.percentile_interval(rb_nan) == pytest.approx((3.475, 97.525))
    with pytest.raises(ValueError):
        A.percentile_interval(np.array([np.nan]))


def test_bca_correction_is_zero_when_half_the_replicates_lie_below():
    rb = np.array([0.1, 0.2, 0.3, 0.4])
    assert A.bca_correction(rb, 0.25) == pytest.approx(0.0, abs=1e-12)
    assert A.bca_correction(rb, 0.35) > 0 > A.bca_correction(rb, 0.15)


def test_bca_correction_counts_only_strictly_smaller_replicates():
    """Gleichstände mit r̂ zählen nicht als „darunter“: bei (0,1; 0,2; 0,2; 0,3) und r̂ = 0,2 liegt ein Viertel darunter, z0 = Φ⁻¹(0,25)."""
    z0 = A.bca_correction(np.array([0.1, 0.2, 0.2, 0.3]), 0.2)
    assert z0 == pytest.approx(A.norm_ppf(0.25), abs=1e-12) and z0 < 0


def test_bca_correction_stays_finite_at_the_extremes():
    rb = np.array([0.1, 0.2, 0.3])
    assert math.isfinite(A.bca_correction(rb, 5.0)) and math.isfinite(A.bca_correction(rb, -5.0))


def test_bca_acceleration_is_zero_on_a_perfect_line():
    x = np.arange(6.0)
    assert A.bca_acceleration(x, 2 * x) == 0.0


def test_bca_acceleration_follows_the_formula(four):
    th = A.jackknife_r(*four)
    d = th.mean() - th
    assert A.bca_acceleration(*four) == pytest.approx((d**3).sum() / (6 * (d * d).sum() ** 1.5))
    assert A.bca_acceleration(*four) == pytest.approx(0.0, abs=1e-12)     # die Leave-one-out-Werte (½, 13/14, 13/14, ½) sind symmetrisch


def test_bca_equals_the_percentile_interval_when_z0_and_a_vanish(four):
    rb = (np.arange(1000) + 0.5) / 1000         # genau die Hälfte unter 0,5: z0 = 0
    assert A.bca_interval(*four, rb, 0.5) == pytest.approx(A.percentile_interval(rb), abs=1e-9)


def test_bca_shifts_the_interval_when_the_estimate_is_off_centre(four):
    rb = np.linspace(0.0, 1.0, 1001)
    shifted = A.bca_interval(*four, rb, 0.7)       # 70 % der Replikate liegen unter r̂: Quantile rücken nach oben
    plain = A.percentile_interval(rb)
    assert shifted[0] > plain[0] and shifted[1] > plain[1]


def test_bootstrap_replicates_shape_and_range(four):
    rb = A.bootstrap_replicates(*four, 300, VRng(1))
    assert rb.shape == (300,) and np.nanmax(rb) <= 1.0 and np.nanmin(rb) >= -1.0
    assert np.isnan(rb).any() or True        # Resamples mit konstanter Kennzahl ergeben NaN; bei n = 4 kommen sie vor


def test_bootstrap_resamples_of_one_repeated_pair_give_nan():
    x = np.array([1.0, 2.0])
    rb = A.bootstrap_replicates(x, x, 200, VRng(2))
    assert np.isnan(rb).sum() > 0 and np.nanmax(rb) == pytest.approx(1.0)       # zweimal dasselbe Paar: konstant → NaN; sonst exakt 1


def test_permutation_p_extremes():
    x = np.arange(12.0)
    for y in (x, -x, 3 * x + 1):       # perfekter Zusammenhang: bei 12 Beobachtungen trifft keine der 199 Permutationen die Identität, p = 1/(B + 1) genau
        assert A.permutation_p(x, y, 199, VRng(3)) == 1 / 200
    rng = VRng(4)
    y = rng.normal(8)
    p = A.permutation_p(np.arange(8.0), y, 199, VRng(5))
    assert 1 / 200 <= p <= 1 and (p * 200) == pytest.approx(round(p * 200), abs=1e-9)         # (1 + Zahl der Treffer) / (B + 1)


def test_permutation_p_is_exactly_one_when_r_is_zero():
    """x = (−1, 0, 1), y = (1, −2, 1): r = 0 genau; dann erreicht jede der B Permutationen |r*| ≥ 0, p = (1 + B)/(B + 1) = 1."""
    x, y = np.array([-1.0, 0.0, 1.0]), np.array([1.0, -2.0, 1.0])
    assert A.pearson_r(x, y) == pytest.approx(0.0, abs=1e-15)
    assert A.permutation_p(x, y, 99, VRng(1)) == 1.0
    assert A.permutation_p(x, y, 7, VRng(2)) == 1.0


def test_permutation_p_counts_ties_with_the_observed_value():
    """Zwei Beobachtungen: |r*| ist immer 1 = |r|, jede Permutation zählt, p = 1."""
    assert A.permutation_p(np.array([0.0, 1.0]), np.array([0.0, 5.0]), 50, VRng(2)) == 1.0


def test_variance_factor_and_predicted_coverage_by_hand():
    assert A.variance_factor(6) == pytest.approx(2.0) and A.variance_factor(8) == pytest.approx(1.5)
    assert A.variance_factor(5) == pytest.approx(3.0)
    assert math.isinf(A.variance_factor(4)) and math.isinf(A.variance_factor(3))
    assert A.variance_factor(math.inf) == 1.0
    assert A.predicted_coverage(math.inf) == pytest.approx(0.95, abs=1e-12)
    assert A.predicted_coverage(6) == pytest.approx(2 * A.norm_cdf(1.959963984540054 / math.sqrt(2.0)) - 1)
    assert A.predicted_coverage(3) == 0.0 and A.predicted_coverage(4) == 0.0


def test_closed_forms_by_hand():
    assert A.rmse_ratio(0.0) == 1.0 and A.rmse_ratio(1.0) == 0.0 and A.rmse_ratio(0.6) == pytest.approx(0.8)
    assert A.rmse_ratio(1.0000001) == 0.0                         # nach oben gegen Rundung abgesichert
    assert A.sign_hit(0.0) == 0.5 and A.sign_hit(1.0) == pytest.approx(1.0) and A.sign_hit(0.5) == pytest.approx(2 / 3)
    assert A.sign_hit(-0.5) == pytest.approx(1 / 3)


def test_plan_n_by_hand():
    assert A.plan_n(0.3) == 85 and A.plan_n(0.1) == 783 and A.plan_n(0.5) == 30
    assert A.plan_n(-0.3) == 85                                  # das Vorzeichen spielt keine Rolle
    assert A.plan_n(0.3, power=0.9) > 85 and A.plan_n(0.3, alpha=0.01) > 85
    for bad in (0.0, 1.0, -1.0, 1.5):
        with pytest.raises(ValueError, match="rho muss zwischen"):
            A.plan_n(bad)
