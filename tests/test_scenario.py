"""Szenario: der vektorisierte Generator gegen die skalare Klasse, Verteilungseigenschaften, Hebelpunkte, Teilströme."""

import math

import numpy as np
import pytest

import prs_algorithm as A
import prs_constants as C
import prs_scenario as S
from prs_rng import SplitMix64


def test_vectorised_stream_equals_the_scalar_class():
    for seed in (0, 7, 2**63 + 5, 2**64 - 1):
        scalar = SplitMix64(seed)
        assert [int(v) for v in S.VRng(seed).u64(200)] == [scalar.next() for _ in range(200)]


def test_stream_continues_across_calls():
    a = S.VRng(9)
    first, second = a.u64(10), a.u64(10)
    assert [int(v) for v in np.concatenate([first, second])] == [int(v) for v in S.VRng(9).u64(20)]


def test_uniform_is_in_the_unit_interval_with_53_bits():
    u = S.VRng(3).uniform(100_000)
    assert u.min() >= 0.0 and u.max() < 1.0 and u.mean() == pytest.approx(0.5, abs=0.01)
    assert S.VRng(3).uniform((4, 5)).shape == (4, 5)


def test_normal_moments_and_shape():
    z = S.VRng(4).normal(200_000)
    assert z.mean() == pytest.approx(0.0, abs=0.01) and z.std() == pytest.approx(1.0, abs=0.01)
    assert S.VRng(4).normal((3, 7)).shape == (3, 7) and S.VRng(4).normal(5).shape == (5,)
    assert np.mean(np.abs(z) > 1.959963984540054) == pytest.approx(0.05, abs=0.005)


def test_below_stays_in_range():
    k = S.VRng(5).below(7, (1000, 10))
    assert k.min() == 0 and k.max() == 6 and k.dtype.kind == "i"


def test_substream_seeds_are_distinct_and_not_shifted_copies():
    seeds = S.substream_seeds(1, 500)
    assert len(set(seeds)) == 500
    assert all(abs(a - b) % S.GAMMA != 0 for a, b in zip(seeds, seeds[1:]))


def test_same_seed_same_dataset_and_other_seed_other_dataset():
    a = S.make_dataset("t3", 40, 0.5, 11)
    assert a == S.make_dataset("t3", 40, 0.5, 11) and a != S.make_dataset("t3", 40, 0.5, 12)


@pytest.mark.parametrize("family", list(C.ALL_FAMILIES))
def test_each_family_has_the_documented_true_correlation(family):
    x, y = S.draw_pairs(family, 1, 400_000, 0.5, S.VRng(21))
    assert A.pearson_r(x[0], y[0]) == pytest.approx(S.true_r(family, 0.5), abs=0.03 if family != "lognormal1" else 0.05)


def test_true_r_by_hand():
    assert S.true_r("normal", 0.3) == 0.3 and S.true_r("t3", -0.4) == -0.4 and S.true_r("tindep3", 0.9) == 0.0
    assert S.true_r("lognormal1", 0.0) == 0.0
    assert S.true_r("lognormal1", 1.0) == pytest.approx(1.0)
    assert S.true_r("lognormal1", 0.5) == pytest.approx((math.e**0.5 - 1) / (math.e - 1))
    assert S.true_r("lognormal05", 0.5) == pytest.approx((math.exp(0.125) - 1) / (math.exp(0.25) - 1))


def test_t_family_has_the_documented_kurtosis():
    x, _ = S.draw_pairs("t10", 1, 600_000, 0.0, S.VRng(22))
    x = x[0]
    kurt = float(np.mean(x**4) / np.mean(x**2) ** 2)
    assert kurt == pytest.approx(3.0 * (10 - 2) / (10 - 4), rel=0.1)          # t(ν): Kurtosis 3(ν−2)/(ν−4) = 4


def test_common_scale_makes_squares_correlated_but_independent_margins_do_not():
    for family, expect_dependent in (("t3", True), ("tindep3", False)):
        x, y = S.draw_pairs(family, 1, 300_000, 0.0, S.VRng(23))
        c = A.pearson_r(np.abs(x[0]), np.abs(y[0]))
        assert (c > 0.2) == expect_dependent, family


def test_contaminated_share_and_scale():
    x, _ = S.draw_pairs("kontaminiert", 1, 400_000, 0.0, S.VRng(24))
    assert x[0].var() == pytest.approx((1 - 0.05) + 0.05 * 25, rel=0.03)


def test_draw_pairs_validates_input():
    with pytest.raises(ValueError, match="unbekannte Familie"):
        S.draw_pairs("gamma", 1, 10, 0.0, S.VRng(1))
    with pytest.raises(ValueError, match="rho"):
        S.draw_pairs("normal", 1, 10, 1.0, S.VRng(1))
    with pytest.raises(ValueError, match="rho"):
        S.draw_pairs("normal", 1, 10, -1.0, S.VRng(1))


def test_t_df_parses_families():
    assert S.t_df("t3") == 3 and S.t_df("t20") == 20 and S.t_df("tindep3") == 3
    assert S.t_df("t2") is None and S.t_df("normal") is None and S.t_df("tx") is None


def test_add_leverage_places_the_points_in_units_of_the_rest():
    x = np.arange(10.0)
    y = np.arange(10.0)[::-1] * 2
    x2, y2, idx = S.add_leverage(x, y, 2, 5.0, 1)
    assert idx == (0, 1)
    assert x2[0] == pytest.approx(x[2:].mean() + 5 * x[2:].std()) and y2[0] == pytest.approx(y[2:].mean() + 5 * y[2:].std())
    assert np.array_equal(x2[2:], x[2:]) and np.array_equal(y2[2:], y[2:])
    x3, y3, _ = S.add_leverage(x, y, 1, 5.0, -1)
    assert y3[0] == pytest.approx(y[1:].mean() - 5 * y[1:].std())


def test_add_leverage_without_effect_returns_copies():
    x, y = np.arange(6.0), np.arange(6.0)
    for count, k in ((0, 5.0), (1, 0.0)):
        x2, y2, idx = S.add_leverage(x, y, count, k, 1)
        assert idx == () and np.array_equal(x2, x) and x2 is not x


def test_add_leverage_keeps_three_untouched_points():
    x, y = np.arange(5.0), np.arange(5.0)
    _, _, idx = S.add_leverage(x, y, 4, 3.0, 1)
    assert idx == (0, 1)


def test_leverage_raises_r_for_independent_data():
    ds0 = S.make_dataset("normal", 30, 0.0, 5)
    ds1 = S.make_dataset("normal", 30, 0.0, 5, 1, 8.0, 1)
    ds2 = S.make_dataset("normal", 30, 0.0, 5, 1, 8.0, -1)
    r0 = A.pearson_r(ds0.x, ds0.y)
    assert A.pearson_r(ds1.x, ds1.y) > r0 + 0.3 and A.pearson_r(ds1.x, ds1.y) > 0.3
    assert A.pearson_r(ds2.x, ds2.y) < r0 - 0.3 and A.pearson_r(ds2.x, ds2.y) < -0.3
    assert ds1.lever == (0,) and ds0.lever == ()
