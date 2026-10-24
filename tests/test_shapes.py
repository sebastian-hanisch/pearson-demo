"""Formen: exakt gleiches r, Erreichbarkeit, Eigenheiten der Formen."""

import numpy as np
import pytest

import prs_algorithm as A
import prs_constants as C
import prs_shapes as SH


def test_orth_pair_is_exactly_orthogonal_and_standardised():
    a, b = SH.orth_pair(3, 40)
    assert abs(a.mean()) < 1e-14 and abs(b.mean()) < 1e-14
    assert a.std() == pytest.approx(1.0) and b.std() == pytest.approx(1.0)
    assert abs(float(a @ b)) < 1e-12 and len(a) == 40


@pytest.mark.parametrize("key", list(SH.SHAPES))
@pytest.mark.parametrize("target", [0.3, 0.5, 0.8])
def test_every_shape_hits_the_target_exactly(key, target):
    solved = 0
    for seed in range(1, 11):
        res = SH.solve_shape(key, target, seed)
        if res is None:
            continue
        solved += 1
        assert abs(A.pearson_r(res.x, res.y) - target) < 1e-12
        assert res.r == pytest.approx(target, abs=1e-12) and len(res.x) == C.SHAPE_N
    # Die U-Form mit Schräge hat schon bei t = 0 ein zufälliges r (bis etwa 0,3): kleine Ziele sind in einem Teil der Seeds nicht erreichbar (Vormessung: 18 von 20 bei r = 0,3).
    assert solved >= (6 if (key, target) == ("parabel", 0.3) else 9), (key, target)


def test_unreachable_targets_give_none_and_reachable_range_explains_it():
    assert SH.solve_shape("kurve", 0.99, 1) is None
    lo, hi = SH.reachable_range("kurve", 1)
    assert hi < 0.99 and lo < 0.01
    assert SH.solve_shape("gegenausreisser", -0.5, 1) is not None
    assert SH.solve_shape("wolke", 0.999, 1) is not None


def test_target_at_the_range_end_is_accepted():
    lo, hi = SH.reachable_range("wolke", 2)
    assert SH.solve_shape("wolke", hi - 1e-13, 2) is not None


def test_solve_is_deterministic_and_seed_dependent():
    a, b = SH.solve_shape("zwei_gruppen", 0.8, 5), SH.solve_shape("zwei_gruppen", 0.8, 5)
    assert a == b and a != SH.solve_shape("zwei_gruppen", 0.8, 6)


def test_the_single_point_carries_the_whole_correlation():
    res = SH.solve_shape("hebelpunkt", 0.8, 1)
    x, y = np.array(res.x), np.array(res.y)
    assert abs(A.pearson_r(x[:-1], y[:-1])) < 1e-12              # ohne den Punkt sind die übrigen exakt unkorreliert
    assert np.abs(A.influence(x, y)).argmax() == len(x) - 1 and np.abs(A.influence(x, y)).max() == pytest.approx(0.8)


def test_hebelpunkt_has_small_spearman_over_many_seeds():
    values = [A.spearman_r(*[np.array(v) for v in (r.x, r.y)]) for r in (SH.solve_shape("hebelpunkt", 0.8, s) for s in range(1, 21))]
    assert max(abs(v) for v in values) < 0.35


def test_counter_outlier_lowers_r_from_the_bulk():
    res = SH.solve_shape("gegenausreisser", 0.8, 1)
    x, y = np.array(res.x), np.array(res.y)
    assert A.pearson_r(x[:-1], y[:-1]) == pytest.approx(0.97, abs=1e-9)
    assert A.influence(x, y)[-1] < 0                              # der Punkt senkt r


def test_shapes_differ_from_each_other_on_at_least_one_seed():
    """Zweig-Test (Nullspalten-Regel): jede Form unterscheidet sich von jeder anderen in Spearman oder Steigung."""
    stats = {}
    for key in SH.SHAPES:
        res = SH.solve_shape(key, 0.8, 1)
        x, y = np.array(res.x), np.array(res.y)
        stats[key] = (round(A.spearman_r(x, y), 6), round(A.slope(x, y), 6))
    assert len(set(stats.values())) == len(SH.SHAPES)


def test_hebelpunkt_and_gegenausreisser_have_one_point_more_than_their_base():
    assert SH.SHAPES["hebelpunkt"].n_base == C.SHAPE_N - 1 and SH.SHAPES["gegenausreisser"].n_base == C.SHAPE_N - 1
    assert SH.SHAPES["wolke"].n_base == C.SHAPE_N
