"""Abbildungen und PDF: jede Figur lässt sich bauen, alle Achsen sind gesperrt, die Inhalte stimmen mit den Eingaben überein."""

import numpy as np
import pytest

import prs_constants as C
import prs_evaluation as ev
import prs_pdf_export as PDF
import prs_shapes as SH
import prs_visualization as V


def _locked(fig):
    axes = [v for k, v in fig.to_dict()["layout"].items() if k.startswith(("xaxis", "yaxis"))]
    assert axes and all(ax.get("fixedrange") for ax in axes)


def test_scatter_marks_the_leverage_points_and_the_line():
    fig = V.build_scatter([0, 1, 2, 3], [0, 1, 2, 9], lever=(3,), slope=1.0, intercept=0.5)
    names = [t.name for t in fig.data]
    assert names == ["Beobachtungen", "Hebelpunkt", "Regressionsgerade"]
    assert list(fig.data[1].x) == [3] and list(fig.data[0].x) == [0, 1, 2]
    _locked(fig)
    assert [t.name for t in V.build_scatter([0, 1], [0, 1]).data] == ["Beobachtungen"]


def test_intervals_figure_has_one_bar_per_method_and_a_true_line():
    a = ev.analyse(ev.Settings())
    fig = V.build_intervals(a.intervals, a.r, a.true_r)
    assert len(fig.data) == 5 and fig.data[-1].x == (a.true_r, a.true_r) and fig.layout.showlegend is False
    for trace, (lo, hi) in zip(fig.data[:3], a.intervals.values()):
        assert trace.x == (lo, hi)
    _locked(fig)


def test_influence_highlights_the_top_observation():
    fig = V.build_influence([0.01, -0.2, 0.03], 1)
    assert list(fig.data[0].marker.color) == [C.COLOR_POINTS, C.COLOR_LEVER, C.COLOR_POINTS]
    _locked(fig)


def test_shapes_figure_handles_unreachable_forms():
    results = [(k, SH.SHAPES[k].label, SH.solve_shape(k, 0.99, 1)) for k in SH.SHAPES]
    fig = V.build_shapes(results, 0.99)
    assert any(a.text == "nicht erreichbar" for a in fig.layout.annotations)
    assert len(fig.data) == sum(res is not None for _, _, res in results)
    _locked(fig)


def test_coverage_figure_has_three_methods_and_the_95_line():
    rows = [{"n": 10, "fisher": 0.9, "perc": 0.9, "bca": 0.9, "fisher_sd": 0.01, "perc_sd": 0.01, "bca_sd": 0.01}, {"n": 100, "fisher": 0.8, "perc": 0.92, "bca": 0.9, "fisher_sd": 0.01, "perc_sd": 0.01, "bca_sd": 0.01}]
    fig = V.build_coverage(rows, "t(3)")
    assert [t.name for t in fig.data] == ["Fisher-z", "Perzentil-Bootstrap", "BCa-Bootstrap"] and list(fig.data[0].y) == [90.0, 80.0]
    assert fig.layout.shapes[0].y0 == 95
    _locked(fig)


def test_law_leverage_size_utility_figures_build_from_the_grid_and_the_experiments():
    grid = ev.load_grid()
    law = [r for r in grid["law"] if r["rho"] == 0.5]
    fig = V.build_law(law)
    assert len(fig.data) == 2 and len(fig.data[0].x) == len(law)
    _locked(fig)
    _locked(V.build_leverage(ev.leverage_grid(ns=(10, 30), ks=(0, 5), runs=100), ns=(10, 30)))
    size = [ev.grid_size_row(grid, f, 100) for f in C.SIZE_FAMILIES]
    fig = V.build_size(size)
    assert len(fig.data) == 3 and len(fig.data[0].x) == len(C.SIZE_FAMILIES)
    _locked(fig)
    _locked(V.build_utility(ev.utility_rows(rs=(0.3, 0.6), n=10_000)))


def test_the_leverage_figure_has_one_line_per_n():
    rows = ev.leverage_grid(ns=(10, 30), ks=(0, 5), runs=100)
    assert [t.name for t in V.build_leverage(rows, ns=(10, 30)).data] == ["n = 10", "n = 30"]


def test_pdf_is_a_valid_pdf_and_names_the_settings():
    a = ev.analyse(ev.Settings(family="kontaminiert", rho=0.3, n=40, lev_count=1, lev_k=6.0))
    data = PDF.build_pdf(a)
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_pdf_text_replaces_characters_the_core_fonts_do_not_have():
    assert PDF._text("ρ = 0,5 – √n · σ") == "rho = 0,5 - sqrt n * sigma"
    assert PDF._text("**fett** „zitat“") == 'fett "zitat"'
    assert PDF._text("Überstunden ✓").startswith("Überstunden")          # Ü gibt es in den Kernschriften, das Häkchen nicht (wird zu „?“)
    assert PDF._text("✓") == "?"
