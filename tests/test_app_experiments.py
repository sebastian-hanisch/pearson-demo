"""App: jedes Preset, der Permalink, jeder Experiment-Knopf einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d, Lehre aus Stück 10: jeder Knopf einzeln im AppTest)."""

import os
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import prs_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    assert at.session_state["n_slider"] == preset["n"] and at.session_state["family_select"] == preset["family"]
    assert len(at.get("plotly_chart")) >= 8


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["fam"] = "t3"
    at.query_params["n"] = "99999"
    at.query_params["rho"] = "-0.4"
    at.query_params["ld"] = "quer"          # ungültig: wird ignoriert
    at.run()
    assert not at.exception
    assert at.session_state["family_select"] == "t3" and at.session_state["n_slider"] == C.N_MAX
    assert at.session_state["rho_slider"] == -0.4 and at.session_state["lev_dir_select"] == "gleichläufig"


def test_the_address_line_mirrors_the_settings():
    at = _app()
    at.sidebar.slider(key="n_slider").set_value(77).run()
    assert at.query_params["n"] == ["77"] or at.query_params["n"] == "77"


def test_leverage_widgets_are_disabled_without_leverage_points():
    at = _app()
    assert at.sidebar.slider(key="lev_k_slider").proto.disabled and at.sidebar.selectbox(key="lev_dir_select").proto.disabled
    at.sidebar.slider(key="lev_count_slider").set_value(1).run()
    assert not at.sidebar.slider(key="lev_k_slider").proto.disabled and not at.sidebar.selectbox(key="lev_dir_select").proto.disabled


@pytest.mark.parametrize("direction", list(C.LEV_DIRS))
@pytest.mark.parametrize("edge", ["min", "max"])
def test_leverage_position_at_its_limits_with_one_and_two_points(edge, direction):
    for count in (1, 2):
        at = _app()
        at.sidebar.slider(key="lev_count_slider").set_value(count).run()
        w = at.sidebar.slider(key="lev_k_slider")
        w.set_value(w.min if edge == "min" else w.max).run()
        at.sidebar.selectbox(key="lev_dir_select").set_value(direction).run()
        assert not at.exception, (count, edge, direction, [str(e) for e in at.exception])


def test_main_result_changes_with_a_leverage_point():
    at = _app()
    before = at.metric[0].value
    at.sidebar.slider(key="lev_count_slider").set_value(1).run()
    at.sidebar.slider(key="lev_k_slider").set_value(8.0).run()
    assert at.metric[0].value != before and not at.exception


def test_coverage_button_shows_a_verdict():
    at = _app()
    at.sidebar.slider(key="reps_slider").set_value(100).run()
    at.sidebar.slider(key="boot_slider").set_value(200).run()
    at.button(key="btn_cov").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "Abdeckung der 95 %-Intervalle über" in _texts(at)


def test_coverage_result_disappears_when_the_settings_change():
    at = _app()
    at.sidebar.slider(key="reps_slider").set_value(100).run()
    at.button(key="btn_cov").click().run()
    assert "Abdeckung der 95 %-Intervalle über" in _texts(at)
    at.sidebar.slider(key="n_slider").set_value(31).run()
    assert "Abdeckung der 95 %-Intervalle über" not in _texts(at)


def test_law_button_shows_the_live_table():
    at = _app()
    at.button(key="btn_law").click().run()
    assert not at.exception and "Live (n = 1 000" in " ".join(c.value for c in at.caption)


def test_size_button_shows_a_verdict_for_the_current_family():
    at = _app()
    at.sidebar.selectbox(key="family_select").set_value("t3").run()
    at.button(key="btn_size").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "Bei wahrem ρ = 0 schlägt der Test (t(3), gemeinsame Skala, n = 30)" in _texts(at)


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    at.slider(key="shape_target").set_value(0.95).run()
    assert not at.exception
    row = re.search(r"\| Gekrümmte Kurve \| (\d+) von 50 \|", " ".join(m.value for m in at.markdown))
    assert row and int(row.group(1)) < 50                      # bei r = 0,95 ist die gekrümmte Kurve nicht mehr in jedem Seed erreichbar
    for key, value in (("grid_family", "lognormal1"), ("grid_rho", 0.9), ("law_rho", 0.0), ("lev_n", 300), ("lev_k_sel", 15), ("size_n", 300), ("size_family", "kontaminiert")):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, [str(e) for e in at.exception])


def test_corner_combination_of_all_minima_and_all_maxima():
    for edge in ("min", "max"):
        at = _app()
        w = at.sidebar.slider(key="lev_count_slider")
        w.set_value(int(w.min if edge == "min" else w.max)).run()         # erst die Zahl der Hebelpunkte: ohne Hebelpunkt sind Lage und Richtung gesperrt
        for key in ("rho_slider", "n_slider", "lev_k_slider", "boot_slider", "reps_slider"):
            w = at.sidebar.slider(key=key)
            if w.proto.disabled:
                continue
            value = w.min if edge == "min" else w.max
            w.set_value(int(value) if isinstance(w.value, int) else value)
        at.sidebar.selectbox(key="family_select").set_value("t3" if edge == "min" else "lognormal1")
        at.run()
        assert not at.exception, (edge, [str(e) for e in at.exception])
