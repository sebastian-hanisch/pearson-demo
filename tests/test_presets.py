"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Hilfetexte mit gemessenen Zahlen."""

import prs_constants as C
import prs_evaluation as ev
import prs_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert preset["family"] in C.FAMILIES
        assert C.RHO_MIN <= preset["rho"] <= C.RHO_MAX and C.N_MIN <= preset["n"] <= C.N_MAX
        assert 0 <= preset["lev_count"] <= C.LEV_COUNT_MAX and 0 <= preset["lev_k"] <= C.LEV_K_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    assert ev.Settings(family=p["family"], rho=p["rho"], n=p["n"], lev_count=p["lev_count"], lev_k=p["lev_k"], seed=p["seed"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_defaults_of_the_specs_match_the_default_settings():
    s = ev.Settings()
    assert [P.SETTING_SPECS[k].default for k in ("family_select", "rho_slider", "n_slider", "lev_count_slider", "lev_k_slider", "lev_dir_select", "boot_slider", "reps_slider", "seed_input")] == [
        s.family, s.rho, s.n, s.lev_count, s.lev_k, s.lev_dir, s.boot_b, s.reps, s.seed]


def test_url_parameters_are_unique():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_the_presets_show_what_their_help_text_promises():
    """Die Hilfetexte nennen Zahlen; hier stehen die Größen, die sie behaupten (Seed 35, wie im Preset)."""
    one = ev.analyse(ev.Settings(**{k: v for k, v in C.PRESETS["Ein einzelner Punkt (k = 5, n = 30)"].items()}))
    assert one.true_r == 0.0 and one.lever == (0,)
    heavy = C.PRESETS["Schwerer Rand (t(3), n = 300)"]
    assert heavy["family"] == "t3" and heavy["n"] == 300
    zero = C.PRESETS["Unkorreliert, aber nicht unabhängig (t(3), ρ = 0)"]
    assert zero["rho"] == 0.0 and zero["family"] == "t3"
