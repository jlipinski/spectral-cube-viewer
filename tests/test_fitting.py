import numpy as np
import pytest

from spectral_cube_viewer.fitting import (MODEL_GAUSS, MODEL_HFS_FIXED, MODEL_HFS_FREE, MODEL_HFS_TAU, LineModel,
                                    auto_guess, fit_lines, gaussian)
from spectral_cube_viewer.lines import HFS_PRESETS, preset_hfs, preset_reference

V = np.arange(-60, 20, 0.2)


def test_two_gaussians_with_auto_guess():
    rng = np.random.default_rng(1)
    y = gaussian(V, 2.0, -25.0, 2.5) + gaussian(V, 1.2, -21.0, 1.5) + rng.normal(0, 0.1, V.size)
    model = LineModel(MODEL_GAUSS, 2)
    sel = (V > -35) & (V < -12)
    guesses = auto_guess(V[sel], y[sel], 2, model, 0.1)
    assert len(guesses) == 2
    fit = fit_lines(V[sel], y[sel], model, guesses, (-35, -12), 0.1)
    p = fit.popt.reshape(2, 3)
    np.testing.assert_allclose(p[:, 1], [-25, -21], atol=0.2)      # sorted by velocity
    assert 0.6 < fit.chi2r < 1.5
    area, err = fit.group_area(0)
    assert area == pytest.approx(1.0645 * p[0, 0] * p[0, 2], rel=1e-3) and err > 0
    assert 'Component 2' in fit.report()


def test_absorption_component_allowed():
    y = gaussian(V, 3.0, -20, 4.0) + gaussian(V, -1.0, -20, 1.0)
    fit = fit_lines(V, y, LineModel(MODEL_GAUSS, 2), [(3, -20, 4), (-1, -20, 1)], (-60, 20))
    assert fit.popt.reshape(2, 3)[:, 0].min() < -0.8


@pytest.mark.parametrize('kind, ptrue, guess', [
    (MODEL_HFS_FIXED, [1.5, -24.0, 2.0], (1.2, -23.5, 2.5)),
    (MODEL_HFS_FREE, [0.8, 1.0, 0.5, -24.0, 2.0], (1.0, -23.5, 2.5)),
    (MODEL_HFS_TAU, [3.0, -24.0, 2.0, 2.5], (2.5, -23.5, 2.5)),
])
def test_hcn_hyperfine_models(kind, ptrue, guess):
    preset = HFS_PRESETS['HCN 1-0']
    hfs = preset_hfs(preset, preset_reference(preset)[0])
    model = LineModel(kind, 1, hfs)
    rng = np.random.default_rng(2)
    y = model(V, *ptrue) + rng.normal(0, 0.05, V.size)
    sel = (V > -45) & (V < 0)
    fit = fit_lines(V[sel], y[sel], model, [guess], (-45, 0), 0.05)
    np.testing.assert_allclose(fit.popt, ptrue, rtol=0.15, atol=0.05)


def test_n2hp_opacity_fit_with_catalog_preset():
    preset = HFS_PRESETS['N2H+ 1-0']
    ref, from_cube = preset_reference(preset, 93173.7637)
    assert from_cube
    model = LineModel(MODEL_HFS_TAU, 1, preset_hfs(preset, ref))
    v = np.arange(-30, 30, 0.06)
    ptrue = [5.0, 3.0, 0.4, 4.0]
    y = model(v, *ptrue) + np.random.default_rng(3).normal(0, 0.1, v.size)
    fit = fit_lines(v, y, model, [(4.0, 3.0, 0.6)], (-30, 30), 0.1)
    np.testing.assert_allclose(fit.popt, ptrue, rtol=0.15)


def test_too_few_channels():
    with pytest.raises(ValueError):
        fit_lines(V[:3], np.ones(3), LineModel(MODEL_GAUSS, 1), [(1, V[1], 1)], (V[0], V[2]))
