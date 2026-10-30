import warnings

import numpy as np
import pytest

from spectral_cube_viewer.core import (LazyCube, baseline_cube, fit_baseline, format_windows, merge_windows, moment0,
                                 open_cube, parse_windows, region_mean_spectrum, windows_mask)
from conftest import LINE_FWHM, LINE_V, synthetic_data


def test_open_cube_float(cube_file):
    path, data, vel = cube_file
    hdul, cube, lazy = open_cube(str(path))
    try:
        assert lazy.shape == data.shape
        np.testing.assert_allclose(lazy[:], data, equal_nan=True)
        v = cube.with_spectral_unit('km/s', velocity_convention='radio').spectral_axis.value
        np.testing.assert_allclose(v, vel, atol=1e-9)
    finally:
        hdul.close()


def test_open_cube_scaled_int_with_blanks(scaled_cube_file):
    path, expected, _ = scaled_cube_file
    hdul, _, lazy = open_cube(str(path))
    try:
        got = lazy[:]
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, expected, atol=1e-4, equal_nan=True)   # BLANK and -1000 -> NaN
    finally:
        hdul.close()


def test_lazy_cube_slicing_matches_full():
    raw = np.arange(60, dtype='>i4').reshape(5, 3, 4)
    lazy = LazyCube(raw, bscale=0.5, bzero=1.0, blank=7)
    full = lazy[:]
    assert np.isnan(full.ravel()[7])
    np.testing.assert_array_equal(lazy[1:3, :, 2], full[1:3, :, 2])


def test_moment0_matches_direct_sum():
    data, vel = synthetic_data()
    dv = np.abs(np.gradient(vel))
    i0, i1 = 50, 300
    m0 = moment0(data, i0, i1, dv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.nansum(data[i0:i1] * dv[i0:i1, None, None], axis=0)
    ref[np.all(np.isnan(data[i0:i1]), axis=0)] = np.nan
    np.testing.assert_allclose(m0, ref, rtol=1e-5, equal_nan=True)
    assert np.isnan(m0[:, 0]).all()


def test_region_mean_spectrum():
    data, _ = synthetic_data()
    mask = np.zeros(data.shape[1:], bool)
    mask[2:6, 0:5] = True                       # includes the NaN column
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.nanmean(data[:, mask].astype(np.float64), axis=1)
    np.testing.assert_allclose(region_mean_spectrum(data, mask), ref)
    with pytest.raises(ValueError):
        region_mean_spectrum(data, np.zeros_like(mask))


def test_windows_parsing():
    assert parse_windows('-80:-40, 10:50; 60 : 55') == [(-80, -40), (10, 50), (55, 60)]
    assert merge_windows([(0, 5), (3, 8), (10, 12)]) == [(0, 8), (10, 12)]
    assert parse_windows(format_windows([(-1.5, 2.25)])) == [(-1.5, 2.25)]
    with pytest.raises(ValueError):
        parse_windows('1-2')


def test_baseline_spectrum_and_cube_agree():
    data, vel = synthetic_data()
    wmask = windows_mask(vel, [(-19, -5), (12, 19)])
    out, rms, i0, i1 = baseline_cube(data, vel, wmask, 1)
    assert out.shape == (i1 - i0,) + data.shape[1:]
    assert np.isnan(out[:, :, 0]).all()
    spec = data[:, 5, 5].astype(float)
    base, srms = fit_baseline(vel, spec, wmask, 1)
    np.testing.assert_allclose(out[:, 5, 5], (spec - base)[i0:i1], atol=1e-4)
    assert abs(rms[5, 5] - srms) < 1e-4
    # the linear baseline is removed: line-free channels average to ~0, the line peak is preserved
    line_free = (vel[i0:i1] < LINE_V - 3 * LINE_FWHM)
    assert abs(np.nanmean(out[line_free])) < 0.01
    assert np.nanmax(out[:, :, -1]) > 2.5
