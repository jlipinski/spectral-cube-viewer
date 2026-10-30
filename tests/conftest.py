"""Shared fixtures: small synthetic cubes written to temporary FITS files."""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')   # GUI tests without a display

import numpy as np
import pytest
from astropy.io import fits

from spectral_cube_viewer.fitting import gaussian

NCHAN, NY, NX = 400, 12, 10
LINE_V, LINE_FWHM, LINE_T = 5.0, 2.0, 3.0        # km/s, km/s, K


def _header(nchan=NCHAN, ny=NY, nx=NX):
    h = fits.Header()
    h['CTYPE1'], h['CRVAL1'], h['CDELT1'], h['CRPIX1'], h['CUNIT1'] = 'GLON-CAR', 120.0, -0.01, nx / 2, 'deg'
    h['CTYPE2'], h['CRVAL2'], h['CDELT2'], h['CRPIX2'], h['CUNIT2'] = 'GLAT-CAR', 0.5, 0.01, ny / 2, 'deg'
    h['CTYPE3'], h['CRVAL3'], h['CDELT3'], h['CRPIX3'], h['CUNIT3'] = 'VRAD', 0.0, 100.0, nchan / 2 + 1, 'm/s'
    h['RESTFRQ'] = 88.6318473e9                     # HCN 1-0 F=2-1
    h['SPECSYS'] = 'LSRK'
    h['BUNIT'] = 'K'
    return h


def synthetic_data(seed=0, noise=0.05):
    """(data, vel): Gaussian line whose peak scales across the map, linear baseline, NaN border column."""
    rng = np.random.default_rng(seed)
    vel = (np.arange(NCHAN) + 1 - (NCHAN / 2 + 1)) * 0.1           # km/s, matches _header
    yy, xx = np.mgrid[:NY, :NX]
    amp = LINE_T * (0.5 + 0.5 * xx / (NX - 1))                    # 1.5 .. 3 K
    line = gaussian(vel[:, None, None], amp[None], LINE_V, LINE_FWHM)
    base = 0.2 + 0.01 * vel[:, None, None]
    data = (line + base + rng.normal(0, noise, (NCHAN, NY, NX))).astype(np.float32)
    data[:, :, 0] = np.nan
    return data, vel


@pytest.fixture
def cube_file(tmp_path):
    data, vel = synthetic_data()
    path = tmp_path / 'synthetic.fits'
    fits.PrimaryHDU(data, header=_header()).writeto(path)
    return path, data, vel


@pytest.fixture
def scaled_cube_file(tmp_path):
    """GILDAS-style file: int32 with BSCALE/BZERO/BLANK and undeclared -1000 blanks."""
    data, vel = synthetic_data()
    data[:20, 3, 3] = -1000.0
    bscale, bzero, blank = 1e-5, 0.0, -2147483648
    raw = np.round(np.nan_to_num(data, nan=0.0) / bscale).astype(np.int32)
    raw[np.isnan(data)] = blank
    h = _header()
    hdu = fits.PrimaryHDU(raw, header=h)
    hdu.header['BSCALE'], hdu.header['BZERO'], hdu.header['BLANK'] = bscale, bzero, blank
    path = tmp_path / 'scaled.fits'
    hdu.writeto(path)
    expected = data.copy()
    expected[:20, 3, 3] = np.nan
    return path, expected, vel
