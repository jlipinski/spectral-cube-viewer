"""
Cube access and cube-wide operations (no GUI).

Cubes are handled as arrays of shape (nchan, ny, nx), either in RAM or memory-mapped; all cube-wide
operations work in chunks of CHUNK_BYTES, so cubes larger than the available memory can be used.
"""

import re
import warnings

import numpy as np
from numpy.polynomial import legendre
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube

for _name in ('K (Tmb)', 'K (T_MB)', 'K (TMB)'):
    u.add_enabled_units(u.def_unit([_name], represents=u.Kelvin))

CHUNK_BYTES = 64 * 1024 ** 2         # size of the blocks processed at once in cube-wide operations
FALLBACK_RAM_BUDGET = 2 * 1024 ** 3  # max cube size loaded into RAM when psutil is unavailable
GILDAS_BLANK = -1000.0               # blanking value written by GILDAS/CLASS, often not declared in the header

# ----------------------------------------------------------------------------------------------
# Cube operations (chunked, work on in-memory or memory-mapped arrays of shape (nchan, ny, nx))
# ----------------------------------------------------------------------------------------------

class Cancelled(Exception):
    """Raised by a progress callback when the user presses Cancel."""


def chunk_ranges(start, stop, step):
    step = max(1, int(step))
    for i in range(start, stop, step):
        yield i, min(i + step, stop)


def _report(progress, frac):
    if progress is not None:
        progress(frac)


def ram_budget():
    try:
        import psutil
        return 0.5 * psutil.virtual_memory().available
    except Exception:
        return FALLBACK_RAM_BUDGET


class LazyCube:
    """
    Read-on-access view of the raw FITS data (usually memory-mapped): slicing returns native float32
    with BSCALE/BZERO applied and BLANK / GILDAS -1000 blanks turned into NaN.
    """

    def __init__(self, raw, bscale=1.0, bzero=0.0, blank=None):
        self.raw, self.bscale, self.bzero, self.blank = raw, float(bscale), float(bzero), blank
        self.scaled = self.bscale != 1.0 or self.bzero != 0.0
        self.shape, self.ndim, self.size = raw.shape, raw.ndim, raw.size
        self.dtype = np.dtype(np.float32)

    def __getitem__(self, key):
        r = np.asarray(self.raw[key])
        out = (r * self.bscale + self.bzero).astype(np.float32) if self.scaled else r.astype(np.float32)
        if self.blank is not None:
            out[r == self.blank] = np.nan
        out[np.abs(out - GILDAS_BLANK) < 0.01] = np.nan
        return out


def open_cube(path):
    """
    Open a FITS cube without reading the data. Returns (hdul, cube, data): `cube` is a SpectralCube used
    for metadata only, `data` a LazyCube of shape (nchan, ny, nx).
    """
    # do_not_scale_image_data keeps scaled-integer cubes (BSCALE/BZERO/BLANK) memory-mappable
    hdul = fits.open(path, memmap=True, do_not_scale_image_data=True)
    try:
        hdu = next(h for h in hdul if h.is_image and h.header.get('NAXIS', 0) >= 3)
        hdr = hdu.header
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cube = SpectralCube.read(hdul, hdu=hdul.index(hdu))
            spec_axis = WCS(hdr).wcs.spec
        scale = dict(bscale=hdr.get('BSCALE', 1.0), bzero=hdr.get('BZERO', 0.0),
                     blank=hdr.get('BLANK') if hdu.data.dtype.kind in 'iu' else None)
        raw = hdu.data
        while raw.ndim > 3 and raw.shape[0] == 1:   # degenerate Stokes axis
            raw = raw[0]
        if raw.shape != cube.shape or spec_axis != 2:
            # unusual axis order: take spectral_cube's re-oriented array (reads integer data into RAM)
            raw = cube.unmasked_data[:].value
        return hdul, cube, LazyCube(raw, **scale)
    except Exception:
        hdul.close()
        raise


def moment0(data, i0, i1, dv, progress=None):
    """sum(T * dv) over channels i0:i1 ignoring NaNs; NaN where a pixel has no valid channel."""
    ny, nx = data.shape[1:]
    acc = np.zeros((ny, nx))
    nvalid = np.zeros((ny, nx), dtype=np.int64)
    for c0, c1 in chunk_ranges(i0, i1, CHUNK_BYTES // (4 * ny * nx)):
        blk = np.asarray(data[c0:c1], dtype=np.float32)
        finite = np.isfinite(blk)
        acc += np.tensordot(dv[c0:c1], np.where(finite, blk, 0), axes=1)
        nvalid += finite.sum(axis=0)
        _report(progress, (c1 - i0) / (i1 - i0))
    acc[nvalid == 0] = np.nan
    return acc


def region_mean_spectrum(data, mask, progress=None):
    """NaN-ignoring mean spectrum over the pixels where mask (ny, nx) is True."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError('The region contains no pixel centres.')
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = mask[y0:y1, x0:x1]
    nchan = data.shape[0]
    out = np.empty(nchan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN channels
        for c0, c1 in chunk_ranges(0, nchan, CHUNK_BYTES // (4 * sub.size)):
            blk = np.asarray(data[c0:c1, y0:y1, x0:x1], dtype=np.float32)[:, sub]
            out[c0:c1] = np.nanmean(blk, axis=1, dtype=np.float64)
            _report(progress, c1 / nchan)
    return out


# ----------------------------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------------------------

def parse_windows(text):
    """'-80:-40, 10:50' -> [(-80.0, -40.0), (10.0, 50.0)] (km/s)."""
    windows = []
    for part in re.split(r'[,;]', text):
        part = part.strip()
        if not part:
            continue
        bits = part.split(':')
        if len(bits) != 2:
            raise ValueError(f'Bad window "{part}": use "vlo:vhi" pairs separated by commas.')
        lo, hi = float(bits[0]), float(bits[1])
        windows.append((min(lo, hi), max(lo, hi)))
    return windows


def merge_windows(windows):
    merged = []
    for lo, hi in sorted(windows):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def format_windows(windows):
    return ', '.join(f'{lo:.2f}:{hi:.2f}' for lo, hi in windows)


def windows_mask(vel, windows):
    mask = np.zeros(vel.shape, dtype=bool)
    for lo, hi in windows:
        mask |= (vel >= lo) & (vel <= hi)
    return mask


def _baseline_basis(vel, wmask, order):
    """Legendre design matrix for all channels, velocity scaled to [-1, 1] over the windows."""
    vw = vel[wmask]
    centre, half = 0.5 * (vw.max() + vw.min()), 0.5 * (vw.max() - vw.min())
    return legendre.legvander((vel - centre) / (half or 1.0), order)


def fit_baseline(vel, y, wmask, order):
    """Fit a polynomial to y in the windows; return (baseline on all channels, rms in the windows)."""
    good = wmask & np.isfinite(y)
    if good.sum() <= order + 1:
        raise ValueError(f'Need more than {order + 1} valid channels inside the baseline windows.')
    basis = _baseline_basis(vel, good, order)
    coef, *_ = np.linalg.lstsq(basis[good], y[good], rcond=None)
    baseline = basis @ coef
    return baseline, np.std(y[good] - baseline[good], ddof=order + 1)


def baseline_cube(data, vel, wmask, order, progress=None):
    """
    Subtract a polynomial baseline from every spectrum of the cube.

    Only the channels spanned by the windows (first to last window channel) are kept, so the
    result is usually much smaller than the input. Returns (corrected sub-cube, rms map, i0, i1).
    Spectra with no valid window channel become NaN.
    """
    idx = np.nonzero(wmask)[0]
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    nchan, (ny, nx) = i1 - i0, data.shape[1:]
    wm = wmask[i0:i1]
    nwin = int(wm.sum())
    if nwin <= order + 1:
        raise ValueError(f'Need more than {order + 1} channels inside the baseline windows.')
    basis = _baseline_basis(vel[i0:i1], wm, order)
    basis_w = basis[wm]
    solver = np.linalg.pinv(basis_w)            # same channels for every pixel -> one pseudo-inverse

    out = np.empty((nchan, ny, nx), dtype=np.float32)
    rms = np.full((ny, nx), np.nan, dtype=np.float32)
    rows = CHUNK_BYTES // (8 * nchan * nx)
    for y0, y1 in chunk_ranges(0, ny, rows):
        blk = np.asarray(data[i0:i1, y0:y1, :], dtype=np.float64).reshape(nchan, -1)
        yw = blk[wm]
        good = np.isfinite(yw).all(axis=0)
        if good.any():
            blk[:, good] -= basis @ (solver @ yw[:, good])
        rms_blk = np.full(blk.shape[1], np.nan)
        rms_blk[good] = np.std(blk[wm][:, good], axis=0, ddof=order + 1)
        for j in np.nonzero(~good)[0]:          # spectra with NaNs inside the windows
            fin = np.isfinite(yw[:, j])
            if fin.sum() > order + 1:
                coef, *_ = np.linalg.lstsq(basis_w[fin], yw[fin, j], rcond=None)
                blk[:, j] -= basis @ coef
                rms_blk[j] = np.std(blk[wm][fin, j], ddof=order + 1)
            else:
                blk[:, j] = np.nan
        out[:, y0:y1, :] = blk.reshape(nchan, y1 - y0, nx)
        rms[y0:y1, :] = rms_blk.reshape(y1 - y0, nx)
        _report(progress, y1 / ny)
    return out, rms, i0, i1
