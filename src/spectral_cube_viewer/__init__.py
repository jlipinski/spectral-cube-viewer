"""
Spectral Cube Viewer: interactive moment-0 maps, spectra, baseline correction and (hyperfine) line fitting
for radio spectral-line FITS cubes.

The GUI lives in :mod:`spectral_cube_viewer.gui` (``spectral-cube-viewer`` command); the numerical parts can be
used without Qt from :mod:`spectral_cube_viewer.core` (cube access, moment 0, region averages, baselines),
:mod:`spectral_cube_viewer.fitting` (line models and fitting) and :mod:`spectral_cube_viewer.lines` (line catalogs).
"""

__version__ = '0.1.0'
